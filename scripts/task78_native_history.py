"""Recover student inputs from the actual native rollout, without a second game path."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

POLICY = 'native_student_trace_v1'


def json_digest(value):
    def numbers(item):
        if isinstance(item, float) and item.is_integer():
            return int(item)
        if isinstance(item, dict):
            return {k: numbers(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [numbers(v) for v in item]
        return item
    return hashlib.sha256(json.dumps(numbers(value), sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def context_digest(record):
    """The previous action contributes its plant flag, exactly as student.features does."""
    result = deepcopy(record)
    action = result.get('previous_action') or {}
    result['previous_action'] = dict(type='plant' if action.get('type') == 'plant' else 'wait')
    if result.get('previous_frame') is not None:
        action = result['previous_frame'].get('action') or {}
        result['previous_frame']['action'] = dict(type='plant' if action.get('type') == 'plant' else 'wait')
    return json_digest(result)


def recover(trace_path, initial_state, root_frame, leaf, response, context, obs_version,
            *, identity_packet=False):
    """Verify history-dependent input bits while recovering the history they encode.

    TRACE records each pre-action input. The last action's accepted-plant flag is
    total native plant actions minus the preceding flags already present in TRACE.
    These are the two previous-action inputs consumed by pvz_rollout_policy.features;
    no packet/cell or unrecorded physical action is invented.
    """
    import numpy as np
    from pvz_rollout_policy import features, identity_scalars, SCALARS
    from pvz_student_observation import StudentHistory
    from render_episode import _frame
    from value_teacher_feasibility import history_record

    data = Path(trace_path).read_bytes()
    rows = [json.loads(line) for line in data.splitlines()]
    if len(rows) != response['decisions'] or not rows:
        raise ValueError('native trace decision coverage differs')
    if len(response['substeps']) != len(rows):
        raise ValueError('native trace substep coverage differs')
    if obs_version not in ('v1', 'v2lite', 'v2', 'v2fog'):
        raise ValueError('unsupported native student observation version')
    history = deepcopy(initial_state.get('history'))
    if obs_version != 'v1' and history is None:
        history = StudentHistory(obs_version)
    state = {}
    flags = []
    compared = 0
    previous_tick = -1
    for index, row in enumerate(rows):
        frame = _frame(row['observation'])
        frame['sun_income_rate'] = row['income']
        flag = row['previous_plant']
        if flag not in (0, 1) or row['previous_tick'] != previous_tick:
            raise ValueError(f'native trace action context differs at {index}')
        frame['action'] = dict(type='plant' if flag else 'wait')
        previous = dict(tick=previous_tick) if previous_tick >= 0 else None
        board, scalars = (history.features(frame, previous, context) if history is not None
                          else features(frame, previous, context))
        if identity_packet:
            scalars = identity_scalars(frame, scalars)
        scalar_indices = [SCALARS, SCALARS + 1]
        if obs_version == 'v2lite':
            scalar_indices.append(len(scalars) - 1)
        elif obs_version != 'v1':
            scalar_indices.append(len(scalars) - 2)
        if board.numel() != len(row['board']) or len(scalars) != len(row['scalars']):
            raise ValueError('native student input shape differs')
        pairs = [(scalars[scalar_indices], np.asarray(row['scalars'])[scalar_indices])]
        if obs_version == 'v2lite':
            pairs.append((board[-1].flatten(), np.asarray(row['board'])[-board[-1].numel():]))
        for encoded, native in pairs:
            actual = encoded.numpy().astype(np.float32, copy=False).view(np.uint32)
            expected = np.asarray(native, dtype=np.float32).view(np.uint32)
            if actual.shape != expected.shape or not np.array_equal(actual, expected):
                raise ValueError(f'native student input bits differ at decision {index}: '
                                 f'shapes {actual.shape} / {expected.shape}')
            compared += actual.size
        if index == 0:
            start = deepcopy(frame); start.pop('action')
            if json_digest(start) != json_digest(root_frame):
                raise ValueError('native trace starts from another root frame')
        if index and frame['tick'] != response['substeps'][index-1][0]:
            raise ValueError('native trace timeline differs from native substeps')
        previous_tick = frame['tick']
        flags.append(int(flag))
        state['frame'] = frame
    if flags[0] != 0 or response['substeps'][-1][0] != leaf['tick']:
        raise ValueError('native trace boundary context differs')
    final_plant = sum(response['planted_slots']) - sum(flags[1:])
    if final_plant not in (0, 1):
        raise ValueError('native final action context is inconsistent')
    state['action'] = dict(type='plant' if final_plant else 'wait')
    if history is not None:
        state['history'] = history
    proof = dict(policy=POLICY, trace_sha256=hashlib.sha256(data).hexdigest(),
                 obs_version=obs_version, identity_packet=bool(identity_packet),
                 context_dimensions=0 if context is None else len(context),
                 decisions=len(rows), compared_float32=compared,
                 initial_history_sha256=json_digest(history_record(initial_state)),
                 context_sha256=context_digest(history_record(state)),
                 root_tick=root_frame['tick'], leaf_tick=leaf['tick'],
                 previous_tick=previous_tick, previous_plant=bool(final_plant),
                 planted_slots=response['planted_slots'], history_input_bits_equal=True,
                 checked_input_scope='temporal_history_and_previous_action_context',
                 previous_action_context_only=True)
    return state, proof


def audit_record(record, root, directory):
    """Bind a collected proof to its source and lossless trace; do not rerun recovery."""
    proof = record.get('native_history', {})
    if record.get('history_source') != POLICY or proof.get('policy') != POLICY:
        raise ValueError('native student history proof differs from its leaf or root')
    version = (root['history'].get('history') or {}).get('obs', 'v1')
    decisions = proof.get('decisions')
    per_decision = 2 + (version != 'v1') + (60 if version == 'v2lite' else 0)
    previous = record['leaf_history']['previous_frame']
    expected_trace = f"native_candidate_{record['candidate']}.jsonl.gz"
    if (record.get('history_source') != POLICY or proof.get('policy') != POLICY
            or record['history_checks'].get('native_history') is not True
            or proof.get('history_input_bits_equal') is not True
            or proof.get('checked_input_scope') != 'temporal_history_and_previous_action_context'
            or type(decisions) is not int or decisions < 1
            or proof.get('compared_float32') != decisions * per_decision
            or proof.get('obs_version') != version
            or proof.get('initial_history_sha256') != json_digest(root['history'])
            or proof.get('context_sha256') != context_digest(record['leaf_history'])
            or proof.get('root_tick') != root['root_frame']['tick']
            or proof.get('leaf_tick') != record['leaf_frame']['tick']
            or proof.get('previous_tick') != previous['tick']
            or type(proof.get('previous_plant')) is not bool
            or proof['previous_plant'] != (record['leaf_history']['previous_action'].get('type') == 'plant')
            or proof.get('trace_file') != expected_trace):
        raise ValueError('native student history proof differs from its leaf or root')
    path = Path(directory) / expected_trace
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != proof.get('packed_trace_sha256'):
        raise ValueError('native student trace changed after history recovery')
    return {str(path): digest}
