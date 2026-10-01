"""Fresh full reevaluation with source-verified native conveyor classification.

The failed v1 protocol and helper remain unchanged. Native sampling, original
inventory validation and paired row checks delegate to that preserved helper.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import json
from pathlib import Path
import time
import traceback

import research_packet_cost_reevaluation as original

ROOT = original.ROOT
CONVEYOR_ADVENTURE_LEVELS = (5, 10, 20, 25, 30, 40, 45, 50)


def conveyor_task_ids(tasks):
    return [t['task_id'] for t in tasks if t['level'] in CONVEYOR_ADVENTURE_LEVELS]


def persisted_inventory(inventory):
    # JSON turns integer initialization keys into strings. Compare the actual
    # on-disk representation, while preserving every state/checkpoint digest.
    return json.loads(json.dumps(inventory, allow_nan=False))


def check_scope(tasks, protocol, gate):
    expected = conveyor_task_ids(tasks)
    if protocol['allowed_changed_task_ids'] != expected or gate['conveyor_task_ids'] != expected:
        raise ValueError('scope must include exactly native conveyor tasks, regardless of task names')
    if gate['adventure_conveyor_levels'] != list(CONVEYOR_ADVENTURE_LEVELS):
        raise ValueError('native classification proof changed')
    if (gate['result'] != 'pass' or gate['roof_native_equivalence']['paired_jobs'] != 576
            or gate['roof_native_equivalence']['recorded_traces'] != 32
            or not gate['roof_policy_repeatability']['guarded32_all_observation_output_action_rng_exact']):
        raise ValueError('complete roof native and fixed-policy evidence required')


def checked_protocol(path, resource_dir):
    protocol = json.loads(path.read_text())
    if original.sha256_file(Path(__file__)) != protocol['helper_sha256']:
        raise ValueError('v2 helper changed after preregistration')
    for name, expected in protocol['required_fingerprints'].items():
        if original.sha256_file(ROOT / name) != expected:
            raise ValueError(f'frozen reevaluation source/evidence changed: {name}')
    for name, expected in protocol['resource_fingerprints'].items():
        if original.sha256_file(resource_dir / name) != expected:
            raise ValueError(f'frozen resource changed: {name}')
    for path in protocol['required_gates']:
        if json.loads((ROOT / path).read_text()).get('result') != 'pass':
            raise ValueError(f'native prerequisite failed: {path}')
    gate = json.loads((ROOT / protocol['classification_gate']).read_text())
    if gate['simulator_sha256'] != original.sha256_file(Path(protocol['executable'])):
        raise ValueError('classification gate belongs to another native binary')
    for name, expected in gate['required_fingerprints'].items():
        if original.sha256_file(ROOT / name) != expected:
            raise ValueError(f'native classification source/evidence changed: {name}')
    queue = json.loads((ROOT / protocol['queue']).read_text())
    original.storage_revision(queue)
    if len(queue['order']) != 12 or queue['required_initializations'] != [0, 1, 2]:
        raise ValueError('all original reward/initialization candidates required')
    tasks = json.loads((ROOT / protocol['evaluation_manifest']).read_text())['tasks']
    if (len(tasks) != 35 or any(len(t['seeds']) != 64 for t in tasks)
            or [t['task_id'] for t in tasks] != protocol['task_ids']):
        raise ValueError('all original tasks and seeds required')
    check_scope(tasks, protocol, gate)
    prior = json.loads((ROOT / protocol['previous_protocol']).read_text())
    for key in ('queue', 'queue_state', 'evaluation_manifest', 'task_ids', 'modes', 'runtime', 'executable',
                'original_comparable_fingerprints', 'resource_fingerprints', 'already_existing_checkpoints'):
        if protocol[key] != prior[key]:
            raise ValueError(f'v2 changed science or runtime beyond native classification: {key}')
    return protocol, queue, tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--resource-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    output, summary = args.output_dir.resolve(), args.summary.resolve()
    output.relative_to(ROOT)
    summary.relative_to(ROOT)
    if not args.resume and (output.exists() or summary.exists()):
        raise ValueError('fresh v2 evidence paths required; v1 is not reused')
    protocol, queue, tasks = checked_protocol(args.protocol, args.resource_dir.resolve())
    if (json.loads((ROOT / protocol['queue_state']).read_text())['status'] != 'matrix_budget_complete'
            or not original.matrix_ready(queue)):
        raise ValueError('complete original matrix required; no workers started')
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.execution.lock').open('a') as lock, ExitStack() as originals:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = original.sha256_file(args.protocol)
        state = {'protocol_sha256': identity, 'status': 'preflight', 'results': [], 'invocations': []}
        if args.resume:
            state = json.loads((output / 'state.json').read_text())
            if state['protocol_sha256'] != identity or state['status'] in ('complete', 'failed'):
                raise ValueError('resume only same interrupted unfinished v2; stopped gates remain stopped')
        state['invocations'].append({'commit': original.git_metadata(ROOT)[0], 'started_ns': time.time_ns()})
        original.publish(output, summary, state, protocol)
        try:
            for entry in queue['order']:
                run_lock = originals.enter_context((ROOT / entry['output_dir'] / '.execution.lock').open('a'))
                fcntl.flock(run_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            checked_protocol(args.protocol, args.resource_dir.resolve())
            inventory = persisted_inventory(original.make_inventory(queue, protocol, tasks))
            if inventory != json.loads((ROOT / protocol['previous_inventory']).read_text()):
                raise ValueError('any original frozen state/checkpoint/raw changed since failed v1')
            inventory_path = output / 'inventory.json'
            if inventory_path.exists():
                if json.loads(inventory_path.read_text()) != inventory:
                    raise ValueError('inventory changed during interruption')
            else:
                original.atomic_json(inventory_path, inventory)
            state.update(inventory_sha256=original.sha256_file(inventory_path), status='reevaluating')
            original.publish(output, summary, state, protocol)
            for index, node in enumerate(inventory['nodes']):
                if index < len(state['results']):
                    prior = state['results'][index]
                    if (prior['experiment_id'] != node['experiment_id'] or prior['node_index'] != node['node_index']
                            or original.sha256_file(ROOT / prior['raw_path']) != prior['raw_sha256']
                            or prior['changes']['unexpected_changed_rows']):
                        raise ValueError('cannot skip invalid completed evaluation evidence')
                    continue
                result = original.evaluate_node(node, protocol, tasks, args.resource_dir.resolve(), output)
                state['results'].append(result)
                original.publish(output, summary, state, protocol)
                if result['changes']['unexpected_changed_rows']:
                    raise ValueError('non-conveyor rows changed; stop before capability acceptance')
                print(f"completed v2 guarded node {index+1}/60 changes={result['changes']}", flush=True)
            for path, expected in inventory['original_states'].items():
                if original.sha256_file(ROOT / path) != expected:
                    raise ValueError('original state changed during readonly evaluation')
            state['status'] = 'complete'
            original.publish(output, summary, state, protocol)
            original.atomic_json(output / 'report.json', state)
        except BaseException:
            state['status'] = 'failed'
            original.atomic_json(output / f'failure_{time.time_ns()}.json',
                {'error': traceback.format_exc(), 'protocol_sha256': identity, 'completed_nodes': len(state['results'])})
            original.publish(output, summary, state, protocol)
            raise


if __name__ == '__main__':
    main()
