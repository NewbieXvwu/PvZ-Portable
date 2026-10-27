"""Frozen real-runner contracts, resumability and corrupted episode rejection (no engines)."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from pvz_promotion import PANEL_TASKS, PANEL_TASKS_V3, SEED_POOLS_V2
from test_curriculum_pipeline import setup
import promotion_gate as gate


V3_UNITS = 20      # 10 tasks x 2 sides
V2_UNITS = 16      # 8 tasks x 2 sides


def arguments(root):
    common, _ = setup(root)
    candidate = root / 'candidate.pt'
    candidate.write_bytes(b'candidate')
    # These tests issue v3/v4 certificates, whose panel membership is version-pinned.
    panel = root / 'v3_panel.json'
    panel.write_text(json.dumps(dict(panel_version='v2', groups={name: zone for zone, name in PANEL_TASKS_V3},
        tasks=[dict(task_id=name, seeds=list(range(30000, 30128)), wave_cap=None,
                    zombie_count_multiplier=1, level=1, deck=[0, 1, 2, 3, 4, 5]) for _, name in PANEL_TASKS_V3])))
    return SimpleNamespace(candidate=candidate, incumbent=common.rollout_policy,
        source_panel=panel, executable=common.executable, value=common.value,
        resource_dir=common.resource_dir, workers=1, out=root / 'evaluation',
        gate_version=3, seed_ledger=root / 'gate_ledger.json')


def fake_games(args, command, **kwargs):
    # Preregistration must exist before the first episode starts.
    registration = gate.read(args.out / 'registration.json')
    if registration['version'] in (3, 4, 5):
        assert registration['policy']['pair_count'] == 1920
        block = registration['seed_block']
        assert block[1] - block[0] == 10 * 192
        assert any(low <= block[0] and block[1] <= high for low, high in SEED_POOLS_V2)
    elif registration['version'] == 2:
        assert registration['policy']['pair_count'] == 1536
        block = registration['seed_block']
        assert block[1] - block[0] == 8 * 192
        assert any(low <= block[0] and block[1] <= high for low, high in SEED_POOLS_V2)
    else:
        assert registration['policy']['pair_count'] == 64
    assert command[command.index('--arms') + 1] == 'fast'
    assert command[command.index('--dagger-beta') + 1] == '0'
    assert '--frames' not in command
    model = command[command.index('--rollout-policy') + 1]
    name = command[command.index('--task-id') + 1]
    folder = Path(command[command.index('--out') + 1])
    task = next(task for task in gate.read(args.out / 'manifest.json')['tasks'] if task['task_id'] == name)
    (folder / 'config.json').write_text(json.dumps(dict(rollout_policy=model, task_id=name, arms='fast',
        dagger_beta='0', executable=str(args.executable), workers='1')))
    (folder / 'games.jsonl').write_text('\n'.join(json.dumps(dict(seed=seed, arm='fast', searched=0,
        won=(model == str(args.candidate)) != (registration['version'] == 5 and
            name in gate.read(args.out / 'manifest.json').get('monitor', [])))) for seed in task['seeds']))
    return SimpleNamespace(returncode=0)


def test_v5_roof_panel_issues_resumes_and_records_heldout_collapse(tmp_path):
    args = arguments(tmp_path)
    args.gate_version = 5
    args.source_panel = Path(__file__).resolve().parents[1] / 'configs/gate_v5_panel.json'
    with patch.object(gate.subprocess, 'run', side_effect=lambda cmd, **kw: fake_games(args, cmd, **kw)) as run:
        result = gate.evaluate(args)
        assert run.call_count == 80                       # 40 tasks, both sides
        assert result['approved'] and result['overall']['pairs'] == 1728
        assert result['zones']['roof']['pairs'] == 90
        assert all(cell['harm'] == 96 for cell in result['monitor'].values())
        assert gate.valid_certificate(result, result['contract'], args.out)
        gate.evaluate(args)
        assert run.call_count == 80                       # resume uses the issued block


def test_complete_evaluation_resumes_and_rechecks_raw_episode_certificates():
    with tempfile.TemporaryDirectory() as folder:
        args = arguments(Path(folder))
        with patch.object(gate.subprocess, 'run', side_effect=lambda cmd, **kw: fake_games(args, cmd, **kw)) as run:
            result = gate.evaluate(args)
            assert run.call_count == V3_UNITS
            assert result['approved'] and len(result['paired']) == 1920
            assert len(result['episode_files_sha256']) == V3_UNITS
            assert gate.valid_certificate(result, result['contract'], args.out)
            gate.evaluate(args)
            assert run.call_count == V3_UNITS  # exact coverage reuses episodes without rerunning any game
        for field, value in [('approved', False), ('overall', {})]:
            broken = copy.deepcopy(result)
            broken[field] = value
            assert not gate.valid_certificate(broken, broken['contract'], args.out)
        path = next((args.out / 'candidate').glob('*/games.jsonl'))
        lines = path.read_text().splitlines()
        lines[-1] = lines[0]  # same row count, but one missing and one duplicated seed
        path.write_text('\n'.join(lines))
        assert not gate.valid_certificate(result, result['contract'], args.out)
        with patch.object(gate.subprocess, 'run', side_effect=AssertionError('must reject corruption')), pytest.raises(ValueError):
            gate.evaluate(args)


def test_changed_checkpoint_cannot_reuse_registered_evaluation():
    with tempfile.TemporaryDirectory() as folder:
        args = arguments(Path(folder))
        with patch.object(gate.subprocess, 'run', side_effect=lambda cmd, **kw: fake_games(args, cmd, **kw)):
            result = gate.evaluate(args)
        args.candidate.write_bytes(b'changed bytes')
        assert not gate.valid_certificate(result, result['contract'], args.out)
        with pytest.raises(ValueError, match='different model'):
            gate.evaluate(args)


def test_parallel_units_produce_the_identical_certificate_to_serial_units():
    """task73c: --eval-units is an execution detail; the decision must not move."""
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        serial = arguments(root)
        # Same models/panel paths (one contract), separate out directory and ledger.
        parallel = SimpleNamespace(**{**serial.__dict__, 'out': root / 'evaluation-parallel',
                                      'seed_ledger': root / 'parallel-ledger.json'})
        outcomes = {}
        for label, args, units in (('serial', serial, 1), ('parallel', parallel, 7)):
            with patch.object(gate.subprocess, 'run', side_effect=lambda cmd, **kw: fake_games(args, cmd, **kw)) as run:
                result = gate.evaluate(args)
                assert run.call_count == V3_UNITS
            outcomes[label] = result
        a, b = outcomes['serial'], outcomes['parallel']
        assert a['approved'] and b['approved']
        assert a['paired'] == b['paired']
        assert a['contract'] == b['contract']
        assert a['episode_files_sha256'] == b['episode_files_sha256']
        for key in a:
            if key not in ('started', 'finished'):   # wall-clock fields only
                assert a[key] == b[key], key
        assert gate.valid_certificate(b, b['contract'], parallel.out)


def test_parallel_units_cap_concurrency_and_run_every_unit_once():
    with tempfile.TemporaryDirectory() as folder:
        args = arguments(Path(folder))
        args.eval_units = 5
        constructed = []
        real_executor = gate.ThreadPoolExecutor

        class SpyExecutor(real_executor):
            def __init__(self, *constructor, **kwargs):
                constructed.append((constructor, kwargs))
                super().__init__(*constructor, **kwargs)

        with patch.object(gate, 'ThreadPoolExecutor', SpyExecutor), patch.object(
                gate.subprocess, 'run', side_effect=lambda cmd, **kw: fake_games(args, cmd, **kw)):
            result = gate.evaluate(args)
        assert constructed == [((), dict(max_workers=5, thread_name_prefix='gate-eval'))]
        assert result['approved'] and len(result['paired']) == 1920
        # Every unit ran exactly once: 20 unit folders, one command.json each.
        assert len(list(args.out.glob('*/*/command.json'))) == V3_UNITS


# ---- task73b: ledger allocation, exhaustion, incumbent rerun, v1 certificate compatibility ----

def test_seed_ledger_hands_out_fresh_blocks_and_resume_never_reallocates():
    with tempfile.TemporaryDirectory() as folder:
        args = arguments(Path(folder))
        with patch.object(gate.subprocess, 'run', side_effect=lambda cmd, **kw: fake_games(args, cmd, **kw)) as run:
            first = gate.evaluate(args)
            assert min(row['seed'] for row in first['paired']) == 31024
            assert max(row['seed'] for row in first['paired']) == 31024 + 1920 - 1
            # both sides played the fresh seeds: no cached incumbent outcome exists
            assert run.call_count == V3_UNITS
            incumbent_rows = next((args.out / 'incumbent').glob('*/games.jsonl')).read_text().splitlines()
            assert len(incumbent_rows) == 192
            gate.evaluate(args)  # resumed run: same contract, no new block
        ledger = gate.read(args.seed_ledger)
        assert ledger['pool'] == [31024, 65536] and len(ledger['allocated']) == 1
        args2 = arguments(Path(folder) / 'second-run')
        args2.seed_ledger = args.seed_ledger  # same pool ledger across rounds
        args2.out = Path(folder) / 'evaluation2'
        with patch.object(gate.subprocess, 'run', side_effect=lambda cmd, **kw: fake_games(args2, cmd, **kw)):
            second = gate.evaluate(args2)
        ledger = gate.read(args.seed_ledger)
        assert ledger['allocated'] == [[31024, 31024 + 1920], [31024 + 1920, 31024 + 3840]]
        assert not ({row['seed'] for row in first['paired']} & {row['seed'] for row in second['paired']})
        assert gate.valid_certificate(second, second['contract'], args2.out)


def test_seed_pool_exhaustion_aborts_instead_of_wrapping():
    import pvz_promotion
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        args = arguments(root)
        tiny = (31024, 31024 + 1920 + 100)
        # A single-pool registration still has to abort, not continue somewhere else.
        with patch.object(pvz_promotion, 'SEED_POOLS_V2', (tiny,)):
            gate.allocate_seed_block(args.seed_ledger, 1920)
            with pytest.raises(gate.SeedPoolExhausted, match='refusing to reuse'):
                gate.allocate_seed_block(args.seed_ledger, 1920)
            args.out = root / 'evaluation3'
            with patch.object(gate.subprocess, 'run', side_effect=AssertionError('gate must not run')):
                with pytest.raises(gate.SeedPoolExhausted):
                    gate.evaluate(args)
        assert not (args.out / 'registration.json').exists()


def test_v1_and_v2_gate_flows_still_run_and_old_certificates_stay_readable():
    """GATE_VERSION moved on to 3; the v1/v2 issuance paths and their certificates survive."""
    with tempfile.TemporaryDirectory() as folder:
        args = arguments(Path(folder))
        # The v1/v2 panel spec is the 8-task PANEL_TASKS, not the v3 superset.
        args.source_panel.write_text(json.dumps(dict(panel_version='v2',
            groups={name: zone for zone, name in PANEL_TASKS},
            tasks=[dict(task_id=name, seeds=list(range(30000, 30128)), wave_cap=None,
                        zombie_count_multiplier=1, level=1, deck=[0, 1, 2, 3, 4, 5]) for _, name in PANEL_TASKS])))
        args.gate_version = 1
        with patch.object(gate.subprocess, 'run', side_effect=lambda cmd, **kw: fake_games(args, cmd, **kw)) as run:
            result = gate.evaluate(args)
            assert result['version'] == 1 and len(result['paired']) == 64
            assert result['contract']['policy']['pair_count'] == 64
            assert 'seed_block' not in result['contract']
            assert run.call_count == V2_UNITS  # incumbent re-evaluated here too
        assert not args.seed_ledger.exists()  # v1 never touches the ledger
        assert gate.valid_certificate(result, result['contract'], args.out)
        manifest = gate.read(args.out / 'manifest.json')
        assert manifest['panel_version'] == 'v2-task49A'
        assert {row['seed'] for row in result['paired']} == set(range(30000, 30064))
        # A v2 round, issued after GATE_VERSION=3: ledger block of 1536, 16 units.
        args2 = arguments(Path(folder) / 'v2-run')
        args2.source_panel.write_text(args.source_panel.read_text())
        args2.gate_version = 2
        with patch.object(gate.subprocess, 'run', side_effect=lambda cmd, **kw: fake_games(args2, cmd, **kw)) as run:
            v2 = gate.evaluate(args2)
            assert run.call_count == V2_UNITS
        assert v2['version'] == 2 and len(v2['paired']) == 1536
        assert gate.read(args2.seed_ledger)['allocated'] == [[31024, 31024 + 1536]]
        assert gate.valid_certificate(v2, v2['contract'], args2.out)


# ---- task88: the gate continues into a second pool window instead of aborting ----

def test_allocation_falls_through_to_the_second_window_and_never_overlaps():
    import pvz_promotion
    with tempfile.TemporaryDirectory() as folder:
        ledger = Path(folder) / 'gate_ledger.json'
        first = pvz_promotion.SEED_POOLS_V2[0]
        # A first window with room for exactly two blocks of 1920.
        narrow = (first[0], first[0] + 2 * 1920)
        pools = (narrow, pvz_promotion.SEED_POOLS_V2[1])
        blocks = [gate.allocate_seed_block(ledger, 1920, pools=pools) for _ in range(3)]
        assert blocks[0] == (first[0], first[0] + 1920)
        assert blocks[1] == (first[0] + 1920, first[0] + 2 * 1920)
        assert blocks[2] == (pools[1][0], pools[1][0] + 1920)  # new window, from its start
        document = gate.read(ledger)
        assert document['pool'] == list(pools[0]) and document['pools'] == [list(p) for p in pools]
        assert document['allocated'] == [list(block) for block in blocks]
        # Blocks from different windows are disjoint and every block stays inside a pool.
        spans = sorted(blocks)
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
        assert all(any(low <= a and b <= high for low, high in pools) for a, b in blocks)


def test_legacy_single_pool_ledger_is_read_and_upgraded_in_place():
    import pvz_promotion
    with tempfile.TemporaryDirectory() as folder:
        ledger = Path(folder) / 'gate_ledger.json'
        legacy = dict(pool=[31024, 65536], allocated=[[31024, 32560], [32560, 34736]])
        ledger.write_text(json.dumps(legacy))
        block = gate.allocate_seed_block(ledger, 1920)
        assert block == (34736, 34736 + 1920)  # continues where the legacy file stopped
        document = gate.read(ledger)
        assert document['allocated'][:2] == legacy['allocated']  # nothing rewritten
        assert document['pool'] == [31024, 65536]
        assert document['pools'] == [list(p) for p in pvz_promotion.SEED_POOLS_V2]
        # The legacy key can only vouch for the first window; a different first
        # window, or a "pools" document that disagrees, is still refused.
        ledger.write_text(json.dumps(dict(pool=[20000, 30000], allocated=[])))
        with pytest.raises(ValueError, match='was created for pool'):
            gate.allocate_seed_block(ledger, 1920)
        ledger.write_text(json.dumps(dict(pool=[31024, 65536],
                                          pools=[[31024, 65536], [200_000_000, 201_000_000]],
                                          allocated=[[31024, 32560]])))
        with pytest.raises(ValueError, match='was created for pools'):
            gate.allocate_seed_block(ledger, 1920)


def test_two_hundred_fifty_rounds_are_pairwise_disjoint_across_both_windows():
    """Dry-run shape of task88 step 4: 250 v3 rounds (1920 seeds each) from a copy."""
    import pvz_promotion
    with tempfile.TemporaryDirectory() as folder:
        ledger = Path(folder) / 'gate_ledger.json'
        first = pvz_promotion.SEED_POOLS_V2[0]
        # Start near the end of the first window, as the live ledger is on 2026-10-26.
        ledger.write_text(json.dumps(dict(pool=list(first), allocated=[[first[0], 65000]])))
        blocks = [gate.allocate_seed_block(ledger, 1920) for _ in range(250)]
        spans = sorted(blocks)
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
        assert len({seed for low, high in blocks for seed in (low,)}) == 250
        assert sum(high - low for low, high in blocks) == 250 * 1920
        assert any(low >= pvz_promotion.SEED_POOLS_V2[1][0] for low, _ in blocks)
        assert gate.read(ledger)['allocated'][-1] == list(blocks[-1])


def test_v4_issues_and_resumes_a_version_pinned_certificate(tmp_path):
    args = arguments(tmp_path)
    args.gate_version = 4
    with patch.object(gate.subprocess, 'run', side_effect=lambda cmd, **kw: fake_games(args, cmd, **kw)) as run:
        result = gate.evaluate(args)
        assert result['version'] == 4 and result['approved']
        assert gate.valid_certificate(result, result['contract'], args.out)
        gate.evaluate(args)
        assert run.call_count == V3_UNITS


def test_controller_can_finish_an_old_round_using_explicit_v3(tmp_path):
    from curriculum_queue import TrainingPipeline
    common, state = setup(tmp_path)
    common.gate_version = 3
    pipeline = TrainingPipeline(common, state)
    try:
        job = dict(model=str(common.rollout_policy), seed_block=[31024, 32944])
        frozen = pipeline.evaluation_contract(job)
        assert frozen['version'] == 3 and frozen['policy'] == gate.policy(3)
    finally:
        pipeline.close()
