"""Frozen real-runner contracts, resumability and corrupted episode rejection (no engines)."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from pvz_promotion import SEED_POOLS_V2, V5_MONITOR_TASKS
from test_curriculum_pipeline import setup
import promotion_gate as gate


V5_UNITS = 80      # 40 tasks x 2 sides


def arguments(root):
    common, _ = setup(root)
    candidate = root / 'candidate.pt'
    candidate.write_bytes(b'candidate')
    return SimpleNamespace(candidate=candidate, incumbent=common.rollout_policy,
        source_panel=root / 'panel.json', executable=common.executable, value=common.value,
        resource_dir=common.resource_dir, workers=1, out=root / 'evaluation',
        gate_version=5, seed_ledger=root / 'gate_ledger.json')


def fake_games(args, command, **kwargs):
    # Preregistration must exist before the first episode starts.
    registration = gate.read(args.out / 'registration.json')
    assert registration['version'] == 5
    assert registration['policy']['pair_count'] == 1920
    block = registration['seed_block']
    assert block[1] - block[0] == 1920
    assert any(low <= block[0] and block[1] <= high for low, high in SEED_POOLS_V2)
    assert command[command.index('--arms') + 1] == 'fast'
    assert command[command.index('--dagger-beta') + 1] == '0'
    assert '--frames' not in command
    model = command[command.index('--rollout-policy') + 1]
    name = command[command.index('--task-id') + 1]
    folder = Path(command[command.index('--out') + 1])
    task = next(task for task in gate.read(args.out / 'manifest.json')['tasks'] if task['task_id'] == name)
    (folder / 'games.jsonl').write_text('\n'.join(json.dumps(dict(seed=seed, arm='fast', searched=0,
        won=(model == str(args.candidate)) != (name in V5_MONITOR_TASKS))) for seed in task['seeds']))
    return SimpleNamespace(returncode=0)


def test_systemic_unit_failure_keeps_engine_diagnostics(tmp_path):
    args = arguments(tmp_path)
    task = dict(task_id='failed', seeds=[1])
    marker = 'Inconsistency detected by ld.so: Assertion failed'

    def crash(command, **kwargs):
        kwargs['stdout'].write(marker + '\n')
        return SimpleNamespace(returncode=127)

    with (tmp_path / 'lock').open('w') as lock, \
         patch.object(gate.subprocess, 'run', side_effect=crash):
        with pytest.raises(RuntimeError, match='Inconsistency detected by ld.so'):
            gate._run_unit(args, lock, args.source_panel, task, 'candidate', args.candidate)


def test_v5_panel_issues_and_resumes_a_certificate(tmp_path):
    args = arguments(tmp_path)
    with patch.object(gate.subprocess, 'run', side_effect=lambda cmd, **kw: fake_games(args, cmd, **kw)) as run:
        result = gate.evaluate(args)
        assert run.call_count == V5_UNITS                 # 40 tasks, both sides
        assert result['approved'] and result['overall']['pairs'] == 1728
        assert result['zones']['roof']['pairs'] == 90
        assert all(cell['harm'] == 96 for cell in result['monitor'].values())
        assert gate.valid_certificate(result, result['contract'])
        gate.evaluate(args)
        assert run.call_count == V5_UNITS                 # resume uses the issued block


def test_complete_evaluation_resumes_and_rejects_incomplete_episode_files():
    with tempfile.TemporaryDirectory() as folder:
        args = arguments(Path(folder))
        with patch.object(gate.subprocess, 'run', side_effect=lambda cmd, **kw: fake_games(args, cmd, **kw)) as run:
            result = gate.evaluate(args)
            assert run.call_count == V5_UNITS
            assert result['approved'] and len(result['paired']) == 1920
            assert gate.valid_certificate(result, result['contract'])
            gate.evaluate(args)
            assert run.call_count == V5_UNITS  # exact coverage reuses episodes without rerunning any game
        for field, value in [('approved', False), ('overall', {})]:
            broken = dict(result)
            broken[field] = value
            assert not gate.valid_certificate(broken, result['contract'])
        path = next((args.out / 'candidate').glob('*/games.jsonl'))
        lines = path.read_text().splitlines()
        lines[-1] = lines[0]  # same row count, but one missing and one duplicated seed
        path.write_text('\n'.join(lines))
        with patch.object(gate.subprocess, 'run', side_effect=AssertionError('must reject corruption')), pytest.raises(ValueError):
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
                assert run.call_count == V5_UNITS
            outcomes[label] = result
        a, b = outcomes['serial'], outcomes['parallel']
        assert a['approved'] and b['approved']
        assert a['paired'] == b['paired']
        assert a['contract'] == b['contract']
        for key in ('overall', 'zones', 'monitor', 'reasons', 'version', 'policy'):
            assert a[key] == b[key], key
        assert gate.valid_certificate(b, b['contract'])


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
        # Every unit ran exactly once: 80 unit folders, one games file each.
        assert len(list(args.out.glob('*/*/games.jsonl'))) == V5_UNITS


# ---- ledger allocation, exhaustion, resume ----

def test_seed_ledger_hands_out_fresh_blocks_and_resume_never_reallocates():
    with tempfile.TemporaryDirectory() as folder:
        args = arguments(Path(folder))
        with patch.object(gate.subprocess, 'run', side_effect=lambda cmd, **kw: fake_games(args, cmd, **kw)) as run:
            first = gate.evaluate(args)
            assert min(row['seed'] for row in first['paired']) == 31024
            assert max(row['seed'] for row in first['paired']) == 31024 + 1920 - 1
            # both sides played the fresh seeds: no cached incumbent outcome exists
            assert run.call_count == V5_UNITS
            incumbent_rows = (args.out / 'incumbent/task101_l1_day_d7/games.jsonl').read_text().splitlines()
            assert len(incumbent_rows) == 46
            gate.evaluate(args)  # resumed run: same contract, no new block
        ledger = gate.read(args.seed_ledger)
        assert ledger['allocated'] == [[31024, 31024 + 1920]]
        args2 = arguments(Path(folder) / 'second-run')
        args2.seed_ledger = args.seed_ledger  # same pool ledger across rounds
        args2.out = Path(folder) / 'evaluation2'
        with patch.object(gate.subprocess, 'run', side_effect=lambda cmd, **kw: fake_games(args2, cmd, **kw)):
            second = gate.evaluate(args2)
        ledger = gate.read(args.seed_ledger)
        assert ledger['allocated'] == [[31024, 31024 + 1920], [31024 + 1920, 31024 + 3840]]
        assert not ({row['seed'] for row in first['paired']} & {row['seed'] for row in second['paired']})
        assert gate.valid_certificate(second, second['contract'])


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


def test_ledger_refuses_a_different_pool_list_and_overlaps():
    with tempfile.TemporaryDirectory() as folder:
        ledger = Path(folder) / 'gate_ledger.json'
        ledger.write_text(json.dumps(dict(pools=[[31024, 65536], [200_000_000, 201_000_000]],
                                          allocated=[[31024, 32560]])))
        with pytest.raises(ValueError, match='was created for pools'):
            gate.allocate_seed_block(ledger, 1920)
        ledger.write_text(json.dumps(dict(pools=[list(p) for p in SEED_POOLS_V2],
                                          allocated=[[31024, 32560], [32000, 34000]])))
        with pytest.raises(ValueError, match='overlapping'):
            gate.allocate_seed_block(ledger, 1920)


def test_two_hundred_fifty_rounds_are_pairwise_disjoint_across_both_windows():
    """Dry-run shape of task88 step 4: 250 rounds (1920 seeds each) from a copy."""
    import pvz_promotion
    with tempfile.TemporaryDirectory() as folder:
        ledger = Path(folder) / 'gate_ledger.json'
        first = pvz_promotion.SEED_POOLS_V2[0]
        # Start near the end of the first window, as the live ledger is on 2026-10-26.
        ledger.write_text(json.dumps(dict(pools=[list(p) for p in SEED_POOLS_V2],
                                          allocated=[[first[0], 65000]])))
        blocks = [gate.allocate_seed_block(ledger, 1920) for _ in range(250)]
        spans = sorted(blocks)
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
        assert len({low for low, _ in blocks}) == 250
        assert sum(high - low for low, high in blocks) == 250 * 1920
        assert any(low >= pvz_promotion.SEED_POOLS_V2[1][0] for low, _ in blocks)
        assert gate.read(ledger)['allocated'][-1] == list(blocks[-1])
