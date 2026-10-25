"""Frozen real-runner contracts, resumability and corrupted episode rejection (no engines)."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from test_curriculum_pipeline import setup
import promotion_gate as gate


def arguments(root):
    common, _ = setup(root)
    candidate = root / 'candidate.pt'
    candidate.write_bytes(b'candidate')
    return SimpleNamespace(candidate=candidate, incumbent=common.rollout_policy,
        source_panel=common.promotion_panel, executable=common.executable, value=common.value,
        resource_dir=common.resource_dir, workers=1, out=root / 'evaluation',
        gate_version=2, seed_ledger=root / 'gate_ledger.json')


def fake_games(args, command, **kwargs):
    # Preregistration must exist before the first episode starts.
    registration = gate.read(args.out / 'registration.json')
    if registration['version'] == 2:
        assert registration['policy']['pair_count'] == 1536
        block = registration['seed_block']
        assert block[1] - block[0] == 8 * 192 and 31024 <= block[0] and block[1] <= 65536
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
        won=model == str(args.candidate))) for seed in task['seeds']))
    return SimpleNamespace(returncode=0)


def test_complete_evaluation_resumes_and_rechecks_raw_episode_certificates():
    with tempfile.TemporaryDirectory() as folder:
        args = arguments(Path(folder))
        with patch.object(gate.subprocess, 'run', side_effect=lambda cmd, **kw: fake_games(args, cmd, **kw)) as run:
            result = gate.evaluate(args)
            assert run.call_count == 16
            assert result['approved'] and len(result['paired']) == 1536
            assert len(result['episode_files_sha256']) == 16
            assert gate.valid_certificate(result, result['contract'], args.out)
            gate.evaluate(args)
            assert run.call_count == 16  # exact coverage reuses episodes without rerunning any game
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


# ---- task73b: ledger allocation, exhaustion, incumbent rerun, v1 certificate compatibility ----

def test_seed_ledger_hands_out_fresh_blocks_and_resume_never_reallocates():
    with tempfile.TemporaryDirectory() as folder:
        args = arguments(Path(folder))
        with patch.object(gate.subprocess, 'run', side_effect=lambda cmd, **kw: fake_games(args, cmd, **kw)) as run:
            first = gate.evaluate(args)
            assert min(row['seed'] for row in first['paired']) == 31024
            assert max(row['seed'] for row in first['paired']) == 31024 + 1536 - 1
            # both sides played the fresh seeds: no cached incumbent outcome exists
            assert run.call_count == 16
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
        assert ledger['allocated'] == [[31024, 31024 + 1536], [31024 + 1536, 31024 + 3072]]
        assert not ({row['seed'] for row in first['paired']} & {row['seed'] for row in second['paired']})
        assert gate.valid_certificate(second, second['contract'], args2.out)


def test_seed_pool_exhaustion_aborts_instead_of_wrapping():
    import pvz_promotion
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        args = arguments(root)
        tiny = (31024, 31024 + 1536 + 100)
        with patch.object(pvz_promotion, 'SEED_POOL_V2', tiny):
            gate.allocate_seed_block(args.seed_ledger, 1536)
            with pytest.raises(gate.SeedPoolExhausted, match='refusing to reuse'):
                gate.allocate_seed_block(args.seed_ledger, 1536)
            args.out = root / 'evaluation3'
            with patch.object(gate.subprocess, 'run', side_effect=AssertionError('gate must not run')):
                with pytest.raises(gate.SeedPoolExhausted):
                    gate.evaluate(args)
        assert not (args.out / 'registration.json').exists()


def test_v1_gate_flow_still_runs_and_old_style_certificates_stay_readable():
    with tempfile.TemporaryDirectory() as folder:
        args = arguments(Path(folder))
        args.gate_version = 1
        with patch.object(gate.subprocess, 'run', side_effect=lambda cmd, **kw: fake_games(args, cmd, **kw)) as run:
            result = gate.evaluate(args)
            assert result['version'] == 1 and len(result['paired']) == 64
            assert result['contract']['policy']['pair_count'] == 64
            assert 'seed_block' not in result['contract']
            assert run.call_count == 16  # incumbent re-evaluated here too
        assert not args.seed_ledger.exists()  # v1 never touches the ledger
        assert gate.valid_certificate(result, result['contract'], args.out)
        manifest = gate.read(args.out / 'manifest.json')
        assert manifest['panel_version'] == 'v2-task49A'
        assert {row['seed'] for row in result['paired']} == set(range(30000, 30064))
