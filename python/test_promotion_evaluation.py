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
        resource_dir=common.resource_dir, workers=1, out=root / 'evaluation')


def fake_games(args, command, **kwargs):
    # Preregistration must exist before the first episode starts.
    registration = gate.read(args.out / 'registration.json')
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
            assert result['approved'] and len(result['paired']) == 64
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
