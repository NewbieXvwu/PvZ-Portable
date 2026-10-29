"""Small CPU fixtures for interruption and production-priority scheduling."""
import json
import os
from pathlib import Path
import signal
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import train_rollout_policy as trainer
import test_identity_training as fixtures
from pack_training_rows import packed_load
from pvz_recipe import write_recipe


def test_window_resume_matches_uninterrupted_optimizer(tmp_path, monkeypatch):
    fixtures.IdentityTrainingTest().fixture(tmp_path / 'data')
    packed_load(tmp_path / 'data', {'example'}, obs='v2', packet_head='identity', cells=True, rebuild=True)
    base = dict(schema_version=1, kind='training', data=str(tmp_path / 'data'), labels=['example'],
                device='cpu', width=8, epochs=3, seed=17, obs='v2', pool='meanmax', row_pool=True,
                packet_head='identity', packet_balance=.5, cell_soft_weight=3, spread_weight=8,
                batch_size=16, validation_batch_size=16, lr_schedule='cosine')
    reference, resumed = tmp_path / 'reference', tmp_path / 'resumed'
    losses = []
    for out in (reference, resumed):
        recipe = tmp_path / (out.name + '.json')
        write_recipe(recipe, dict(base, out=str(out)))
        argv = ['train', '--recipe', str(recipe)]
        if out == resumed:
            argv += ['--checkpoint-seconds', '3600']
        monkeypatch.setattr(sys, 'argv', argv)
        def observe(step, loss):
            if out == reference:
                losses.append(loss)
            elif step == 3:
                os.kill(os.getpid(), signal.SIGUSR1)
        if out == reference:
            trainer.main(step_observer=observe)
        else:
            import pytest
            with pytest.raises(SystemExit) as stopped:
                trainer.main(step_observer=observe)
            assert stopped.value.code == 75
            saved = torch.load(out / 'resume.pt', weights_only=False)
            assert saved['completed_steps'] == 3
            monkeypatch.setattr(sys, 'argv', argv + ['--resume', str(out / 'resume.pt')])
            tail = []
            trainer.main(step_observer=lambda step, loss: tail.append(loss))
            assert tail == losses[3:]
    expected = torch.load(reference / 'model.pt', weights_only=True)
    actual = torch.load(resumed / 'model.pt', weights_only=True)
    for name, weight in expected['state_dict'].items():
        torch.testing.assert_close(weight, actual['state_dict'][name], rtol=0, atol=0)
