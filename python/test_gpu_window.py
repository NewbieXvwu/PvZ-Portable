"""Small CPU fixtures for interruption and production-priority scheduling."""
import json
import os
from pathlib import Path
import signal
import sys
from argparse import Namespace

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


def test_window_phase_and_reserves_never_admit_training_or_transition():
    from gpu_window_runner import production_phase, may_run, GIB
    def proc(script, parent=1):
        return dict(script=script, parent=parent, argv=['python', script])
    table = {10: proc('curriculum_queue.py'), 11: proc('pack_training_rows.py', 10),
             20: proc('gpu_window_runner.py'), 21: proc('train_rollout_policy.py', 20)}
    phase = production_phase(table, 20)
    assert phase['open'] and phase['phase'] == 'pack'  # Our child is excluded.
    args = Namespace(host_reserve_gib=12, gpu_reserve_gib=8)
    job = dict(host_gib=5, gpu_gib=4)
    assert may_run(phase, 17 * GIB, 12 * GIB, job, args)[0]
    assert not may_run(phase, 16 * GIB, 12 * GIB, job, args)[0]
    assert not may_run(phase, 17 * GIB, None, job, args)[0]
    table[12] = proc('train_rollout_policy.py', 10)
    assert not production_phase(table, 20)['open']
    del table[12]; del table[11]
    assert not production_phase(table, 20)['open']


def test_window_cpu_metrics_use_plant_denominators_and_deck_identity():
    from eval_window_policy import deck_keys, empty_metrics, add_metrics, report_metrics, terrain_for_level
    import numpy as np
    keys = deck_keys(np.array([[0, 1, 1], [1, 0, 1], [0, 1, 2]], dtype=np.int8))
    assert keys[0] == keys[1] and keys[0] != keys[2]
    counts = empty_metrics()
    add_metrics(counts, torch.tensor([False, True, True]), torch.tensor([False, True, True]),
                torch.tensor([False, True, False]), torch.tensor([False, True, True]), torch.tensor([1., 2., 3.]))
    report = report_metrics(counts)
    assert report['packet_acc'] == .5 and report['cell_acc'] == 1
    assert report['type_acc'] == 1 and report['joint_acc'] == 2 / 3
    assert report_metrics(empty_metrics())['packet_acc'] is None
    assert terrain_for_level(35) == 'night' and terrain_for_level(50) == 'roof'


def test_window_yield_signals_only_child_group_and_persists_progress(tmp_path, monkeypatch):
    import subprocess
    import time
    import gpu_window_runner as window
    args = Namespace(queue=tmp_path / 'queue', handoff=tmp_path / 'mac', production_log=tmp_path / 'prod.log',
                     yield_timeout=2)
    args.queue.mkdir()
    out = tmp_path / 'job'
    recipe = tmp_path / 'recipe.json'
    child = ('import signal,time,sys; '
             'signal.signal(signal.SIGUSR1,lambda *a:sys.exit(75)); print("ready",flush=True); time.sleep(30)')
    job = dict(id='fixture', out=str(out), recipe=str(recipe), command=[sys.executable, '-c', child])
    runner = window.Runner(args)
    state = dict(status='queued', seconds=0, attempts=[])
    runner.start(job, state, dict(phase='pack'))
    deadline = time.monotonic() + 2
    while 'ready' not in (out / 'train.log').read_text():
        assert time.monotonic() < deadline
        time.sleep(.01)
    actual_kill = os.killpg
    signalled = []
    def kill(pid, sig):
        signalled.append(pid); actual_kill(pid, sig)
    monkeypatch.setattr(os, 'killpg', kill)
    pid = runner.child.pid
    runner.yield_child('training')
    runner.child.wait(timeout=2)
    runner.finish()
    assert signalled == [pid]
    saved = json.loads((out / 'window.json').read_text())
    assert saved['status'] == 'paused' and saved['attempts'][0]['returncode'] == 75
    assert saved['attempts'][0]['release_seconds'] < 2


def test_window_cpu_eval_continues_counts_after_yield(tmp_path, monkeypatch):
    import eval_window_policy as evaluator
    from pvz_rollout_policy import RolloutNet
    fixtures.IdentityTrainingTest().fixture(tmp_path / 'data')
    packed_load(tmp_path / 'data', {'example'}, obs='v2', packet_head='identity', cells=True, rebuild=True)
    pack = next((tmp_path / 'data/packs').glob('*.npz'))
    recipe = tmp_path / 'eval_recipe.json'
    write_recipe(recipe, dict(schema_version=1, kind='training', data=str(tmp_path / 'data'), labels=['example'],
                              out=str(tmp_path / 'eval'), width=8, obs='v2', packet_head='identity'))
    torch.manual_seed(1)
    checkpoint = dict(state_dict=RolloutNet(width=8, obs='v2', packet_head='identity').state_dict(),
                      width=8, obs='v2', packet_head='identity')
    original_add = evaluator.add_metrics
    results = []
    for name in ('continuous', 'resumed'):
        out = tmp_path / name
        out.mkdir()
        torch.save(checkpoint, out / 'model.pt')
        peer = tmp_path / (name + '_peer')
        peer.mkdir()
        torch.save(checkpoint, peer / 'model.pt')
        monkeypatch.setattr(sys, 'argv', ['eval', '--recipe', str(recipe), '--pack', str(pack),
                           '--merge-dir', str(tmp_path / 'merge'), '--out', str(out), '--batch-size', '2',
                           '--peer-out', str(peer), '--preview-rows', '2'])
        if name == 'resumed':
            def interrupt(*args):
                original_add(*args)
                os.kill(os.getpid(), signal.SIGUSR1)
            monkeypatch.setattr(evaluator, 'add_metrics', interrupt)
            import pytest
            with pytest.raises(SystemExit) as stopped:
                evaluator.main()
            assert stopped.value.code == 75
            monkeypatch.setattr(evaluator, 'add_metrics', original_add)
        evaluator.main()
        results.append(json.loads((out / 'heldout.json').read_text())['groups'])
        compared = json.loads((peer / 'heldout.json').read_text())
        assert compared['groups'] == results[-1] and compared['evaluation_complete'] is True
        assert compared['heldout_rows'] == compared['evaluated_rows'] == 7
    assert results[0] == results[1]
    assert results[0]['all']['rows'] == 7
    assert results[0]['seen']['rows'] == 7 and results[0]['unseen']['rows'] == 0


def test_window_training_completion_hands_off_model_before_cpu_evaluation(tmp_path):
    from gpu_window_runner import Runner
    args = Namespace(queue=tmp_path / 'queue', handoff=tmp_path / 'mac', production_log=tmp_path / 'prod.log')
    args.queue.mkdir()
    out = tmp_path / 'job'
    recipe = tmp_path / 'recipe.json'
    recipe.write_text('{}')
    job = dict(id='fixture', out=str(out), recipe=str(recipe),
        command=[sys.executable, '-c', f'from pathlib import Path; Path({str(out / "model.pt")!r}).write_bytes(b"model")'],
        evaluate=[sys.executable, '-c', f'from pathlib import Path; Path({str(out / "heldout.json")!r}).write_text("{{}}")'])
    runner = Runner(args)
    state = dict(status='queued', seconds=0, attempts=[])
    runner.start(job, state, dict(phase='pack'))
    runner.child.wait(timeout=2); runner.finish()
    assert (args.handoff / 'fixture/model.pt').stat().st_ino == (out / 'model.pt').stat().st_ino
    assert state['status'] == 'paused'
    peer = dict(job, id='peer')
    (args.queue / 'peer.json').write_text(json.dumps(peer))
    job['evaluation_peers'] = ['peer']
    runner.start(job, state, dict(phase='selection'))
    assert runner.active[2]['stage'] == 'evaluate'
    runner.child.wait(timeout=2); runner.finish()
    assert state['status'] == 'complete'
    assert (args.handoff / 'fixture.json').is_file()
    assert (args.handoff / 'peer.json').is_file()
    job['defer_evaluation'] = True
    job['id'] = 'deferred'
    (out / 'model.pt').unlink()
    state = dict(status='queued', seconds=0, attempts=[])
    runner.start(job, state, dict(phase='pack'))
    runner.child.wait(timeout=2); runner.finish()
    assert state['status'] == 'complete'
    assert json.loads((args.handoff / 'deferred/ready.json').read_text())['stage'] == 'trained'
