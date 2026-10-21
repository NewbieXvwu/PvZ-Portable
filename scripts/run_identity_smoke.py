"""Eight v33 perturbed teacher games, archive/deck checks and real training-loader/replay smoke."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]
from run_identity_validation import guard, OUT, SOURCE
from render_episode import load_training_task, training_task_kwargs


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'artifacts/stage4/day_panel_identity.json')
    args = parser.parse_args()
    frames_dir = OUT / 'perturbed_frames'
    model = SOURCE / 'artifacts/student/v33/model.pt'
    identity = sha(model)
    manifest_data = json.loads(args.manifest.read_text())
    report = dict(model=str(model), model_sha256=identity, manifest_sha256=sha(args.manifest),
                  capture_manifest_executable_sha256=manifest_data['generation_audit']['executable_sha256'],
                  replay_executable_sha256=sha(ROOT / 'build/pvz-portable'), games=[])
    labels = set()
    for level in (7, 8, 9, 10):
        task_id = f'day_l{level}_train_d7p'
        definition = load_training_task(args.manifest, task_id)
        dest = OUT / f'perturb_smoke_l{level}'
        label = f'task19_l{level}p'
        labels.add(label + '_fast_search')
        if not (dest / 'summary.json').exists():
            command = [guard.PYTHON, 'scripts/search_play.py', '--rollout-policy', str(model), '--verify-samples', '9',
                       '--student-candidates', '2', '--static-value', str(SOURCE / 'artifacts/hard/value_branch1/model.pt'),
                       '--value', str(SOURCE / 'artifacts/value_net/v3/model.pt'), '--manifest', str(args.manifest),
                       '--task-id', task_id, '--seeds', '2', '--dagger-beta', '0', '--workers', '2',
                       '--resource-dir', guard.ENV['PVZ_RESOURCE_DIR'], '--executable', str(ROOT / 'build/pvz-portable'),
                       '--frames', str(frames_dir), '--frames-label', label, '--arms', 'fast_search', '--out', str(dest)]
            result = guard.run(command, f'perturb_l{level}', extra={'CUDA_VISIBLE_DEVICES': ''})
            assert result['returncode'] == 0
        for seed in definition['seeds'][:2]:
            archive = frames_dir / f'seed_{seed}' / (label + '_fast_search')
            meta = json.loads((archive / 'meta.json').read_text())
            actual = training_task_kwargs(definition, seed)['deck']
            assert meta['task']['deck'] == actual and meta['task']['level'] == level
            assert meta['task']['checkpoint_sha256'] == identity
            from pvz_frame_store import read_frames
            frames = read_frames(archive / 'frames.npz')
            assert len(frames) > 1
            assert [p[1] for p in sorted(frames[0]['packets'])] == actual
            changes = sum([p[1] for p in sorted(frame['packets'])] != actual for frame in frames)
            report['games'].append(dict(level=level, seed=seed, actual_deck=actual, frames=len(frames),
                                         archive=str(archive), won=meta['outcome']['won'],
                                         frames_with_changed_seed_bank=changes))
    assert len(report['games']) == 8 and sha(model) == identity
    from train_rollout_policy import load
    loaded = load(frames_dir, labels, cache_dir=OUT / 'smoke_cache', obs='v2', cells=True, packet_head='identity')
    assert len(loaded) == 10 and loaded[-1].shape == (len(loaded[2]), 10, 4)
    assert loaded[0].shape[0] > 8
    report['training_loader'] = dict(rows=loaded[0].shape[0], obs='v2', cell_targets=True, packet_head='identity')
    import torch
    import torch.nn.functional as F
    from pvz_rollout_policy import RolloutNet
    torch.set_num_threads(1)
    torch.manual_seed(19019)
    free = torch.nonzero((loaded[2] > .5) &
                         (loaded[-1][torch.arange(len(loaded[2])), loaded[3].clamp(min=0), 3] == 0)).flatten()
    assert len(free) > 0
    selected = free[:128]
    net = RolloutNet(8, 'meanmax', obs='v2', row_pool=True, packet_head='identity')
    output = net(loaded[0].batch(selected, 'cpu'), loaded[1].batch(selected, 'cpu'),
                 loaded[3][selected], packet_info=loaded[-1][selected])
    loss = F.cross_entropy(output['packet'], loaded[3][selected])
    assert torch.isfinite(loss) and loss < 100
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
    report['free_card_cpu_gradient_preflight'] = dict(rows=len(selected), eligible_rows=len(free),
                                                     packet_cross_entropy=float(loss.detach()), finite_gradients=True)
    # The replay entry uses the archive's deck and checkpoint; never consult a changed task manifest.
    from episode_query import _replay_task_extra
    from render_episode import collect
    from pvz_frame_store import read_frames
    first = report['games'][0]
    meta = json.loads((Path(first['archive']) / 'meta.json').read_text())
    archived = read_frames(Path(first['archive']) / 'frames.npz')
    length = min(8, len(archived) - 1)
    replay = collect(guard.ENV['PVZ_RESOURCE_DIR'], first['seed'], first['level'], 'student',
                     max_actions=length, deck=meta['task']['deck'], checkpoint=meta['task']['checkpoint'],
                     prefix_actions=[f['action'] for f in archived[1:length + 1]],
                     task_extra=_replay_task_extra(meta['task']))
    assert replay['task']['deck'] == meta['task']['deck']
    assert [p[1] for p in replay['frames'][0]['packets']] == meta['task']['deck']
    report['live_replay_archived_deck_verified'] = True
    report['complete'] = True
    target = ROOT / 'MEMORY/docs/task19-perturbation-smoke.json'
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'perturbed_games': 8, 'rows_loaded': report['training_loader']['rows'],
                      'live_replay_archived_deck_verified': True}), flush=True)


if __name__ == '__main__':
    main()
