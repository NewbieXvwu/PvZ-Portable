"""Paired native Predict timing on identical legal live states, excluding IPC/simulation."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'scripts')]
import torch
from pvz_env import TaskSpec
from pvz_event_env import EventWaitEnv
from pvz_rollout_policy import export_student, load_student
from pvz_tasks import profile_for_deck
from pvz_student_observation import StudentHistory
from render_episode import _frame


def benchmark(narrow, wide, out):
    torch.set_num_threads(1)
    models = {64: narrow, 128: wide}
    bins = {}
    for width, path in models.items():
        net = load_student(path).eval()
        assert net.trunk[0].out_channels == width and net.packet_head == 'identity'
        bins[width] = out.parent/f'benchmark_w{width}.pvzrn'
        export_student(net, bins[width])
    records = []
    with EventWaitEnv('/home/newbiexvwu/.cache/pvz-research-resources', executable=ROOT/'build/pvz-portable') as env:
        env.load_rollout_net(bins[64])
        deck = (0,1,2,3,4,5,7)
        obs, _ = env.reset(deck=deck, task=TaskSpec(level=9, seed=30000, playthrough=2, profile=profile_for_deck(deck)))
        history = StudentHistory('v2')
        prior_tick = None
        prior_plant = False
        for decision in range(1000):
            if obs['terminal']: break
            if decision >= 16 and decision % 12 == 0 and obs['legal_actions']['plants'] and len(records) < 24:
                root = env.snapshot()
                times = {64: [], 128: []}
                # ABBA / BAAB blocks keep resident models warm and balance load/drift.
                order = (64,128,128,64) if len(records) % 2 == 0 else (128,64,64,128)
                for width in order:
                    env.load_rollout_net(bins[width])
                    for repetition in range(16):
                        env.restore(root)
                        env.rollout(None, 0, max_actions=1, student_history=copy.deepcopy(history))
                        result = env._last_response
                        assert result['net_calls'] == result['decisions'] == 1
                        if repetition >= 4: times[width].append(result['network_seconds']*1e6)
                env.restore(root); env.release_snapshot(root)
                records.append(dict(decision=decision, tick=obs['tick'], samples_us={str(k):v for k,v in times.items()}))
            env.load_rollout_net(bins[64])
            previous = _frame(obs)
            chosen = env.rollout_predict(prior_tick, prior_plant, student_history=history)['action']
            # Keep public plant-age and wave history from the narrow model's live trajectory.
            history.extra(previous)
            prior_tick, prior_plant = previous['tick'], chosen['type'] == 'plant'
            obs, *_, info = env.step(chosen)
            assert info.get('ok')
            if len(records) == 24: break
    assert len(records) == 24
    summaries = {}
    for width in (64,128):
        values = [x for r in records for x in r['samples_us'][str(width)]]
        summaries[str(width)] = dict(samples=len(values), median_us=statistics.median(values), mean_us=statistics.mean(values),
                                    model_bytes=models[width].stat().st_size,
                                    model_sha256=hashlib.sha256(models[width].read_bytes()).hexdigest())
    result = dict(complete=True, states=24, states_source='width64 dw4i_s1 live L9 seed30000', repeats_per_state=24,
                  warmup_per_state=8, widths=summaries, median_ratio_128_over_64=summaries['128']['median_us']/summaries['64']['median_us'],
                  paired_state_median_ratio=statistics.median(statistics.median(r['samples_us']['128'])/statistics.median(r['samples_us']['64']) for r in records),
                  records=records, methodology='Same native binary, one process, identical snapshots with legal plants. '
                  'Existing C++ network_seconds times Predict (features, network and action selection); '
                  'IPC, snapshot restore, model load and simulation are outside this timer. '
                  'One real network call per sample; this is deployed single-step cost, not isolated matrix multiplication.')
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--narrow', type=Path, required=True)
    parser.add_argument('--wide', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); benchmark(args.narrow, args.wide, args.out)
