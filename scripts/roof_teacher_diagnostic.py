"""Record task102 roof decisions with the unchanged production search."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]
os.environ.setdefault('PVZ_NO_POPUP', '1')

import search_play as sp
from roof_phase2 import Timeline


def capture(config_path, seed, arm, out):
    config = json.loads(config_path.read_text())
    out.mkdir(parents=True, exist_ok=True)
    flags = []
    for name in ('rollout_policy', 'value', 'static_value', 'manifest', 'task_id',
                 'resource_dir', 'executable', 'per_card', 'student_candidates',
                 'samples', 'horizon', 'verify_samples', 'min_gap', 'rollout_sim_skip'):
        flags.extend(['--' + name.replace('_', '-'), str(config[name])])
    args = sp.parser().parse_args(flags + ['--out', str(out / 'run'), '--frames', str(out / 'frames')])
    args.halving = str(config['halving']).lower() == 'true'
    args.dagger_beta = 1 if arm == 'fast_search' else 0
    original_choice, original_static = sp._verified_choice, sp._static_scores
    static_scores = []

    def trace_static(*positional, **keywords):
        scores = original_static(*positional, **keywords)
        static_scores[:] = [dict(action=action, value=value) for value, action in scores]
        return scores

    with (out / 'search.jsonl').open('w') as stream:
        def trace_choice(env, obs, root, candidates, totals, counts, best, policy_index,
                         policy_action, search_args, decision_seed, stats, **keywords):
            before = list(totals)
            chosen = original_choice(env, obs, root, candidates, totals, counts, best,
                                     policy_index, policy_action, search_args, decision_seed,
                                     stats, **keywords)
            stream.write(json.dumps(dict(decision=decision_seed % 10000 + 1,
                tick=obs['tick'], sun=obs['sun'], wave=obs['wave'], candidates=candidates,
                static_scores=static_scores, values_before=before, values_after=totals,
                counts=counts, best=best, policy_index=policy_index,
                gap_before=before[best] - before[policy_index],
                gap_after=totals[best] - totals[policy_index], chosen=chosen,
                verification=stats.get('last_verification')), allow_nan=False) + '\n')
            stream.flush()
            return chosen

        class RoofTimeline(Timeline):
            def before(self, env, obs, state, decision):
                static_scores.clear()
                if decision == 0:
                    deck = [packet['type'] for packet in obs['packets'] if packet['type'] >= 0]
                    (out / 'initial.json').write_text(json.dumps(dict(observation=obs,
                        warnings=env.deck_warnings(deck), options=env.deck_options()), indent=2))
                super().before(env, obs, state, decision)

        observer = RoofTimeline(out / 'timeline.jsonl')
        sp._verified_choice, sp._static_scores = trace_choice, trace_static
        keep, values = sp._export_values(config['value'], config['static_value'])
        try:
            sp._load_nets(values[0], config['rollout_policy'], values[1])
            sp._init_worker(values[0], config['rollout_policy'], args.rollout_sim_skip, values[1])
            result = sp._play(dict(args=args, seed=seed, arm=arm), observer=observer)
        finally:
            observer.close()
            sp._verified_choice, sp._static_scores = original_choice, original_static
            sp._close_worker_env()
            if sp._W.get('rollout_tmp'):
                sp._W['rollout_tmp'].cleanup()
            sp._W.clear()
            keep.cleanup()
    result['events'] = dict(observer.events)
    result['identity'] = {name: dict(path=config[name], sha256=hashlib.sha256(
        Path(config[name]).read_bytes()).hexdigest())
        for name in ('rollout_policy', 'value', 'static_value', 'executable', 'manifest')}
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--arm', required=True, choices=('fast', 'fast_search'))
    parser.add_argument('--out', required=True, type=Path)
    options = parser.parse_args()
    capture(options.config, options.seed, options.arm, options.out)
