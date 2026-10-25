"""Statistics for task72b's frozen 94 fog states; the old replay path is disabled.

The reference decision is a continuation index, not a human-demo index. Restore the
original probe inputs or validate task78_task72_rebuild.py's state mapping and v5
probe protocol before producing any candidate scores. Pure agreement helpers remain
available; a complete-looking legacy result is not a validated task72b evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'scripts')]
from task78_data import sha

def run_group(job):
    raise RuntimeError('legacy replay disabled: continuation decision is not a demo decision; '
                       'restore/validate original task72b inputs before scoring')


def agreement(rows, arm):
    """Sign agreement with the terminal truth on wait vs static_best; ties score 0.5."""
    scored, ties, pairs = [], 0, 0
    for row in rows:
        pair = row.get('truth_pair')
        values = row['values'].get(arm) or {}
        if not pair or pair['better'] not in values or pair['worse'] not in values:
            continue
        pairs += 1
        diff = values[pair['better']] - values[pair['worse']]
        if abs(diff) < 1e-9:
            ties += 1
            scored.append(0.5)
        else:
            scored.append(1.0 if diff > 0 else 0.0)
    return dict(pairs=pairs, ties=ties, correct=sum(scored), rate=(sum(scored)/pairs if pairs else None))


def cluster_bootstrap(rows, arms, seed=780, draws=10000):
    """Resample whole future trajectories, keeping decisions from a world together.

    The original rows omit the takeover point. Grouping all occurrences of the same
    task/future together is conservative; decisions are never independent clusters.
    """
    import numpy as np
    groups = {}
    for row in rows:
        if not row.get('truth_pair'):
            continue
        groups.setdefault((row['task'], row['seed']), []).append(row)
    keys = sorted(groups)
    per_world = []
    for key in keys:
        world = groups[key]
        per_world.append([agreement(world, arm) for arm in arms])
    if not per_world:
        return dict(draws=0, worlds=0, seed=seed, lower=None, upper=None,
                    status='insufficient_evidence', reason='no truth pairs meet the preregistered gap')
    if any(a['pairs'] != b['pairs'] for a,b in per_world):
        raise ValueError('arms do not cover the same terminal-truth pairs')
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(draws):
        pick = rng.integers(0, len(per_world), len(per_world))
        a = sum(per_world[i][0]['correct'] for i in pick)
        b = sum(per_world[i][1]['correct'] for i in pick)
        n = sum(per_world[i][0]['pairs'] for i in pick)
        deltas.append((b-a)/n if n else 0.0)
    deltas = np.sort(np.array(deltas))
    return dict(draws=draws, worlds=len(per_world), seed=seed,
                lower=float(deltas[int(0.025*draws)]), upper=float(deltas[int(0.975*draws)]))


def report(path, arms):
    document = json.loads(Path(path).read_text())
    if document.get('complete') is not True or document.get('passed') is not True:
        raise ValueError('all frozen truth-state jobs must finish successfully before reporting')
    if (document.get('original_probe_protocol_verified') is not True or
            document.get('original_reference_sha256')!=sha(ROOT/'MEMORY/docs/task72b-probe-rows.jsonl')):
        raise ValueError('original task72b mapping, common-future protocol and v5 values must be verified')
    rows = [row for result in document['results'] if result['status'] == 'ok' for row in result['rows']]
    frozen = [json.loads(line) for line in
              (ROOT/'MEMORY/docs/task72b-probe-rows.jsonl').read_text().splitlines() if line.strip()]
    identity = lambda r: (r['task'], r['seed'], r['decision'], r['tick'])
    if len(rows) != len(frozen) or {identity(r) for r in rows} != {identity(r) for r in frozen}:
        raise ValueError('truth report must cover all 94 original state identities exactly once')
    summary = {arm: agreement(rows, arm) for arm in arms}
    boot = cluster_bootstrap(rows, list(arms))
    rates = [summary[name]['rate'] for name in arms]
    out = dict(horizon=document['horizon'], arms=summary, bootstrap=boot,
               improvement=rates[1]-rates[0] if all(v is not None for v in rates) else None,
               status='scored' if all(v is not None for v in rates) else 'insufficient_evidence')
    Path(path).with_name(f"agreement_h{document['horizon']}.json").write_text(json.dumps(out, indent=1)+'\n')
    print(json.dumps(out, indent=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', type=int, default=2500)
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--arms', default='v5=/home/newbiexvwu/PvZ-search/artifacts/value_net/v5/model.pt,'
                                         f'v6={ROOT}/artifacts/task78/v6/model.pt')
    parser.add_argument('--out', default=str(ROOT/'artifacts/task78/truth'))
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 2:
        parser.error('task78 currently permits at most two game processes')
    arms = dict(spec.split('=', 1) for spec in args.arms.split(','))
    if args.report:
        return report(Path(args.out)/f"h{args.horizon}.json", arms)
    parser.error('legacy replay disabled: use task78_task72_rebuild.py to recover the '
                 'original takeover mapping, then validate the original probe protocol')


if __name__ == '__main__':
    main()
