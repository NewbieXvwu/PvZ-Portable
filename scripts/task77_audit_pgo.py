"""Recompute every number in search-hotpath.md section 7.5 from the raw evidence."""
import argparse
import json
import math
import os
import random
from collections import Counter
from pathlib import Path

_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1] / 'artifacts/task77')
P = _parser.parse_args().out
main = json.loads((P / 'abba_pgo/abba_evidence.json').read_text())
day = json.loads((P / 'abba_pgo_day/abba_evidence.json').read_text())
audit = json.loads((P / 'abba_pgo/abba_audit.json').read_text())
daudit = json.loads((P / 'abba_pgo_day/abba_audit.json').read_text())


def close(a, b, tol=5e-5):
    assert abs(a - b) <= tol, (a, b)


print('== 7.5 table')
cpu = {r['cell']: r for r in main['cpu_ratios']}
for cell, v in main['by_cell'].items():
    c = cpu[cell]
    print(f"  {cell:9s} {v['geomean']:.4f} [{v['bootstrap95'][0]:.4f},{v['bootstrap95'][1]:.4f}] "
          f"med {v['median']:.4f} cpu {c['ratio']:.4f} {c['old']:.1f}->{c['new']:.1f}")
    close(v['geomean'], {'day': 1.1152, 'night': 0.9642, 'pool': 0.9737, 'conveyor': 1.0270,
                         'roof': 1.0200, 'fog': 0.9932}[cell], 1e-4)
    close(c['ratio'], {'day': 1.1342, 'night': 0.9733, 'pool': 0.9855, 'conveyor': 1.0295,
                       'roof': 0.9915, 'fog': 0.9912}[cell], 1e-4)
close(main['summary']['geomean'], 1.0144, 1e-4)
close(main['summary']['median'], 1.0037, 1e-4)
close(main['summary']['bootstrap95'][0], 0.9886, 1e-4)
close(main['summary']['bootstrap95'][1], 1.0446, 1e-4)
close(main['summary']['family_bootstrap95'][0], 0.9808, 1e-4)
close(main['summary']['family_bootstrap95'][1], 1.0578, 1e-4)
close(main['cpu_summary']['geomean'], 1.0161, 1e-4)
close(main['cpu_summary']['median'], 0.9913, 1e-4)
close(main['cpu_summary']['bootstrap95'][0], 0.9834, 1e-4)
close(main['cpu_summary']['bootstrap95'][1], 1.0638, 1e-4)
print('  summary', main['summary'])
print('  cpu    ', main['cpu_summary'])

print('== audit')
print('  main', {k: v for k, v in audit.items() if k != 'cells'})
assert audit['pass'] and audit['paired_games'] == 12 and audit['frame_comparisons'] == 36
assert audit['decision_frames_compared'] == 40686 and audit['state_hash_differ'] == 13
assert daudit['pass'] and daudit['paired_games'] == 2 and daudit['frame_comparisons'] == 6
assert daudit['decision_frames_compared'] == 6411

print('== load')
rows = [json.loads(l) for l in (P / 'abba_pgo/load_monitor.jsonl').read_text().splitlines()]
loads = [float(r['loadavg'].split()[0]) for r in rows]
qce = [r['qc_engines'] for r in rows]
print(f'  samples {len(rows)} controllers {Counter(tuple(r["qc_controllers"]) for r in rows).most_common(2)}')
print(f'  loadavg {min(loads):.2f}-{max(loads):.2f} qc_engines {min(qce)}-{max(qce)}')
assert len(rows) == 552 and {tuple(r['qc_controllers']) for r in rows} == {(626824,)}
assert sum(1 for r in rows if r['qc_engines'] == 0) == 2
assert sorted(set(qce)) == [0, 7, 9]
bounds = [float(l[side]['loadavg'].split()[0]) for l in main['load'] for side in ('before', 'after')]
bq = [l[side]['qc_engines'] for l in main['load'] for side in ('before', 'after')]
print(f'  boundary loadavg {min(bounds):.2f}-{max(bounds):.2f} qc_engines {min(bq)}-{max(bq)}')
assert 12.2 < min(bounds) < 12.4 and 25.8 < max(bounds) < 26.0
assert min(bq) == 7 and max(bq) == 10
assert all(tuple(l['before']['qc_controllers']) == (626824,) and tuple(l['after']['qc_controllers']) == (626824,)
           for l in main['load']) and len(main['load']) == 24

print('== day A0 window')
a0 = P / 'abba_pgo/abba_0_A/day_abba_0_A_2'
start, end = os.stat(a0 / 'config.json').st_mtime, os.stat(a0 / 'games.jsonl').st_mtime
first = rows[0]['timestamp']
print(f'  A0 {start:.0f}-{end:.0f}, monitor first {first:.0f}, covered samples '
      f'{sum(1 for r in rows if start <= r["timestamp"] <= end)}')
assert end < first
arms = {r['index']: r for r in main['cpu'] if r['cell'] == 'day'}
print('  day arms', [(i, r['arm'], round(r['seconds'], 1), round(r['wall_seconds'], 1)) for i, r in arms.items()])
assert round(arms[0]['seconds'], 1) == 158.3 and round(arms[3]['seconds'], 1) == 204.5
assert round(arms[0]['wall_seconds'], 1) == 96.9 and round(arms[3]['wall_seconds'], 1) == 122.2
close(arms[0]['seconds'] / arms[3]['seconds'], 0.774, 1e-3)

print('== day recheck')
print('  ', day['summary'], day['cpu_ratios'])
close(day['summary']['geomean'], 0.9790, 1e-4)
close(day['summary']['bootstrap95'][0], 0.9760, 1e-4)
close(day['summary']['bootstrap95'][1], 0.9820, 1e-4)
close(day['cpu_ratios'][0]['ratio'], 0.9780, 1e-4)
drows = [json.loads(l) for l in (P / 'abba_pgo_day/load_monitor.jsonl').read_text().splitlines()]
dloads = [float(r['loadavg'].split()[0]) for r in drows]
dqce = [r['qc_engines'] for r in drows]
print(f'  samples {len(drows)} loadavg {min(dloads):.2f}-{max(dloads):.2f} qc_engines {min(dqce)}-{max(dqce)}')
assert len(drows) == 85 and {tuple(r['qc_controllers']) for r in drows} == {(626824,)}
assert 15.2 < min(dloads) < 15.4 and 22.8 < max(dloads) < 23.0
assert min(dqce) == 1 and max(dqce) == 8
darms = {r['index']: r for r in day['cpu']}
print('  recheck arms', [(i, r['arm'], round(r['seconds'], 1)) for i, r in darms.items()])
assert round(darms[0]['seconds'], 1) == 176.6 and round(darms[3]['seconds'], 1) == 181.6
close(darms[0]['seconds'] / darms[3]['seconds'], 0.9725, 1e-3)

print('== aggregates')
rest = [g['ratio'] for g in main['games'] if g['cell'] != 'day']
rest_g = math.exp(sum(map(math.log, rest)) / len(rest))
print('  excluding day', round(rest_g, 4), len(rest))
close(rest_g, 0.9953, 1e-4)
vals = rest + [g['ratio'] for g in day['games']]
sens = math.exp(sum(map(math.log, vals)) / len(vals))
rng = random.Random(77)
boot = []
for _ in range(4000):
    s = [vals[rng.randrange(len(vals))] for _ in vals]
    boot.append(math.exp(sum(map(math.log, s)) / len(s)))
boot.sort()
lo, hi = boot[100], boot[3899]
print(f'  sensitivity {sens:.4f} [{lo:.4f},{hi:.4f}] n={len(vals)}')
close(sens, 0.9926, 1e-4)
close(lo, 0.9796, 1e-3)
close(hi, 1.0064, 1e-4)
print('all 7.5 numbers reproduce')
