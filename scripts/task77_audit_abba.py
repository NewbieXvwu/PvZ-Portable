"""Recompute every number in search-hotpath.md sections 7.1-7.4 from the evidence."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from task77_search_hotpath import family_interval, paired_interval

_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1] / 'artifacts/task77')
P = _parser.parse_args().out

failures = []


def check(name, got, want, tol):
    ok = abs(got - want) <= tol
    print(('  ok    ' if ok else '  FAIL  ') + f'{name}: {got:.6g} vs {want:.6g}')
    if not ok:
        failures.append(name)


def check3(name, got, want):
    """The report prints three decimals; compare at that precision."""
    ok = f'{got:.3f}' == want
    print(('  ok    ' if ok else '  FAIL  ') + f'{name}: {got:.6f} vs {want}')
    if not ok:
        failures.append(name)


def section(title):
    print(f'== {title}')


section('7.1 gate')
g = json.loads((P / 'gate_evidence.json').read_text())
games = g['games']
check('games', len(games), 48, 0)
check('pass', float(g['pass']), 1.0, 0)
check('cells', len(Counter(x['cell'] for x in games)), 6, 0)
assert all(n == 8 for n in Counter(x['cell'] for x in games).values()), 'cell counts'
check('teacher', Counter(x['arm'] for x in games)['teacher'], 24, 0)
check('student', Counter(x['arm'] for x in games)['student'], 24, 0)
check('state_hash same', Counter(x['state_hash_same'] for x in games)[True], 35, 0)
check('state_hash differ', Counter(x['state_hash_same'] for x in games)[False], 13, 0)
assert all(x['frame_mismatch'] is None for x in games), 'frame mismatch'
assert all(x['pass_'] for x in games), 'gate row failed'

section('7.2 abba_final')
f = json.loads((P / 'abba_final/abba_evidence.json').read_text())
s = f['summary']
check('geomean', s['geomean'], 0.9934, 1e-4)
check('median', s['median'], 1.0027, 1e-4)
check('ci lo', s['bootstrap95'][0], 0.9485, 1e-4)
check('ci hi', s['bootstrap95'][1], 1.0347, 1e-4)
check('family lo', s['family_bootstrap95'][0], 0.9261, 1e-4)
check('family hi', s['family_bootstrap95'][1], 1.0517, 1e-4)
for cell, want in (('day', 1.1041), ('night', 0.8515), ('pool', 1.0075),
                   ('conveyor', 1.0167), ('roof', 1.0), ('fog', 0.9978)):
    check(f'{cell} geomean', f['by_cell'][cell]['geomean'], want, 1e-4)
q = f['qc_running']
check('qc pairs', q['paired_games'], 4, 0)
check('qc geomean', q['summary']['geomean'], 0.9696, 1e-4)
check('qc ci lo', q['summary']['bootstrap95'][0], 0.8515, 1e-4)
check('qc ci hi', q['summary']['bootstrap95'][1], 1.1041, 1e-4)
stable = [r for r in f['games'] if r['cell'] in ('conveyor', 'roof', 'fog')]
ss = paired_interval([r['ratio'] for r in stable])
check('stable pairs', len(stable), 6, 0)
check('stable geomean', ss['geomean'], 1.0048, 1e-4)
check('stable ci lo', ss['bootstrap95'][0], 0.9985, 1e-4)
check('stable ci hi', ss['bootstrap95'][1], 1.0115, 1e-4)
sf = family_interval(stable)
check('stable family lo', sf[0], 0.9978, 1e-4)
check('stable family hi', sf[1], 1.0167, 1e-4)
for cell, lo, hi in (('day', 11.9, 34.3), ('night', 14.4, 29.2), ('pool', 8.4, 14.4),
                     ('conveyor', 6.6, 8.5), ('roof', 6.4, 6.6), ('fog', 6.1, 6.5)):
    loads = [float(l[side]['loadavg'].split()[0]) for l in f['load'] if l['cell'] == cell
             for side in ('before', 'after')]
    check(f'{cell} load min', min(loads), lo, 0.05)
    check(f'{cell} load max', max(loads), hi, 0.05)
for cell, lo, hi in (('day', 13, 21), ('night', 8, 19), ('pool', 2, 8),
                     ('conveyor', 0, 2), ('roof', 0, 0), ('fog', 0, 0)):
    qc = [l[side]['qc_engines'] for l in f['load'] if l['cell'] == cell for side in ('before', 'after')]
    check(f'{cell} qc engines min', min(qc), lo, 0)
    check(f'{cell} qc engines max', max(qc), hi, 0)
# The per-arm table: games.jsonl keeps the A and B means, so the report's arms must average to them.
arms = {('day', 5012288): (111.4, 135.5, 138.6, 132.7),
        ('day', 5012289): (138.6, 173.0, 154.2, 162.8),
        ('night', 5157696): (221.2, 147.8, 173.7, 150.6),
        ('night', 5157697): (256.2, 167.7, 186.7, 166.5)}
by_key = {(r['cell'], r['seed']): r for r in f['games']}
for key, (a1, b1, b2, a2) in arms.items():
    check(f'{key} A mean', (a1 + a2) / 2, by_key[key]['old'], 0.06)
    check(f'{key} B mean', (b1 + b2) / 2, by_key[key]['new'], 0.06)
a = f['identity_audit']
check('identity pairs', a['paired_games'], 12, 0)
check('identity comparisons', a['frame_comparisons'], 36, 0)
check('identity frames', a['decision_frames_compared'], 40686, 0)
check('identity state_hash differ', a['state_hash_differ'], 12, 0)
assert a['pass']

section('7.3 abba_cpu')
c = json.loads((P / 'abba_cpu/abba_evidence.json').read_text())
s = c['summary']
check('wall geomean', s['geomean'], 0.9899, 1e-4)
check('wall ci lo', s['bootstrap95'][0], 0.9692, 1e-4)
check('wall ci hi', s['bootstrap95'][1], 1.0117, 1e-4)
check('wall family lo', s['family_bootstrap95'][0], 0.9611, 1e-4)
check('wall family hi', s['family_bootstrap95'][1], 1.0204, 1e-4)
check('cpu geomean', c['cpu_summary']['geomean'], 0.9848, 1e-4)
check('cpu ci lo', c['cpu_summary']['bootstrap95'][0], 0.9594, 1e-4)
check('cpu ci hi', c['cpu_summary']['bootstrap95'][1], 1.0114, 1e-4)
cpu = {r['cell']: r['ratio'] for r in c['cpu_ratios']}
for cell, wall, cpu_ratio in (('day', '1.037', '1.036'), ('night', '0.995', '0.994'),
                              ('pool', '0.938', '0.938'), ('conveyor', '0.969', '0.968'),
                              ('roof', '1.041', '1.011'), ('fog', '0.964', '0.964')):
    check3(f'{cell} wall', c['by_cell'][cell]['geomean'], wall)
    check3(f'{cell} cpu', cpu[cell], cpu_ratio)
a = c['identity_audit']
check('identity pairs', a['paired_games'], 12, 0)
check('identity comparisons', a['frame_comparisons'], 36, 0)
check('identity frames', a['decision_frames_compared'], 40686, 0)
check('identity state_hash differ', a['state_hash_differ'], 13, 0)
assert a['pass']

section('7.4 abba_qc')
q = json.loads((P / 'abba_qc/abba_evidence.json').read_text())
s = q['summary']
check('wall geomean', s['geomean'], 1.0045, 1e-4)
check('wall ci lo', s['bootstrap95'][0], 0.9874, 1e-4)
check('wall ci hi', s['bootstrap95'][1], 1.0221, 1e-4)
check('wall family lo', s['family_bootstrap95'][0], 0.9823, 1e-4)
check('wall family hi', s['family_bootstrap95'][1], 1.0298, 1e-4)
check('cpu geomean', q['cpu_summary']['geomean'], 1.0050, 1e-4)
check('cpu ci lo', q['cpu_summary']['bootstrap95'][0], 0.9846, 1e-4)
check('cpu ci hi', q['cpu_summary']['bootstrap95'][1], 1.0286, 1e-4)
cpu = {r['cell']: r['ratio'] for r in q['cpu_ratios']}
for cell, wall, cpu_ratio in (('day', '0.988', '0.989'), ('night', '1.016', '1.014'),
                              ('pool', '1.017', '1.017'), ('conveyor', '0.984', '0.984'),
                              ('roof', '0.967', '0.973'), ('fog', '1.057', '1.056')):
    check3(f'{cell} wall', q['by_cell'][cell]['geomean'], wall)
    check3(f'{cell} cpu', cpu[cell], cpu_ratio)
a = q['identity_audit']
check('identity pairs', a['paired_games'], 12, 0)
check('identity comparisons', a['frame_comparisons'], 36, 0)
check('identity frames', a['decision_frames_compared'], 40686, 0)
check('identity state_hash differ', a['state_hash_differ'], 18, 0)
assert a['pass']
rows = [json.loads(line) for line in (P / 'abba_qc/load_monitor.jsonl').read_text().splitlines()]
check('monitor samples', len(rows), 535, 0)
assert {tuple(r['qc_controllers']) for r in rows} == {(626824,)}, 'monitor controllers'
loads = [float(r['loadavg'].split()[0]) for r in rows]
check('monitor load min', min(loads), 12.7, 0.05)
check('monitor load max', max(loads), 19.8, 0.05)
print(f'  monitor loadavg {min(loads):.2f}-{max(loads):.2f}')
bounds = [float(l[side]['loadavg'].split()[0]) for l in q['load'] for side in ('before', 'after')]
qce = [l[side]['qc_engines'] for l in q['load'] for side in ('before', 'after')]
check('boundary blocks', len(q['load']), 24, 0)
check('boundary qc engines min', min(qce), 6, 0)
check('boundary qc engines max', max(qce), 9, 0)
check('boundary load min', min(bounds), 9.7, 0.05)
check('boundary load max', max(bounds), 19.6, 0.05)
print(f'  boundary loadavg {min(bounds):.2f}-{max(bounds):.2f}')
assert all(tuple(l[side]['qc_controllers']) == (626824,) for l in q['load'] for side in ('before', 'after'))

if failures:
    print(f'{len(failures)} mismatches: {failures}')
    sys.exit(1)
print('7.1-7.4 numbers reproduce')
