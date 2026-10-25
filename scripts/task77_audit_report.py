"""Recompute every number in search-hotpath.md sections 3, 3.1 and 4 from the evidence."""
import argparse
import json
from pathlib import Path

_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1] / 'artifacts/task77')
P = _parser.parse_args().out
s = json.loads((P/'profile_summary.json').read_text())
h = json.loads((P/'hotspots.json').read_text())
print('== hotspots categories (share of all samples)')
for k,v in h['categories'].items():
    print(f'  {k}: {v*100:.2f}%')
print('== engine cpu share of wall')
for cell, v in s['cells'].items():
    cpu = sum(v['commands_engine_cpu_seconds'].values()) if isinstance(v['commands_engine_cpu_seconds'], dict) else v['commands_engine_cpu_seconds']
    print(f'  {cell}: engine_cpu/wall {cpu/v["totals"]["wall"]*100:.0f}%')
print('== combined rollout+student share')
for cell, v in s['cells'].items():
    m = v['milliseconds_per_decision']
    total = sum(m.values())
    print(f'  {cell}: total {total:.1f} ms, rollout+student {(m["rollout_simulation"]+m["cpp_student"])/total*100:.1f}%, leaf+static {(m["leaf_features_forward"]+m["static_features_forward"])/total*100:.1f}%')
print('== per-cell sample counts', {k: v['samples'] for k, v in s['cells'].items()})

# §3.1 conveyor supplement: every number in the report table must come from its evidence.
c = json.loads((P/'profile_summary_conveyor.json').read_text())['cells']['conveyor']
m = c['milliseconds_per_decision']
t = c['totals']
rows = {'rollout_simulation': 80.44, 'cpp_student': 40.97, 'leaf_features_forward': 1.08,
        'static_features_forward': 8.22, 'snapshot_explicit_roundtrip': 0.33,
        'static_restore_action_ipc': 7.89, 'rollout_restore_control_ipc': 3.04,
        'other_commands': 2.18, 'python_outside_commands': 7.73}
print('== 3.1 conveyor table')
for key, expected in rows.items():
    assert round(m[key], 2) == expected, (key, m[key], expected)
    print(f'  {key}: {m[key]:.2f}')
total = sum(m.values())
assert round(total, 1) == 151.9, total
assert round((m['rollout_simulation']+m['cpp_student'])/total*100, 1) == 79.9
assert round((m['rollout_simulation'])/total*100, 1) == 53.0
assert round((m['cpp_student'])/total*100, 1) == 27.0
assert round((m['static_features_forward']+m['static_restore_action_ipc']), 1) == 16.1
assert round((m['static_features_forward']+m['static_restore_action_ipc'])/total*100, 1) == 10.6
assert round(t['static_calls']/t['decisions'], 1) == 37.3
print(f'  total {total:.1f} ms, rollout+student 79.9%, static 16.1 ms / 10.6%, {t["static_calls"]/t["decisions"]:.1f} static calls per decision')
print('3.1 conveyor numbers reproduce')

# ---- section 1: the ranges quoted in the conclusion must be the min/max of section 3.
rows = {}
for cell, v in s['cells'].items():
    m = v['milliseconds_per_decision']
    total = sum(m.values())
    rows[cell] = dict(total=total, rollout=m['rollout_simulation'] / total * 100,
                      student=m['cpp_student'] / total * 100,
                      leaf=m['leaf_features_forward'] / total * 100,
                      static=m['static_features_forward'] / total * 100,
                      snap=m['snapshot_explicit_roundtrip'] / total * 100,
                      ipc=(m['static_restore_action_ipc'] + m['rollout_restore_control_ipc']) / total * 100,
                      py=m['python_outside_commands'] / total * 100,
                      rest=100 - (m['rollout_simulation'] + m['cpp_student']) / total * 100)
print('== section 1 ranges (min/max over the five terrains)')
for key, lo, hi, digits in (('total', 74.8, 161.0, 1), ('rollout', 50.3, 65.3, 1),
                            ('student', 19.7, 32.0, 1), ('leaf', 0.8, 1.6, 1),
                            ('static', 0.9, 4.3, 1), ('snap', 0.3, 0.7, 1),
                            ('ipc', 3.6, 6.9, 1), ('py', 5.7, 10.0, 1),
                            ('rest', 14.0, 24.0, 0)):
    values = [r[key] for r in rows.values()]
    got = (round(min(values), digits), round(max(values), digits))
    assert got == (lo, hi), (key, got, (lo, hi))
    print(f'  {key}: {got[0]}-{got[1]}')
cpu_share = {cell: sum(v['commands_engine_cpu_seconds'].values()) / v['totals']['wall'] * 100
             for cell, v in s['cells'].items()}
rollout_share = {cell: v['commands_engine_cpu_seconds']['ROLLOUT_VALUE']
                 / sum(v['commands_engine_cpu_seconds'].values()) * 100
                 for cell, v in s['cells'].items()}
assert (round(min(cpu_share.values()), 1), round(max(cpu_share.values()), 1)) == (56.6, 77.7), cpu_share
assert (round(min(rollout_share.values()), 1), round(max(rollout_share.values()), 1)) == (89.3, 95.5), rollout_share
print(f'  engine cpu/wall {min(cpu_share.values()):.1f}-{max(cpu_share.values()):.1f}%, '
      f'ROLLOUT_VALUE share {min(rollout_share.values()):.1f}-{max(rollout_share.values()):.1f}%')

# ---- section 6: the offline kernel check.
k = json.loads((P / 'value_kernel_check_final/evidence.json').read_text())
assert k['pass'] and set(k['models']) == {'value', 'static'}
for name, prefix in (('value', '9762c586'), ('static', 'f4f8f4a0')):
    model = k['models'][name]
    assert model['pass'] and model['logits'] == 480
    assert model['sha256']['before'] == model['sha256']['after']
    assert model['sha256']['before'].startswith(prefix), (name, model['sha256']['before'][:8])
print('== section 6 kernel check: value/static 480 logits identical (9762c586 / f4f8f4a0)')

# ---- section 7.3b: the interleaved component timing.
t = json.loads((P / 'value_kernel_timing/timing.json').read_text())
before = [x / 1000 for x in t['rounds']['before']]
after = [x / 1000 for x in t['rounds']['after']]
assert len(before) == len(after) == 18, (len(before), len(after))
assert round(t['median_nanoseconds']['before'] / 1000, 1) == 164.1
assert round(t['median_nanoseconds']['after'] / 1000, 1) == 136.1
assert round(t['ratio'], 3) == 0.829
assert (round(min(before), 1), round(max(before), 1)) == (162.9, 172.3), (min(before), max(before))
assert (round(min(after), 1), round(max(after), 1)) == (134.8, 140.4), (min(after), max(after))
ratios = [b / a for a, b in zip(before, after)]
assert (round(min(ratios), 3), round(max(ratios), 3)) == (0.794, 0.854), (min(ratios), max(ratios))
assert min(before) > max(after), 'the two builds overlap'
print(f'== section 7.3b timing: 164.1 -> 136.1 us (0.829), before {min(before):.1f}-{max(before):.1f}, '
      f'after {min(after):.1f}-{max(after):.1f}, per-round ratio {min(ratios):.3f}-{max(ratios):.3f}')
print('section 1 / 6 / 7.3b numbers reproduce')


