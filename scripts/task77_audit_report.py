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

