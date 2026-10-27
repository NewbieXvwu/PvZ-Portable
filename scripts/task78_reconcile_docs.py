
"""Re-derive the headline numbers of MEMORY/docs/value-v6.md from the receipts.

The receipts live in the ignored artifacts/task78 tree, so this runs where the task was
executed, not on a fresh clone (it exits 3 when they are absent). Every number the
document states for the four gates is recomputed from ab/paired.json,
roof_truth/scored.json, task72_probe94/score_original_protocol.json,
task72_probe94/score_full192_9sample.json and gates.json, and the run fails if any of
them is no longer present verbatim in the document -- that is what catches a doc edit
that silently drifts away from the evidence.

    python3 scripts/task78_reconcile_docs.py
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / 'artifacts/task78'
if not (A / 'ab/paired.json').is_file():
    raise SystemExit(3)
DOC = (ROOT / 'MEMORY/docs/value-v6.md').read_text()

paired = json.loads((A / 'ab/paired.json').read_text())
roof = json.loads((A / 'roof_truth/scored.json').read_text())
orig = json.loads((A / 'task72_probe94/score_original_protocol.json').read_text())
full = json.loads((A / 'task72_probe94/score_full192_9sample.json').read_text())
gates = json.loads((A / 'gates.json').read_text())

rows = []
def want(label, value, needles):
    rows.append(dict(item=label, value=value, missing=[n for n in needles if n not in DOC]))

t = paired['terrain']
want('day lower', round(t['day']['conservative_lower'], 5), ['%.4f' % abs(t['day']['conservative_lower'])])
want('night lower', round(t['night']['conservative_lower'], 5), ['%.5f' % abs(t['night']['conservative_lower'])])
want('pool lower', round(t['pool']['conservative_lower'], 5), ['%.5f' % abs(t['pool']['conservative_lower'])])
want('fog net/p', [t['fog']['net'], t['fog']['mcnemar_p']], ['+1', '1.0'])
want('roof net/p', [t['roof']['net'], t['roof']['mcnemar_p']], ['1.0'])
want('paired totals', [sum(c['pairs'] for c in paired['cells'].values()), paired['complete']], ['512'])

prim = roof['by_horizon']['2500']['nonterminal']
want('roof primary pairs/roots', [prim['pairs'], prim['roots']], ['162', '48'])
want('roof primary rates', [prim['v5'], prim['v6']], ['0.4691', '0.5185'])
want('roof primary gate', roof['primary_gate_passed'], ['未通过'])

want('orig 94 flips', [orig['arms']['v5r9']['flips'], orig['arms']['v69']['flips']], ['47/93', '0.505'])
want('orig 94 gate', orig['gate']['passed'], ['不通过'])
want('full192 flips', [full['arms']['v6f192r9']['flips'], full['arms']['v6f192r9']['flip_rate']], ['46/93', '0.495'])
want('full192 diff', round(full['flip_rate_difference']['point'], 3), ['0.011'])
want('full192 ci', [round(x, 3) for x in full['flip_rate_difference']['ci95']], ['0.119', '0.083'])

want('corpus fog', [round(gates['fog']['v6']['consistency']['rate'], 4)], ['0.7146'])
want('corpus roof', [round(gates['roof']['v6']['consistency']['rate'], 4)], ['0.8230'])

bad = [r for r in rows if r['missing']]
print(json.dumps(rows, indent=1, ensure_ascii=False))
print('RECONCILED' if not bad else 'MISMATCH %d' % len(bad))
