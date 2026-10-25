#!/usr/bin/env python3
"""task84: per-round training data composition behind the heldout regression.

The promotion gate saw the heldout zone (day_l7 d7p deck holdout) regress
significantly in rounds 13 and 14 while day/night stayed flat. This script
compares what each background round actually trained on, from the durable
source_rows.json reports the packer leaves next to every student.

Families (label grammar from curriculum_queue.label_for + the legacy DAgger corpus):
- c-labels (c<round><short>...) are curriculum collection; everything else is legacy.
- perturbed decks: day *_d7q, night *_n10q (the q-variants of the standard decks).
- multiplier tiers m<nnn>: m100 = 1.0, higher = harder variants.
- demo start cells: ...s<decision> (task71).
Run: python3 scripts/task84_data_composition.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROUNDS = (11, 12, 13, 14)
PIPELINE = Path('/home/newbiexvwu/pvz-logs/curriculum_state.pipeline')


def family(label: str) -> str:
    base = re.sub(r'_fast_search$', '', label)
    if not base.startswith('c'):
        return 'legacy_dag'
    body = re.sub(r'^c\d+', '', base)
    if re.search(r's\d+$', body):
        return 'demo_start'
    mult = re.search(r'm(\d{3})$', body)
    tier = 'm100' if (mult is None or mult.group(1) == '100') else 'm_high'
    if 'foguser' in body:
        fam = 'fog_user'
    elif 'roof' in body:
        fam = 'roof'
    elif 'demofog' in body:
        fam = 'demo_fog'
    elif re.search(r'd\d+', body):
        fam = 'day_d7q' if 'd7q' in body else 'day'
    elif re.search(r'n\d+', body):
        fam = 'night_n10q' if 'n10q' in body else 'night'
    elif re.search(r'p\d+', body):
        fam = 'pool'
    else:
        fam = 'other'
    return fam + '/' + tier


def main() -> None:
    tables = {}
    totals = {}
    for round_index in ROUNDS:
        folder = PIPELINE / ('round_%06d' % round_index)
        report = json.loads((folder / 'student/source_rows.json').read_text())
        table = {}
        for label, row in report['sources'].items():
            fam = family(label)
            table[fam] = table.get(fam, 0) + row['rows']
        tables[round_index] = table
        totals[round_index] = report['rows']
    families = sorted({fam for table in tables.values() for fam in table})
    header = '%-22s' % 'family'
    for r in ROUNDS:
        header += ('r%d rows' % r).rjust(12) + ('r%d share' % r).rjust(11)
    print(header)
    for fam in families:
        cells = ''
        for r in ROUNDS:
            cells += '%12d' % tables[r].get(fam, 0)
        for r in ROUNDS:
            cells += ('%11.1f%%' % (100.0 * tables[r].get(fam, 0) / totals[r]))
        print('%-22s' % fam + cells)
    for r in ROUNDS:
        print('round %d: total rows %d' % (r, totals[r]))


if __name__ == '__main__':
    main()
