"""task23 step 6.3: why the teacher never picks the sun-shroom (9) on n9 night games.

Read-only. The teacher archives store, for every search decision, 'targets' = the search's candidate
values as [packet, row, col, value] (packet == -1 is "do nothing now"). This script reads them and
reports, per family:

  * how many decisions had a sun-shroom candidate at all, and how often it was the best plant candidate;
  * the value gap between the best plant candidate / the chosen action and the sun-shroom candidate;
  * the same for grave buster (11) on the n10 family, to show whether 11 is even offered or chosen;
  * affordability at those decisions, from the frame's own packet list and sun.

No games, no writes outside the report.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'scripts')]
import numpy as np
from pvz_frame_store import read_frames

SUN_SHROOM = 9
GRAVE_BUSTER = 11


def frame_path(archive: Path) -> Path:
    return archive/'frames.npz' if (archive/'frames.npz').exists() else archive/'frames.jsonl'


def archives(root: Path, first: int, last: int):
    for folder in sorted(root.iterdir()):
        if not folder.name.startswith('seed_'):
            continue
        seed = int(folder.name.split('_', 1)[1])
        if not first <= seed <= last:
            continue
        for archive in sorted(folder.iterdir()):
            if frame_path(archive).exists():
                yield archive, seed


def value_of(targets, slot):
    best = None
    for target_slot, _row, _col, value in targets:
        if target_slot == slot:
            best = value if best is None else max(best, value)
    return best


def analyze(archive, seed, wanted, rows):
    frames = read_frames(frame_path(archive))
    meta = json.loads((archive/'meta.json').read_text())
    level = meta['context']['level']
    for index, frame in enumerate(frames[:-1]):
        targets = frame.get('targets')
        if not targets:
            continue
        # A target's packet field is the seed-bank slot, not the plant type; the same slot can hold a
        # different type under a different deck. Translate through this frame's own packet list.
        slot_type = {int(entry[0]): int(entry[1]) for entry in frame['packets']}
        plants = [(slot_type.get(slot, -1), row, col, value) for slot, row, col, value in targets if slot >= 0]
        wait = next((value for slot, _r, _c, value in targets if slot < 0), None)
        label = frames[index + 1].get('label') or {}
        chosen = value_of(targets, label.get('packet', -1)) if label.get('type') == 'plant' else wait
        row = dict(level=level, seed=seed, tick=frame.get('tick'), sun=frame.get('sun'),
                   wait=wait, chosen_value=chosen, label_type=label.get('type'),
                   label_packet=label.get('packet'), candidates=len(plants))
        for packet in wanted:
            values = [value for kind, _r, _c, value in plants if kind == packet]
            if not values:
                continue
            slots = {slot for slot, kind in slot_type.items() if kind == packet}
            row[f'p{packet}_value'] = max(values)
            row[f'p{packet}_best'] = max(values) >= max(value for _s, _r, _c, value in plants)
            row[f'p{packet}_affordable'] = any(entry[0] in slots and entry[2] and frame['sun'] >= entry[5]
                                              for entry in frame['packets'])
            row[f'p{packet}_chosen'] = label.get('type') == 'plant' and label.get('packet') in slots
            row[f'p{packet}_gap_to_best'] = max(value for _s, _r, _c, value in plants) - max(values)
            row[f'p{packet}_gap_to_chosen'] = (chosen - max(values)) if chosen is not None else None
            if row[f'p{packet}_best'] and chosen is not None:
                row[f'p{packet}_best_minus_chosen'] = max(values) - chosen
        rows.append(row)


def summarize(rows, packet):
    subset = [row for row in rows if f'p{packet}_value' in row]
    if not subset:
        return dict(decisions_with_candidate=0)
    values = np.array([row[f'p{packet}_value'] for row in subset])
    gap_best = np.array([row[f'p{packet}_gap_to_best'] for row in subset])
    gap_chosen = np.array([row[f'p{packet}_gap_to_chosen'] for row in subset if row[f'p{packet}_gap_to_chosen'] is not None])
    best = [row for row in subset if row[f'p{packet}_best'] and f'p{packet}_best_minus_chosen' in row]
    margin = np.array([row[f'p{packet}_best_minus_chosen'] for row in best])
    return dict(
        decisions_where_best=len(best),
        best_but_not_chosen_share=float(np.mean([not row[f'p{packet}_chosen'] for row in best])) if best else None,
        best_minus_chosen_mean=float(margin.mean()) if len(margin) else None,
        best_minus_chosen_median=float(np.median(margin)) if len(margin) else None,
        best_minus_chosen_over_margin_share=float((margin > 0.02).mean()) if len(margin) else None,
        decisions_with_candidate=len(subset),
        share_of_all_search_decisions=len(subset)/len(rows),
        mean_candidate_value=float(values.mean()), median_candidate_value=float(np.median(values)),
        best_plant_candidate_share=float(np.mean([row[f'p{packet}_best'] for row in subset])),
        chosen_share=float(np.mean([row[f'p{packet}_chosen'] for row in subset])),
        mean_gap_to_best_plant=float(gap_best.mean()),
        median_gap_to_best_plant=float(np.median(gap_best)),
        mean_gap_chosen_minus_candidate=float(gap_chosen.mean()) if len(gap_chosen) else None,
        share_gap_within_0_02=float((gap_best <= 0.02).mean()),
        share_candidate_affordable=float(np.mean([row[f'p{packet}_affordable'] for row in subset])),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--frames-root', type=Path, default=Path('/home/newbiexvwu/PvZ-search/artifacts/student/frames_v2'))
    p.add_argument('--n9-first', type=int, default=1900000)
    p.add_argument('--n9-last', type=int, default=1929439)
    p.add_argument('--n10-first', type=int, default=2200000)
    p.add_argument('--n10-last', type=int, default=2200127)
    p.add_argument('--limit', type=int, default=40, help='archives per family')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if not args.out.is_absolute():
        p.error('--out must be absolute')
    report = {}
    for family, first, last, wanted in (('n9', args.n9_first, args.n9_last, [SUN_SHROOM]),
                                        ('n10', args.n10_first, args.n10_last, [SUN_SHROOM, GRAVE_BUSTER])):
        rows = []
        used = []
        for archive, seed in archives(args.frames_root, first, last):
            if len(used) >= args.limit:
                break
            analyze(archive, seed, wanted, rows)
            used.append(str(archive))
            print(json.dumps({'family': family, 'archives': len(used), 'decisions': len(rows)}), flush=True)
        report[family] = dict(archives=len(used), archive_list=used, search_decisions=len(rows),
                              levels={str(level): count for level, count in sorted(Counter(r['level'] for r in rows).items())},
                              labels={str(kind): count for kind, count in sorted(Counter(r['label_type'] for r in rows).items())},
                              sun_shroom=summarize(rows, SUN_SHROOM),
                              **({'grave_buster': summarize(rows, GRAVE_BUSTER)} if GRAVE_BUSTER in wanted else {}))
    report['note'] = ('values are the search leaf values stored in each teacher frame; packet -1 is the '
                      '"do nothing now" candidate; gaps are best_plant_value - sun_shroom_value, so a '
                      'positive gap means the sun-shroom scored below the best plant candidate.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({family: {key: value for key, value in part.items() if key != 'archive_list'}
                      for family, part in report.items() if isinstance(part, dict)}, indent=2))


if __name__ == '__main__':
    main()
