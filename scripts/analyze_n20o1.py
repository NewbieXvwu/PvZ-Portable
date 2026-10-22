"""汇总 task28 的 2-10 DAgger 采集（beta=0：学生走、搜索打标签）。

`planted` 在 beta=0 下恒为空（执行者是学生，而学生在 2-10 上一棵不种），所以这里按**标签侧**统计：
每个决策的 `label` 是搜索的选择，`label.packet` 是当时的卡槽号，用同一帧 `packets[slot][1]` 翻成植物类型。
同时也统计实际执行（`action`）的种植数，用来对照“老师想种 vs 学生种了”。
"""
import argparse
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python')]
from pvz_frame_store import frame_bytes


def load_frames(archive: Path):
    return [json.loads(line) for line in frame_bytes(archive / 'frames.npz').decode().splitlines() if line.strip()]


def plant_type(packets, slot):
    if slot is None or not 0 <= slot < len(packets):
        return None
    entry = packets[slot]
    return entry[1] if isinstance(entry, list) else entry.get('type')


def summarize(archive: Path):
    frames = load_frames(archive)
    meta = json.loads((archive / 'meta.json').read_text())
    label_types = Counter()
    label_plants = Counter()
    action_plants = Counter()
    dealt = {}
    belt_full_tick = None
    mower_losses = []
    previous = None
    for frame in frames:
        packets = frame.get('packets') or []
        for entry in packets:
            slot, kind = (entry[0], entry[1]) if isinstance(entry, list) else (entry.get('index'), entry.get('type'))
            if kind is not None and kind >= 0 and kind not in dealt:
                dealt[kind] = frame['tick']
        if belt_full_tick is None and len(packets) >= 10 and all(
                ((e[1] if isinstance(e, list) else e.get('type')) or -1) >= 0 for e in packets):
            belt_full_tick = frame['tick']
        label = frame.get('label') or {}
        label_types[label.get('type')] += 1
        if label.get('type') == 'plant':
            kind = plant_type(packets, label.get('packet'))
            if kind is not None and kind >= 0:
                label_plants[kind] += 1
        action = frame.get('action') or {}
        if action.get('type') == 'plant' and not frame.get('action_rejected'):
            kind = plant_type(packets, action.get('packet'))
            if kind is not None and kind >= 0:
                action_plants[kind] += 1
        if previous is not None:
            before = {row: state for row, state in previous.get('mowers') or []}
            after = {row: state for row, state in frame.get('mowers') or []}
            for row, state in before.items():
                if state == 1 and after.get(row) == 2:
                    mower_losses.append(dict(tick=frame['tick'], wave=frame.get('wave'), row=row))
        previous = frame
    outcome = meta.get('outcome') or {}
    return dict(
        seed=meta['task']['seed'],
        won=bool(outcome.get('won')),
        final_wave=outcome.get('final_wave'),
        decisions=len(frames),
        label_types=dict(label_types),
        label_plants={str(k): v for k, v in sorted(label_plants.items())},
        label_plant_total=sum(label_plants.values()),
        action_plants={str(k): v for k, v in sorted(action_plants.items())},
        action_plant_total=sum(action_plants.values()),
        dealt_types=sorted(dealt),
        first_deal_tick=min(dealt.values()) if dealt else None,
        belt_full_tick=belt_full_tick,
        mower_losses=mower_losses,
        mowers_used=len(mower_losses),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--label', default='n20o1')
    parser.add_argument('--arm', default='fast_search')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    archives = sorted(args.frames.glob(f'seed_*/{args.label}_{args.arm}'))
    rows = [summarize(a) for a in archives]
    label_totals = Counter()
    action_totals = Counter()
    for row in rows:
        label_totals.update({int(k): v for k, v in row['label_plants'].items()})
        action_totals.update({int(k): v for k, v in row['action_plants'].items()})
    for row in rows:
        print(f"{row['seed']} won={row['won']} wave={row['final_wave']} dec={row['decisions']} "
              f"label_plants={row['label_plant_total']} action_plants={row['action_plant_total']} "
              f"mowers={row['mowers_used']} belt_full={row['belt_full_tick']}")
    summary = dict(games=len(rows), wins=sum(r['won'] for r in rows),
                   mean_final_wave=round(sum(r['final_wave'] or 0 for r in rows) / max(1, len(rows)), 2),
                   label_plants_by_type={str(k): v for k, v in sorted(label_totals.items())},
                   label_plant_total=sum(label_totals.values()),
                   action_plants_by_type={str(k): v for k, v in sorted(action_totals.items())},
                   action_plant_total=sum(action_totals.values()),
                   per_game=rows)
    print(json.dumps({k: v for k, v in summary.items() if k != 'per_game'}, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=1) + '\n')


if __name__ == '__main__':
    main()
