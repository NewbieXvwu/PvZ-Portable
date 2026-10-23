#!/usr/bin/env python3
"""task26: 从帧档案里读 2-10 原版传送带局的发卡 / 种植 / 丢车。"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/home/newbiexvwu/PvZ-search/.claude/worktrees/l20diag')
sys.path.insert(0, str(ROOT / 'python'))
from pvz_frame_store import frame_bytes  # noqa: E402

PLANTS = {0: 'peashooter', 1: 'sunflower', 2: 'cherry', 3: 'wallnut', 4: 'potatomine', 5: 'snowpea',
          6: 'chomper', 7: 'repeater', 8: 'puff', 9: 'sunshroom', 10: 'fume', 11: 'gravebuster',
          12: 'hypno', 13: 'scaredy', 14: 'ice', 15: 'doom', 16: 'lily', 19: 'tangle', 43: 'cattail'}
ZOMBIES = {0: 'normal', 1: 'flag', 2: 'cone', 3: 'pole', 4: 'pail', 5: 'newspaper', 6: 'door',
           7: 'football', 8: 'dancer', 9: 'backup', 23: 'garg'}


def name(mapping: dict[int, str], value: int) -> str:
    return mapping.get(value, f'#{value}')


def analyse(archive: Path) -> dict:
    lines = frame_bytes(archive / 'frames.npz').splitlines()
    frames = [json.loads(line) for line in lines]
    first = frames[0]
    last = frames[-1]

    dealt: Counter = Counter()
    first_seen: dict[int, int] = {}
    prev = Counter()
    planted: Counter = Counter()
    planted_cells: dict[tuple[int, int], tuple[int, int]] = {}
    mower_losses = []
    # mowers: [row, state]，1 = 待命，2 = 正在跑；跑完这一行直接从列表里消失。
    mower_state = {row: state for row, state in first['mowers']}
    max_sun = 0
    last_zombies: list = []
    graves0 = len(first.get('graves') or [])

    for frame in frames:
        current = Counter(packet[1] for packet in frame['packets'] if packet[1] != -1)
        for seed_type, count in current.items():
            if count > prev.get(seed_type, 0):
                dealt[seed_type] += count - prev.get(seed_type, 0)
                first_seen.setdefault(seed_type, frame['tick'])
        prev = current
        for row, col, seed_type, *_ in frame['plants']:
            key = (row, col)
            if key not in planted_cells:
                planted_cells[key] = (seed_type, frame['tick'])
                planted[seed_type] += 1
        # 丢车 = 该行状态从 1 变成 2（车开始跑）；用上一帧的僵尸说明是哪只触发的。
        for row, state in frame['mowers']:
            if mower_state.get(row) == 1 and state != 1:
                nearby = [(name(ZOMBIES, z[1]), round(z[2], 1)) for z in last_zombies
                          if z[0] == row and z[2] < 320]
                mower_losses.append(dict(row=row, tick=frame['tick'], wave=frame['wave'],
                                         zombies=nearby))
            mower_state[row] = state
        for row in set(mower_state) - {row for row, _state in frame['mowers']}:
            mower_state.pop(row, None)
        max_sun = max(max_sun, frame['sun'])
        last_zombies = frame['zombies']

    return dict(
        ticks=last['tick'], final_wave=last['wave'], won=bool(last.get('result') == 1),
        dealt={name(PLANTS, k): v for k, v in sorted(dealt.items())},
        first_seen={name(PLANTS, k): v for k, v in sorted(first_seen.items())},
        planted={name(PLANTS, k): v for k, v in sorted(planted.items())},
        mower_losses=mower_losses, max_sun=max_sun, graves0=graves0,
        graves_end=len(last.get('graves') or []), sun_end=last['sun'],
        lanes_with_plants=sorted({row for row, _ in planted_cells}),
        cells=len(planted_cells),
    )


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / 'artifacts/task26/frames')
    label = sys.argv[2] if len(sys.argv) > 2 else 'l20s_fast'
    rows = []
    for seed_dir in sorted(root.glob('seed_*')):
        archive = seed_dir / label
        if not (archive / 'frames.npz').exists():
            continue
        meta = json.loads((archive / 'meta.json').read_text())
        row = analyse(archive)
        row['seed'] = meta['task']['seed']
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
    if rows:
        print('--- summary')
        print('games', len(rows), 'won', sum(r['won'] for r in rows),
              'mean final wave', round(sum(r['final_wave'] for r in rows) / len(rows), 2),
              'mean mowers lost', round(sum(len(r['mower_losses']) for r in rows) / len(rows), 2))
        dealt_total: Counter = Counter()
        planted_total: Counter = Counter()
        for r in rows:
            dealt_total.update(r['dealt']); planted_total.update(r['planted'])
        print('dealt total', dict(dealt_total))
        print('planted total', dict(planted_total))
        loss_waves = Counter(m['wave'] for r in rows for m in r['mower_losses'])
        print('mower loss waves', dict(sorted(loss_waves.items())))
        loss_rows = Counter(m['row'] for r in rows for m in r['mower_losses'])
        print('mower loss rows', dict(sorted(loss_rows.items())))
        print('first seen ticks', {k: round(sum(r['first_seen'].get(k, 0) for r in rows if k in r['first_seen']) / max(1, sum(1 for r in rows if k in r['first_seen']))) for k in sorted({k for r in rows for k in r['first_seen']})})


if __name__ == '__main__':
    main()
