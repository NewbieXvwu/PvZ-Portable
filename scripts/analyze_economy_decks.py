"""task28 第 5 步：用已有档案统计“缺经济卡的扰动卡组”表现（不跑新局）。

数据源 `~/PvZ-search/artifacts/student/frames_v2/seed_*/<label>_fast_search`：
`meta.json` 里有**实际初始卡组**（`task.deck`）与结果（`outcome.won`），`frames.npz` 里有逐决策的 `label`。

分组：白天 d7p（`d{7,8,9,10}p{r}{d,m}`，经济卡 = 向日葵 1）、夜间 n10（`n{r}t{r}`，经济卡 = 阳光菇 9）、
泳池 p9k（`p{r}dag{r}`，经济卡 = 向日葵 1）。
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from pvz_frame_store import frame_bytes

FAMILIES = [
    ('day_d7p', re.compile(r'^d(?:7|8|9|10)p\d+[dm]$'), 1),
    ('night_n10', re.compile(r'^n\d+t\d+$'), 9),
    ('pool_p9k', re.compile(r'^p\d+dag\d+$'), 1),
]

# 这些族的 meta 里没有 deck（不是 deck_perturb 任务），卡组是固定的。
FAMILY_DECKS = {
    'night_n10': [0, 1, 2, 3, 4, 5, 7, 8, 9, 11],
    'pool_p9k': [0, 1, 2, 3, 4, 5, 7, 16, 19],
}


def family_of(label: str):
    for name, pattern, economy in FAMILIES:
        if pattern.match(label):
            return name, economy
    return None, None


def deck_table(manifests):
    table = {}
    for path in manifests:
        data = json.loads(Path(path).read_text())
        entries = data['tasks'] if isinstance(data, dict) and 'tasks' in data else data
        for task in entries:
            if isinstance(task, dict) and task.get('task_id') and task.get('deck'):
                table[task['task_id']] = list(task['deck'])
    return table


def scan(root: Path, decks=None, only=None):
    rows = []
    decks = decks or {}
    for seed_dir in sorted(root.glob('seed_*')):
        for archive in sorted(seed_dir.iterdir()):
            if not archive.is_dir() or not archive.name.endswith('_fast_search'):
                continue
            label = archive.name[:-len('_fast_search')]
            family, economy = family_of(label)
            if family is None or (only and family not in only):
                continue
            meta_path = archive / 'meta.json'
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text())
            task = meta.get('task') or {}
            deck = task.get('deck') or decks.get(task.get('task_id')) or FAMILY_DECKS.get(family)
            if deck is None:
                continue
            rows.append(dict(family=family, economy=economy, label=label, archive=archive,
                             deck=list(deck), won=bool((meta.get('outcome') or {}).get('won')),
                             wave=(meta.get('outcome') or {}).get('final_wave'),
                             task_id=task.get('task_id')))
    return rows


def label_stats(archive: Path):
    """搜索在整局里判“种”多少次、判的是什么植物（槽号用同一帧 packets 翻）。"""
    plants = Counter()
    waits = 0
    total = 0
    for line in frame_bytes(archive / 'frames.npz').decode().splitlines():
        if not line.strip():
            continue
        frame = json.loads(line)
        total += 1
        label = frame.get('label') or {}
        if label.get('type') == 'plant':
            packets = frame.get('packets') or []
            slot = label.get('packet')
            kind = None
            if isinstance(slot, int) and 0 <= slot < len(packets):
                entry = packets[slot]
                kind = entry[1] if isinstance(entry, list) else entry.get('type')
            plants[kind] += 1
        elif label.get('type') == 'wait':
            waits += 1
    return total, waits, plants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=Path, default=Path.home() / 'PvZ-search/artifacts/student/frames_v2')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--manifest', type=Path, nargs='*', default=[],
                        help='manifests with task_id -> deck for archives that predate deck_perturb')
    parser.add_argument('--family', nargs='*', default=[], help='restrict to these families')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    rows = scan(args.frames, deck_table(args.manifest), set(args.family) or None)
    print(f'archives: {len(rows)}')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        stats = list(pool.map(lambda row: label_stats(row['archive']), rows))
    groups = defaultdict(list)
    for row, (total, waits, plants) in zip(rows, stats):
        row.update(decisions=total, label_waits=waits, label_plants=sum(plants.values()),
                   label_plant_types=dict(plants))
        groups[(row['family'], row['economy'] in row['deck'])].append(row)
    summary = {}
    for (family, has_economy), items in sorted(groups.items()):
        wins = sum(item['won'] for item in items)
        key = f'{family}|economy={has_economy}'
        summary[key] = dict(
            games=len(items), wins=wins, win_rate=round(wins / len(items), 3),
            mean_final_wave=round(sum((item['wave'] or 0) for item in items) / len(items), 2),
            mean_label_plants=round(sum(item['label_plants'] for item in items) / len(items), 2),
            games_with_zero_label_plants=sum(1 for item in items if item['label_plants'] == 0),
            mean_decisions=round(sum(item['decisions'] for item in items) / len(items), 1),
            label_plant_types=dict(sorted(Counter(
                {k: v for item in items for k, v in item['label_plant_types'].items()}).items())),
        )
    for key, value in summary.items():
        print(key, json.dumps(value, ensure_ascii=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=1) + '\n')


if __name__ == '__main__':
    main()
