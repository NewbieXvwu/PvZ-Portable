#!/usr/bin/env python3
"""task81c：登记 LLM 打全程的任务分层与 40 种子块。

做三件事（都可重复执行）：

1. 把 llmplay 需要、但仓库清单里还没有的任务族补进 `configs/human_demo_tasks.json`
   （单行格式，追加在末尾，不重排已有条目）：
   - B 层：L41 的 doom 族（`demo_roof_41_doom_m200`）；
   - C 层：L46–L49 doom 族的 3.0 / 4.0 倍（`_m300` / `_m400`）；
   - D 层：fog79 族 L36–L39 的 2.0 倍（`_m200`）；
   - E 层：原生传送带 L30 / L40（`deck: []` + `conveyor_start: "original"`）。
2. 给每个 llmplay 任务分配 40 个种子，写 `configs/llmplay_seeds.json`
   （新独立块 81,000,000 起，每任务步长 1000）。
3. 机械核对：新块与仓库里**所有**清单的种子不相交、任务之间不相交、
   与稳健性块（`configs/llmplay_robustness.json`）不相交。

卡组不手写：B/C/D 直接复制已登记族的 deck（AGENTS §8：改卡组 = 换任务族），
E 用原版传送带（引擎发牌，没有任务卡组）。

用法：
  python3 scripts/llmplay/make_llmplay_tasks.py --dry-run
  python3 scripts/llmplay/make_llmplay_tasks.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
sys.path.insert(0, str(ROOT / 'scripts'))

MANIFEST = ROOT / 'configs' / 'human_demo_tasks.json'
SEEDS = ROOT / 'configs' / 'llmplay_seeds.json'
ROBUSTNESS = ROOT / 'configs' / 'llmplay_robustness.json'


def _refuse_cycle_plan():
    """task139 起种子计划换了 schema；旧生成器再跑一次会静默毁掉循环计划。"""
    if SEEDS.is_file():
        schema = json.loads(SEEDS.read_text(encoding='utf-8')).get('schema')
        if schema and schema != 'llmplay_seeds_v1':
            raise SystemExit('configs/llmplay_seeds.json 现在是 %s；本脚本只写 llmplay_seeds_v1，'
                             '新计划请用 scripts/llmplay/make_llmplay3_tasks.py。' % schema)

BLOCK_BASE = 81_000_000
BLOCK_STRIDE = 1_000
SEEDS_PER_TASK = 40

# 层 -> 任务。C/D 的族定义从 base 复制（同关卡、同卡组、同观测），只改 id / 倍率。
LAYERS = {
    'A': ['demo_roof_46_doom_m200', 'demo_roof_47_doom_m200',
          'demo_roof_48_doom_m200', 'demo_roof_49_doom_m200'],
    # kernel 三个族不打（主控 2026-10-26：用户定过玉米套不打，群怪下溅射远强于单体控制）。
    # 族登记留在 configs/human_demo_tasks.json，只是不进 llmplay 任务表。
    'B': ['demo_roof_41_doom_m200',
          'demo_roof_42_melon_m200', 'demo_roof_42_gloom_m200',
          'demo_roof_42_gloommelon_m200',
          'demo_roof_43_melon_m200', 'demo_roof_43_gloom_m200',
          'demo_roof_44_melon_m200', 'demo_roof_44_gloom_m200'],
    'C': ['demo_roof_46_doom_m300', 'demo_roof_46_doom_m400',
          'demo_roof_47_doom_m300', 'demo_roof_47_doom_m400',
          'demo_roof_48_doom_m300', 'demo_roof_48_doom_m400',
          'demo_roof_49_doom_m300', 'demo_roof_49_doom_m400'],
    'D': ['fog79_l36_snow_magnet_v2_m200', 'fog79_l36_snow_split_v2_m200',
          'fog79_l37_snow_magnet_v2_m200', 'fog79_l37_snow_split_v2_m200',
          'fog79_l38_snow_magnet_v2_m200', 'fog79_l38_snow_tall_v2_m200',
          'fog79_l39_snow_cactus_v2_m200', 'fog79_l39_snow_blover_v2_m200'],
    # E 层命名对齐主控：task85c 在 boss50 上登记的是 demo_pool_30_orig_m100 /
    # demo_boss_50_orig_m100，这里 L30 用同一个名字，避免两条线合并后出现两个同义族。
    # L50 的族定义随 boss50 合入 pvz-env（2026-10-26 合并）进入清单，种子合并成
    # [20261050, 81033000]（主控指示：两条线的种子合成一条）。
    'E': ['demo_pool_30_orig_m100', 'demo_storm_40_orig_m100', 'demo_boss_50_orig_m100'],
}

# 需要新建的族：id -> 复制来源（None = 自己写全字段）
NEW_FAMILIES = {
    'demo_roof_41_doom_m200': ('demo_roof_46_doom_m200', dict(level=41)),
    'demo_roof_46_doom_m300': ('demo_roof_46_doom_m200', dict(zombie_count_multiplier=3.0)),
    'demo_roof_46_doom_m400': ('demo_roof_46_doom_m200', dict(zombie_count_multiplier=4.0)),
    'demo_roof_47_doom_m300': ('demo_roof_47_doom_m200', dict(zombie_count_multiplier=3.0)),
    'demo_roof_47_doom_m400': ('demo_roof_47_doom_m200', dict(zombie_count_multiplier=4.0)),
    'demo_roof_48_doom_m300': ('demo_roof_48_doom_m200', dict(zombie_count_multiplier=3.0)),
    'demo_roof_48_doom_m400': ('demo_roof_48_doom_m200', dict(zombie_count_multiplier=4.0)),
    'demo_roof_49_doom_m300': ('demo_roof_49_doom_m200', dict(zombie_count_multiplier=3.0)),
    'demo_roof_49_doom_m400': ('demo_roof_49_doom_m200', dict(zombie_count_multiplier=4.0)),
    'fog79_l36_snow_magnet_v2_m200': ('fog79_l36_snow_magnet_v2', dict(zombie_count_multiplier=2.0)),
    'fog79_l36_snow_split_v2_m200': ('fog79_l36_snow_split_v2', dict(zombie_count_multiplier=2.0)),
    'fog79_l37_snow_magnet_v2_m200': ('fog79_l37_snow_magnet_v2', dict(zombie_count_multiplier=2.0)),
    'fog79_l37_snow_split_v2_m200': ('fog79_l37_snow_split_v2', dict(zombie_count_multiplier=2.0)),
    'fog79_l38_snow_magnet_v2_m200': ('fog79_l38_snow_magnet_v2', dict(zombie_count_multiplier=2.0)),
    'fog79_l38_snow_tall_v2_m200': ('fog79_l38_snow_tall_v2', dict(zombie_count_multiplier=2.0)),
    'fog79_l39_snow_cactus_v2_m200': ('fog79_l39_snow_cactus_v2', dict(zombie_count_multiplier=2.0)),
    'fog79_l39_snow_blover_v2_m200': ('fog79_l39_snow_blover_v2', dict(zombie_count_multiplier=2.0)),
    # 原生传送带：卡由引擎发，任务没有卡组（pvz_tasks.deck_for_level 的约定）。
    'demo_pool_30_orig_m100': (None, dict(level=30, deck=[], conveyor_start='original',
                                          zombie_count_multiplier=1.0)),
    'demo_storm_40_orig_m100': (None, dict(level=40, deck=[], conveyor_start='original',
                                           zombie_count_multiplier=1.0)),
    'demo_boss_50_orig_m100': (None, dict(level=50, deck=[], conveyor_start='original',
                                          zombie_count_multiplier=1.0)),
}


# 一次性改名遗留（2026-10-26）：E 层原来叫 demo_conv_*，与 task85c 的
# demo_pool_30_orig_m100 同义；生成器会把旧条目从清单里清掉。
LEGACY_TASK_IDS = {'demo_conv_30_orig_m100', 'demo_conv_40_orig_m100'}


def ordered_tasks():
    return [(layer, task_id) for layer in 'ABCDE' for task_id in LAYERS[layer]]


def block_for(slot: int) -> list[int]:
    start = BLOCK_BASE + slot * BLOCK_STRIDE
    return [start + i for i in range(SEEDS_PER_TASK)]


def build_missing(known: dict) -> list[dict]:
    missing = []
    for layer, task_id in ordered_tasks():
        if task_id in known:
            continue
        source_id, overrides = NEW_FAMILIES[task_id]
        if source_id is None:
            entry = dict(task_id=task_id, **overrides)
        else:
            source = known[source_id]
            entry = dict(source)
            entry.pop('seeds', None)
            # A demonstration belongs to its original task/seed/environment.
            entry.pop('demo_start', None)
            entry.update(overrides)
            entry['task_id'] = task_id
        missing.append(entry)
    return missing


def read_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding='utf-8'))


def write_manifest(document: dict, tasks: list[dict]) -> None:
    """保持现有单行格式：每条任务一行，不重排、不整文件 json.dumps。"""
    lines = ['{', '  "tasks": [']
    for index, task in enumerate(tasks):
        body = json.dumps(task, ensure_ascii=False, separators=(',', ':'))
        lines.append('    ' + body + (',' if index < len(tasks) - 1 else ''))
    lines += ['  ]', '}']
    MANIFEST.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def scan_roots() -> list[Path]:
    """本工作树 + 主仓库（工作树里的 artifacts 往往不全，主仓库才有机器的全部清单）。"""
    roots = [ROOT]
    for parent in ROOT.parents:
        if (parent / 'configs' / 'human_demo_tasks.json').is_file() and parent != ROOT:
            roots.append(parent)
            break
    return roots


def all_seeds_except(exclude: set[Path]) -> dict[int, str]:
    found: dict[int, str] = {}
    paths: list[Path] = []
    for root in scan_roots():
        paths.extend(root.rglob('*.json'))
    for path in paths:
        if path in exclude or '.git' in path.parts:
            continue
        try:
            text = path.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError):
            continue
        if '"seeds"' not in text:
            continue
        try:
            document = json.loads(text)
        except json.JSONDecodeError:
            continue
        stack = [document]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == 'seeds' and isinstance(value, list):
                        for seed in value:
                            if isinstance(seed, int):
                                found.setdefault(seed, str(path))
                    else:
                        stack.append(value)
            elif isinstance(node, list):
                stack.extend(node)
    return found


def robustness_block() -> list[int]:
    if not ROBUSTNESS.is_file():
        return []
    document = json.loads(ROBUSTNESS.read_text(encoding='utf-8'))
    seeds = []
    for entry in document.get('seeds') or []:
        seeds.extend(entry.get('seeds') or [])
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    _refuse_cycle_plan()
    document = read_manifest()
    tasks = [task for task in document['tasks'] if task['task_id'] not in LEGACY_TASK_IDS]
    known = {task['task_id']: task for task in tasks}
    missing = build_missing(known)
    print('已有 %d 条，新增 %d 条：%s' % (len(tasks), len(missing),
                                       ', '.join(task['task_id'] for task in missing) or '（无）'))

    # 新块分配：按层顺序，每个任务 40 个连续种子。
    plan = {}
    for slot, (layer, task_id) in enumerate(ordered_tasks()):
        plan[task_id] = dict(layer=layer, slot=slot, seeds=block_for(slot))
    flat = [seed for item in plan.values() for seed in item['seeds']]
    if len(flat) != len(set(flat)):
        raise SystemExit('新块内部有重复种子')

    # 新族默认种子 = 自己的第一个块种子（清单里只写一个）。
    for task in missing:
        task['seeds'] = [plan[task['task_id']]['seeds'][0]]

    # 生成器管理的族（NEW_FAMILIES）：清单默认种子同步为当前块首。
    # 槽位前移（例如 kernel 出表）时自愈；演示种子（< BLOCK_BASE，如 20261030）保留在前。
    seed_synced = False
    for task in tasks:
        if task['task_id'] not in NEW_FAMILIES:
            continue
        demo = [x for x in task.get('seeds', []) if x < BLOCK_BASE][:1]
        want = demo + [plan[task['task_id']]['seeds'][0]]
        if task.get('seeds') != want:
            task['seeds'] = want
            seed_synced = True
    if seed_synced:
        print('清单默认种子同步：生成器管理的族改为当前块首（演示种子保留在前）')

    known_after = dict(known)
    known_after.update({task['task_id']: task for task in missing})
    unknown = [task_id for _, task_id in ordered_tasks() if task_id not in known_after]
    if unknown:
        raise SystemExit('缺任务定义：%s' % unknown)

    # 机械核对：新块与仓库其它所有清单不相交。
    others = all_seeds_except({SEEDS})
    # 本方案自己的种子（含清单里 33 个任务的默认种子）不算「外来」。
    other_values = set(others) - set(flat)
    clash = sorted(set(flat) & other_values)
    if clash:
        raise SystemExit('新块与现有清单冲突：%s（例 %s）' % (len(clash), clash[:5]))
    robust = set(robustness_block())
    robust_clash = sorted(set(flat) & robust)
    if robust_clash:
        raise SystemExit('新块与稳健性块冲突：%s' % robust_clash[:5])
    print('块 %d..%d，与现有 %d 个种子条目、稳健性 %d 个种子零冲突'
          % (min(flat), max(flat), len(others), len(robust)))

    if args.dry_run:
        for layer in 'ABCDE':
            ids = LAYERS[layer]
            print('  %s 层 %d 任务：%s' % (layer, len(ids), ', '.join(ids)))
        return 0

    if missing or seed_synced or len(tasks) != len(document['tasks']):
        write_manifest(document, tasks + missing)
    SEEDS.write_text(json.dumps(dict(
        schema='llmplay_seeds_v1',
        block=dict(base=BLOCK_BASE, stride=BLOCK_STRIDE, per_task=SEEDS_PER_TASK),
        layers={layer: list(ids) for layer, ids in LAYERS.items()},
        tasks={task_id: plan[task_id]['seeds'] for _, task_id in ordered_tasks()},
    ), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('写入 %s 与 %s' % (MANIFEST.relative_to(ROOT), SEEDS.relative_to(ROOT)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
