"""task133 修订1：只追加新的关卡任务族，F/H 卡组由玩家选择。

种子为 task133 独立块 133,000,000 起，每关 1000 个位置中的前40个。
不扫描机器、不重放旧存档；旧任务、旧种子清单保留在 git 历史和原条目中。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEVELS = {"F": [42, 43, 44, 45, 47, 48, 49],
          "H": [2, 6, 8, 12, 16, 18, 22, 26, 28, 32, 36, 38]}


def generate(unlock_field, roof45_mode):
    manifest = ROOT / "configs" / "human_demo_tasks.json"
    document = json.loads(manifest.read_text())
    scoreboard = json.loads((ROOT / "configs" / "adventure_scoreboard.json").read_text())
    reference = {task["level"]: task for task in scoreboard["tasks"]}
    layers, seeds, new = {}, {}, []
    for layer, levels in LEVELS.items():
        layers[layer] = []
        for level in levels:
            task_id = "llmplay2_%s_l%d_m100" % (layer.lower(), level)
            block = 133_000_000 + len(new) * 1000
            seeds[task_id] = list(range(block, block + 40))
            layers[layer].append(task_id)
            ref = reference.get(level, {})
            task = dict(task_id=task_id, level=level, deck=[],
                        deck_selection="player", unlock_field=unlock_field,
                        reference_deck=ref.get("deck", []),
                        reference_task_id=ref.get("task_id"),
                        zombie_count_multiplier=1.0, seeds=[block],
                        description="卡组自选；参考卡组不强制。每关三套不同流派各赢两个种子，三种子稳健。")
            if ref.get("fog_observation"):
                task["fog_observation"] = ref["fog_observation"]
            if level == 45:
                task["reference_task_id"] = "demo_roof_45_orig_m100"
                task["description"] += " L45 原参考为蹦极闪电战原生传送带，无固定参考卡组。"
                if roof45_mode == "fixed":
                    task["description"] += " 本新族按修订1改为自选卡种卡模式，旧原生传送带族保留。"
                if roof45_mode == "original":
                    task.pop("deck_selection")
                    task["conveyor_start"] = "original"
            new.append(task)
    ids = {task["task_id"] for task in new}
    # 重新运行只更新本生成器管理的新族；旧条目逐字保留。
    lines = manifest.read_text().splitlines()
    lines = [line for line in lines if not any('"task_id":"%s"' % task_id in line for task_id in ids)]
    end = next(i for i in range(len(lines)-1, -1, -1) if lines[i].strip() == "]")
    if not lines[end-1].rstrip().endswith(","):
        lines[end-1] += ","
    added = ["    " + json.dumps(task, ensure_ascii=False, separators=(",", ":")) +
             ("," if i < len(new)-1 else "") for i, task in enumerate(new)]
    manifest.write_text("\n".join(lines[:end] + added + lines[end:]) + "\n")
    plan = dict(schema="llmplay_seeds_v1", block=dict(base=133_000_000, stride=1000, per_task=40),
                layers=layers, tasks=seeds)
    (ROOT / "configs" / "llmplay_seeds.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    print("已追加 F=%d / H=%d 个新族" % (len(layers["F"]), len(layers["H"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unlock-field", choices=["available_plants", "first_adventure_available"], required=True)
    parser.add_argument("--roof45-mode", choices=["fixed", "original"], required=True)
    args = parser.parse_args()
    generate(args.unlock_field, args.roof45_mode)
