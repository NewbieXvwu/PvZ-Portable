"""task139/task182：循环式五层工作流的任务族与种子计划。

第5版：第1圈原定义保留；当前第2圈保留已完成的旧21关、追加7关；第3圈起主线22关。
C轨按配置的previous保留旧定义，新卡组使用llmplay5_新族和追加种子块。

产物（两个文件都由本脚本重写，其它地方不要手改）：

* `configs/human_demo_tasks.json` —— 追加 `llmplay3_` 前缀的任务族（A/C/E 三层）。
  旧任务族逐字保留：`AGENTS.md` §8.4「改卡组 = 换任务族，不覆盖旧名」。
* `configs/llmplay_seeds.json` —— 种子块与层内必做量（`llmplay_cycle_v1`）。

任务族
------
A 层（自选卡组）：屋顶 L41–L49 共 9 关 + 白天/夜间/泳池/雾夜 12 关，共 21 关。
E 层（加压）：每关 1.25 / 1.5 两档出怪倍数（旧 2.0 族保留供前次进度计数），卡组由 B 层下传。
C 层（指定卡组挑战）：`configs/llmplay3_challenge_decks.json` 的 50 条，
按关卡分三批（batch 1/2/3），第 N 圈用 `(N-1) % 3 + 1` 批，种子按圈换窗口。
第 1 圈 5 个种子全输的 `RETRY` 那几套在第 2 圈再给一次机会：任务 id 加 `_retry`、
进 batch 2、排在种子计划末尾（种子块全新），原任务与其余批次一字不动。

种子
----
每个任务 160 个种子，每圈用 40 个的窗口（`per_cycle`）；圈与圈之间不重叠。
窗口用完还没达标时允许继续往后取（不回退到更早的种子）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from pvz_constants import background_for_level  # noqa: E402

ROOF = [41, 42, 43, 44, 45, 46, 47, 48, 49]
H_LEVELS = [2, 6, 8, 12, 16, 18, 22, 26, 28, 32, 36, 38]
A_LEVELS = ROOF + H_LEVELS
RETIRED_LEVELS = {2, 6, 8, 12, 16, 18}
NEW_LEVELS = [33, 34, 39, 40, 27, 29, 50]
BASE = 139_000_000
STRIDE = 1000
PER_TASK = 160
PER_CYCLE = 40
BATCHES = 3

REQUIREMENTS = {
    "A": {"decks": 4, "wins_per_deck": 2, "min_card_changes": 3,
          "note": "4 套彼此差≥3 张、核心思路不同的卡组，每套赢 2 个不同种子"},
    "B": {"decks": 2, "robust_seeds_per_deck": 2, "checkpoint": "10", "k": 8, "wins": 5,
          "note": "A 层胜局最多的 2 套，各 2 个种子过第 10 波 k=8 赢≥5"},
    "C": {"seeds": 5, "wins": 2,
          "note": "指定卡组每套 5 个种子里赢≥2；5 个全输就写失败诊断，也算完成"},
    "D": {"strategies": 3, "wins_per_strategy": 2,
          "note": "B 层下传的每套卡组，3 种声明的不同开局节奏各赢 2 个种子"},
    "E": {"multipliers": [1.25, 1.5], "wins_per_multiplier": 2, "loss_streak": 5,
          "note": "B 层下传的每套卡组，1.25 与 1.5 倍各赢 2 个种子；连续 5 个新种子全输则写诊断完成"},
}


def reference(scoreboard):
    return {task["level"]: task for task in scoreboard["tasks"]}


def a_tasks(scoreboard):
    out = []
    for level in A_LEVELS:
        ref = reference(scoreboard).get(level) or {}
        task = dict(task_id="llmplay3_a_l%d_m100" % level, level=level, deck=[],
                    deck_selection="player", unlock_field="available_plants", layer="A",
                    reference_deck=list(ref.get("deck") or []),
                    reference_task_id=ref.get("task_id"),
                    zombie_count_multiplier=1.0,
                    description="自选卡组。每关 4 套彼此差≥3 张、核心思路不同的卡组，各赢 2 个种子；"
                                "胜局最多的 2 套下传 B/D/E 层。")
        if level == 45:
            task["reference_task_id"] = "demo_roof_45_orig_m100"
            task["description"] += " L45 原参考为蹦极闪电战原生传送带；本族按修订改为自选卡组，保留关卡自身出怪。"
        if ref.get("fog_observation"):
            task["fog_observation"] = ref["fog_observation"]
        out.append(task)
    return out


def e_tasks(*, reduced=False):
    out = []
    for level in A_LEVELS:
        for multiplier, tag in (((1.25, "m125"),) if reduced else ((1.5, "m150"), (2.0, "m200"))):
            out.append(dict(task_id="llmplay3_e_l%d_%s" % (level, tag), level=level, deck=[],
                            deck_selection="player", unlock_field="available_plants", layer="E",
                            pair_task="llmplay3_a_l%d_m100" % level,
                            zombie_count_multiplier=multiplier,
                            description="加压层：出怪 %g 倍，卡组必须用 B 层下传的那套；赢 2 个种子，或连续 5 个新种子全输写失败诊断。" % multiplier))
    return out


def c_tasks(challenges):
    """按关卡分三批：前 3 套进第 1 圈，其余对半分到第 2、3 圈。"""
    by_level = {}
    for row in challenges["decks"]:
        row = dict(row, **row.get("previous", {}))
        by_level.setdefault(row["level"], []).append(row)
    out, order = [], []
    for level in sorted(by_level):
        rows = by_level[level]
        hold = [row for row in rows if row["source"] == "holdout"]
        graded = sorted((row for row in rows if row["source"] != "holdout"),
                        key=lambda row: (row["won"], row["source_task_id"]))
        batches = {1: hold + graded[:3]}
        rest = graded[3:]
        half = (len(rest) + 1) // 2
        batches[2] = rest[:half]
        batches[3] = rest[half:]
        counter = 0
        for batch in (1, 2, 3):
            for row in batches[batch]:
                if row["source"] == "holdout":
                    kind = "hold"
                else:
                    counter += 1
                    kind = "d%02d" % counter
                task_id = "llmplay3_c_l%d_%s" % (level, kind)
                if task_id in order:
                    raise SystemExit("重复的 C 层任务 id：%s" % task_id)
                order.append(task_id)
                description = "指定卡组挑战（第 %d 圈批次）：不能换卡，5 个种子里赢≥2；5 个全输写失败诊断。" % batch
                if row["source"] == "holdout":
                    description += " 留出卡组 %s。" % row["source_task_id"]
                else:
                    description += " 来源 task129 %s（老师 %s/%s）。" % (
                        row["source_task_id"], row["won"], row["games"])
                out.append(dict(task_id=task_id, level=level, deck=list(row["deck"]), layer="C",
                                batch=batch, challenge_source=row["source"],
                                challenge_source_task_id=row["source_task_id"],
                                zombie_count_multiplier=1.0, description=description))
    return out, order


# 第 1 圈（batch 1）5 个指定种子全输、已写失败诊断的卡组：第 2 圈换新种子再给一次机会。
RETRY = ("llmplay3_c_l14_d02", "llmplay3_c_l14_d03", "llmplay3_c_l24_d02",
         "llmplay3_c_l42_d01", "llmplay3_c_l48_d02", "llmplay3_c_l48_d03")


def retry_tasks(tasks):
    """把 RETRY 里的卡组复制成 batch 2 的新任务：同一套卡、同一关，种子块全新。

    不改原任务 id，也不动其它批次，历史存档与 carry 里的记录继续对得上。
    """
    by_id = {task["task_id"]: task for task in tasks}
    out = []
    for task_id in RETRY:
        origin = by_id.get(task_id)
        if origin is None:
            raise SystemExit("重试任务找不到原卡组：%s" % task_id)
        out.append(dict(origin, task_id=task_id + "_retry", batch=2, retry_of=task_id,
                        description="第 2 圈重试（第 1 圈这套指定卡组 5 个种子全输并写了失败诊断）："
                                    "不能换卡，换 5 个新种子，赢≥2 才算过；再全输就再写一份诊断。"))
    return out


def write_manifest(tasks):
    path = ROOT / "configs" / "human_demo_tasks.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    kept = [task for task in document["tasks"]
            if not task["task_id"].startswith(("llmplay3_", "llmplay5_"))]
    kept.extend(tasks)
    lines = ["{", '  "tasks": [']
    body = ["    " + json.dumps(task, ensure_ascii=False, separators=(",", ":")) for task in kept]
    lines.append(",\n".join(body))
    lines.append("  ]")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plan(order):
    path = ROOT / "configs" / "llmplay_seeds.json"
    prior = json.loads(path.read_text(encoding="utf-8"))
    # 原任务顺序与种子块逐项沿用；新增族只追加，不挪动历史种子。
    existing = prior["tasks"]
    order = list(prior["order"]) + [task_id for task_id in order if task_id not in existing]
    tasks = dict(existing)
    block = max(max(seeds) for seeds in tasks.values()) // STRIDE * STRIDE + STRIDE
    for task_id in order:
        if task_id not in tasks:
            tasks[task_id] = list(range(block, block + PER_TASK))
            block += STRIDE
    document = dict(schema="llmplay_cycle_v1",
                    block=dict(base=BASE, stride=STRIDE, per_task=PER_TASK, per_cycle=PER_CYCLE),
                    requirements=REQUIREMENTS, batches=BATCHES, order=list(order), tasks=tasks)
    (ROOT / "configs" / "llmplay_seeds.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return document


def fifth_version(a, e, c, challenges):
    """task182：第1圈定义留用；换卡组新开族，第2圈起生效。"""
    for task in a + e:
        if task["level"] in RETIRED_LEVELS:
            task["last_cycle"] = 2
    added = []
    for level in NEW_LEVELS:
        from pvz_tasks import deck_for_level
        task_id = "llmplay5_a_l%d_m100" % level
        task = dict(task_id=task_id, level=level, deck=[], deck_selection="player",
                    unlock_field="available_plants", layer="A", first_cycle=2,
                    reference_deck=list(deck_for_level(level)), zombie_count_multiplier=1.0,
                    description=a[0]["description"])
        if level in (33, 34, 39, 40):
            task["fog_observation"] = "human_v2"
        if level == 50:
            task["robustness_checkpoint"] = "tick:6000"
        added.append(task)
        for multiplier, tag in ((1.25, "m125"), (1.5, "m150")):
            added.append(dict(task_id="llmplay5_e_l%d_%s" % (level, tag), level=level,
                              deck=[], deck_selection="player", unlock_field="available_plants",
                              layer="E", first_cycle=2, pair_task=task_id,
                              zombie_count_multiplier=multiplier,
                              **({"fog_observation": "human_v2"} if level in (33, 34, 39, 40) else {}),
                              description="加压层：沿用 B 层卡组，两胜或连续五败后写诊断。"))
    changed = {row["previous"]["source_task_id"]: row
               for row in challenges["decks"] if row.get("previous")}
    for task in c:
        row = changed.get(task["challenge_source_task_id"])
        if row is None:
            continue
        # 同一个旧留出源曾用于三关，按关卡找回对应替换项。
        row = next(row for row in challenges["decks"]
                   if row.get("previous", {}).get("source_task_id") == task["challenge_source_task_id"]
                   and row["level"] == task["level"])
        task["last_cycle"] = 1
        replacement = dict(task, task_id=task["task_id"].replace("llmplay3_", "llmplay5_", 1),
                           deck=list(row["deck"]), first_cycle=2,
                           challenge_source=row["source"], challenge_source_task_id=row["source_task_id"],
                           replaces=task["task_id"],
                           description="指定胜局卡组挑战（第 %d 批）：不能换卡，5 个种子赢≥2；"
                                       "未达标则按真实败局写诊断。来源 %s（L%d）。" %
                                       (task["batch"], row["source_task_id"], row["source_level"]))
        replacement.pop("last_cycle")
        replacement.pop("retry_of", None)
        added.append(replacement)
    return added


def validate(tasks):
    """卡组硬约束在生成期就查一遍：长度 6–10、不重复、地形覆盖。"""
    problems = []
    for task in tasks:
        deck = task["deck"]
        if task.get("layer") != "C":
            continue
        scene = background_for_level(task["level"])
        if not 6 <= len(deck) <= 10 or len(set(deck)) != len(deck) or any(not 0 <= card <= 47 for card in deck):
            problems.append((task["task_id"], "长度/重复/编号"))
        if scene in ("BACKGROUND_3_POOL", "BACKGROUND_4_FOG") and 16 not in deck:
            problems.append((task["task_id"], "水路缺睡莲"))
        if scene in ("BACKGROUND_5_ROOF", "BACKGROUND_6_BOSS") and 33 not in deck:
            problems.append((task["task_id"], "屋顶缺花盆"))
    if problems:
        raise SystemExit("C 层卡组违反硬约束：%s" % problems)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只校验，不写文件")
    args = parser.parse_args()
    scoreboard = json.loads((ROOT / "configs" / "adventure_scoreboard.json").read_text(encoding="utf-8"))
    challenges = json.loads((ROOT / "configs" / "llmplay3_challenge_decks.json").read_text(encoding="utf-8"))
    a, e = a_tasks(scoreboard), e_tasks()
    c, c_order = c_tasks(challenges)
    retry = retry_tasks(c)
    reduced = e_tasks(reduced=True)
    tasks = a + e + c + retry + reduced
    added = fifth_version(a, e + reduced, c + retry, challenges)
    tasks += added
    # task187：所有层的雾关统一使用生产观测；历史计数与存档不变。
    for task in tasks:
        if background_for_level(task["level"]) == "BACKGROUND_4_FOG":
            task["fog_observation"] = "human_v2"
    validate(tasks)
    # 重试任务排在最后：种子块按 order 序号分配，排在后面才拿得到没用过的新种子。
    order = ([task["task_id"] for task in a + e] + c_order
             + [task["task_id"] for task in retry + reduced + added])
    if args.check:
        print(json.dumps(dict(A=len(a), E=len(e), C=len(c), retry=len(retry), total=len(tasks)),
                         ensure_ascii=False))
        return 0
    write_manifest(tasks)
    plan = write_plan(order)
    print(json.dumps(dict(A=len(a), E=len(e), C=len(c), retry=len(retry), total=len(tasks),
                          seeds=len(plan["tasks"]), per_task=PER_TASK), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
