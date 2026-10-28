"""task139：循环式五层工作流（A 探索 → B 稳健 → C 挑战 → D 打法 → E 加压）。

这一层只做**判定与排序**，不碰引擎，也不写交付文件：

* `derive()` 把存档列表算成「第几圈 / 第几层 / 每项还差什么」；
* `next_task()` **只**返回当前圈当前层的未完成项，所以不能跳层、不能自选顺序；
* `check()` 在 `Game()` 建局时把关：层不对、卡组差 <3 张、打法重复、指定卡组被换掉，
  一律报错并直接告诉你现在该做什么；
* `report()` 每层自动生成小结（做了什么、下传了什么）。

一圈做完自动进下一圈，`progress()["complete"]` 永远是 `False`：这个任务没有终点，
停止只由用户在对话里决定。

存档里带 `cycle` 字段（导出时由 `delivery.current_cycle()` 写入）。同一套卡组在不同圈
的胜局分开算；C 层的挑战卡组按批循环，第 N 圈用 `(N-1) % batches + 1` 批。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .game import ROOT, load_tasks, required_checkpoint
from .runtime import output_path

LAYERS = ("A", "B", "C", "D", "E")
PLAN_PATH = ROOT / "configs" / "llmplay_seeds.json"
SCHEMA = "llmplay_cycle_v1"
PROGRESS_SCHEMA = "llmplay_progress_v2"
NO_END_NOTE = "循环式任务没有终点：一圈做完自动进下一圈，停止只由用户决定。"

_CACHE: dict = {}


# ----------------------------------------------------------------- 计划

def plan() -> dict:
    if "plan" not in _CACHE:
        document = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        if document.get("schema") != SCHEMA:
            raise ValueError("不是 %s 计划：%s" % (SCHEMA, PLAN_PATH))
        _CACHE["plan"] = document
    return _CACHE["plan"]


def requirements() -> dict:
    return plan()["requirements"]


def per_cycle() -> int:
    return int(plan()["block"]["per_cycle"])


def batch_for_cycle(cycle: int) -> int:
    return (int(cycle) - 1) % int(plan().get("batches") or 1) + 1


def seed_order(task_id, run_id, seeds) -> list:
    """按运行ID散列排序：不同 run 不会都从同一个种子开始。"""
    return sorted(seeds, key=lambda seed: hashlib.sha256(
        (str(run_id) + ":" + task_id + ":" + str(seed)).encode("utf-8")).digest())


def seeds_for(task_id, cycle, run_id=None) -> list:
    """第 cycle 圈的种子窗口（默认 40 个）。窗口用完见 `_fresh` 的顺延规则。"""
    seeds = plan()["tasks"].get(task_id) or []
    start = max(0, (int(cycle) - 1) * per_cycle())
    window = list(seeds[start:start + per_cycle()])
    return seed_order(task_id, run_id, window) if run_id else window


def ordered_tasks() -> list:
    tasks = load_tasks()
    return [(task_id, tasks[task_id]) for task_id in plan()["order"] if task_id in tasks]


def diagnoses() -> dict:
    """C 层失败诊断：`out/diagnosis/<task_id>.json`。文件缺失即视为没写。"""
    root = output_path(ROOT) / "diagnosis"
    found = {}
    if root.is_dir():
        for path in sorted(root.glob("*.json")):
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            found[path.stem] = document
    return found


# ----------------------------------------------------------------- 统计

def deck_key(deck) -> str:
    return ",".join(str(card) for card in sorted(deck or []))


def card_diff(left, right) -> int:
    return len(set(left or []) ^ set(right or []))


def _suppressed(entries) -> set:
    """同一种子同一卡组放弃 ≥2 次的组合不再计数（防反复放弃刷种子）。"""
    counts: dict = {}
    for entry in entries:
        if not entry.get("abandoned"):
            continue
        key = (entry["task_id"], int(entry["seed"]), deck_key(entry.get("deck")),
               int(entry.get("cycle") or 1))
        counts[key] = counts.get(key, 0) + 1
    return {key for key, count in counts.items() if count >= 2}


def _drop_suppressed(entries):
    suppressed = _suppressed(entries)
    if not suppressed:
        return list(entries)
    kept = []
    for entry in entries:
        key = (entry["task_id"], int(entry["seed"]), deck_key(entry.get("deck")),
               int(entry.get("cycle") or 1))
        if key not in suppressed:
            kept.append(entry)
    return kept


def _decks(entries) -> dict:
    groups: dict = {}
    for entry in entries:
        key = deck_key(entry.get("deck"))
        group = groups.setdefault(key, dict(deck=sorted(entry.get("deck") or []), winning=set(),
                                            robust=set(), attempts=0, losses=0, abandoned=0,
                                            strategy="", reason="", review="", seeds={}))
        group["attempts"] += 1
        if entry.get("abandoned"):
            group["abandoned"] += 1
        elif entry.get("won"):
            group["winning"].add(int(entry["seed"]))
            group["seeds"][int(entry["seed"])] = entry
        else:
            group["losses"] += 1
        if entry.get("robust"):
            group["robust"].add(int(entry["seed"]))
        for field in ("strategy", "reason", "review"):
            source = "deck_reason" if field == "reason" else "deck_" + field if field == "review" else field
            value = (entry.get(source) or "").strip()
            if value:
                group[field] = value
    out = {}
    for key, group in groups.items():
        group["wins"] = len(group["winning"])
        group["robust_wins"] = len(group["robust"])
        group["winning_seeds"] = sorted(group["winning"])
        group["robust_seeds"] = sorted(group["robust"])
        group["winning"] = sorted(group["winning"])
        group["robust"] = sorted(group["robust"])
        group["strategies"] = sorted({(entry.get("strategy") or "").strip()
                                      for entry in group["seeds"].values()} - {""})
        group["seeds"] = {str(seed): dict(seed=seed, archive=entry.get("archive"),
                                          robust=bool(entry.get("robust")),
                                          strategy=(entry.get("strategy") or "").strip())
                          for seed, entry in sorted(group["seeds"].items())}
        out[key] = group
    return out


def _select(decks, previous, spec) -> list:
    """挑出 A 层合格的卡组：够胜局、有思路与评价、两两差≥3 张、与本关历史卡组也差≥3 张。"""
    need, changes = int(spec["decks"]), int(spec["min_card_changes"])
    eligible = [key for key, group in decks.items()
                if group["wins"] >= int(spec["wins_per_deck"]) and group["strategy"] and group["review"]]
    eligible.sort(key=lambda key: (-decks[key]["wins"], -decks[key]["robust_wins"], key))
    chosen = []
    for key in eligible:
        if any(card_diff(decks[key]["deck"], decks[other]["deck"]) < changes for other in chosen):
            continue
        if any(card_diff(decks[key]["deck"], other) < changes for other in previous):
            continue
        chosen.append(key)
        if len(chosen) >= need:
            break
    return chosen


def _top(decks, chosen, count) -> list:
    ranked = sorted(chosen, key=lambda key: (-decks[key]["wins"], -decks[key]["robust_wins"], key))
    return ranked[:int(count)]


# ----------------------------------------------------------------- 单圈

def _cycle(number, entries, tasks, req, run_id=None) -> dict:
    in_cycle = [entry for entry in entries if int(entry.get("cycle") or 1) == number]
    a_ids = [task_id for task_id, task in tasks if task.get("layer") == "A"]
    e_tasks: dict = {}
    for task_id, task in tasks:
        if task.get("layer") == "E":
            e_tasks.setdefault(task["pair_task"], {})[float(task["zombie_count_multiplier"])] = task_id
    a_set = set(a_ids)
    previous: dict = {}
    for entry in entries:
        if entry["task_id"] in a_set and int(entry.get("cycle") or 1) < number:
            bucket = previous.setdefault(entry["task_id"], {})
            bucket.setdefault(deck_key(entry.get("deck")), sorted(entry.get("deck") or []))
    previous = {task_id: list(bucket.values()) for task_id, bucket in previous.items()}

    # ---- A 探索
    a_detail = {}
    for task_id in a_ids:
        decks = _decks([entry for entry in in_cycle if entry["task_id"] == task_id])
        chosen = _select(decks, previous.get(task_id, []), req["A"])
        passed = _top(decks, chosen, req["B"]["decks"])
        a_detail[task_id] = dict(task_id=task_id, level=tasks and _level(tasks, task_id),
                                 decks=decks, selected=chosen, passed=passed,
                                 complete=len(chosen) >= int(req["A"]["decks"]))
    a_complete = all(item["complete"] for item in a_detail.values())

    # ---- B 稳健
    b_detail, b_passed = {}, {}
    for task_id in a_ids:
        items = {}
        for key in a_detail[task_id]["passed"]:
            group = a_detail[task_id]["decks"][key]
            need = int(req["B"]["robust_seeds_per_deck"])
            items[key] = dict(deck=group["deck"], robust_seeds=group["robust_seeds"], need=need,
                              wins=group["wins"], complete=len(group["robust"]) >= need,
                              pending=[dict(seed=seed, archive=info["archive"])
                                       for seed, info in group["seeds"].items() if not info["robust"]])
        b_passed[task_id] = [key for key in a_detail[task_id]["passed"] if items[key]["complete"]]
        b_detail[task_id] = dict(task_id=task_id, items=items,
                                 complete=bool(items) and all(item["complete"] for item in items.values()))
    b_complete = a_complete and all(item["complete"] for item in b_detail.values())

    # ---- C 挑战
    batch = batch_for_cycle(number)
    diagnosis = diagnoses()
    c_detail = {}
    for task_id, task in tasks:
        if task.get("layer") != "C" or int(task.get("batch") or 1) != batch:
            continue
        entries_c = [entry for entry in in_cycle if entry["task_id"] == task_id]
        window = seeds_for(task_id, number, run_id)[:int(req["C"]["seeds"])]
        wins = len({int(entry["seed"]) for entry in entries_c if entry.get("won")})
        attempted = {int(entry["seed"]) for entry in entries_c}
        report = diagnosis.get(task_id)
        complete = wins >= int(req["C"]["wins"]) or (set(window) <= attempted and bool(report))
        c_detail[task_id] = dict(task_id=task_id, level=int(task["level"]), deck=sorted(task["deck"]),
                                 window=window, wins=wins, attempted=sorted(attempted),
                                 diagnosis=report, complete=complete)
    c_complete = all(item["complete"] for item in c_detail.values())

    # ---- D 打法
    d_detail = {}
    for task_id in a_ids:
        items = {}
        for key in b_passed[task_id]:
            entries_d = [entry for entry in in_cycle
                         if entry["task_id"] == task_id and deck_key(entry.get("deck")) == key
                         and entry.get("won")]
            strategies: dict = {}
            for entry in entries_d:
                name = (entry.get("strategy") or "").strip()
                if name:
                    strategies.setdefault(name, set()).add(int(entry["seed"]))
            good = sorted(name for name, seeds in strategies.items()
                          if len(seeds) >= int(req["D"]["wins_per_strategy"]))
            items[key] = dict(deck=a_detail[task_id]["decks"][key]["deck"],
                              strategies={name: sorted(seeds) for name, seeds in strategies.items()},
                              done=good, complete=len(good) >= int(req["D"]["strategies"]))
        d_detail[task_id] = dict(task_id=task_id, items=items,
                                 complete=bool(items) and all(item["complete"] for item in items.values()))
    d_complete = b_complete and all(item["complete"] for item in d_detail.values())

    # ---- E 加压
    e_detail = {}
    for task_id in a_ids:
        items = {}
        for key in b_passed[task_id]:
            for multiplier, e_task in sorted(e_tasks.get(task_id, {}).items()):
                entries_e = [entry for entry in in_cycle
                             if entry["task_id"] == e_task and deck_key(entry.get("deck")) == key]
                wins = len({int(entry["seed"]) for entry in entries_e if entry.get("won")})
                items["%s@%.1f" % (key, multiplier)] = dict(
                    task_id=e_task, multiplier=multiplier, deck=a_detail[task_id]["decks"][key]["deck"],
                    wins=wins, complete=wins >= int(req["E"]["wins_per_multiplier"]))
        e_detail[task_id] = dict(task_id=task_id, items=items,
                                 complete=bool(items) and all(item["complete"] for item in items.values()))
    e_complete = b_complete and all(item["complete"] for item in e_detail.values())

    layers = {
        "A": _summary(a_detail, a_complete, True),
        "B": _summary(b_detail, b_complete, a_complete),
        "C": _summary(c_detail, c_complete, True),
        "D": _summary(d_detail, d_complete, b_complete),
        "E": _summary(e_detail, e_complete, b_complete),
    }
    return dict(number=number, batch=batch, complete=all(item["complete"] for item in layers.values()),
                layers=layers,
                detail=dict(A=a_detail, B=b_detail, C=c_detail, D=d_detail, E=e_detail))


def _level(tasks, task_id) -> int:
    for candidate, task in tasks:
        if candidate == task_id:
            return int(task["level"])
    return 0


def _summary(detail, complete, upstream) -> dict:
    total = len(detail)
    done = sum(1 for item in detail.values() if item["complete"])
    return dict(completed=done, total=total, upstream=bool(upstream),
                complete=bool(upstream) and total > 0 and done == total)


# ----------------------------------------------------------------- 文档

def derive(entries, run_id=None) -> dict:
    tasks = ordered_tasks()
    req = requirements()
    usable = [entry for entry in entries if entry.get("valid", True)]
    usable = _drop_suppressed(usable)
    highest = max([int(entry.get("cycle") or 1) for entry in usable] or [1])
    cycles = {number: _cycle(number, usable, tasks, req, run_id) for number in range(1, highest + 2)}
    current = next(number for number in sorted(cycles) if not cycles[number]["complete"])
    cycle = cycles[current]
    layer = next(name for name in LAYERS if not cycle["layers"][name]["complete"])

    flat = {}
    for task_id, task in tasks:
        kind = task.get("layer")
        active = True
        if kind == "C":
            active = int(task.get("batch") or 1) == cycle["batch"]
            item = cycle["detail"]["C"].get(task_id)
            complete = bool(item and item["complete"])
        elif kind == "E":
            item = next((entry for group in cycle["detail"]["E"].values()
                         for entry in group["items"].values() if entry["task_id"] == task_id), None)
            complete = bool(item and item["complete"])
        else:
            item = cycle["detail"]["A"].get(task_id)
            complete = bool(item and item["complete"])
        seeds_map = _seed_map(usable, task_id, current)
        flat[task_id] = dict(layer=kind, cycle=current, active=active, complete=complete,
                             available_seeds=_fresh(task_id, current, usable, run_id),
                             seeds=seeds_map,
                             wins=sum(1 for row in seeds_map.values() if row["won"]),
                             robust=sum(1 for row in seeds_map.values() if row["robust"]),
                             decks=(item or {}).get("decks"))
    return dict(schema=PROGRESS_SCHEMA, run_id=run_id, cycle=current, layer=layer,
                requirements=req, batch=cycle["batch"], layers=cycle["layers"],
                detail=cycle["detail"], tasks=flat,
                cycles={str(number): dict(complete=cycles[number]["complete"],
                                          layers=cycles[number]["layers"]) for number in cycles},
                complete=False, note=NO_END_NOTE, archives=list(entries))


def _seed_map(entries, task_id, cycle) -> dict:
    """本圈每个种子只留最好的一条（稳健 > 通关 > 其它），给报告和默认种子用。"""
    out = {}
    for entry in entries:
        if entry["task_id"] != task_id or int(entry.get("cycle") or 1) != cycle:
            continue
        row = dict(seed=int(entry["seed"]), won=bool(entry.get("won")), robust=bool(entry.get("robust")),
                   result=entry.get("result"), archive=entry.get("archive"), deck=entry.get("deck"),
                   abandoned=bool(entry.get("abandoned")), manual_patch=bool(entry.get("manual_patch")),
                   note=entry.get("note", ""), strategy=entry.get("strategy", ""),
                   deck_review=entry.get("deck_review", ""))
        previous = out.get(str(entry["seed"]))
        if previous is None or (row["robust"], row["won"]) >= (previous["robust"], previous["won"]):
            out[str(entry["seed"])] = row
    return out


def _used(entries, task_id) -> set:
    return {int(entry["seed"]) for entry in entries if entry["task_id"] == task_id}


# ----------------------------------------------------------------- 排序

def _fresh(task_id, cycle, entries, run_id=None) -> list:
    """本圈窗口里还没用过的种子；窗口用完了就顺延到更后面的种子（不回退到更早的圈）。"""
    used = _used(entries, task_id)
    window = [seed for seed in seeds_for(task_id, cycle, run_id) if seed not in used]
    if window:
        return window
    later = (plan()["tasks"].get(task_id) or [])[int(cycle) * per_cycle():]
    later = [seed for seed in later if seed not in used]
    return seed_order(task_id, run_id, later) if run_id else later


def next_task(document) -> dict:
    """只返回当前圈当前层的未完成项；永远不会返回 None。"""
    layer, cycle = document["layer"], document["cycle"]
    detail = document["detail"]
    archives = document["archives"]
    req = document["requirements"]

    if layer == "A":
        for task_id in _ids(detail["A"]):
            item = detail["A"][task_id]
            if item["complete"]:
                continue
            fresh = _fresh(task_id, cycle, archives, document["run_id"])
            for key in sorted(item["decks"]):
                group = item["decks"][key]
                if group["wins"] == 1 and fresh:
                    return _item(document, task_id, "second_seed", seed=fresh[0], deck=group["deck"],
                                 note="这套卡组已赢 1 个种子；同卡组再赢 1 个不同种子就满足 A 层的每套 2 胜")
            if fresh:
                return _item(document, task_id, "new_deck", seed=fresh[0],
                             note="自选一套与已有卡组至少差 %d 张、核心思路不同的新卡组" % req["A"]["min_card_changes"])
        return _item(document, None, "blocked", note="A 层还有任务没种子可用，请检查种子计划")

    if layer == "B":
        for task_id in _ids(detail["B"]):
            item = detail["B"][task_id]
            if item["complete"]:
                continue
            for key in sorted(item["items"]):
                group = item["items"][key]
                if group["complete"]:
                    continue
                pending = [entry for entry in group["pending"] if entry["archive"]]
                if pending:
                    return _item(document, task_id, "improve_robustness", seed=pending[0]["seed"],
                                 deck=group["deck"], archive=pending[0]["archive"],
                                 note="这套卡组还差 %d 个稳健种子（第 10 波 k=8 赢≥5）；读这个存档，用闭环打法跑 robustness()"
                                      % (group["need"] - len(group["robust_seeds"])))
        return _item(document, None, "blocked", note="B 层没有可推进的项")

    if layer == "C":
        for task_id in _ids(detail["C"]):
            item = detail["C"][task_id]
            if item["complete"]:
                continue
            remaining = [seed for seed in item["window"] if seed not in set(item["attempted"])]
            if remaining:
                return _item(document, task_id, "play", seed=remaining[0], deck=item["deck"],
                             note="指定卡组：不能换卡；5 个种子里赢 2 个，或 5 个全输后写失败诊断")
            if not item["diagnosis"]:
                return _item(document, task_id, "diagnose", deck=item["deck"],
                             note="5 个种子都没赢 2 局：写失败诊断（卡在第几波、缺什么），用 llmplay.diagnose()")
        return _item(document, None, "blocked", note="C 层没有可推进的项")

    if layer == "D":
        for task_id in _ids(detail["D"]):
            item = detail["D"][task_id]
            if item["complete"]:
                continue
            fresh = _fresh(task_id, cycle, archives, document["run_id"])
            for key in sorted(item["items"]):
                group = item["items"][key]
                if group["complete"] or not fresh:
                    continue
                return _item(document, task_id, "new_strategy", seed=fresh[0], deck=group["deck"],
                             used=sorted(group["strategies"]), done=sorted(group["done"]),
                             note="这套卡组还差 %d 种开局节奏（经济优先 / 防线优先 / 速攻或爆发清场），"
                                  "strategy 必须与已用的不重复"
                                  % (int(req["D"]["strategies"]) - len(group["done"])))
        return _item(document, None, "blocked", note="D 层没有可推进的项")

    for task_id in _ids(detail["E"]):
        item = detail["E"][task_id]
        if item["complete"]:
            continue
        for key in sorted(item["items"]):
            group = item["items"][key]
            if group["complete"]:
                continue
            fresh = _fresh(group["task_id"], cycle, archives, document["run_id"])
            if not fresh:
                continue
            return _item(document, group["task_id"], "play", seed=fresh[0], deck=group["deck"],
                         pair_task=task_id,
                         note="加压层：出怪 %.1f 倍，只能用 B 层下传的这套卡组" % group["multiplier"])
    return _item(document, None, "blocked", note="E 层没有可推进的项")


def _ids(detail) -> list:
    return list(detail)


def _item(document, task_id, action, **kwargs) -> dict:
    out = dict(task_id=task_id, action=action, layer=document["layer"], cycle=document["cycle"],
               batch=document["batch"])
    out.update(kwargs)
    out.setdefault("seed", None)
    out.setdefault("deck", None)
    return out


def describe(item) -> str:
    """把 next_task() 的结果写成一句人话，给异常消息和报告用。"""
    if not item:
        return "没有待办项"
    if item.get("action") == "blocked":
        return item.get("note") or "当前层没有可推进的项"
    parts = ["第 %d 圈 %s 层 %s：%s" % (item["cycle"], item["layer"], item["action"], item["task_id"])]
    if item.get("seed") is not None:
        parts.append("seed=%s" % item["seed"])
    if item.get("deck"):
        parts.append("卡组=%s" % item["deck"])
    if item.get("note"):
        parts.append(item["note"])
    return "；".join(parts)


# ----------------------------------------------------------------- 建局把关

def _refuse(document, reason) -> None:
    raise ValueError("%s。%s" % (reason, describe(next_task(document))))


def check(document, task_id, deck, strategy="") -> None:
    """Game() 建局前调用：层不对 / 卡组不合格 / 打法重复 一律拒绝，并指出当前该做什么。"""
    tasks = load_tasks()
    task = tasks.get(task_id)
    if task is None or task_id not in (plan()["tasks"] or {}):
        return
    layer, cycle = document["layer"], document["cycle"]
    kind = task.get("layer")
    req = document["requirements"]
    if kind == "C":
        if layer != "C":
            _refuse(document, "C 层现在不能建局（当前是 %s 层）" % layer)
        if int(task.get("batch") or 1) != document["batch"]:
            _refuse(document, "这套挑战卡组不在本圈的批次里")
        if sorted(deck or []) != sorted(task["deck"]):
            _refuse(document, "C 层是指定卡组任务，不能换卡")
        return
    if kind == "E":
        if layer != "E":
            _refuse(document, "E 层现在不能建局（当前是 %s 层）" % layer)
        passed = set(document["detail"]["B"][task["pair_task"]]["items"])
        if deck_key(deck) not in {key for key in passed
                                  if document["detail"]["B"][task["pair_task"]]["items"][key]["complete"]}:
            _refuse(document, "E 层只能用 B 层稳健复验通过的卡组")
        return
    if layer not in ("A", "B", "D"):
        _refuse(document, "%s 层现在不能建局" % kind)
    item = document["detail"]["A"].get(task_id) or {}
    key = deck_key(deck)
    if layer == "D":
        if key not in set(document["detail"]["B"].get(task_id, {}).get("items") or {}):
            _refuse(document, "D 层只能用 B 层下传的卡组")
        if not document["detail"]["B"][task_id]["items"][key]["complete"]:
            _refuse(document, "这套卡组还没通过 B 层稳健复验，D 层要用过了 B 的卡组")
        if not (strategy or "").strip():
            _refuse(document, "D 层必须在 Game(..., strategy=\"...\") 里声明开局节奏")
        done = set((document["detail"]["D"].get(task_id, {}).get("items", {}).get(key) or {}).get("done") or [])
        if strategy.strip() in done:
            _refuse(document, "这个打法在这套卡组上已经赢够 2 个种子了；D 层要 3 种不同的开局节奏，换一种")
        return
    if layer == "B":
        if key not in set(item.get("passed") or []):
            _refuse(document, "B 层只复验 A 层胜局最多的 %d 套卡组" % int(req["B"]["decks"]))
        return
    decks = item.get("decks") or {}
    if key in decks:
        return
    changes = int(req["A"]["min_card_changes"])
    if any(card_diff(deck, group["deck"]) < changes for group in decks.values()):
        _refuse(document, "这套卡组和本关已有卡组差不到 %d 张；A 层要求每套明显不同" % changes)


def archive_requirements(task_id, strategy="", review="", abandoned=False) -> None:
    """导出存档前的必填项：A/D 层要思路与评价，C 层全输要诊断。"""
    if abandoned:
        return
    task = load_tasks().get(task_id) or {}
    if task_id not in (plan()["tasks"] or {}):
        return
    kind = task.get("layer")
    if kind in ("A", "E") and not (strategy or "").strip():
        raise ValueError("A/E 层导出前请填写 strategy（核心思路），例如 Game(..., strategy=\"投手压制\")")
    if kind in ("A", "D") and not (review or "").strip():
        raise ValueError("导出前请填写 game.deck_review（一句话：这套卡组好打吗、为什么），"
                         "缺评价不能进下一层")


# ----------------------------------------------------------------- 报告

def report(document) -> str:
    lines = ["# LLM 对局进度（循环式）", "",
             "运行ID：%s" % document["run_id"], "",
             "第 **%d** 圈 · 当前层 **%s** · 本圈 C 层批次 %d" % (document["cycle"], document["layer"], document["batch"]),
             "", "接口自动生成；详细结果以 progress.json 为准。", "",
             "**这个任务没有终点。** 一圈做完自动进下一圈，不会出现「全部完成」。", "",
             "## 现在该做什么", "", "- " + describe(next_task(document)), "",
             "## 本圈各层", "", "| 层 | 完成项 | 总数 | 达标 |", "|---|---|---|---|"]
    for name in LAYERS:
        info = document["layers"][name]
        lines.append("| %s | %d | %d | %s |" % (name, info["completed"], info["total"],
                                                "是" if info["complete"] else ("上游未完成" if not info["upstream"] else "否")))
    lines += ["", "下传关系：A 层胜局最多的 2 套 → B（稳健）→ D（三种打法）/ E（加压）；C 层独立，按批循环。", ""]
    lines += ["## A 层：试过的卡组", "",
              "| 任务 | 卡组 | 核心思路 | 胜种子 | 稳健种子 | 选卡理由 | 优缺点评价 | 入选 |",
              "|---|---|---|---|---|---|---|---|"]
    for task_id, item in document["detail"]["A"].items():
        for key, group in item["decks"].items():
            cells = [task_id, str(group["deck"]), group["strategy"] or "待填",
                     "%d/%d" % (group["wins"], group["attempts"]), str(group["robust_wins"]),
                     group["reason"] or "", group["review"] or "待填",
                     "下传" if key in item["passed"] else ("入选" if key in item["selected"] else "")]
            lines.append("| " + " | ".join(_cell(cell) for cell in cells) + " |")
    lines += ["", "## C 层：指定卡组挑战", "", "| 任务 | 卡组 | 胜 | 已试种子 | 诊断 |", "|---|---|---|---|---|"]
    for task_id, item in document["detail"]["C"].items():
        cells = [task_id, str(item["deck"]), "%d/%d" % (item["wins"], len(item["window"])),
                 str(item["attempted"]), "已写" if item["diagnosis"] else ""]
        lines.append("| " + " | ".join(_cell(cell) for cell in cells) + " |")
    lines += ["", "## B / D / E 层", "", "| 层 | 任务 | 卡组 | 进度 |", "|---|---|---|---|"]
    for task_id, item in document["detail"]["B"].items():
        for key, group in item["items"].items():
            lines.append("| B | %s | %s | 稳健种子 %d/%d |" % (task_id, group["deck"], len(group["robust_seeds"]), group["need"]))
    for task_id, item in document["detail"]["D"].items():
        for key, group in item["items"].items():
            lines.append("| D | %s | %s | 打法 %d/%d（%s） |" % (task_id, group["deck"], len(group["done"]),
                                                              int(document["requirements"]["D"]["strategies"]),
                                                              "、".join(group["done"]) or "—"))
    for task_id, item in document["detail"]["E"].items():
        for key, group in item["items"].items():
            lines.append("| E | %s | %s | %.1f 倍 胜 %d |" % (group["task_id"], group["deck"], group["multiplier"], group["wins"]))
    lines += ["", "## 全部存档", "", "| 任务 | 种子 | 圈 | 结果 | 稳健 | 卡组 | 存档 | 说明 |", "|---|---|---|---|---|---|---|---|"]
    for entry in document["archives"]:
        status = ("放弃：" + str(entry.get("abandon_reason"))) if entry.get("abandoned") else ("通关" if entry["won"] else "未通关")
        cells = [entry["task_id"], str(entry["seed"]), str(entry.get("cycle") or 1), status,
                 "是" if entry["robust"] else "否", str(entry.get("deck") or []), entry["archive"],
                 (entry.get("strategy") or "") + " " + (entry.get("note") or "")]
        lines.append("| " + " | ".join(_cell(cell) for cell in cells) + " |")
    return "\n".join(lines) + "\n"


def _cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
