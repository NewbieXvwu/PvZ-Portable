"""task139：主线 A → B → D → E，与每圈必做的 C 挑战轨并行。

这一层只做**判定与排序**，不碰引擎，也不写交付文件：

* `derive()` 把存档列表算成「第几圈 / 第几层 / 每项还差什么」；
* `next_task()` 同时返回主线与挑战轨的下一项；两轨都完成才能进下一圈；
* `check()` 在 `Game()` 建局时把关：层不对、卡组差 <3 张、打法重复、指定卡组被换掉，
  一律报错并直接告诉你现在该做什么；
* `report()` 每层自动生成小结（做了什么、下传了什么）。

一圈做完自动进下一圈，`progress()["complete"]` 永远是 `False`：这个任务没有终点，
停止只由用户在对话里决定。

存档里带开局时固定的 `cycle` 与 `workflow_layer` 字段。同一套卡组在不同圈
的胜局分开算；C 层的挑战卡组按批循环，第 N 圈用 `(N-1) % batches + 1` 批。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .game import ROOT, load_tasks, required_checkpoint
from .runtime import output_path

try:                                    # 只用来串行化 claims.json 的读改写
    import fcntl as _FCNTL
except ImportError:                     # Windows 上退化为无锁（占位记录本身容忍竞争）
    _FCNTL = None

LAYERS = ("A", "B", "C", "D", "E")
MAIN_LAYERS = ("A", "B", "D", "E")
STRATEGIES = ("经济优先", "防线优先", "速攻清场")
PLAN_PATH = ROOT / "configs" / "llmplay_seeds.json"
SCHEMA = "llmplay_cycle_v1"
PROGRESS_SCHEMA = "llmplay_progress_v2"
NO_END_NOTE = "循环式任务没有终点：一圈做完自动进下一圈，停止只由用户决定。"
PARALLEL = 4            # 同一个待办项最多同时放几个种子给不同子代理
CLAIM_TTL = 900         # 领了还没建局：15 分钟后自动释放
HOLD_TTL = 1800         # 建局后每次推进续期 30 分钟

_CACHE: dict = {}


# ----------------------------------------------------------------- 计划

def plan() -> dict:
    if "plan" not in _CACHE:
        document = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        if document.get("schema") != SCHEMA:
            raise ValueError("不是 %s 计划：%s" % (SCHEMA, PLAN_PATH))
        if 'seed_ranges' in document:
            document['tasks'] = {key: list(range(bounds[0], bounds[-1] + 1))
                                 for key, bounds in document.pop('seed_ranges').items()}
        _CACHE["plan"] = document
    return _CACHE["plan"]


def requirements() -> dict:
    return plan()["requirements"]


def decks_mode():
    return plan().get("mode") == "decks"


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
    """C/E 层失败诊断；E 层另按卡组区分。文件缺失即视为没写。"""
    root = output_path(ROOT) / "diagnosis"
    found = {}
    if root.is_dir():
        for path in sorted(root.rglob("*.json")):
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            key = (document.get("task_id"), document.get("cycle"))
            if document.get("deck") is not None:
                key += (deck_key(document["deck"]),)
            found[key] = document
    return found


# ----------------------------------------------------------------- 统计

def deck_key(deck) -> str:
    return ",".join(str(card) for card in sorted(deck or []))


def card_diff(left, right) -> int:
    return max(len(set(left or []) - set(right or [])), len(set(right or []) - set(left or [])))


def _suppressed(entries) -> set:
    """同一种子同一卡组放弃 ≥2 次的组合不再计数（防反复放弃刷种子）。"""
    counts: dict = {}
    for entry in entries:
        if not entry.get("abandoned"):
            continue
        key = (entry["task_id"], int(entry["seed"]), deck_key(entry.get("deck")),
               int(entry.get("cycle") or 1))
        counts.setdefault(key, set()).add(entry.get("attempt_id") or entry.get("archive") or id(entry))
    return {key for key, attempts in counts.items() if len(attempts) >= 2}


def _drop_suppressed(entries):
    suppressed = _suppressed(entries)
    if not suppressed:
        return list(entries)
    kept = []
    for entry in entries:
        key = (entry["task_id"], int(entry["seed"]), deck_key(entry.get("deck")),
               int(entry.get("cycle") or 1))
        kept.append(dict(entry, won=False, robust=False) if key in suppressed else entry)
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
        if entry.get("won") and not entry.get("abandoned") and entry.get("robust"):
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

def _challenge_detail(task_id, task, number, in_cycle, req, run_id, diagnosis):
    window = seeds_for(task_id, number, run_id)[:int(req["C"]["seeds"])]
    entries_c = [entry for entry in in_cycle if entry["task_id"] == task_id
                 and deck_key(entry.get("deck")) == deck_key(task["deck"])
                 and int(entry["seed"]) in window and not entry.get("abandoned")
                 and entry.get("result") in (1, 2)]
    wins = len({int(entry["seed"]) for entry in entries_c if entry.get("won")})
    attempted = {int(entry["seed"]) for entry in entries_c}
    report = diagnosis.get((task_id, number))
    if not (report and report.get("cycle") == number and report.get("task_id") == task_id
            and report.get("waves") and str(report.get("missing") or "").strip()):
        report = None
    complete = wins >= int(req["C"]["wins"]) or (set(window) <= attempted and bool(report))
    return dict(task_id=task_id, level=int(task["level"]), deck=sorted(task["deck"]),
                             window=window, wins=wins, attempted=sorted(attempted),
                             diagnosis=report, complete=complete)


def _cycle(number, entries, tasks, req, run_id=None, diagnosis=None) -> dict:
    all_tasks = dict(tasks)
    tasks = [(task_id, task) for task_id, task in tasks
             if int(task.get("first_cycle", 1)) <= number <= int(task.get("last_cycle", number))]
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
    diagnosis = diagnoses() if diagnosis is None else diagnosis
    c_detail = {}
    for task_id, task in tasks:
        if task.get("layer") != "C" or int(task.get("batch") or 1) != batch:
            continue
        item = _challenge_detail(task_id, task, number, in_cycle, req, run_id, diagnosis)
        # 半成品已完成的旧挑战位置不重做；保留原结论，不把旧胜局冒充新卡组胜局。
        if task.get("replaces"):
            origin = all_tasks[task["replaces"]]
            done = _challenge_detail(origin["task_id"], origin, number, in_cycle, req, run_id, diagnosis)
            if done["complete"]:
                item.update(complete=True, carried_completion=done)
        c_detail[task_id] = item
    c_complete = all(item["complete"] for item in c_detail.values())

    # ---- D 打法
    d_detail = {}
    for task_id in a_ids:
        items = {}
        for key in b_passed[task_id]:
            entries_d = [entry for entry in in_cycle
                         if entry["task_id"] == task_id and deck_key(entry.get("deck")) == key
                         and entry.get("workflow_layer") == "D" and entry.get("won")
                         and not entry.get("abandoned")]
            strategies: dict = {}
            openings: dict = {}
            for entry in entries_d:
                name = (entry.get("strategy") or "").strip()
                opening = tuple(entry.get("opening") or [])
                if name in STRATEGIES and opening and not any(
                        opening in used for other, used in openings.items() if other != name):
                    strategies.setdefault(name, set()).add(int(entry["seed"]))
                    openings.setdefault(name, set()).add(opening)
            good = sorted(name for name, seeds in strategies.items()
                          if len(seeds) >= int(req["D"]["wins_per_strategy"]))
            items[key] = dict(deck=a_detail[task_id]["decks"][key]["deck"],
                              strategies={name: sorted(seeds) for name, seeds in strategies.items()},
                              openings={name: [list(opening) for opening in sorted(used)]
                                        for name, used in openings.items()},
                              done=good, complete=len(good) >= int(req["D"]["strategies"]))
        d_detail[task_id] = dict(task_id=task_id, items=items,
                                 complete=bool(items) and all(item["complete"] for item in items.values()))
    d_complete = b_complete and all(item["complete"] for item in d_detail.values())

    # ---- E 加压：旧 2.0 倍成绩抵 1.25 倍；五败旧组只补诊断，不再开局。
    e_detail = {}
    for task_id in a_ids:
        items = {}
        for key in b_passed[task_id]:
            cards = a_detail[task_id]["decks"][key]["deck"]
            for multiplier in req["E"]["multipliers"]:
                e_task = e_tasks[task_id][float(multiplier)]
                old = None
                if float(multiplier) == 1.25 and 2.0 in e_tasks[task_id]:
                    old_task = e_tasks[task_id][2.0]
                    old = _pressure_group(old_task, 2.0, cards, in_cycle, diagnosis, number, req["E"])
                if old and (old["wins"] >= int(req["E"]["wins_per_multiplier"]) or old["diagnosable"]):
                    group = old
                else:
                    group = _pressure_group(e_task, multiplier, cards, in_cycle, diagnosis, number, req["E"])
                    if old:
                        group["prior_wins"] = old["wins"]
                        group["wins"] += old["wins"]
                        group["complete"] |= group["wins"] >= int(req["E"]["wins_per_multiplier"])
                items["%s@%g" % (key, multiplier)] = group
        e_detail[task_id] = dict(task_id=task_id, items=items,
                                 complete=bool(items) and all(item["complete"] for item in items.values()))
    e_complete = b_complete and all(item["complete"] for item in e_detail.values())

    layers = {
        "A": _summary(a_detail, a_complete, True),
        "B": _summary(b_detail, b_complete, a_complete),
        "C": _summary(c_detail, c_complete, True),
        "D": _summary(d_detail, d_complete, b_complete),
        "E": _summary(e_detail, e_complete, d_complete),
    }
    return dict(number=number, batch=batch, complete=all(item["complete"] for item in layers.values()),
                layers=layers,
                detail=dict(A=a_detail, B=b_detail, C=c_detail, D=d_detail, E=e_detail))


def _pressure_group(task_id, multiplier, cards, entries, diagnosis, cycle, req) -> dict:
    key = deck_key(cards)
    rows = [entry for entry in entries if entry["task_id"] == task_id
            and deck_key(entry.get("deck")) == key and not entry.get("abandoned")
            and entry.get("result") in (1, 2)]
    wins = {int(entry["seed"]) for entry in rows if entry.get("won")}
    seen, streak = set(), []
    for entry in rows:
        seed = int(entry["seed"])
        if entry.get("won"):
            streak = []
        elif seed not in seen:
            streak.append(seed)
        seen.add(seed)
    report = diagnosis.get((task_id, cycle, key))
    if not (report and report.get("waves") and str(report.get("missing") or "").strip()):
        report = None
    diagnosable = len(streak) >= int(req["loss_streak"])
    return dict(task_id=task_id, multiplier=multiplier, deck=cards, wins=len(wins),
                loss_seeds=streak, diagnosable=diagnosable, diagnosis=report,
                complete=len(wins) >= int(req["wins_per_multiplier"]) or (diagnosable and bool(report)))


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

def derive(entries, run_id=None, *, diagnosis=None) -> dict:
    if decks_mode():
        from . import decks
        return decks.derive(entries, run_id)
    tasks = ordered_tasks()
    req = requirements()
    usable = [entry for entry in entries if entry.get("valid", True)]
    usable = _drop_suppressed(usable)
    highest = max([int(entry.get("cycle") or 1) for entry in usable] or [1])
    cycles = {number: _cycle(number, usable, tasks, req, run_id, diagnosis)
              for number in range(1, highest + 2)}
    current = next(number for number in sorted(cycles) if not cycles[number]["complete"])
    cycle = cycles[current]
    layer = next((name for name in MAIN_LAYERS if not cycle["layers"][name]["complete"]), "C")

    flat = {}
    for task_id, task in tasks:
        kind = task.get("layer")
        active = int(task.get("first_cycle", 1)) <= current <= int(task.get("last_cycle", current))
        if not active:
            flat[task_id] = dict(layer=kind, cycle=current, active=False, complete=False,
                                 available_seeds=[], seeds={}, wins=0, robust=0, decks=None)
            continue
        if kind == "C":
            active = int(task.get("batch") or 1) == cycle["batch"]
            item = cycle["detail"]["C"].get(task_id)
            complete = bool(item and item["complete"])
        elif kind == "E":
            groups = [entry for group in cycle["detail"]["E"].values()
                      for entry in group["items"].values() if entry["task_id"] == task_id]
            item = groups[0] if groups else None
            active = bool(groups)
            complete = bool(groups) and all(group["complete"] for group in groups)
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
    current = [entry for entry in entries if entry["task_id"] == task_id
               and int(entry.get("cycle") or 1) == cycle]
    finished = {int(entry["seed"]) for entry in current
                if not entry.get("abandoned") and entry.get("result") in (1, 2)}
    pending = {int(entry["seed"]) for entry in current
               if entry.get("abandoned") or entry.get("result") not in (1, 2)} - finished
    if pending:
        order = seeds_for(task_id, cycle, run_id)
        return sorted(pending, key=lambda seed: order.index(seed) if seed in order else seed)
    used = _used(entries, task_id)
    window = [seed for seed in seeds_for(task_id, cycle, run_id) if seed not in used]
    if window:
        return window
    later = (plan()["tasks"].get(task_id) or [])[int(cycle) * per_cycle():]
    later = [seed for seed in later if seed not in used]
    return seed_order(task_id, run_id, later) if run_id else later


def _slot(task_id, seed) -> str:
    return "%s|%s" % (task_id, seed)


@contextmanager
def _claim_lock():
    path = _claims_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent / ".claims.lock").open("a+b") as handle:
        if _FCNTL is not None:
            _FCNTL.flock(handle.fileno(), _FCNTL.LOCK_EX)
        try:
            yield
        finally:
            if _FCNTL is not None:
                _FCNTL.flock(handle.fileno(), _FCNTL.LOCK_UN)


def _claims_path() -> Path:
    return output_path(ROOT) / "claims.json"


def _read_claims() -> dict:
    """读占位记录；过期的自动丢弃（进程崩了不会永久占着任务）。"""
    try:
        document = json.loads(_claims_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    now = time.time()
    return {key: value for key, value in document.items()
            if float(value.get("expires") or 0) > now}


def _write_claims(data) -> None:
    path = _claims_path()
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def held() -> dict:
    """当前还没过期的占位记录，键是 task_id|seed。"""
    return _read_claims()


def hold(task_id, seed, ttl=None) -> None:
    """建局后占住这一项并续期；游戏每推进一步都会再续一次。"""
    ttl = float(ttl or os.environ.get("LLMPLAY_CLAIM_HOLD") or HOLD_TTL)
    with _claim_lock():
        data = _read_claims()
        key = _slot(task_id, seed)
        record = data.get(key) or dict(task_id=task_id, seed=seed, action="play",
                                       token=uuid.uuid4().hex, claimed_at=time.time(),
                                       pid=os.getpid())
        record["expires"] = time.time() + ttl
        data[key] = record
        _write_claims(data)


def release(task_id, seed) -> None:
    """这一项打完了（导出/放弃/关闭）：把占位还回队列。"""
    with _claim_lock():
        data = _read_claims()
        if data.pop(_slot(task_id, seed), None) is not None:
            _write_claims(data)


def claim_item(document=None, *, track=None, ttl=None) -> dict:
    """领一项还没被领走的任务；全部被领走时返回 action=busy，不阻塞。"""
    document = document if document is not None else _progress()
    main_items, c_items = _ordered(document)
    # C 轨专职子代理先领 C，C 做完自动接主线；主线子代理反之。
    order = (c_items + main_items) if track == "C" and document.get("mode") != "decks" else (main_items + c_items)
    ttl = float(ttl or os.environ.get("LLMPLAY_CLAIM_TTL") or CLAIM_TTL)
    with _claim_lock():
        data = _read_claims()
        for candidate in order:
            key = _slot(candidate["task_id"], candidate.get("seed"))
            if key in data:
                continue
            token = uuid.uuid4().hex
            data[key] = dict(task_id=candidate["task_id"], seed=candidate.get("seed"),
                             cycle=document["cycle"], layer=candidate["layer"],
                             action=candidate["action"], token=token,
                             claimed_at=time.time(), expires=time.time() + ttl, pid=os.getpid())
            _write_claims(data)
            return dict(candidate, claim=token)
        return _busy(main_items + c_items) or _item(document, None, "blocked",
                                                    note="本圈没有可推进的项")


def _ordered(document) -> tuple:
    """两轨的候选项（按优先顺序）：主线在前，C 轨在后，互不阻塞（decks 模式为三轨优先序）。"""
    if document.get("mode") == "decks":
        from . import decks
        # task211 三轨：重放 > 传送带支线 > 主线新卡组，合成一条按优先序领取的队列。
        return (decks.replay_items(document) + decks.native_items(document)
                + decks.items(document)), []
    return _main_items(document), _challenge_items(document)


def _interleave(groups) -> list:
    """按任务轮流发牌：先让不同任务各领一项，再回头领第二项。"""
    groups = [list(group) for group in groups if group]
    out = []
    for index in range(max((len(group) for group in groups), default=0)):
        for group in groups:
            if index < len(group):
                out.append(group[index])
    return out


def _main_items(document) -> list:
    layer, cycle = document["layer"], document["cycle"]
    detail, archives, req = document["detail"], document["archives"], document["requirements"]
    run_id = document.get("run_id")
    out = []

    if layer == "A":
        groups = []
        for task_id in _ids(detail["A"]):
            item = detail["A"][task_id]
            if item["complete"]:
                continue
            fresh = _fresh(task_id, cycle, archives, run_id)
            group = []
            for key in sorted(item["decks"]):
                deck = item["decks"][key]
                if deck["wins"] == 1 and deck["strategy"] and deck["review"]:
                    for seed in fresh[:PARALLEL]:
                        group.append(_item(document, task_id, "second_seed", seed=seed, deck=deck["deck"],
                                           note="这套卡组已赢 1 个种子；同卡组再赢 1 个不同种子就满足 A 层的每套 2 胜"))
            for seed in fresh[:PARALLEL]:
                group.append(_item(document, task_id, "new_deck", seed=seed,
                                   note="自选一套与已有卡组至少差 %d 张、核心思路不同的新卡组"
                                        % req["A"]["min_card_changes"]))
            groups.append(group)
        return _interleave(groups)

    if layer == "B":
        groups = []
        for task_id in _ids(detail["B"]):
            item = detail["B"][task_id]
            if item["complete"]:
                continue
            for key in sorted(item["items"]):
                deck = item["items"][key]
                if deck["complete"]:
                    continue
                groups.append([_item(document, task_id, "improve_robustness", seed=row["seed"],
                                     deck=deck["deck"], archive=row["archive"],
                                     note="这套卡组还差 %d 个稳健种子（%s k=8 赢≥5）；"
                                          "读这个存档，用闭环打法跑 robustness()"
                                          % (deck["need"] - len(deck["robust_seeds"]),
                                             required_checkpoint(load_tasks()[task_id])))
                               for row in deck["pending"] if row["archive"]])
        return _interleave(groups)

    if layer == "D":
        groups = []
        for task_id in _ids(detail["D"]):
            item = detail["D"][task_id]
            if item["complete"]:
                continue
            fresh = _fresh(task_id, cycle, archives, run_id)
            for key in sorted(item["items"]):
                deck = item["items"][key]
                if deck["complete"]:
                    continue
                remaining = [name for name in STRATEGIES if name not in deck["done"]]
                groups.append([_item(document, task_id, "new_strategy", seed=seed, deck=deck["deck"],
                                     strategy=name, strategies=remaining, done=sorted(deck["done"]),
                                     note="这套卡组还差 %d 种开局节奏；用返回的 strategy，每种赢 2 个种子"
                                          % (int(req["D"]["strategies"]) - len(deck["done"])))
                               for name in remaining for seed in fresh[:2]])
        return _interleave(groups)

    if layer == "E":
        groups = []
        for task_id in _ids(detail["E"]):
            item = detail["E"][task_id]
            if item["complete"]:
                continue
            for key in sorted(item["items"]):
                deck = item["items"][key]
                if deck["complete"]:
                    continue
                if deck["diagnosable"]:
                    groups.append([_item(document, deck["task_id"], "diagnose", deck=deck["deck"],
                                         pair_task=task_id, multiplier=deck["multiplier"],
                                         note="同一卡组同一倍数连续 5 个新种子全输：补写失败波次和缺什么，诊断后完成该组")])
                    continue
                groups.append([_item(document, deck["task_id"], "play", seed=seed, deck=deck["deck"],
                                     pair_task=task_id,
                                     note="加压层：出怪 %g 倍，只能用 B 层下传的这套卡组" % deck["multiplier"])
                               for seed in _fresh(deck["task_id"], cycle, archives, run_id)[:2]])
        return _interleave(groups)
    return out


def _challenge_items(document) -> list:
    track = dict(document, layer="C")
    detail = document["detail"]["C"]
    groups = []
    for task_id in _ids(detail):
        item = detail[task_id]
        if item["complete"]:
            continue
        attempted = set(item["attempted"])
        remaining = [seed for seed in item["window"] if seed not in attempted]
        group = [_item(track, task_id, "play", seed=seed, deck=item["deck"],
                       note="指定卡组：不能换卡；5 个种子里赢 2 个。放弃不算尝试，必须同种子重开直到真实终局")
                 for seed in remaining]
        if not remaining and not item["diagnosis"]:
            group.append(_item(track, task_id, "diagnose", deck=item["deck"],
                               note="5 个种子都没赢 2 局：写失败诊断（卡在第几波、缺什么），用 llmplay.diagnose()"))
        groups.append(group)
    return _interleave(groups)


def _first_free(items, claims):
    for item in items:
        if _slot(item["task_id"], item.get("seed")) not in claims:
            return item
    return None


def _busy(items):
    if not items:
        return None
    return dict(items[0], action="busy", busy=True,
                note="当前所有待办项都已被其他子代理领走；手上没活就直接结束，不要轮询等待")


def _pick(document, main, challenge, main_items, c_items):
    if main is not None:
        return main
    if challenge is not None:
        return challenge
    for items in (main_items, c_items):
        item = _busy(items)
        if item:
            return item
    return _item(document, None, "blocked", note="本圈没有可推进的项")


def _view(document, claims=None) -> dict:
    """只读视图：两轨各自第一个还没被领走的项，不写占位记录。"""
    if document.get("mode") == "decks":
        from . import decks
        return decks.view(document)
    main_items, c_items = _ordered(document)
    claims = _read_claims() if claims is None else claims
    main = _first_free(main_items, claims)
    challenge = _first_free(c_items, claims)
    return dict(cycle=document["cycle"], main=main, challenge=challenge,
                item=_pick(document, main, challenge, main_items, c_items),
                challenge_quota=document["layers"]["C"], complete=False, note=NO_END_NOTE)


def next_task(document=None, *, track=None, claim=True, ttl=None) -> dict:
    """领下一项：在 out/claims.json 里占位，多个子代理同时调用会各领到不同项。

    主线与 C 轨合成一条队列（主线优先）；全部被领走时 item.action 是 busy，不阻塞。
    claim=False 只查看队列，不占位（报告、测试用）。
    """
    document = document if document is not None else _progress()
    view = _view(document)
    if claim:
        view["item"] = claim_item(document, track=track, ttl=ttl)
    return view


def _progress():
    from .delivery import progress
    return progress()


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
    view = _view(document)
    raise ValueError("%s。主线：%s。C 轨：%s" % (reason, describe(view["main"]), describe(view["challenge"])))


def check(document, task_id, deck, strategy="", seed=None) -> dict:
    """Game() 建局前调用：层序 / 种子窗口 / 卡组差异 / 打法去重，当场计数判断。"""
    if document.get("mode") == "decks":
        from . import decks
        return decks.check(document, task_id, deck, strategy, seed)
    tasks = load_tasks()
    task = tasks.get(task_id)
    if (task is None or task_id not in (plan()["tasks"] or {})
            or not int(task.get("first_cycle", 1)) <= document["cycle"]
            <= int(task.get("last_cycle", document["cycle"]))):
        if any(task_id in tasks for task_id in plan()["tasks"]):
            _refuse(document, "循环任务请使用 next_task() 返回的任务，旧任务不计入本轮")
        return {}
    layer, cycle = document["layer"], document["cycle"]
    kind = task.get("layer")
    req = document["requirements"]
    context = dict(cycle=cycle, workflow_layer="C" if kind == "C" else layer)
    if kind == "C":
        if int(task.get("batch") or 1) != document["batch"]:
            _refuse(document, "这套挑战卡组不在本圈的批次里")
        window = set(seeds_for(task_id, cycle, document.get("run_id"))[:int(req["C"]["seeds"])])
        if seed is None or int(seed) not in window:
            _refuse(document, "C 轨只能用本圈指定的那 5 个种子")
        if sorted(deck or []) != sorted(task["deck"]):
            _refuse(document, "C 层是指定卡组任务，不能换卡")
        return context
    if kind == "E":
        if layer != "E":
            _refuse(document, "E 层现在不能建局（当前是 %s 层）" % layer)
        passed = document["detail"]["B"][task["pair_task"]]["items"]
        if deck_key(deck) not in {key for key, item in passed.items() if item["complete"]}:
            _refuse(document, "E 层只能用 B 层稳健复验通过的卡组")
        group = next((item for item in document["detail"]["E"][task["pair_task"]]["items"].values()
                      if item["task_id"] == task_id and deck_key(item["deck"]) == deck_key(deck)), None)
        if not group or group["complete"] or group["diagnosable"]:
            _refuse(document, "该加压组已完成或须写失败诊断，请接下一项")
        return context
    # A/B/D 三层共用 A 层的任务；当前层决定用哪套规则。
    seeds = plan()["tasks"].get(task_id) or []
    start = max(0, (int(cycle) - 1) * per_cycle())
    if seed is None or int(seed) not in set(seeds[start:]):
        _refuse(document, "种子由队列指定：只能用本圈窗口内的种子，不能自己挑或刷新")
    item = document["detail"]["A"].get(task_id) or {}
    key = deck_key(deck)
    if layer == "D":
        group = (document["detail"]["B"].get(task_id, {}).get("items") or {}).get(key)
        if not group:
            _refuse(document, "D 层只能用 B 层下传的卡组")
        if not group["complete"]:
            _refuse(document, "这套卡组还没通过 B 层稳健复验，D 层要用过了 B 的卡组")
        if (strategy or "").strip() not in STRATEGIES:
            _refuse(document, "D 层 strategy 只能选：经济优先 / 防线优先 / 速攻清场")
        done = set(((document["detail"]["D"].get(task_id, {}).get("items", {}).get(key) or {})
                    .get("done") or []))
        if strategy.strip() in done:
            _refuse(document, "这个打法在这套卡组上已经赢够 2 个种子了；D 层要 3 种不同的开局节奏，换一种")
        return context
    if layer == "B":
        if key not in set(item.get("passed") or []):
            _refuse(document, "B 层只复验 A 层胜局最多的 %d 套卡组" % int(req["B"]["decks"]))
        return context
    if layer != "A":
        _refuse(document, "%s 层任务现在不能建局（当前 %s 层）" % (kind, layer))
    decks = item.get("decks") or {}
    changes = int(req["A"]["min_card_changes"])
    if any(card_diff(deck, entry.get("deck")) < changes for entry in document["archives"]
           if entry["task_id"] == task_id and int(entry.get("cycle") or 1) < cycle):
        _refuse(document, "A 层新卡组须与本关之前所有圈的卡组至少更换 %d 张" % changes)
    if key in decks:
        if decks[key]["wins"] >= int(req["A"]["wins_per_deck"]):
            _refuse(document, "这套卡组已赢够种子，请继续探索新卡组")
        return context
    if any(card_diff(deck, group["deck"]) < changes for group in decks.values()):
        _refuse(document, "这套卡组和本关已有卡组差不到 %d 张；A 层要求每套明显不同" % changes)
    return context


def archive_requirements(task_id, strategy="", review="", abandoned=False, *, layer=None,
                         opening=None, cycle=None, deck=None, reason="", note="", result=None) -> None:
    """导出存档前的必填项：A/D 层要思路与评价，C 层全输要诊断。"""
    if layer == "decks":
        from . import decks
        return decks.archive_requirements(task_id, deck, reason=reason, note=note,
                                          result=result, abandoned=abandoned)
    if abandoned:
        return
    task = load_tasks().get(task_id) or {}
    if task_id not in (plan()["tasks"] or {}):
        return
    kind = layer or task.get("layer")
    if kind in ("A", "E") and not (strategy or "").strip():
        raise ValueError("A/E 层导出前请填写 strategy（核心思路），例如 Game(..., strategy=\"投手压制\")")
    if kind in ("A", "D") and not (review or "").strip():
        raise ValueError("导出前请填写 game.deck_review（一句话：这套卡组好打吗、为什么），"
                         "缺评价不能进下一层")
    if kind == "D":
        if strategy not in STRATEGIES or not opening:
            raise ValueError("D 层须声明规定的开局节奏，并有实际成功种植的开局记录")
        from .delivery import progress
        document = progress()
        for entry in document["archives"]:
            if (entry["task_id"] == task_id and int(entry.get("cycle") or 1) == cycle
                    and entry.get("workflow_layer") == "D" and entry.get("won")
                    and deck_key(entry.get("deck")) == deck_key(deck)
                    and entry.get("strategy") != strategy and entry.get("opening") == opening):
                raise ValueError("这段开局已用于另一种打法，仅改名或移动种植行列不计新打法")


# ----------------------------------------------------------------- 报告

def report(document) -> str:
    if document.get("mode") == "decks":
        from . import decks
        return decks.report(document)
    view = _view(document)
    lines = ["# LLM 对局进度（循环式）", "",
             "运行ID：%s" % document["run_id"], "",
             "第 **%d** 圈 · 当前层 **%s** · 本圈 C 层批次 %d" % (document["cycle"], document["layer"], document["batch"]),
             "", "接口自动生成；详细结果以 progress.json 为准。", "",
             "**这个任务没有终点。** 一圈做完自动进下一圈，不会出现「全部完成」。", "",
             "## 现在该做什么", "", "- 主线：" + describe(view["main"]),
             "- C 轨：" + describe(view["challenge"]), "",
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
            lines.append("| E | %s | %s | %g 倍 胜 %d，连续败 %d，诊断%s |" % (group["task_id"], group["deck"], group["multiplier"], group["wins"], len(group["loss_seeds"]), "已写" if group["diagnosis"] else "待写" if group["diagnosable"] else "—"))
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
