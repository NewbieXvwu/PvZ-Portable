"""task186 扩卡组任务：只计算结论、选下一项，不启动或重放引擎。"""
from __future__ import annotations

import json
import os
from functools import lru_cache

from pvz_constants import background_for_level
from pvz_tasks import deck_pool_for_level
from .game import ROOT, load_tasks, validate_deck
from . import workflow as wf

NOTE = "扩卡组任务没有终点；达标后继续增加胜利卡组，优先补覆盖缺口。只有用户说停才停。"


def config():
    return wf.plan()["decks"]


def region(level):
    return background_for_level(int(level)).split("_")[-1].lower()


def configure_shard(shard=None, shards=None):
    """0 起编号；环境变量须在 import 前设置，参数也可在领任务前设置。"""
    count = int(shards if shards is not None else
                os.environ.get("LLMPLAY_SHARDS", config()["default_shards"]))
    index = int(shard if shard is not None else os.environ.get("LLMPLAY_SHARD", 0))
    if not 1 <= count <= len(config()["order"]) or not 0 <= index < count:
        raise ValueError("分片要求 1 <= shards <= 41 且 0 <= shard < shards")
    if shard is not None or shards is not None:
        os.environ["LLMPLAY_SHARD"] = str(index)
        os.environ["LLMPLAY_SHARDS"] = str(count)
    return index, count


def assigned_tasks():
    index, count = configure_shard()
    tasks = load_tasks()
    order = config()["order"]
    if count == len(config()["terrain_shards"]):
        return [key for key in order
                if region(tasks[key]["level"]) in config()["terrain_shards"][index]]
    return order[index::count]


@lru_cache(maxsize=1)
def holdouts():
    document = json.loads((ROOT / "configs/deploy17_deck_holdouts.json").read_text())
    return {wf.deck_key(item["deck"]) for item in document["tasks"]}


@lru_cache(maxsize=50)
def legal_cards(level):
    return deck_pool_for_level(level, config()["blacklist"])


def seed_at(level, index):
    cfg = config()
    epoch, offset = divmod(index, cfg["seed_window"])
    return cfg["seed_base"] + epoch * cfg["seed_epoch_stride"] + (
        level - 1) * cfg["seed_window"] + offset


def seed_index(level, seed):
    cfg = config()
    epoch, offset = divmod(int(seed) - cfg["seed_base"], cfg["seed_epoch_stride"])
    offset -= (level - 1) * cfg["seed_window"]
    return epoch * cfg["seed_window"] + offset if epoch >= 0 and 0 <= offset < cfg["seed_window"] else None


def _latest(entries):
    """同次局的多次导出取最远结论；同种子同卡组不会刷计数。"""
    attempts = {}
    for entry in entries:
        if not entry.get("valid", True):
            continue
        key = (entry["task_id"], entry["seed"], wf.deck_key(entry.get("deck")))
        previous = attempts.get(key)
        def rank(item):
            return (item.get("result") in (1, 2),
                    int((item.get("outcome") or {}).get("final_tick") or 0))
        if previous is None or rank(entry) >= rank(previous):
            attempts[key] = entry
    return list(attempts.values())


def _won(entry):
    return entry.get("won") and not entry.get("abandoned") and entry.get("result") == 1


def derive(entries, run_id=None):
    cfg, definitions = config(), load_tasks()
    by_level = {}
    for entry in _latest(entries):
        task = definitions.get(entry["task_id"], {})
        level = int(entry.get("level") or task.get("level") or 0)
        multiplier = entry.get("zombie_count_multiplier") or task.get("zombie_count_multiplier")
        if multiplier == 1.0 and level:
            by_level.setdefault(level, []).append(entry)

    levels, coverage = {}, {}
    candidate_count = 8 + len(wf.held())
    # Count all shards' registered conclusions, even when only one shard is active.
    for task_id in cfg["order"]:
        task = definitions[task_id]
        level, terrain = int(task["level"]), region(task["level"])
        pool = legal_cards(level)
        groups, registered, used, pending = {}, {}, set(), None
        for entry in by_level.get(level, []):
            key, cards = wf.deck_key(entry.get("deck")), entry.get("deck") or []
            if _won(entry):
                registered[key] = list(cards)
            if entry["task_id"] != task_id:
                continue
            group = groups.setdefault(key, dict(deck=list(cards), losses=0, won=False,
                                                loss_notes=[], seeds=[]))
            group["seeds"].append(entry["seed"])
            if _won(entry):
                group["won"] = True
            elif not entry.get("abandoned") and entry.get("result") == 2:
                group["losses"] += 1
                group["loss_notes"].append(entry.get("note") or entry.get("deck_review") or "")
            used.add(int(entry["seed"]))
            if not entry.get("abandoned") and entry.get("result") == 0:
                pending = dict(seed=entry["seed"], deck=list(cards), archive=entry["archive"])
        wins = {key: cards for key, cards in registered.items()
                if key not in holdouts() and set(cards) <= set(pool)}
        small = sum(len(cards) <= 7 for cards in wins.values())
        eight = sum(len(cards) == 8 for cards in wins.values())
        target = cfg["targets"][terrain]
        goal = target + max(0, (len(wins) - target) // cfg["growth_step"] + 1) * cfg["growth_step"] if len(wins) >= target else target
        indices = [seed_index(level, seed) for seed in used]
        cursor = max([value for value in indices if value is not None], default=-1) + 1
        fresh = [seed_at(level, cursor + n) for n in range(candidate_count)]
        retry = next((group for group in reversed(list(groups.values()))
                      if not group["won"] and 0 < group["losses"] < cfg["max_losses"]), None)
        if pending and (wf.deck_key(pending["deck"]) in registered or
                        groups[wf.deck_key(pending["deck"])]["losses"] >= cfg["max_losses"]):
            pending = None
        levels[task_id] = dict(level=level, terrain=terrain, victory_decks=len(wins),
                              target=target, goal=goal, small_decks=small, eight_decks=eight,
                              baseline_complete=len(wins) >= target and small >= cfg["min_small_decks"]
                              and eight >= cfg["min_eight_decks"], registered_decks=registered,
                              winning_decks=wins, decks=groups, retry=retry, pending=pending,
                              legal_cards=pool, available_seeds=fresh, attempts=len(used),
                              required_lengths=([6, 7] if small < cfg["min_small_decks"] else
                                                [8] if eight < cfg["min_eight_decks"] else list(range(6, 11))))
        terrain_cards = coverage.setdefault(terrain, {})
        for card in pool:
            signatures = terrain_cards.setdefault(card, set())
            signatures.update(key for key, cards in wins.items() if card in cards)

    terrain_progress = {}
    for terrain, cards in coverage.items():
        focus = cfg.get("coverage_priority", {}).get(terrain, {})
        counts = {str(card): len(signatures) for card, signatures in sorted(cards.items())}
        targets = {card: max(cfg["coverage_per_card"], int(focus.get(card, cfg["coverage_per_card"])))
                   for card in counts}
        missing = {card: targets[card] - counts[card] for card in counts if counts[card] < targets[card]}
        terrain_progress[terrain] = dict(counts=counts, targets=targets, missing=missing,
                                         covered=sum(counts[card] >= targets[card] for card in counts),
                                         total=len(counts), complete=not missing)
    active = assigned_tasks()
    index, count = configure_shard()
    flat = {task_id: dict(item, active=task_id in active, complete=False, layer="decks")
            for task_id, item in levels.items()}
    return dict(schema=wf.PROGRESS_SCHEMA, mode="decks", run_id=run_id, cycle=1, batch=1,
                layer="decks", complete=False, note=NOTE, requirements=cfg,
                shard=index, shards=count, active_tasks=active, levels=levels,
                coverage=terrain_progress, tasks=flat, archives=list(entries))


def items(document):
    levels = document["levels"]
    order = sorted(document["active_tasks"], key=lambda key: (
        not bool(document["coverage"][levels[key]["terrain"]]["missing"]),
        levels[key]["victory_decks"] / levels[key]["goal"], levels[key]["attempts"], levels[key]["level"]))
    candidates = []
    for task_id in order:
        info = levels[task_id]
        missing = document["coverage"][info["terrain"]]["missing"]
        priority = sorted((int(card) for card in missing if int(card) in info["legal_cards"]),
                          key=lambda card: (-missing[str(card)], card))
        base = dict(task_id=task_id, level=info["level"], terrain=info["terrain"],
                    cycle=1, batch=1, layer="decks", multiplier=1.0,
                    victory_decks=info["victory_decks"], target=info["goal"],
                    required_lengths=info["required_lengths"], priority_cards=priority,
                    legal_cards=info["legal_cards"], registered_decks=list(info["registered_decks"].values()),
                    failed_decks=[group for group in info["decks"].values() if group["losses"] >= config()["max_losses"]])
        if info["pending"]:
            candidates.append(dict(base, action="resume", **info["pending"]))
        for seed in info["available_seeds"]:
            retry = info["retry"]
            candidates.append(dict(base, seed=seed, deck=retry["deck"] if retry else None,
                                   action="retry_deck" if retry else "new_deck",
                                   losses=retry["losses"] if retry else 0))
    # One candidate per level before allocating second seeds of the same level.
    grouped = [[item for item in candidates if item["task_id"] == key] for key in order]
    return [group[n] for n in range(max(map(len, grouped), default=0)) for group in grouped if n < len(group)]


def view(document):
    candidates = items(document)
    item = wf._first_free(candidates, wf.held()) or wf._busy(candidates)
    return dict(item=item, main=item, challenge=None, complete=False, cycle=1,
                shard=document["shard"], shards=document["shards"], note=NOTE)


def check(document, task_id, deck, strategy="", seed=None):
    if task_id not in document["active_tasks"]:
        raise ValueError("请使用本分片 next_task() 给出的新任务族")
    info = document["levels"][task_id]
    validate_deck(load_tasks()[task_id], deck)
    key = wf.deck_key(deck)
    if key in holdouts():
        raise ValueError("门禁/挑选留出卡组签名禁止使用（跨关同样禁止）")
    if not set(deck) <= set(info["legal_cards"]):
        raise ValueError("卡组含任务黑名单中的卡；请看 next_task() 的 legal_cards")
    if any(wf.card_diff(deck, cards) < config()["min_card_changes"]
           for cards in info["registered_decks"].values()):
        raise ValueError("新卡组须与本关所有已登记胜利卡组至少差 2 张；顺序不计，换一张只算一张")
    group = info["decks"].get(key, {})
    if group.get("losses", 0) >= config()["max_losses"]:
        raise ValueError("这套卡组已三败，请记录失败原因并换卡组")
    if len(deck) not in info["required_lengths"] and not group:
        raise ValueError("本关卡组大小尚缺配额；新卡组长度须为 %s" % info["required_lengths"])
    if seed is None or int(seed) not in info["available_seeds"]:
        raise ValueError("请使用 next_task() 分配的本关新种子；中盘用 Game.load_archive()")
    return dict(cycle=1, workflow_layer="decks")


def archive_requirements(task_id, deck, *, reason="", note="", result=None, **kwargs):
    if not str(reason).strip():
        raise ValueError("deck_reason 必填：用 reason 写经济、主力、防守、应急、地形专用卡的核心思路")
    if result != 2 or kwargs.get("abandoned"):
        return
    document = wf._progress()
    group = document["levels"].get(task_id, {}).get("decks", {}).get(wf.deck_key(deck), {})
    if group.get("losses", 0) >= config()["max_losses"] - 1 and not str(note).strip():
        raise ValueError("第三次败局请先填写 game.note 的一句失败原因，再导出，然后换卡组")


def report(document):
    lines = ["# 扩卡组进度", "", NOTE, "",
             "当前分片 %d/%d（0 起编号）；原版 1.0 倍，一套一胜。" % (document["shard"], document["shards"]),
             "", "| 关卡 | 地形 | 胜利卡组 | 当前目标 | ≤7张 | 8张 |", "|---|---|---|---|---|---|"]
    for key in document["active_tasks"]:
        info = document["levels"][key]
        lines.append("| L%d | %s | %d | %d | %d/3 | %d/3 |" % (
            info["level"], info["terrain"], info["victory_decks"], info["goal"], info["small_decks"], info["eight_decks"]))
    lines += ["", "| 地形 | 达到覆盖目标的卡 | 各卡胜利卡组数 | 覆盖缺口 |", "|---|---|---|---|"]
    for terrain, info in document["coverage"].items():
        lines.append("| %s | %d/%d | %s | %s |" % (terrain, info["covered"], info["total"], info["counts"], info["missing"]))
    lines += ["", "## 已登记胜利卡组", ""]
    for key in document["active_tasks"]:
        info = document["levels"][key]
        lines.append("L%d：%s" % (info["level"], list(info["registered_decks"].values())))
    lines += ["", "## 三败卡组及原因", ""]
    for key in document["active_tasks"]:
        for group in document["levels"][key]["decks"].values():
            if group["losses"] >= config()["max_losses"]:
                lines.append("%s %s：%s" % (key, group["deck"], group["loss_notes"][-1]))
    return "\n".join(lines) + "\n"
