"""task186 扩卡组任务：只计算结论、选下一项，不启动或重放引擎。

task220 派发次序（用户 2026-11-01 定）：学生败局中盘接回（resume）> 前沿倍率
（frontier，只派当前倍率 > 1.0 的关卡）> 传送带/Boss 支线 > 主线新卡组（升级植物
优先，雾/屋顶广度留在主线）。普通 ×1 的日/夜/池重放已退役。
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

from pvz_constants import background_for_level
from pvz_tasks import deck_pool_for_level, terrain_covered
from .game import ROOT, load_tasks
from . import workflow as wf

NOTE = ("扩卡组任务没有终点；派发顺序：学生败局中盘接回（resume）> 前沿倍率（frontier，仅 >1.0 的关卡）"
        "> 传送带/Boss 支线 > 主线新卡组（升级植物优先、雾/屋顶广度在后）。"
        "普通 ×1 重放退役；只有用户说停才停。")


def config():
    return wf.plan()["decks"]


@lru_cache(maxsize=64)
def region(level):
    return background_for_level(int(level)).split("_")[-1].lower()


def configure_shard(shard=None, shards=None):
    """0 起编号；环境变量须在 import 前设置，参数也可在领任务前设置。"""
    count = int(shards if shards is not None else
                os.environ.get("LLMPLAY_SHARDS", config()["default_shards"]))
    index = int(shard if shard is not None else os.environ.get("LLMPLAY_SHARD", 0))
    if not 1 <= count <= len(config()["order"]) or not 0 <= index < count:
        raise ValueError("分片要求 1 <= shards <= 40 且 0 <= shard < shards")
    if shard is not None or shards is not None:
        os.environ["LLMPLAY_SHARD"] = str(index)
        os.environ["LLMPLAY_SHARDS"] = str(count)
    return index, count


def assigned_tasks(*, native=False, frontier=False):
    index, count = configure_shard()
    tasks = load_tasks()
    if frontier:
        order = config()["frontier"]["order"]
        order = [key for key in order if key in tasks]
    else:
        order = config()["native_order" if native else "order"]
    if count == len(config()["terrain_shards"]):
        terrains = config()["terrain_shards"][index]
        return [key for key in order if region(tasks[key]["level"]) in terrains]
    return order[index::count]


@lru_cache(maxsize=1)
def holdouts():
    document = json.loads((ROOT / "configs/deploy17_deck_holdouts.json").read_text())
    return {wf.deck_key(item["deck"]) for item in document["tasks"]}


@lru_cache(maxsize=50)
def legal_cards(level):
    return deck_pool_for_level(level, config()["blacklist"])


def upgrade_cards(terrain):
    return list((config().get("upgrades", {}).get("cards") or {}).get(terrain) or [])


def upgrade_target(terrain):
    return int((config().get("upgrades", {}).get("targets") or {}).get(terrain, 0))


def _seed_base(level, native=False):
    cfg = config()
    if native:
        slot = cfg["native_order"].index("llmplay_decks_native_l%d_m100" % level)
        return cfg["native_seed_base"] + slot * cfg["seed_window"]
    return cfg["seed_base"] + (level - 1) * cfg["seed_window"]


def seed_at(level, index, *, native=False):
    cfg = config()
    epoch, offset = divmod(index, cfg["seed_window"])
    return _seed_base(level, native) + epoch * cfg["seed_epoch_stride"] + offset


def seed_index(level, seed, *, native=False):
    cfg = config()
    epoch, offset = divmod(int(seed) - _seed_base(level, native), cfg["seed_epoch_stride"])
    return epoch * cfg["seed_window"] + offset if epoch >= 0 and 0 <= offset < cfg["seed_window"] else None


def frontier_tasks():
    """前沿倍率任务表：每个 (关卡, 倍率, 层) 一条，自带种子窗口起点。"""
    return list(config()["frontier"]["tasks"])


def frontier_task(task_id):
    return next((task for task in frontier_tasks() if task["task_id"] == task_id), None)


def frontier_base(task_id):
    task = frontier_task(task_id)
    if task is None:
        raise ValueError("不是前沿倍率任务：%s" % task_id)
    return int(task["seed_base"])


def frontier_seed_at(task_id, index):
    cfg = config()["frontier"]
    epoch, offset = divmod(index, cfg["seed_window"])
    return frontier_base(task_id) + epoch * cfg["seed_epoch_stride"] + offset


def frontier_seed_index(task_id, seed):
    cfg = config()["frontier"]
    epoch, offset = divmod(int(seed) - frontier_base(task_id), cfg["seed_epoch_stride"])
    return epoch * cfg["seed_window"] + offset if epoch >= 0 and 0 <= offset < cfg["seed_window"] else None


def main_task(level):
    """关卡 -> 主线任务族 id。"""
    for task_id in config()["order"]:
        if int(load_tasks().get(task_id, {}).get("level") or 0) == int(level):
            return task_id
    return None


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


def _public_win(entry, level):
    # 未记录遮挡模式的旧雾局只保留历史去重，不抵扣 human_v2 的目标。
    return _won(entry) and (region(level) != "fog" or entry.get("fog_observation") == "human_v2")


def attempt_seeds(entries, task_id, deck, seed, prior=()):
    key = wf.deck_key(deck)
    seeds = set(prior) | {int(seed)}
    for entry in entries:
        if (entry.get("valid", True) and entry["task_id"] == task_id
                and wf.deck_key(entry.get("deck")) == key):
            seeds.add(int(entry["seed"]))
            seeds.update(entry.get("deck_attempt_seeds") or [])
    return sorted(seeds)


@lru_cache(maxsize=1)
def resume_catalog():
    """包内中盘接回清单（configs/llmplay_resume.json）。

    每条对应 `resume/<archive>/` 里的一份存档：学生败局在首次破防前 30–60 秒截断的
    tick 日志 + 截断点那一帧。卡组与倍率都照原局，不能再自选。
    """
    relative = (config().get("resume") or {}).get("catalog") or "configs/llmplay_resume.json"
    path = ROOT / relative
    if not path.is_file():
        return []
    document = json.loads(path.read_text(encoding="utf-8"))
    return list(document.get("items") or [])


def resume_progress(document):
    catalog = resume_catalog()
    known = {item["id"] for item in catalog}
    played = {entry.get("resume_id") for entry in document.get("archives") or []
              if entry.get("resume_id") in known}
    items = [item for item in catalog if item["id"] not in played]
    terrain_order = (config().get("resume") or {}).get("terrain_order") or []
    rank = {name: index for index, name in enumerate(terrain_order)}
    def terrain(item):
        return item.get("terrain") or region(item["level"])
    # task226：priority 0 = 当前 v8 策略的新鲜败局，1 = 旧 r86–r89 存档；新鲜的先打。
    items.sort(key=lambda item: (int(item.get("priority", 1)),
                                 rank.get(terrain(item), len(rank)),
                                 item.get("cutoff_tick", 0), item["id"]))
    counts = {}
    for item in catalog:
        name = terrain(item)
        counts.setdefault(name, [0, 0])
        counts[name][0] += 1
        if item["id"] in played:
            counts[name][1] += 1
    return dict(total=len(catalog), played=len(played), remaining=len(items),
                by_terrain=counts, items=items)


def resume_items(document):
    candidates = []
    index, count = configure_shard()
    terrains = (config()["terrain_shards"][index]
                if count == len(config().get("terrain_shards") or []) else None)
    # 目录有数千条，只把队首能领的一段变成项；领走后 resume_progress 会重新排序。
    for item in (document.get("resume", {}).get("items") or [])[:256]:
        if terrains is not None and region(item["level"]) not in terrains:
            continue
        archive = item.get("archive") or ""
        candidates.append(dict(task_id=item["id"], level=int(item["level"]),
                               terrain=item.get("terrain") or region(item["level"]),
                               cycle=1, batch=1, layer="resume", multiplier=item.get("multiplier", 1.0),
                               seed=int(item["seed"]), deck=list(item.get("deck") or []),
                               action="resume_archive", archive=archive,
                               path=str(ROOT / "resume" / archive),
                               base_task_id=item.get("base_task_id"),
                               breach_tick=item.get("breach_tick"), cutoff_tick=item.get("cutoff_tick"),
                               source_arm=item.get("source_arm"), source_checkpoint=item.get("source_checkpoint"),
                               source_executable=item.get("source_executable"),
                               source_result=item.get("source_result")))
    return candidates


# ----------------------------------------------------------------- task238 必刷题 / 自出题

def _catalog(name):
    relative = (config().get(name) or {}).get("catalog")
    if not relative:
        return []
    path = ROOT / relative
    if not path.is_file():
        return []
    document = json.loads(path.read_text(encoding="utf-8"))
    return list(document.get("items") or [])


@lru_cache(maxsize=1)
def drill_catalog():
    return _catalog("drill")


@lru_cache(maxsize=1)
def scenario_catalog():
    return _catalog("scenario")


def catalog_tasks():
    """题库里的 task_id，给 carry 保留这些记录（它们不在种子表里）。"""
    return {item["task_id"] for item in drill_catalog() + scenario_catalog()}


def _played(entries, kind, key="drill_id"):
    out = set()
    for entry in entries:
        if not entry.get(key):
            continue
        if entry.get("future_seed") is None:
            continue
        out.add((entry[key], int(entry["future_seed"])))
    return out


def drill_progress(document):
    entries = document.get("archives") or []
    played = _played(entries, "drill")
    items = []
    for item in drill_catalog():
        for index, seed in enumerate(item.get("future_seeds") or []):
            key = (item["task_id"], int(seed))
            if key in played:
                continue
            items.append(dict(item, seed=int(seed), seed_index=index))
    items.sort(key=lambda item: (int(item.get("priority", 9)), item.get("terrain", ""),
                                 item.get("type", ""), item["task_id"], item["seed_index"]))
    counts = {}
    for item in drill_catalog():
        name = item.get("terrain", "?")
        counts.setdefault(name, [0, 0])
        counts[name][0] += len(item.get("future_seeds") or [])
        counts[name][1] += sum(1 for seed in item.get("future_seeds") or []
                               if (item["task_id"], int(seed)) in played)
    return dict(total=sum(len(item.get("future_seeds") or []) for item in drill_catalog()),
                played=len(played), remaining=len(items), by_terrain=counts, items=items)


def drill_items(document):
    candidates = []
    index, count = configure_shard()
    terrains = (config()["terrain_shards"][index]
                if count == len(config().get("terrain_shards") or []) else None)
    # 题库可能上千条；只把队首一段变成项，领走后 progress 重新排序。
    for item in (document.get("drill", {}).get("items") or [])[:512]:
        if terrains is not None and item.get("terrain") not in terrains:
            continue
        candidates.append(dict(
            task_id=item["task_id"], level=int(item["level"]), terrain=item.get("terrain"),
            cycle=1, batch=1, layer="drill", multiplier=item.get("multiplier", 1.0),
            seed=int(item["seed"]), source_seed=int(item.get("source_seed") or 0),
            deck=list(item.get("deck") or []),
            action="drill_archive", archive=item.get("archive"),
            path=str(ROOT / item["archive"]), future_seed=int(item["seed"]),
            drill_id=item["task_id"], knowledge=item.get("knowledge"),
            horizon=int(item.get("horizon") or 900), targets=list(item.get("targets") or []),
            priority=int(item.get("priority", 9)), source=item.get("source"),
            suits=item.get("suits")))
    return candidates


def scenario_progress(document):
    entries = document.get("archives") or []
    played = _played(entries, "scenario", "scenario_id")
    items = []
    for item in scenario_catalog():
        for index, seed in enumerate(item.get("future_seeds") or []):
            key = (item["task_id"], int(seed))
            if key in played:
                continue
            items.append(dict(item, seed=int(seed), seed_index=index))
    items.sort(key=lambda item: (item.get("terrain", ""), item["task_id"], item["seed_index"]))
    counts = {}
    for item in scenario_catalog():
        name = item.get("terrain", "?")
        counts.setdefault(name, [0, 0])
        counts[name][0] += len(item.get("future_seeds") or [])
        counts[name][1] += sum(1 for seed in item.get("future_seeds") or []
                               if (item["task_id"], int(seed)) in played)
    return dict(total=sum(len(item.get("future_seeds") or []) for item in scenario_catalog()),
                played=len(played), remaining=len(items), by_terrain=counts, items=items)


def scenario_items(document):
    candidates = []
    index, count = configure_shard()
    terrains = (config()["terrain_shards"][index]
                if count == len(config().get("terrain_shards") or []) else None)
    for item in (document.get("scenario", {}).get("items") or [])[:512]:
        if terrains is not None and item.get("terrain") not in terrains:
            continue
        candidates.append(dict(
            task_id=item["task_id"], level=int(item["level"]), terrain=item.get("terrain"),
            cycle=1, batch=1, layer="scenario", multiplier=item.get("multiplier", 1.0),
            seed=int(item["seed"]), source_seed=int(item.get("source_seed") or 0),
            deck=list(item.get("deck") or []),
            action="scenario_sandbox", archive=item.get("archive"),
            path=str(ROOT / item["archive"]), future_seed=int(item["seed"]),
            scenario_id=item["task_id"], base_id=item.get("base_id"),
            knowledge_candidates=list(item.get("knowledge_candidates") or []),
            knowledge=(item.get("knowledge_candidates") or [None])[0],
            horizon=int(item.get("horizon") or 900), suits=item.get("suits")))
    return candidates


def derive(entries, run_id=None):
    cfg, definitions = config(), load_tasks()
    by_level = {}
    for entry in _latest(entries):
        if entry.get("resume_id") or entry.get("drill_id") or entry.get("scenario_id"):
            continue
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
        groups, registered, counted, used, pending = {}, {}, {}, set(), None
        for entry in by_level.get(level, []):
            key, cards = wf.deck_key(entry.get("deck")), entry.get("deck") or []
            if _won(entry):
                registered[key] = list(cards)
                if _public_win(entry, level):
                    counted[key] = list(cards)
            if entry["task_id"] != task_id:
                continue
            group = groups.setdefault(key, dict(deck=list(cards), losses=0, won=False,
                                                loss_notes=[], seeds=[]))
            group["seeds"].append(entry["seed"])
            group["seeds"].extend(entry.get("deck_attempt_seeds") or [])
            if _won(entry):
                group["won"] = True
            elif not entry.get("abandoned") and entry.get("result") == 2:
                group["losses"] += 1
                group["loss_notes"].append(entry.get("note") or entry.get("deck_review") or "")
            used.add(int(entry["seed"]))
            if not entry.get("abandoned") and entry.get("result") == 0:
                pending = dict(seed=entry["seed"], deck=list(cards), archive=entry["archive"])
        wins = {key: cards for key, cards in counted.items()
                if key not in holdouts() and set(cards) <= set(pool)}
        small = sum(len(cards) <= 7 for cards in wins.values())
        eight = sum(len(cards) == 8 for cards in wins.values())
        target = cfg["targets"][terrain]
        for group in groups.values():
            group["seed_attempts"] = len(set(group["seeds"]))
        indices = [seed_index(level, seed) for seed in used]
        cursor = max([value for value in indices if value is not None], default=-1) + 1
        fresh = [seed_at(level, cursor + n) for n in range(candidate_count)]
        retry = next((group for group in reversed(list(groups.values()))
                      if not group["won"] and 0 < group["losses"] < cfg["max_losses"]), None)
        if pending and (wf.deck_key(pending["deck"]) in registered or
                        groups[wf.deck_key(pending["deck"])]["losses"] >= cfg["max_losses"]):
            pending = None
        levels[task_id] = dict(level=level, terrain=terrain, victory_decks=len(wins),
                              target=target, goal=target, small_decks=small, eight_decks=eight,
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
    # task220：升级植物按地形配额。只算本关已登记、且卡组里带该地形升级植物的不同胜组。
    upgrade_progress = {}
    for terrain in (cfg.get("upgrades", {}).get("targets") or {}):
        cards = upgrade_cards(terrain)
        decks = set()
        for entry in entries:
            if entry.get("carried") or not _won(entry):
                continue
            if region(int(entry.get("level") or 0)) != terrain:
                continue
            if set(entry.get("deck") or []) & set(cards):
                decks.add(wf.deck_key(entry.get("deck")))
        upgrade_progress[terrain] = dict(wins=len(decks), target=upgrade_target(terrain),
                                         missing=max(0, upgrade_target(terrain) - len(decks)),
                                         cards=cards, decks=sorted(decks))
    active, active_native = assigned_tasks(), assigned_tasks(native=True)
    native_progress = {}
    for task_id in cfg["native_order"]:
        task = definitions[task_id]
        level = task["level"]
        records = [entry for entry in by_level.get(level, []) if entry["task_id"] == task_id]
        wins = {entry["seed"] for entry in by_level.get(level, [])
                if _public_win(entry, level) and (entry["task_id"] == task_id or
                definitions.get(entry["task_id"], {}).get("conveyor_start") == "original")}
        indices = [seed_index(level, entry["seed"], native=True) for entry in records]
        cursor = max([value for value in indices if value is not None], default=-1) + 1
        pending = next((dict(seed=entry["seed"], deck=[], archive=entry["archive"])
                        for entry in reversed(records) if entry.get("result") == 0
                        and not entry.get("abandoned")), None)
        native_progress[task_id] = dict(level=level, terrain=region(level), wins=len(wins),
                                        winning_seeds=sorted(wins), target=cfg["native_wins"],
                                        complete=len(wins) >= cfg["native_wins"], pending=pending,
                                        available_seeds=[seed_at(level, cursor + n, native=True)
                                                         for n in range(candidate_count)])
    # task220：前沿倍率。只给课程当前倍率 > 1.0 的关卡派发；用本关已登记胜组打新种子。
    frontier_cfg = cfg["frontier"]
    active_frontier = assigned_tasks(frontier=True)
    frontier_progress = {}
    for spec in frontier_cfg["tasks"]:
        task_id = spec["task_id"]
        if definitions.get(task_id) is None:
            continue
        level, terrain = int(spec["level"]), region(spec["level"])
        base_task = main_task(level)
        info = levels.get(base_task) or {}
        registered = list((info.get("winning_decks") or {}).values())
        records = [entry for entry in _latest(entries)
                   if entry.get("valid", True) and entry.get("task_id") == task_id
                   and not entry.get("abandoned")]
        wins = sorted({int(entry["seed"]) for entry in records if _won(entry)})
        indices = [frontier_seed_index(task_id, entry["seed"]) for entry in records]
        cursor = max([value for value in indices if value is not None], default=-1) + 1
        frontier_progress[task_id] = dict(level=level, terrain=terrain,
                                          multiplier=float(spec["multiplier"]),
                                          tier=spec.get("tier"),
                                          wins=len(wins), winning_seeds=wins,
                                          target=frontier_cfg["target_wins"],
                                          complete=len(wins) >= frontier_cfg["target_wins"],
                                          decks=[list(cards) for cards in registered],
                                          legal_cards=legal_cards(level),
                                          available_seeds=[frontier_seed_at(task_id, cursor + n)
                                                           for n in range(candidate_count)])
    # task210：传送带支线先派发到各关达标，之后回到主线并按轮次提高主线配额。
    covered = all(terrain_progress[levels[key]["terrain"]]["complete"] for key in active)
    native_done = all(native_progress[key]["complete"] for key in active_native)
    stage = min((max(0, (levels[key]["victory_decks"] - levels[key]["target"]) // cfg["growth_step"] + 1)
                 if levels[key]["baseline_complete"] else 0 for key in active), default=0)
    stage = stage if covered and native_done else 0
    for info in levels.values():
        info["goal"] = info["target"] + stage * cfg["growth_step"]
    index, count = configure_shard()
    flat = {task_id: dict(item, active=task_id in active, complete=False, layer="decks")
            for task_id, item in levels.items()}
    flat.update({task_id: dict(item, active=task_id in active_native, layer="native")
                 for task_id, item in native_progress.items()})
    flat.update({task_id: dict(item, active=task_id in active_frontier, layer="frontier")
                 for task_id, item in frontier_progress.items()})
    resume = resume_progress(dict(archives=list(entries)))
    drill = drill_progress(dict(archives=list(entries)))
    scenario = scenario_progress(dict(archives=list(entries)))
    return dict(schema=wf.PROGRESS_SCHEMA, mode="decks", run_id=run_id, cycle=1, batch=1,
                layer="decks", complete=False, note=NOTE, requirements=cfg,
                shard=index, shards=count, active_tasks=active, levels=levels, stage=stage + 1,
                active_native=active_native, native=native_progress,
                active_frontier=active_frontier, frontier=frontier_progress,
                resume=resume, drill=drill, scenario=scenario, upgrades=upgrade_progress,
                coverage=terrain_progress, tasks=flat, archives=list(entries))


def items(document):
    levels = document["levels"]
    upgrades = document.get("upgrades") or {}
    order = sorted(document["active_tasks"], key=lambda key: (
        not bool(document["coverage"][levels[key]["terrain"]]["missing"]),
        not bool((upgrades.get(levels[key]["terrain"]) or {}).get("missing")),
        levels[key]["victory_decks"] / levels[key]["goal"], levels[key]["attempts"], levels[key]["level"]))
    candidates = []
    for task_id in order:
        info = levels[task_id]
        missing = document["coverage"][info["terrain"]]["missing"]
        if (info["victory_decks"] >= info["goal"] and info["baseline_complete"]
                and not missing and not info["pending"]):
            continue
        priority = sorted((int(card) for card in missing if int(card) in info["legal_cards"]),
                          key=lambda card: (-missing[str(card)], card))
        upgrade = upgrades.get(info["terrain"]) or {}
        base = dict(task_id=task_id, level=info["level"], terrain=info["terrain"],
                    cycle=1, batch=1, layer="decks", multiplier=1.0,
                    victory_decks=info["victory_decks"], target=info["goal"],
                    required_lengths=info["required_lengths"], priority_cards=priority,
                    upgrade_cards=list(upgrade.get("cards") or []),
                    upgrade_wins=upgrade.get("wins"), upgrade_target=upgrade.get("target"),
                    upgrade_required=bool(upgrade.get("missing")),
                    legal_cards=info["legal_cards"], registered_decks=list(info["registered_decks"].values()),
                    failed_decks=[group for group in info["decks"].values() if group["losses"] >= config()["max_losses"]])
        if info["pending"]:
            candidates.append(dict(base, action="resume", **info["pending"]))
            continue
        for seed in info["available_seeds"][:1]:
            retry = info["retry"]
            candidates.append(dict(base, seed=seed, deck=retry["deck"] if retry else None,
                                   action="retry_deck" if retry else "new_deck",
                                   losses=retry["losses"] if retry else 0))
    # One candidate per level before allocating second seeds of the same level.
    grouped = [[item for item in candidates if item["task_id"] == key] for key in order]
    return [group[n] for n in range(max(map(len, grouped), default=0)) for group in grouped if n < len(group)]


def native_items(document):
    candidates = []
    for task_id in document["active_native"]:
        info = document["native"][task_id]
        if info["complete"]:
            continue
        base = dict(task_id=task_id, level=info["level"], terrain=info["terrain"],
                    cycle=1, batch=1, layer="native", multiplier=1.0, deck=[],
                    wins=info["wins"], target=info["target"], conveyor_start="original")
        if info["pending"]:
            candidates.append(dict(base, action="resume", **{key: value for key, value in info["pending"].items() if key != "deck"}))
        else:
            candidates.extend(dict(base, seed=seed, action="native_play")
                              for seed in info["available_seeds"])
    return candidates


def frontier_items(document):
    candidates = []
    for task_id in document.get("active_frontier", []):
        info = document.get("frontier", {}).get(task_id)
        if info is None or info["complete"] or not info["decks"]:
            continue
        base = dict(task_id=task_id, level=info["level"], terrain=info["terrain"],
                    cycle=1, batch=1, layer="frontier", multiplier=info["multiplier"],
                    wins=info["wins"], target=info["target"], registered_decks=info["decks"],
                    legal_cards=info["legal_cards"])
        seed = next((value for value in info["available_seeds"]
                     if wf._slot(task_id, value) not in wf.held()), None)
        if seed is None:
            continue
        candidates.append(dict(base, seed=seed, deck=info["decks"][info["wins"] % len(info["decks"])],
                               action="frontier_deck"))
    return candidates


def view(document):
    claims = wf.held()
    queues = [(drill_items(document), "drill"), (scenario_items(document), "scenario"),
              (resume_items(document), "resume"), (frontier_items(document), "frontier"),
              (native_items(document), "native"), (items(document), "main")]
    pools = {name: wf._first_free(candidates, claims) for candidates, name in queues}
    item = next((found for found in pools.values() if found), None)
    if item is None:
        item = wf._busy([row for candidates, _ in queues for row in candidates]) \
            or wf._item(document, None, "blocked", note="没有可推进的项")
    return dict(item=item, main=pools["main"], drill=pools["drill"],
                scenario=pools["scenario"], resume=pools["resume"],
                frontier=pools["frontier"],
                challenge=None, native=pools["native"], complete=False, cycle=1,
                shard=document["shard"], shards=document["shards"], note=NOTE)


def _earlier_block(document, name):
    """name 之前的队列里还有能领的项就拦下来（drill > scenario > resume > frontier > native > main）。"""
    claims = wf.held()
    queues = {"drill": drill_items, "scenario": scenario_items, "resume": resume_items,
              "frontier": frontier_items, "native": native_items, "main": items}
    order = ["drill", "scenario", "resume", "frontier", "native", "main"]
    labels = {"drill": "真实必刷题", "scenario": "GPT 自出题", "resume": "学生败局中盘接回",
              "frontier": "前沿倍率", "native": "传送带/Boss 支线", "main": "主线"}
    for earlier in order[:order.index(name)]:
        if wf._first_free(queues[earlier](document), claims) is not None:
            raise ValueError("%s优先；请先领取本片%s项" % (labels[earlier], labels[earlier]))
    return None


def check(document, task_id, deck, strategy="", seed=None):
    if task_id in {item["task_id"] for item in drill_catalog()}:
        raise ValueError("必刷题用 Game.load_archive(item['path']) 打开；起点、卡组和视界照题库")
    if task_id in {item["task_id"] for item in scenario_catalog()}:
        raise ValueError("自出题用 Game.load_archive(item['path']) 打开底局，再 spawn 注入威胁")
    if task_id in {item["id"] for item in resume_catalog()}:
        raise ValueError("中盘接回项用 Game.load_archive(item['path']) 打开；卡组与倍率照原局，不能自选")
    if task_id in document.get("active_frontier", []):
        info = document["frontier"][task_id]
        _earlier_block(document, "frontier")
        if info["complete"]:
            raise ValueError("本关前沿倍率已达标；请领取其余项")
        if seed is None or int(seed) not in info["available_seeds"]:
            raise ValueError("请使用 next_task() 分配的本关前沿倍率新种子")
        if wf.deck_key(deck or []) not in {wf.deck_key(cards) for cards in info["decks"]}:
            raise ValueError("前沿倍率只能用本关已登记的胜利卡组；新卡组请领主线项")
        if not terrain_covered(info["level"], deck):
            raise ValueError("卡组没有覆盖本关地形")
        if wf.deck_key(deck) in holdouts():
            raise ValueError("门禁/挑选留出卡组签名禁止使用（跨关同样禁止）")
        if not set(deck) <= set(info["legal_cards"]):
            raise ValueError("卡组含任务黑名单中的卡；请看 next_task() 的 legal_cards")
        return dict(cycle=1, workflow_layer="frontier")
    if task_id in document["active_native"]:
        info = document["native"][task_id]
        _earlier_block(document, "native")
        if deck:
            raise ValueError("原生传送带/固定卡支线由引擎发牌，不能自选卡组")
        if info["complete"] or seed is None or int(seed) not in info["available_seeds"]:
            raise ValueError("支线须使用队列分配的新种子，各关达到 16 胜后不再分配")
        return dict(cycle=1, workflow_layer="native")
    if task_id not in document["active_tasks"]:
        raise ValueError("请使用本分片 next_task() 给出的新任务族")
    info = document["levels"][task_id]
    if wf.deck_key(deck or []) not in info["decks"]:
        _earlier_block(document, "main")
    if not any(item["task_id"] == task_id for item in items(document)):
        raise ValueError("本关本轮目标已达标；请领取其余主线或传送带支线")
    if seed is None or int(seed) not in info["available_seeds"]:
        raise ValueError("请使用 next_task() 分配的本关新种子；中盘用 Game.load_archive()")
    # Task design can prefilter without launching an engine; Game asks C++ at reset.
    if not terrain_covered(info['level'], deck):
        raise ValueError('卡组没有覆盖本关地形')
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
    upgrade = (document.get("upgrades") or {}).get(info["terrain"]) or {}
    if (not group and upgrade.get("missing")
            and not (set(deck) & set(upgrade.get("cards") or []))):
        raise ValueError("本关地形升级植物还缺 %d 套胜组；新卡组必须带一张 %s 升级植物并带它的底座"
                         "（底座规则见 PROMPT）" % (upgrade["missing"], upgrade["cards"]))
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
             "当前分片 %d/%d（0 起编号）；一套一胜。" % (document["shard"], document["shards"]),
             "", "## 学生败局中盘接回（最高优先，一条只打一次）", ""]
    resume = document.get("resume") or {}
    lines.append("共 %d 条，已打 %d，剩 %d。" % (resume.get("total", 0), resume.get("played", 0),
                                              resume.get("remaining", 0)))
    for terrain, (total, played) in sorted((resume.get("by_terrain") or {}).items()):
        lines.append("- %s：%d/%d" % (terrain, played, total))
    lines += ["", "## 前沿倍率（课程当前倍率 > 1.0）", "",
              "| 关卡 | 地形 | 倍率 | 胜局 | 目标 |", "|---|---|---|---|---|"]
    for key in document.get("active_frontier", []):
        info = document["frontier"][key]
        lines.append("| L%d | %s | %.2f | %d | %d |" % (
            info["level"], info["terrain"], info["multiplier"], info["wins"], info["target"]))
    lines += ["", "## 升级植物配额", "", "| 地形 | 升级胜组 | 目标 | 卡 |", "|---|---|---|---|"]
    for terrain, info in sorted((document.get("upgrades") or {}).items()):
        lines.append("| %s | %d | %d | %s |" % (terrain, info["wins"], info["target"], info["cards"]))
    lines += ["", "| 关卡 | 地形 | 胜利卡组 | 当前目标 | ≤7张 | 8张 |", "|---|---|---|---|---|---|"]
    for key in document["active_tasks"]:
        info = document["levels"][key]
        lines.append("| L%d | %s | %d | %d | %d/3 | %d/3 |" % (
            info["level"], info["terrain"], info["victory_decks"], info["goal"], info["small_decks"], info["eight_decks"]))
    lines += ["", "## 原生传送带支线（前沿倍率之后派发，各关 16 胜后停止）", "",
              "| 关卡 | 不同种子胜局 | 目标 |", "|---|---|---|"]
    for key in document["active_native"]:
        info = document["native"][key]
        lines.append("| L%d | %d | %d |" % (info["level"], info["wins"], info["target"]))
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
