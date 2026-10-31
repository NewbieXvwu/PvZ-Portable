"""选卡前查询指定种子的威胁种类与首次波次；不推进、重放或保存对局。"""
from __future__ import annotations

import copy
import re
from functools import lru_cache

import pvz_constants as constants
from pvz_env import PvZEnv
from pvz_tasks import deck_for_level
from .game import (load_tasks, task_spec, _default_executable, _default_resource_dir,
                   _ensure_executable)
from .resources import app_data


@lru_cache(maxsize=1)
def _definitions():
    definitions = constants.zombie_defs()
    return {item["enum"]: item for item in definitions}, {item["type"]: item["enum"] for item in definitions}


def _summary(task_id, spec, observation, state):
    definitions, ids = _definitions()
    entries = {}

    def add(kind, source, wave=None, note=""):
        kind = int(kind)
        item = entries.setdefault(kind, dict(type=kind,
            name=constants.ZOMBIE_NAME.get(kind, definitions[kind]["name"]),
            slug=definitions[kind]["type"].lower(), first_wave=wave,
            sources=[], note=note))
        if wave is not None and (item["first_wave"] is None or wave < item["first_wave"]):
            item["first_wave"] = wave
        if source not in item["sources"]:
            item["sources"].append(source)
        return item

    # 一次读取引擎生成的原始波表；只保留种类与首次波次，不返回逐波数量/顺序或随机数。
    for wave, kinds in enumerate(state["hidden"]["zombies_in_wave"], 1):
        for kind in kinds:
            if kind >= 0:
                add(kind, "wave", wave)
    for zombie in observation.get("zombies") or []:
        if zombie.get("on_board"):
            add(zombie["type"], "initial", 0, "开局已在场")
    for kind in (observation.get("loadout_context") or {}).get("zombie_roster") or []:
        if kind not in entries:
            add(kind, "allowed", note="本关允许的种类；本种子预生成波表没有固定首次波次")

    if spec.level == 50:
        add(ids["BOSS"], "boss", note="Boss关没有普通波次，按实际阶段行动")
        text = constants._read(constants.ZOMBIE_CPP)
        body = constants._array_body(text, "gBossZombieList")
        start, end = constants._function_span(text, "void Zombie::BossBungeeSpawn()")
        for enum in set(re.findall(r"ZombieType::ZOMBIE_(\w+)", body + text[start:end])):
            add(ids[enum], "boss_summon", note="Boss可能召唤，首次出现取决于实际阶段")

    # 这些是引擎的召唤关系，不是选卡或动作规则；召唤发生与否/时间取决于战况。
    for child, parents, note in (
            ("BACKUP_DANCER", ("DANCER",), "由舞王召唤，无固定首次波次"),
            ("IMP", ("GARGANTUAR", "REDEYE_GARGANTUAR"), "由巨人投出，无固定首次波次")):
        if any(ids[parent] in entries for parent in parents):
            add(ids[child], "summon", note=note)

    return dict(task_id=task_id, level=spec.level, seed=spec.seed,
                wave_count=observation.get("wave_count"),
                zombies=sorted(entries.values(), key=lambda item: (
                    item["first_wave"] is None, item["first_wave"] or 0, item["type"])),
                note="first_wave从1起，0表示开局在场，None表示没有固定波次；"
                     "wave来自本种子引擎波表，allowed/summon/boss_summon是可能威胁，非必然发生。")


@lru_cache(maxsize=128)
def _query(task_id, seed):
    task = load_tasks()[task_id]
    # 卡组只用于临时查询棋盘；波表在配置卡槽前生成，不是本次真正选定的卡组。
    deck = (task.get("reference_deck") or deck_for_level(task["level"])) if (
        task.get("deck_selection") == "player") else task.get("deck", [])
    spec, deck, _ = task_spec(task, seed, deck)
    executable, resource_dir = _default_executable(), _default_resource_dir()
    _ensure_executable(executable)
    with PvZEnv(executable=executable, resource_dir=resource_dir, headless=True,
                save_dir=app_data(executable, resource_dir)) as env:
        observation, _ = env.reset(task=spec, deck=deck)
        return _summary(task_id, spec, observation, env.privileged_state())


def roster(task_id, *, seed=None):
    """Game()之前调用：返回本种子威胁列表，含type/name/slug/first_wave/sources。

    默认用任务定义中的首颗种子；应传next_task()给的seed来查询本项。
    一次临时reset和一次波表读取，不推进tick、不占位、不计尝试次数、不写存档。
    相同任务+种子在本进程中缓存；返回副本，不会被调用者修改缓存。
    """
    tasks = load_tasks()
    if task_id not in tasks:
        raise ValueError("没有任务 %r" % task_id)
    seeds = tasks[task_id].get("seeds") or [0]
    actual_seed = int(seeds[0] if seed is None else seed)
    return copy.deepcopy(_query(task_id, actual_seed))
