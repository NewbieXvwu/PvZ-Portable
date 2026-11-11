"""卡片身份：卡槽序号之外，还能按卡片名种植（task144）。

传送带关（原版传送带关卡）用掉一张卡后，引擎会把后面的卡整体前移
（`SeedBank::RemoveSeed`），槽位号因此不是稳定的身份；而 `observe()` 打印的
卡槽行里序号又必须和引擎对齐。这里的办法是：`plant()` 接受**卡片名**，
每次调用时现查当前卡槽，把名字换成那一刻的序号。

名字表从引擎源码现解析，不手抄：

* **身份名**来自 `ConstEnums.h` 的 `enum SeedType`，去掉 ``SEED_`` 前缀。
  这是唯一的一一对应来源——`gPlantDefs` 的显示名可以复用
  （SEED_REPEATER 与 SEED_LEFTPEATER 都写 "REPEATER"），枚举名不会。
* 显示名（`gPlantDefs.mPlantName`）只有在全局唯一时才作为别名保留，
  所以老的 ``doom_shroom`` 仍能种，而 ``repeater`` 只指 7、不会指 52。
* 中文名来自 `pvz_constants.PLANT_NAME`。

可接受的写法是这些别名的任意一种，例如：

    "doom_shroom"   旧显示名（小写下划线）
    "doomshroom"    枚举名
    "毁灭菇"         中文名
"""
from __future__ import annotations

from collections import Counter

from pvz_constants import PLANT_NAME, plant_defs, seed_types

_SLUGS: dict[int, str] | None = None
_ALIASES: dict[int, set[str]] | None = None


def _build() -> tuple[dict[int, str], dict[int, set[str]]]:
    """从枚举名生成唯一卡名；显示名只在全局唯一时当别名。"""
    definitions = plant_defs()
    ids = seed_types()
    display = {int(definition["type"]): str(definition["name"]).lower()
               for definition in definitions}
    display_count = Counter(display.values())
    enum_slug = {}
    for definition in definitions:
        seed_type = int(definition["type"])
        enum_name = definition["seed"]
        for name, value in ids.items():
            if value == seed_type and name.startswith("SEED_"):
                enum_name = name
                break
        enum_slug[seed_type] = enum_name.removeprefix("SEED_").lower()

    # 显示名唯一时保留它（现有 observe 输出不变）；重名的那张改用枚举名。
    slugs: dict[int, str] = {}
    for seed_type, name in display.items():
        slugs[seed_type] = name if display_count[name] == 1 else enum_slug[seed_type]
    duplicates = [name for name, count in Counter(slugs.values()).items() if count > 1]
    if duplicates:
        raise RuntimeError("卡片名仍然重名：%s —— 枚举映射不对，不能靠手写表兜底"
                           % "、".join(sorted(duplicates)))

    aliases: dict[int, set[str]] = {}
    for seed_type, name in slugs.items():
        names = {name, enum_slug[seed_type],
                 name.replace("_", ""), enum_slug[seed_type].replace("_", "")}
        if display_count[display[seed_type]] == 1:
            names.add(display[seed_type])
        chinese = PLANT_NAME.get(seed_type)
        if chinese:
            names.add(chinese)
        aliases[seed_type] = names

    # 最后一道机械检查：任何一个别名都不能指向两张卡。
    owners: dict[str, set[int]] = {}
    for seed_type, names in aliases.items():
        for name in names:
            owners.setdefault(name, set()).add(seed_type)
    conflicts = {name: sorted(types) for name, types in owners.items() if len(types) > 1}
    if conflicts:
        raise RuntimeError("卡片别名有歧义：%s" % "、".join(
            "%s->%s" % (name, types) for name, types in sorted(conflicts.items())))
    return slugs, aliases


def _tables() -> tuple[dict[int, str], dict[int, set[str]]]:
    global _SLUGS, _ALIASES
    if _SLUGS is None or _ALIASES is None:
        _SLUGS, _ALIASES = _build()
    return _SLUGS, _ALIASES


def slugs() -> dict[int, str]:
    """植物编号 -> 英文小写名（如 15 -> "doom_shroom"、52 -> "leftpeater"）。"""
    return _tables()[0]


def slug(seed_type) -> str:
    return slugs().get(int(seed_type), "type%s" % seed_type)


def aliases(seed_type) -> set[str]:
    return set(_tables()[1].get(int(seed_type), set()))


def resolve(text: str) -> int:
    """卡片名 -> 植物编号。认不出或重名就报错，并列出可用的写法。"""
    key = str(text).strip().lower()
    if not key:
        raise ValueError("卡片名不能是空字符串")
    hits = [seed for seed in slugs() if key in aliases(seed)]
    if len(hits) == 1:
        return hits[0]
    if hits:
        raise ValueError("卡片名 %r 有歧义：%s" % (text, "、".join(
            "%s(%d)" % (slug(seed), seed) for seed in sorted(hits))))
    raise ValueError("认不出卡片名 %r。可用写法例如 %s" % (
        text, "、".join(sorted(slugs().values())[:6]) + " …（完整表见 PLANTS.md）"))


def packets(obs) -> list[dict]:
    """当前卡槽，统一成 {index, type, active, elapsed, refresh_time, remaining}。

    完整观测给字典、精简观测给定长数组，两种都要吃：`observe()` 之后是前者，
    推进过一步之后是后者。``elapsed`` 是引擎的 mRefreshCounter（已过的更新数），
    ``refresh_time`` 是总冷却，``remaining`` = max(0, refresh_time - elapsed)。
    """
    out = []
    for entry in (obs or {}).get("packets") or []:
        if isinstance(entry, dict):
            index, seed_type = int(entry["index"]), int(entry["type"])
            imitater = int(entry.get("imitater_type", -1))
            active = bool(entry.get("active"))
            elapsed = int(entry.get("cooldown") or 0)
            refresh = int(entry.get("refresh_time") or 0)
        else:
            index, seed_type = int(entry[0]), int(entry[1])
            imitater = int(entry[6]) if len(entry) > 6 else -1
            active = bool(entry[2])
            elapsed = int(entry[3] or 0)
            refresh = int(entry[4] or 0) if len(entry) > 4 else 0
        out.append(dict(index=index, type=seed_type, imitater=imitater, active=active,
                        elapsed=elapsed, refresh_time=refresh,
                        remaining=max(0, refresh - elapsed)))
    return out


def slot_for(obs, text: str) -> int:
    """在**当前**卡槽里找这张卡：同一张卡有多张时取最左、已就绪的那张。

    An Imitater slot answers to the name of the plant it copies, so a deck with a
    sunflower and an Imitater-sunflower has two ``plant("sunflower")`` targets: the
    first call takes the original, the next (while it is cooling down) the copy.
    ``"imitater:sunflower"`` / ``"模仿者:向日葵"`` forces the copied slot.
    """
    raw = str(text).strip()
    only_imitater = False
    for prefix in ("imitater:", "模仿者:", "模仿者·"):
        if raw.lower().startswith(prefix.lower()):
            raw, only_imitater = raw[len(prefix):].strip(), True
            break
    seed = resolve(raw)
    def matches(entry):
        if entry["type"] == seed:
            return not only_imitater
        return entry["type"] == 48 and entry["imitater"] == seed
    slots = [entry for entry in packets(obs) if matches(entry)]
    if not slots:
        def name(entry):
            if entry["type"] < 0:
                return "空槽"
            if entry["type"] == 48 and entry["imitater"] >= 0:
                return "模仿者·" + slug(entry["imitater"])
            return slug(entry["type"])
        present = [name(entry) for entry in packets(obs)]
        raise ValueError("当前卡槽里没有 %s(%d)；现在的卡槽是 %s" % (
            slug(seed), seed, "、".join("%d:%s" % (i, name_) for i, name_ in enumerate(present))))
    ready = [entry for entry in slots if entry["active"]]
    chosen = (ready or slots)[0]
    return chosen["index"]
