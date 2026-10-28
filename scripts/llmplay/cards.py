"""卡片身份：卡槽序号之外，还能按卡片名种植（task144）。

传送带关（原版传送带关卡）用掉一张卡后，引擎会把后面的卡整体前移
（`SeedBank::RemoveSeed`），槽位号因此不是稳定的身份；而 `observe()` 打印的
卡槽行里序号又必须和引擎对齐。这里的办法是：`plant()` 接受**卡片名**，
每次调用时现查当前卡槽，把名字换成那一刻的序号。

名字表从引擎源码现解析（`pvz_constants.plant_defs` 的 `mPlantName`），
不手抄：源码里改名，这里跟着改。可接受的写法是那三种的任意一种：

    "doom_shroom"   英文名（小写下划线）
    "doomshroom"    去掉下划线
    "毁灭菇"         中文名（pvz_constants.PLANT_NAME）
"""
from __future__ import annotations

from pvz_constants import PLANT_NAME, plant_defs

_SLUGS: dict[int, str] | None = None


def slugs() -> dict[int, str]:
    """植物编号 -> 英文小写名（如 15 -> "doom_shroom"）。"""
    global _SLUGS
    if _SLUGS is None:
        _SLUGS = {int(defn["type"]): str(defn["name"]).lower() for defn in plant_defs()}
    return _SLUGS


def slug(seed_type) -> str:
    return slugs().get(int(seed_type), "type%s" % seed_type)


def aliases(seed_type) -> set[str]:
    seed_type = int(seed_type)
    name = slug(seed_type)
    out = {name, name.replace("_", "")}
    chinese = PLANT_NAME.get(seed_type)
    if chinese:
        out.add(chinese)
    return out


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
    """当前卡槽，统一成 {index, type, active, cooldown}。

    完整观测给字典、精简观测给定长数组，两种都要吃：`observe()` 之后是前者，
    推进过一步之后是后者。
    """
    out = []
    for entry in (obs or {}).get("packets") or []:
        if isinstance(entry, dict):
            out.append(dict(index=int(entry["index"]), type=int(entry["type"]),
                            active=bool(entry.get("active")),
                            cooldown=int(entry.get("cooldown") or 0)))
        else:
            out.append(dict(index=int(entry[0]), type=int(entry[1]),
                            active=bool(entry[2]), cooldown=int(entry[3] or 0)))
    return out


def slot_for(obs, text: str) -> int:
    """在**当前**卡槽里找这张卡：同一张卡有多张时取最左、已就绪的那张。"""
    seed = resolve(text)
    slots = [entry for entry in packets(obs) if entry["type"] == seed]
    if not slots:
        present = [slug(entry["type"]) if entry["type"] >= 0 else "空槽"
                   for entry in packets(obs)]
        raise ValueError("当前卡槽里没有 %s(%d)；现在的卡槽是 %s" % (
            slug(seed), seed, "、".join("%d:%s" % (i, name) for i, name in enumerate(present))))
    ready = [entry for entry in slots if entry["active"] and entry["cooldown"] == 0]
    chosen = (ready or slots)[0]
    return chosen["index"]
