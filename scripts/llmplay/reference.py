"""植物静态参考表（task144）：阳光、冷却、以及屏幕上看得见的动作时长。

数值全部从引擎源码现解析，不手抄：

* 阳光 / 冷却 / 英文名 —— `gPlantDefs`（`pvz_constants.plant_defs()`）
* 血量 / 种下时就设好的倒计时 —— `Plant::PlantInitialize` 的 switch
  （`pvz_constants.plant_specials()`）
* 运行中才设的倒计时 —— `Plant::Update()` 的种子分派表 + 各 `UpdateXxx()` 函数体

第三项是这次新加的：分派表把 `SEED_X` 映射到 `UpdateXxx()`，函数体里的
`mStateCountdown = N` / `mDoSpecialCountdown = N` 就是那一阶段持续多久。
表里的**标签**是人工写的（源码里没有可解析的中文），**数字**全部来自解析结果。
"""
from __future__ import annotations

import re

import pvz_constants as constants

# (植物英文名, 字段, tick) -> 屏幕上看到的是什么。数字不在这里，标签在这里。
LABELS = {
    ("cherrybomb", "do_special_countdown", 100): "种下后原地闪红，倒计时归零爆炸",
    ("jalapeno", "do_special_countdown", 100): "种下后发红，倒计时归零烧掉一整行",
    ("doomshroom", "do_special_countdown", 100): "种下后膨胀，倒计时归零炸 3×3",
    ("iceshroom", "do_special_countdown", 100): "种下后发蓝，倒计时归零全场冰冻",
    ("instantcoffee", "do_special_countdown", 100): "种下后碎裂，归零时把邻居的睡眠计时设为 100",
    ("potatomine", "state_countdown", 1500): "埋在土里武装，归零时出土",
    ("chomper", "state_countdown", 70): "张嘴咬下去，归零时判定咬没咬到",
    ("chomper", "state_countdown", 4000): "咀嚼，归零后才能再咬",
    ("squash", "state_countdown", 80): "盯住目标，归零时起跳",
    ("squash", "state_countdown", 45): "跳起来之前的蓄力",
    ("squash", "state_countdown", 50): "上升，归零开始下落",
    ("squash", "state_countdown", 10): "下落，剩 5 tick 时砸下去",
    ("squash", "state_countdown", 100): "砸完的停顿，归零后消失",
    ("cobcannon", "state_countdown", 500): "架设装填，归零才能开炮",
    ("sunshroom", "state_countdown", 12000): "幼体长大（长大前产小阳光）",
    ("flowerpot", "state_countdown", 100): "刚放下时的无敌帧",
    ("lilypad", "state_countdown", 100): "刚放下时的无敌帧",
    ("marigold", "state_countdown", 6000): "最后一波后的收尾动画",
    ("tanglekelp", "state_countdown", 100): "抓住僵尸往下拖",
    ("gravebuster", "state_countdown", 400): "啃墓碑",
    ("spikeweed", "state_countdown", 100): "攻击动画",
    ("spikerock", "state_countdown", 100): "攻击动画",
    ("umbrella", "state_countdown", 5): "挡下投掷物后的反弹",
    ("imitater", "state_countdown", 200): "变身成模仿目标",
    ("plantern", "state_countdown", 50): "灯光闪烁",
    ("blover", "do_special_countdown", 50): "吹飞所有飞行僵尸",
    ("magnetshroom", "state_countdown", 1500): "吸完一件铁器后重新充能",
}

# 观测里这两个字段叫什么（llmplay 的逐行明细直接打印）。
FIELD_NAME = {"mDoSpecialCountdown": "do_special_countdown",
              "mStateCountdown": "state_countdown"}

_DISPATCH = re.compile(r"mSeedType\s*==\s*SeedType::(\w+)\)\s*(Update\w+)\(\);")
_COUNTDOWN = re.compile(r"m(DoSpecialCountdown|StateCountdown)\s*=\s*(\d+)\s*;")


def _normalize(seed: str) -> str:
    """SEED_INSTANT_COFFEE / instant_coffee / INSTANTCOFFEE 都归到 instantcoffee。"""
    return seed.removeprefix("SEED_").replace("_", "").lower()


def update_timers() -> dict[str, list[tuple[str, int]]]:
    """植物英文名 -> [(字段, tick)]，来自 Update() 分派表与各 UpdateXxx() 函数体。"""
    text = constants._read(constants.PLANT_CPP)
    start, end = constants._function_span(text, "void Plant::UpdateAbilities()")
    body = constants._strip_comments(text[start:end])
    out: dict[str, list[tuple[str, int]]] = {}
    for seed, function in _DISPATCH.findall(body):
        fstart, fend = constants._function_span(text, "void Plant::%s()" % function)
        fbody = constants._strip_comments(text[fstart:fend])
        seen = []
        for field, ticks in _COUNTDOWN.findall(fbody):
            entry = (FIELD_NAME["m" + field], int(ticks))
            if entry not in seen:
                seen.append(entry)
        if seen:
            out.setdefault(_normalize(seed), []).extend(seen)
    return out


def initial_timers() -> dict[str, list[tuple[str, int]]]:
    """种下那一刻就设好的倒计时（PlantInitialize 的 switch）。"""
    out: dict[str, list[tuple[str, int]]] = {}
    for seed, values in constants.plant_specials().items():
        entries = [(FIELD_NAME[field], int(values[field])) for field in FIELD_NAME
                   if values.get(field)]
        if entries:
            out[_normalize(seed)] = entries
    return out


def timers() -> dict[str, list[tuple[str, int]]]:
    """两种来源合并（同一植物同一字段同一 tick 只留一条）。"""
    merged: dict[str, list[tuple[str, int]]] = {}
    for source in (initial_timers(), update_timers()):
        for seed, entries in source.items():
            bucket = merged.setdefault(seed, [])
            for entry in entries:
                if entry not in bucket:
                    bucket.append(entry)
    for entries in merged.values():
        entries.sort(key=lambda item: (item[0], item[1]))
    return merged


def rows() -> list[dict]:
    """一行一个植物：编号、名字、阳光、冷却、血量、可见倒计时。"""
    rate = constants.tick_rate()
    health = {_normalize(seed): values.get("mPlantHealth")
              for seed, values in constants.plant_specials().items()}
    timers_by_seed = timers()
    out = []
    for definition in constants.plant_defs():
        seed = _normalize(definition["seed"])
        out.append(dict(type=int(definition["type"]), slug=str(definition["name"]).lower(),
                        chinese=constants.PLANT_NAME.get(int(definition["type"]), ""),
                        cost=int(definition["cost"]), recharge=int(definition["recharge"]),
                        seconds=round(int(definition["recharge"]) / rate, 2),
                        health=health.get(seed),
                        timers=[dict(field=field, ticks=ticks,
                                     label=LABELS.get((seed, field, ticks), "倒计时"))
                                for field, ticks in timers_by_seed.get(seed, [])]))
    return out


def render_markdown() -> str:
    """包里的 PLANTS.md：模型可以整份读，也可以只看自己卡组那几行。"""
    rate = constants.tick_rate()
    lines = ["# 植物静态参考表（从引擎源码解析，非手抄）", "",
             "时间单位 tick：**%d tick ≈ 1 秒**（引擎 mSyncRefreshRate）。" % rate,
             "本表只是查数用；局面里的实时数值看 observe() 的逐行明细。", "",
             "## 屏幕上看得见的动作倒计时", "",
             "| 编号 | 植物 | 观测字段 | tick | 约几秒 | 屏幕上看到的是什么 |",
             "|---|---|---|---|---|---|"]
    for row in rows():
        for timer in row["timers"]:
            lines.append("| %d | %s %s | %s | %d | %.2f | %s |" % (
                row["type"], row["slug"], row["chinese"], timer["field"], timer["ticks"],
                timer["ticks"] / rate, timer["label"]))
    lines += ["", "## 全部植物（阳光与冷却）", "",
              "冷却 = 卡片用掉后要等多久才能再种（tick / 秒）。", "",
              "| 编号 | 英文名 | 中文名 | 阳光 | 冷却 tick | 冷却秒 | 血量 |",
              "|---|---|---|---|---|---|---|"]
    for row in rows():
        lines.append("| %d | %s | %s | %d | %d | %.2f | %s |" % (
            row["type"], row["slug"], row["chinese"], row["cost"], row["recharge"],
            row["seconds"], row["health"] if row["health"] is not None else "?"))
    lines += ["", "## 怎么用", "",
              "- `plant(15, row, col)` 按槽位；`plant(\"doom_shroom\", row, col)` 按名字（推荐）。",
              "- 名字写法：英文小写（可去下划线）或中文名，例如 `jalapeno`、`jalapeno`、`火爆辣椒`。",
              "- 传送带关用掉一张卡后，后面的卡整体前移，**槽位号会变，名字不会**。", ""]
    return "\n".join(lines)


def main() -> int:
    import sys
    text = render_markdown()
    if len(sys.argv) > 1:
        from pathlib import Path
        Path(sys.argv[1]).write_text(text, encoding="utf-8")
        print("wrote %s" % sys.argv[1])
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
