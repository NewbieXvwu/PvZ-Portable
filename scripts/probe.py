"""把"什么算失败模式"写成可以自动跑、并且能自动验收的条件。

为什么要有这一层
----------------
教师（LLM）看一局要几分钟，而训练一晚上打几万局 —— 不可能每局都给它看。
所以要把教师发现的失败模式变成**能自动跑在每一局上的条件**，先廉价地把可疑的
局挑出来，教师只看挑出来的那几包。

为什么不是"手写启发式"
----------------------
条件**不许自己发明新逻辑**。它只能从下面这张封闭词表里挑字段，而词表里的每个
量都是工具本来就在算、本来就显示给模型看的量（火力、最前僵尸距离、植物类别、
这株活了多久……）。所以一条条件等价于"在日志上做一次筛选"，不是一条新规则：
它不产动作、不进策略、不进奖励。

它还要过机器验收（见 `check_skill`）：
  * 在它派生出来的那一局上**必须命中**（不然这条条件描述的不是它号称的那个现象）；
  * 在它自己列的反例上**必须不命中**（不然它泛化过头，会把正常操作全标红）；
  * 反例必须给到具体存档 + 决策点（散文写的反例机器验不了，等于没写）。

验收过不了会怎样？**不入库**。那条失败模式仍然留在报告里当叙述，只是不能变成
自动筛子 —— 这正是 S6 那条"坚果墙几何反向"的处理方式。

词表
----
  action_type            plant / wait / shovel
  plant_role             producer / shooter / wall / defense / other
                         （**可重叠**：坚果墙既是 wall 也是 defense，
                           写 [defense] 会把坚果墙和土豆雷一起抓到）
  lane_front_zombie_x_max  该路最前僵尸的 x ≤ N（x 越小越靠近房子；每列 80px）
  lane_firepower_max     该路**打得到**最前僵尸的火力 dps ≤ N
  lane_shooters_max      该路射手株数 ≤ N
  lane_zombie_count_max  该路僵尸数 ≤ N（0 = 这条路上没敌人）
  other_lane_zombie_count_min  别的路的僵尸总数 ≥ N（1 = 别的路有敌人）
  other_lane_front_zombie_x_max  别的路最靠房子的僵尸 x ≤ N（别的路的敌人有多近）
  action_col_min         动作落点列 ≥ N（0 起；屋顶预置花盆在 c0–c2，c3+ 是外侧）
  action_col_max         动作落点列 ≤ N
  plant_life_ticks_max   这株种下后活不过 N tick
  sun_max                决策时阳光 ≤ N
  sun_min                决策时阳光 ≥ N
  wave_min               第 N 波之后才适用
  wave_max               第 N 波之前才适用（含第 N 波）

"该路"一律指**这一步动作落在的那一路**。`other_lane_*` 是"除了它以外"的路 ——
这两组配合起来才能说清"种错路了"这类跨路判据：动作所在路没敌人（
`lane_zombie_count_max: 0`），而别的路有敌人（`other_lane_zombie_count_min: 1`）。

动作没有行/列的时候（wait 没有落点），带行/列字段的条件**不算命中** ——
不是"跳过这一条约束"，那样会让每条带行字段的条件都把 wait 全标红。

用法
----
    # 对一局跑条件，看命中哪几步
    python3 scripts/probe.py eval --probe probe.json --archive /tmp/pvz-rl/roof4

    # 验收一条 skill 里的条件（派生局必须命中、反例必须不命中）
    python3 scripts/probe.py check --skill /tmp/pvz-s6/.dsh/skills/prevent-feeding-frenzy

    # 批量扫（Tier-1：给一批 rollout 打标签，挑出给教师看的）
    python3 scripts/probe.py scan --probe probe.json --archives /tmp/pvz-rl/*
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "python"))

from render_episode import (  # noqa: E402
    ECONOMY_TYPES, SHOOTER_TYPES, WALL_TYPES, _reachable_dps)
from pvz_constants import PLANT_NAME  # noqa: E402

# ---------------------------------------------------------------- 词表

# "定点防御"：不发射子弹，只能在本路拦住或炸掉僵尸的植物。
# 前四个就是 render_episode.WALL_TYPES（坚果墙/高坚果/南瓜头/大蒜），
# 后面是地雷与地刺类（土豆雷/窝瓜/地刺/地刺王）。
# 这是一张**植物编号表**，不是新逻辑 —— 编号本来就在 pvz://vocabulary 里，
# 列出来是为了让"什么算防御植物"可核对、不用猜。
#
# ⚠ defense **包含** wall（坚果墙既是"挡路的"也是"防御的"），所以类别之间会重叠。
# 因此一个植物可以有多个类别（见 `_roles_of`），`plant_role: [defense]` 会同时
# 抓到坚果墙和土豆雷。写成互斥的单标签会让"坚果墙算不算防御植物"变成一句谎话。
DEFENSE_TYPES = frozenset(WALL_TYPES | {4, 17, 21, 46})

FIELDS = {
    "action_type": "这一步做了什么：plant / wait / shovel",
    "plant_role": "种的植物属于哪类：producer / shooter / wall / defense / other",
    "lane_front_zombie_x_max": "该路最前僵尸的 x ≤ N（越小越靠近房子，每列 80px）",
    "lane_firepower_max": "该路打得到最前僵尸的火力 dps ≤ N",
    "lane_shooters_max": "该路射手株数 ≤ N",
    "lane_zombie_count_max": "该路僵尸数 ≤ N（0 = 这条路上没敌人）",
    "other_lane_zombie_count_min": "别的路的僵尸总数 ≥ N（1 = 别的路有敌人）",
    "other_lane_front_zombie_x_max": "别的路最靠房子的僵尸 x ≤ N（别的路的敌人有多近）",
    "action_col_min": "动作落点列 ≥ N（0 起）",
    "action_col_max": "动作落点列 ≤ N",
    "plant_life_ticks_max": "种下的这株活不过 N tick",
    "sun_max": "决策时阳光 ≤ N",
    "sun_min": "决策时阳光 ≥ N",
    "wave_min": "第 N 波之后才算",
    "wave_max": "第 N 波之前才算（含第 N 波）",
}

# 上限型字段：值是"最多多少"；*_min 型语义相反。
_MAX_FIELDS = {"lane_front_zombie_x_max", "lane_firepower_max", "lane_shooters_max",
               "lane_zombie_count_max", "other_lane_front_zombie_x_max",
               "action_col_max", "plant_life_ticks_max", "sun_max", "wave_max"}
_MIN_FIELDS = {"other_lane_zombie_count_min", "action_col_min", "sun_min", "wave_min"}

# 需要"动作落点"才能判的字段。动作没有落点（wait）时，带这些字段的条件不算命中。
_ROW_FIELDS = {"lane_front_zombie_x_max", "lane_firepower_max", "lane_shooters_max",
               "lane_zombie_count_max", "other_lane_zombie_count_min",
               "other_lane_front_zombie_x_max"}
_COL_FIELDS = {"action_col_min", "action_col_max"}

_ROLES = ("producer", "shooter", "wall", "defense", "other")


def validate_probe(probe: dict) -> list[str]:
    """检查条件本身写得对不对。返回问题列表，空列表 = 可以跑。

    词表封闭是硬要求：出现一个不认识的字段就整条拒掉 —— 否则"条件"迟早会
    长出只有写它的人看得懂的方言。
    """
    problems = []
    if not isinstance(probe, dict):
        return ["条件必须是一个键值表"]
    known = set(FIELDS)
    for key in probe:
        if key not in known:
            problems.append(f"不认识的字段 {key!r}。可用：{' / '.join(sorted(known))}")
    if not probe:
        problems.append("条件是空的")
    if "action_type" in probe and probe["action_type"] not in ("plant", "wait", "shovel"):
        problems.append("action_type 只能是 plant / wait / shovel")
    if "plant_role" in probe:
        roles = probe["plant_role"]
        roles = roles if isinstance(roles, list) else [roles]
        bad = [r for r in roles if r not in _ROLES]
        if bad:
            problems.append(f"plant_role 只能是 {'/'.join(_ROLES)}，收到 {bad}")
    # 阈值必须是整数：写成 "0" 或 0.5 会让比较静默走另一条路。
    for key in sorted((_MAX_FIELDS | _MIN_FIELDS) & set(probe)):
        value = probe[key]
        if isinstance(value, bool) or not isinstance(value, int):
            problems.append(f"{key} 的阈值必须是整数，收到 {value!r}")
    return problems


def _roles_of(plant_type: int) -> set:
    """这株植物属于哪几类。**是集合不是单值** —— 坚果墙既是 wall 也是 defense。

    类别表之间本来就会重叠（defense ⊃ wall），硬塞成单值就得二选一，
    而被丢掉的那一半会变成"条件抓不到它号称的那个现象"。
    """
    roles = set()
    if plant_type in ECONOMY_TYPES:
        roles.add("producer")
    if plant_type in SHOOTER_TYPES:
        roles.add("shooter")
    if plant_type in WALL_TYPES:
        roles.add("wall")
    if plant_type in DEFENSE_TYPES:
        roles.add("defense")
    return roles or {"other"}


def _front_zombie_x(frame: dict, row: int) -> float | None:
    """这一路最靠房子（x 最小）的那只僵尸在哪。没有僵尸返回 None。"""
    xs = [z[2] for z in frame.get("zombies") or [] if z[0] == row]
    return min(xs) if xs else None


def _lane_zombie_count(frame: dict, row: int) -> int:
    return sum(1 for z in frame.get("zombies") or [] if z[0] == row)


def _other_lane_xs(frame: dict, row: int) -> list:
    """除了这一路以外，其它路上所有僵尸的 x。"""
    return [z[2] for z in frame.get("zombies") or [] if z[0] != row]


def _lane_shooters(frame: dict, row: int) -> int:
    return sum(1 for p in frame.get("plants") or [] if p[0] == row and p[2] in SHOOTER_TYPES)


def _plant_life_ticks(frames: list, i: int) -> int | None:
    """第 i 步种下的那株活了多少 tick。活到终局返回 None。"""
    act = frames[i].get("action") or {}
    r, c = act.get("row"), act.get("col")
    if r is None or c is None:
        return None
    for j in range(i + 1, len(frames)):
        if not any(p[0] == r and p[1] == c for p in frames[j]["plants"]):
            return frames[j]["tick"] - frames[i]["tick"]
    return None


def evaluate(frames: list, deck: list, probe: dict) -> list[dict]:
    """在一局上跑条件，返回命中的决策列表。

    每一步都在**决策前**的那一帧上看局面（策略当时看到的就是它），
    在**决策后**的那一帧上看动作与后果。
    """
    hits = []
    for i in range(1, len(frames)):
        prev, cur = frames[i - 1], frames[i]
        act = cur.get("action") or {}
        kind = act.get("type")

        if "action_type" in probe and kind != probe["action_type"]:
            continue
        row = act.get("row")
        col = act.get("col")
        # 动作没有落点（wait）时，带行/列字段的条件判不了 —— 不算命中。
        # 注意不能"跳过这条约束"，那样会让每条带行字段的条件都把 wait 全标红。
        if (_ROW_FIELDS & set(probe)) and row is None:
            continue
        if (_COL_FIELDS & set(probe)) and col is None:
            continue
        if "action_col_min" in probe and col < probe["action_col_min"]:
            continue
        if "action_col_max" in probe and col > probe["action_col_max"]:
            continue

        plant_type = None
        if kind == "plant":
            pk = act.get("packet")
            plant_type = deck[pk] if isinstance(pk, int) and 0 <= pk < len(deck) else pk
            if "plant_role" in probe:
                want = probe["plant_role"]
                want = want if isinstance(want, list) else [want]
                if not (_roles_of(plant_type) & set(want)):
                    continue
        elif "plant_role" in probe:
            continue  # 条件限定了植物类别，这一步没种东西 → 不算命中

        if row is not None:
            if "lane_shooters_max" in probe and \
                    _lane_shooters(prev, row) > probe["lane_shooters_max"]:
                continue
            if "lane_zombie_count_max" in probe and \
                    _lane_zombie_count(prev, row) > probe["lane_zombie_count_max"]:
                continue
            front_x = _front_zombie_x(prev, row)
            if "lane_front_zombie_x_max" in probe:
                if front_x is None or front_x > probe["lane_front_zombie_x_max"]:
                    continue
            if "lane_firepower_max" in probe:
                dps, _, _ = _reachable_dps(prev, row, front_x)
                if dps > probe["lane_firepower_max"]:
                    continue
            others = _other_lane_xs(prev, row)
            if "other_lane_zombie_count_min" in probe and \
                    len(others) < probe["other_lane_zombie_count_min"]:
                continue
            if "other_lane_front_zombie_x_max" in probe:
                if not others or min(others) > probe["other_lane_front_zombie_x_max"]:
                    continue

        if "plant_life_ticks_max" in probe:
            life = _plant_life_ticks(frames, i) if kind == "plant" else None
            # 活到终局（None）不算"活不过"，所以不命中。
            if life is None or life > probe["plant_life_ticks_max"]:
                continue
        if "sun_max" in probe and (prev.get("sun") or 0) > probe["sun_max"]:
            continue
        if "sun_min" in probe and (prev.get("sun") or 0) < probe["sun_min"]:
            continue
        if "wave_min" in probe and (prev.get("wave") or 0) < probe["wave_min"]:
            continue
        if "wave_max" in probe and (prev.get("wave") or 0) > probe["wave_max"]:
            continue

        hits.append({"decision": i, "tick": cur.get("tick"), "wave": cur.get("wave"),
                     "action": act,
                     "plant": plant_type if kind == "plant" else None})
    return hits


# ---------------------------------------------------------------- 存档读写

# 证据块围栏。**必须与 lint_skills.EVIDENCE_BLOCK 的名字一致** ——
# 两份正则各写一份是为了不让 lint_skills 和 probe 互相 import
# （lint_skills 在函数里 import probe，反向再来一次会加载出第二个模块副本）。
_EVIDENCE_BLOCK = re.compile(r"```pvz-evidence[ \t]*\n(.*?)```", re.S)


def load_archive(path: str | Path) -> tuple[dict, list]:
    meta = json.loads((Path(path) / "meta.json").read_text(encoding="utf-8"))
    frames = [json.loads(line) for line in
              (Path(path) / "frames.jsonl").read_text(encoding="utf-8").splitlines() if line]
    return meta, frames


def _parse_probe_arg(value: str) -> dict:
    """--probe 可以是一个 json 文件，也可以直接是 json 字符串。"""
    p = Path(value)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return json.loads(value)


def _load_skill_probe(skill_dir: str | Path) -> tuple[dict | None, list, list]:
    """从 skill 目录读条件 + 证据点 + 反例点。

    skill 是 markdown：条件写在 YAML frontmatter 的 `probe:` 下（缩进块），
    点和反例写在 ```pvz-evidence 代码块里。
    """
    text = (Path(skill_dir) / "SKILL.md").read_text(encoding="utf-8")
    probe = _frontmatter_probe(text)
    points, counterpoints = [], []
    # 只在 ```pvz-evidence 块里找点和反例 —— 正文里写一句"counterpoint: ..."
    # 不该被当成证据。围栏名必须与 lint_skills.EVIDENCE_BLOCK 一致。
    block = _EVIDENCE_BLOCK.search(text)
    if block is None:
        return probe, points, counterpoints
    for line in block.group(1).splitlines():
        line = line.strip()
        if line.startswith("point:"):
            points.append(line[len("point:"):].strip())
        elif line.startswith("counterpoint:"):
            counterpoints.append(line[len("counterpoint:"):].strip())
    return probe, points, counterpoints


def _frontmatter_probe(text: str) -> dict | None:
    """解析 frontmatter 里的 probe 块。

    只认 `probe:` 后面缩进的 `key: value` 行 —— 不引入 YAML 依赖，
    也避免整份 frontmatter 被当成配置解析。
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return None
    block = lines[1:end]
    out = {}
    inside = False
    for raw in block:
        line = raw.rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        if stripped == "probe:":
            inside = True
            continue
        if inside:
            if not line.startswith((" ", "\t")):
                inside = False
                continue
            if ":" not in stripped:
                continue
            key, _, value = stripped.partition(":")
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                items = [v.strip().strip("'\"") for v in value[1:-1].split(",") if v.strip()]
                out[key.strip()] = items
            else:
                try:
                    out[key.strip()] = int(value)
                except ValueError:
                    out[key.strip()] = value.strip("'\"")
    return out or None


def _point_archive_and_decision(point: str) -> tuple[str, int] | None:
    """`point: <archive> | <decision> | ...` → (存档, 决策号)。"""
    parts = [p.strip() for p in point.split("|")]
    if len(parts) < 2:
        return None
    try:
        return parts[0], int(parts[1])
    except ValueError:
        return None


def check_skill(skill_dir: str) -> tuple[bool, str, bool]:
    """验收一条 skill 的条件。返回 (是否通过, 报告, 有没有写条件)。

    验收规则（这是这一层唯一的权威性来源）：
      1. 条件必须写得合法（封闭词表内）；
      2. 在它派生出来的那些证据点上**至少命中一个** —— 一条在自家证据上
         都抓不到的条件，说明它描述的不是它号称的那个现象；
      3. 在它自己列的每个反例点上**都不许命中** —— 泛化过头的条件会把
         正常操作全标红，扫出来的"可疑局"全是噪声；
      4. 反例点必须存在且能定位 —— 散文写的反例机器验不了。
    """
    probe, points, counterpoints = _load_skill_probe(skill_dir)
    if probe is None:
        # 没写条件**不算错**：很多失败模式说不清成可计算的判据（S6 那条坚果墙
        # 几何就是），它们留在报告里当叙述。只是它们不能变成自动筛子。
        return True, "（这条 skill 没有写 probe 条件 —— 只作叙述，不生成自动筛子）", False
    problems = validate_probe(probe)
    if problems:
        return False, "条件写得不合法：\n  - " + "\n  - ".join(problems), True

    report = [f"条件：{json.dumps(probe, ensure_ascii=False)}", ""]
    ok = True

    hit_points = 0
    for point in points:
        parsed = _point_archive_and_decision(point)
        if not parsed:
            report.append(f"  ？ 证据点格式不对，跳过：{point}")
            continue
        archive, decision = parsed
        if not Path(archive).exists():
            report.append(f"  ！ 证据点的存档不在本机：{archive}（无法验收）")
            ok = False
            continue
        meta, frames = load_archive(archive)
        hits = evaluate(frames, meta["task"].get("deck") or [], probe)
        if any(h["decision"] == decision for h in hits):
            hit_points += 1
            report.append(f"  ✓ 证据点命中：{archive} 第 {decision} 步")
        else:
            report.append(f"  ✗ 证据点**没命中**：{archive} 第 {decision} 步 "
                          f"—— 这条条件抓不到它号称的那个现象")
            ok = False

    if points and hit_points == 0:
        report.append("")
        report.append("  结论：一个证据点都没命中 → 不入库。")

    if not counterpoints:
        report.append("")
        report.append("  ✗ 没有反例点。散文写的反例机器验不了 —— "
                      "给不出具体存档+决策号，这条条件就不能变成自动筛子。")
        ok = False
    for point in counterpoints:
        parsed = _point_archive_and_decision(point)
        if not parsed:
            report.append(f"  ✗ 反例点格式不对：{point}")
            ok = False
            continue
        archive, decision = parsed
        if not Path(archive).exists():
            report.append(f"  ！ 反例点的存档不在本机：{archive}（无法验收）")
            ok = False
            continue
        meta, frames = load_archive(archive)
        hits = evaluate(frames, meta["task"].get("deck") or [], probe)
        if any(h["decision"] == decision for h in hits):
            report.append(f"  ✗ 反例被误判为命中：{archive} 第 {decision} 步 "
                          f"—— 条件泛化过头，会把正常操作也标红")
            ok = False
        else:
            report.append(f"  ✓ 反例正确地没命中：{archive} 第 {decision} 步")

    report.insert(1, f"证据点 {hit_points}/{len([p for p in points if _point_archive_and_decision(p)])} "
                     f"命中，反例 {len(counterpoints)} 条")
    return ok, "\n".join(report), True


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("eval", help="在一局上跑条件，列出命中的决策")
    e.add_argument("--probe", required=True)
    e.add_argument("--archive", required=True)

    c = sub.add_parser("check", help="验收 skill 里的条件（派生局必须命中、反例必须不命中）")
    c.add_argument("--skill", required=True)

    s = sub.add_parser("scan", help="在多个存档上跑条件（挑出给教师看的局）")
    s.add_argument("--probe", required=True)
    s.add_argument("--archives", nargs="+", required=True)

    sub.add_parser("fields", help="列出可用字段（封闭词表）")

    args = ap.parse_args(argv)

    if args.cmd == "fields":
        print("条件只能引用这些字段（每个都是工具本来就在显示的量）：")
        for key, desc in FIELDS.items():
            print(f"  {key:30s} {desc}")
        print("")
        print("plant_role 的类别表（植物编号来自 pvz://vocabulary，不是猜的）：")
        for role, types in (("producer", ECONOMY_TYPES), ("shooter", SHOOTER_TYPES),
                            ("wall", WALL_TYPES), ("defense", DEFENSE_TYPES)):
            names = "、".join(f"{t}={PLANT_NAME.get(t, '?')}" for t in sorted(types))
            print(f"  {role:9s} {names}")
        print("  other     上面四类之外的植物")
        print("")
        print("类别会重叠：defense ⊃ wall（坚果墙两类都算），"
              "所以 plant_role: [defense] 会把坚果墙一起抓上；"
              "写 [wall, defense] 是多余的，写 [wall] 只抓挡路的那四种。")
        return 0

    if args.cmd == "eval":
        probe = _parse_probe_arg(args.probe)
        problems = validate_probe(probe)
        if problems:
            print("条件写得不合法：")
            for p in problems:
                print("  -", p)
            return 2
        meta, frames = load_archive(args.archive)
        hits = evaluate(frames, meta["task"].get("deck") or [], probe)
        print(f"{args.archive}：共 {len(frames) - 1} 个决策，命中 {len(hits)} 个")
        for h in hits:
            print(f"  第 {h['decision']:>4} 步  tick {h['tick']:>7,}  第 {h['wave']} 波  "
                  f"{json.dumps(h['action'], ensure_ascii=False)}")
        return 0

    if args.cmd == "check":
        ok, report, has_probe = check_skill(args.skill)
        print(report)
        print("")
        if not has_probe:
            print("没有条件可验收 —— 这条 skill 只作叙述。")
        elif ok:
            print("验收通过 —— 可以变成自动筛子。")
        else:
            print("验收**不通过** —— 不入库，仅作叙述。")
        return 0 if ok else 1

    if args.cmd == "scan":
        probe = _parse_probe_arg(args.probe)
        problems = validate_probe(probe)
        if problems:
            print("条件写得不合法：")
            for p in problems:
                print("  -", p)
            return 2
        total = 0
        for archive in args.archives:
            try:
                meta, frames = load_archive(archive)
            except Exception as exc:  # noqa: BLE001
                print(f"{archive}: 读不了（{exc}）")
                continue
            hits = evaluate(frames, meta["task"].get("deck") or [], probe)
            outcome = meta.get("outcome", {})
            total += len(hits)
            print(f"{archive}: 命中 {len(hits):>3} / {len(frames) - 1:>4} 决策  "
                  f"[{outcome.get('reason', '?')} 第 {outcome.get('final_wave', '?')} 波]")
        print(f"\n合计命中 {total} 个决策点。命中多的那几局就是该给教师看的证据包。")
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
