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
  plant_role             producer / shooter / wall / other
  lane_front_zombie_x_max  该路最前僵尸的 x ≤ N（x 越小越靠近房子；每列 80px）
  lane_firepower_max     该路**打得到**最前僵尸的火力 dps ≤ N
  lane_shooters_max      该路射手株数 ≤ N
  plant_life_ticks_max   这株种下后活不过 N tick
  sun_max                决策时阳光 ≤ N
  sun_min                决策时阳光 ≥ N
  wave_min               第 N 波之后才适用

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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "python"))

from render_episode import (  # noqa: E402
    ECONOMY_TYPES, SHOOTER_TYPES, WALL_TYPES, _reachable_dps)

# ---------------------------------------------------------------- 词表

FIELDS = {
    "action_type": "这一步做了什么：plant / wait / shovel",
    "plant_role": "种的植物属于哪类：producer / shooter / wall / other",
    "lane_front_zombie_x_max": "该路最前僵尸的 x ≤ N（越小越靠近房子，每列 80px）",
    "lane_firepower_max": "该路打得到最前僵尸的火力 dps ≤ N",
    "lane_shooters_max": "该路射手株数 ≤ N",
    "plant_life_ticks_max": "种下的这株活不过 N tick",
    "sun_max": "决策时阳光 ≤ N",
    "sun_min": "决策时阳光 ≥ N",
    "wave_min": "第 N 波之后才算",
}

# 上限型字段：值是"最多多少"；wave_min 是下限型，语义相反。
_MAX_FIELDS = {"lane_front_zombie_x_max", "lane_firepower_max", "lane_shooters_max",
               "plant_life_ticks_max", "sun_max"}


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
        bad = [r for r in roles if r not in ("producer", "shooter", "wall", "other")]
        if bad:
            problems.append(f"plant_role 只能是 producer/shooter/wall/other，收到 {bad}")
    return problems


def _role_of(plant_type: int) -> str:
    if plant_type in ECONOMY_TYPES:
        return "producer"
    if plant_type in WALL_TYPES:
        return "wall"
    if plant_type in SHOOTER_TYPES:
        return "shooter"
    return "other"


def _front_zombie_x(frame: dict, row: int) -> float | None:
    """这一路最靠房子（x 最小）的那只僵尸在哪。没有僵尸返回 None。"""
    xs = [z[2] for z in frame.get("zombies") or [] if z[0] == row]
    return min(xs) if xs else None


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
        plant_type = None
        if kind == "plant":
            pk = act.get("packet")
            plant_type = deck[pk] if isinstance(pk, int) and 0 <= pk < len(deck) else pk
            if "plant_role" in probe:
                want = probe["plant_role"]
                want = want if isinstance(want, list) else [want]
                if _role_of(plant_type) not in want:
                    continue
        elif "plant_role" in probe:
            continue  # 条件限定了植物类别，这一步没种东西 → 不算命中

        if row is not None:
            if "lane_shooters_max" in probe and \
                    _lane_shooters(prev, row) > probe["lane_shooters_max"]:
                continue
            front_x = _front_zombie_x(prev, row)
            if "lane_front_zombie_x_max" in probe:
                if front_x is None or front_x > probe["lane_front_zombie_x_max"]:
                    continue
            if "lane_firepower_max" in probe:
                dps, _, _ = _reachable_dps(prev, row, front_x)
                if dps > probe["lane_firepower_max"]:
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

        hits.append({"decision": i, "tick": cur.get("tick"), "wave": cur.get("wave"),
                     "action": act,
                     "plant": plant_type if kind == "plant" else None})
    return hits


# ---------------------------------------------------------------- 存档读写

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
    for line in text.splitlines():
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

def main() -> int:
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

    args = ap.parse_args()

    if args.cmd == "fields":
        print("条件只能引用这些字段（每个都是工具本来就在显示的量）：")
        for key, desc in FIELDS.items():
            print(f"  {key:26s} {desc}")
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
