#!/usr/bin/env python3
"""skill 的闸门 —— 写之前过一遍，写完拿它验。

为什么需要闸门
--------------
2026-10-03 的 S3 实验：模型诊断一局败局，在最终回答里给出了 4 条「可复用的判断」，
但**一条 skill 都没写** —— 复盘它的推理轨迹，`skill` 这个词出现 **0 次**。
原因不是它没内容，是**没有判据**：不知道什么样的东西才够格，于是要么写噪音，
要么干脆不写。两种都坏。

所以判据必须**可核查**。喊口号没用（模型不会照口号做，也不会照口号不写）。

判据（三问，全过才写）
----------------------
  1. **反例问**：这条判断在什么情况下**不成立**？答不出 ⇒ 它不是判断，是复述。
  2. **证据问**：它在**至少 2 个不同的决策点**上被 `whatif` 验过吗？
     只有一个点 ⇒ 可能是运气。S3 实测：把雷挪到 (3,3) 通关，但挪到 (3,4) 反而更差 ——
     一个点上的成功**不足以**支撑一条法则。
  3. **迁移问**：它的判据**换一局还能算出来**吗？判据里出现本局的坐标/决策号/seed
     ⇒ 这是一次性答案，不是 skill。

第 1、3 问由写的人（模型）自己回答，写进 `claim` / `falsifier`。
**第 2 问这里真的去查** —— 这是本脚本存在的全部理由。

第 2 问怎么"真的查"
------------------
`frames.jsonl` 的第 N 行就是第 N 个决策，**那一行记着原局当时真实做出的动作**。
所以证据里的 `before` 必须与它对上：对不上就是编的。

这条判据是硬的：存档是原局跑出来的，不是模型写的。模型无法伪造一个决策点。
（它仍然可以伪造 `result` —— 见下面「这个脚本查不了什么」。）

格式
----
`## 证据` 段里放一个 ```pvz-evidence 代码块：

    kind: mechanism
    claim: 土豆雷从种下到能炸约 1620 tick，落点必须给够这么多提前量
    falsifier: 若有雷落在僵尸 1620 tick 内到不了的位置却仍炸到有效目标，此条不成立
    point: /tmp/pvz-s3/run/base | 96 | plant:4:3:6 | plant:4:3:3 | 30/30 通关
    point: /tmp/pvz-s3/run/base | 38 | plant:4:1:6 | plant:4:1:4 | 通关

或（手法类，不可证伪，但顺序有理由）：

    kind: procedure
    rationale: 顺序不能颠倒 —— whatif 之前的任何说法都只是猜测，先落盘才能查任意 tick

`<before>` / `<after>` 用 whatif 的写法（`plant:packet:row:col` / `wait:ticks` /
`shovel:row:col`），由 `episode_query._parse_action` 解析 —— **同一份实现**，
这里不重写一遍（两份必然漂移）。

`claim` / `falsifier` / `rationale` 可以折行：**缩进的续行会接到上一个字段上**。
`point` 必须一条一行。

这个脚本查不了什么（别指望它）
------------------------------
- **查不了 `result` 的真假。** 它只核对"这个决策点存在、当时的动作是这个"，
  不重跑 `whatif` 看结果是否真是那样。要复核结论得自己跑
  `episode_query.py whatif --archive X --decision N --try <after>`。
  之所以不做：重跑一局要秒级、且输出要靠文本解析，脆；而实测出问题的是
  "决策点和动作对不上"，不是"结果数字被改小"。
- **查不了第 1、3 问。** `falsifier` 写得像不像样、判据能不能跨局算，
  这里只看"填了没有"。

用法
----
    lint_skills.py                          # 扫 $PVZ_SKILLS_DIR，默认 ./.dsh/skills
    lint_skills.py <dir> [<dir> ...]        # 扫指定目录（每个子目录是一个 skill）
    lint_skills.py --no-check-archives      # 只查结构（存档不在本机时用）
    lint_skills.py --list                   # 只列 skill 名字，不判

退出码 0 = 全过；1 = 有 skill 没过；2 = 用法错。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# 动作写法**复用** episode_query 的解析器。曾经在别处重写过一份类似的裁剪逻辑，
# 结果和策略漂移、差点删掉续跑要用的检查点（见 AGENTS.md §3）——
# 同一个坑不踩第二次：格式只允许有一份实现。
from episode_query import _parse_action  # noqa: E402

EVIDENCE_BLOCK = re.compile(r"```pvz-evidence[ \t]*\n(.*?)```", re.S)
FRONTMATTER = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*\n", re.S)

KINDS = ("mechanism", "procedure")

# 正文里出现绝对路径 → 提示（不判失败）。理由：skill 是要跨机器复用的，
# 路径写死会让它在别的机器上指向不存在的目录。证据块里的路径是**记录**，
# 本来就该是绝对的，所以排除在外。
ABS_PATH = re.compile(r"(?:^|[\s\"'(=])/(?:Users|home|tmp|var|private)/")

MIN_POINTS = 2


class BadEvidence(Exception):
    """证据块本身写坏了（格式问题，不是判据问题）。"""


# ------------------------------------------------------------------ 解析


def _frontmatter(text: str) -> dict[str, str]:
    m = FRONTMATTER.match(text)
    if not m:
        return {}
    out: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, sep, val = line.partition(":")
        if not sep:
            continue
        out[key.strip()] = val.strip().strip('"').strip("'")
    return out


def _strip_evidence(text: str) -> str:
    """去掉证据块，剩下的才是"给人读的正文"。"""
    return EVIDENCE_BLOCK.sub("", text)


def _parse_evidence(text: str) -> dict | None:
    """解析 ```pvz-evidence 块。没有这个块返回 None。"""
    m = EVIDENCE_BLOCK.search(text)
    if m is None:
        return None

    fields: dict[str, str] = {}
    points: list[str] = []
    counterpoints: list[str] = []
    last_key: str | None = None
    for raw in m.group(1).splitlines():
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        # 缩进行是**续行**（YAML 的折叠写法）。理由：`rationale` / `falsifier`
        # 这类字段本来就该是一句完整的话，硬压成一行既难写也难读，
        # 而写的人（模型）很自然会换行 —— 与其报错，不如认下来。
        if raw[:1] in (" ", "\t") and last_key is not None:
            if last_key in ("point", "counterpoint"):
                raise BadEvidence(f"`{last_key}` 不能折行 —— 一条证据写一行")
            fields[last_key] = f"{fields[last_key]} {raw.strip()}"
            continue
        line = raw.strip()
        key, sep, val = line.partition(":")
        if not sep:
            raise BadEvidence(f"这一行既不是 `key: value` 也不是 `point:`：{line!r}")
        key, val = key.strip(), val.strip()
        # `point` 和 `counterpoint` 都是**可重复**的列表字段 —— 一条一行。
        # 反例如果只能写一条，写的人就会把两条挤进一行（probe 那边读不出来）。
        if key == "point":
            points.append(val)
            last_key = "point"
            continue
        if key == "counterpoint":
            counterpoints.append(val)
            last_key = "counterpoint"
            continue
        if key in fields:
            raise BadEvidence(f"字段 `{key}` 写了两遍")
        fields[key] = val
        last_key = key
    fields["_points"] = points  # type: ignore[assignment]
    fields["_counterpoints"] = counterpoints  # type: ignore[assignment]
    return fields


def _parse_point(raw: str, where: str) -> dict[str, str]:
    parts = [p.strip() for p in raw.split("|")]
    if len(parts) != 5:
        raise BadEvidence(
            f"{where}：`point` 要有 5 段，用 `|` 分隔"
            f"（存档 | 决策号 | 改动前 | 改动后 | 结果），这一条有 {len(parts)} 段：{raw!r}"
        )
    archive, decision, before, after, result = parts
    for name, val in (("存档", archive), ("改动前", before), ("改动后", after), ("结果", result)):
        if not val:
            raise BadEvidence(f"{where}：`point` 的「{name}」是空的")
    if not decision.isdigit():
        raise BadEvidence(f"{where}：决策号必须是整数，给的是 {decision!r}")
    return {"archive": archive, "decision": int(decision),
            "before": before, "after": after, "result": result}


def _load_actions(archive: Path) -> list:
    """读存档，返回每个决策点当时**真实做出**的动作（第 0 项是初始帧，动作为 null）。"""
    frames = archive / "frames.jsonl"
    if not frames.is_file():
        raise BadEvidence(f"{archive} 下没有 frames.jsonl —— 这不是一个 capture 出来的存档")
    actions = []
    with frames.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            actions.append(json.loads(line).get("action"))
    return actions


def _norm(action) -> tuple | None:
    """把动作归一成可比较的元组。None 表示"这一帧没有动作"。"""
    if not isinstance(action, dict):
        return None
    kind = action.get("type")
    if kind == "plant":
        return ("plant", int(action["packet"]), int(action["row"]), int(action["col"]))
    if kind == "wait":
        return ("wait", int(action.get("ticks", 60)))
    if kind == "shovel":
        return ("shovel", int(action["row"]), int(action["col"]))
    return (str(kind),)


# ------------------------------------------------------------------ 判定


def lint_skill(skill_dir: Path, check_archives: bool = True) -> tuple[list[str], list[str]]:
    """检查一个 skill 目录，返回 (失败原因, 提示)。"""
    fails: list[str] = []
    notes: list[str] = []

    path = skill_dir / "SKILL.md"
    if not path.is_file():
        return [f"{skill_dir.name}/ 下没有 SKILL.md"], notes
    text = path.read_text(encoding="utf-8")

    # --- 1. frontmatter（skill-filesystem 的硬要求，缺了它 skill 根本挂不上）---
    fm = _frontmatter(text)
    if not fm:
        fails.append("没有 frontmatter（文件开头 `---` 包起来的那一段）")
    else:
        if not fm.get("name"):
            fails.append("frontmatter 缺 `name`")
        elif fm["name"] != skill_dir.name:
            fails.append(f"frontmatter 的 name={fm['name']!r} 与目录名 {skill_dir.name!r} 不一致")
        if not fm.get("description"):
            fails.append("frontmatter 缺 `description`（模型靠它决定要不要读这个 skill）")

    # --- 2. 证据块 ---
    try:
        ev = _parse_evidence(text)
    except BadEvidence as exc:
        return fails + [f"证据块写坏了：{exc}"], notes

    if ev is None:
        fails.append(
            "没有 ```pvz-evidence 块。判据见 dsh/pvz-teacher/README.md「什么时候写 skill」：\n"
            "        写 skill 必须附上「它在哪几个决策点上被 whatif 验过」，"
            "否则这条 skill 和一句猜测没有区别。"
        )
        return fails, notes

    kind = ev.get("kind", "")
    if kind not in KINDS:
        fails.append(f"`kind` 必须是 {KINDS} 之一，给的是 {kind!r}")
        return fails, notes

    if kind == "procedure":
        # 手法类不可证伪，所以不要反事实证据。但它必须有**可被反驳的理由** ——
        # "为什么是这个顺序"才是它的核心，步骤本身谁都能列。
        if not ev.get("rationale"):
            fails.append("`kind: procedure` 必须写 `rationale`（为什么是这个顺序）—— 这是它唯一可被反驳的部分")
        if ev["_points"]:  # type: ignore[index]
            notes.append("procedure 类带了 point：不需要，但留着也不算错")
        return fails, notes

    # --- 3. mechanism：三问里的第 2 问，这里真的查 ---
    for field in ("claim", "falsifier"):
        if not ev.get(field):
            fails.append(f"`kind: mechanism` 必须写 `{field}`")
    if ev.get("claim") and ev.get("claim") == ev.get("falsifier"):
        fails.append("`claim` 和 `falsifier` 一模一样 —— 说明没想「什么情况下它不成立」")

    raw_points: list[str] = ev["_points"]  # type: ignore[assignment]
    if len(raw_points) < MIN_POINTS:
        fails.append(
            f"只有 {len(raw_points)} 个决策点的证据，至少要有 {MIN_POINTS} 个。\n"
            "        一个点上的成功可能是运气（S3 实测：挪到 (3,3) 通关，挪到 (3,4) 反而更差），"
            "不足以支撑一条法则。"
        )

    parsed = []
    for i, raw in enumerate(raw_points, 1):
        try:
            parsed.append(_parse_point(raw, f"第 {i} 条 point"))
        except BadEvidence as exc:
            fails.append(str(exc))

    decisions = {p["decision"] for p in parsed}
    if parsed and len(decisions) < MIN_POINTS:
        fails.append(
            f"{len(parsed)} 条证据只落在 {len(decisions)} 个决策点上（{sorted(decisions)}）—— "
            f"同一个点重复写不算数，要 {MIN_POINTS} 个**不同**的点。"
        )

    for p in parsed:
        for field in ("before", "after"):
            try:
                _parse_action(p[field])
            except Exception as exc:  # noqa: BLE001 —— 把解析器的原话转给写的人
                fails.append(f"决策 {p['decision']} 的「{'改动前' if field == 'before' else '改动后'}」写法不对：{exc}")
        if p["decision"] == 0:
            fails.append("决策 0 是初始帧，没有动作，不能当反事实的靶子")

    # --- 4. 证据是不是真的（本脚本存在的理由）---
    if check_archives:
        for p in parsed:
            archive = Path(p["archive"])
            if not archive.is_dir():
                fails.append(f"决策 {p['decision']} 的存档不存在：{archive}")
                continue
            try:
                actions = _load_actions(archive)
            except BadEvidence as exc:
                fails.append(str(exc))
                continue
            idx = p["decision"]
            if idx >= len(actions):
                fails.append(
                    f"决策 {idx} 越界：这个存档只到决策 {len(actions) - 1}（共 {len(actions)} 帧）"
                )
                continue
            actual = actions[idx]
            if actual is None:
                fails.append(f"决策 {idx} 在存档里没有动作（初始帧），不能当反事实的靶子")
                continue
            try:
                claimed = _norm(_parse_action(p["before"]))
            except Exception:  # noqa: BLE001 —— 上面已经报过写法错，这里不重复
                continue
            if claimed != _norm(actual):
                fails.append(
                    f"决策 {idx} 的「改动前」与存档对不上：\n"
                    f"          证据里写的是 {p['before']}\n"
                    f"          存档里记的是 {json.dumps(actual, ensure_ascii=False)}\n"
                    f"        存档是原局跑出来的，不是写出来的 —— 对不上就是这条证据编的。"
                )

    # --- 4b. 反例点：给 probe 条件用的"泛化过头"探针，必须能定位到具体决策 ---
    #
    # 散文写的反例机器验不了（"有时候这样种也没事"）。所以反例和证据点一样，
    # 必须是「存档 | 决策号」—— 存档在、决策号在范围内，才有资格当反例。
    for i, raw in enumerate(ev["_counterpoints"], 1):  # type: ignore[index]
        parts = [p.strip() for p in raw.split("|")]
        if len(parts) != 2:
            fails.append(
                f"第 {i} 条 counterpoint 要有 2 段（存档 | 决策号），"
                f"这一条有 {len(parts)} 段：{raw!r}"
            )
            continue
        archive_s, decision_s = parts
        if not decision_s.lstrip("-").isdigit():
            fails.append(f"第 {i} 条 counterpoint 的决策号不是整数：{decision_s!r}")
            continue
        if not check_archives:
            continue
        archive = Path(archive_s)
        if not archive.is_dir():
            fails.append(f"第 {i} 条 counterpoint 的存档不存在：{archive}")
            continue
        try:
            actions = _load_actions(archive)
        except BadEvidence as exc:
            fails.append(f"第 {i} 条 counterpoint：{exc}")
            continue
        idx = int(decision_s)
        if not 0 <= idx < len(actions):
            fails.append(
                f"第 {i} 条 counterpoint 的决策号越界：{idx}，"
                f"这个存档只到决策 {len(actions) - 1}"
            )

    # --- 4c. probe（可选）：写了"什么算这个失败模式"的条件，就要能自动验收 ---
    #
    # 为什么并进闸门而不是单独跑：教师写完 skill 会调一次闸门。如果验收在别处，
    # 它八成不会主动跑 —— S3 的实测就是"没有当场拒，它就不会补"。
    # 条件的判据（派生局必须命中、反例必须不命中）写在 scripts/probe.py 里，
    # 这里只负责把结果接进"过 / 不过"。
    if check_archives:
        try:
            import probe
            ok, report, has_probe = probe.check_skill(skill_dir)
            if has_probe and not ok:
                fails.append("probe 条件验收不通过（不能变成自动筛子）：\n        "
                             + report.replace("\n", "\n        "))
            elif has_probe:
                notes.append("probe 条件验收通过 —— 可以变成自动筛子")
        except Exception as exc:  # noqa: BLE001 - 验收脚本炸了不该让整条 skill 判死
            notes.append(f"（probe 验收没跑起来：{exc}）")

    # --- 5. 提示（不判失败）---
    body = _strip_evidence(text)
    if ABS_PATH.search(body):
        notes.append("正文里有绝对路径 —— skill 要跨机器复用，路径写死会在别的机器上指向不存在的目录")
    if re.search(r"(?m)^##+\s*待验证", body):
        notes.append("带「待验证」段 —— 很好，未经验证的判断就该待在这里，而不是写成规则")

    return fails, notes


def find_skills(root: Path) -> list[Path]:
    """root 下的每个子目录（含 SKILL.md）算一个 skill。"""
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "SKILL.md").is_file())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="skill 闸门：结构 + 证据真伪")
    ap.add_argument("dirs", nargs="*", help="skill 目录（每个子目录一个 skill）。默认 $PVZ_SKILLS_DIR 或 ./.dsh/skills")
    ap.add_argument("--no-check-archives", action="store_true",
                    help="只查结构，不去存档里核对决策点（存档不在本机时用）")
    ap.add_argument("--list", action="store_true", help="只列 skill 名字，不判")
    args = ap.parse_args(argv)

    roots = [Path(d) for d in args.dirs] or [
        Path(os.environ.get("PVZ_SKILLS_DIR") or Path.cwd() / ".dsh" / "skills")
    ]

    skills: list[Path] = []
    for root in roots:
        if not root.is_dir():
            print(f"（跳过 {root}：不是目录）")
            continue
        skills += find_skills(root)

    if not skills:
        print(f"没找到任何 skill。找过的目录：{[str(r) for r in roots]}")
        return 0

    if args.list:
        for s in skills:
            print(s.name)
        return 0

    failed = 0
    for skill in skills:
        fails, notes = lint_skill(skill, check_archives=not args.no_check_archives)
        if fails:
            failed += 1
            print(f"✗ {skill.name}")
            for f in fails:
                print(f"    · {f}")
        else:
            print(f"✓ {skill.name}")
        for n in notes:
            print(f"    （提示）{n}")

    print(f"\n{len(skills) - failed}/{len(skills)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
