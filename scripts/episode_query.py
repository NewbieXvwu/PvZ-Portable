"""把一局对局完整存档，并提供按需查询的接口 —— 给 Agent 用的"可滑时间轴"。

设计立场（回应"报告永远盖不全败局"这条批评）
--------------------------------------------
上一版 `render_episode.py` 把诊断结论直接写死在规则里，只能找到**已经想到的**
失败方式（火力上限低、压力超火力、净损失多……）。它漏掉的东西包括：植物类型选错、
种植位置太靠前、时机太晚、资源花在错的地方、一次樱桃炸弹都没用。这些都没法用
枚举规则覆盖。

这个工具换一个分工：

1. **存档是无损的。** 每一帧都在盘上（`frames.jsonl`，一帧一行），
   模型可以查任意 tick、任意路、任意一只僵尸。实测 567 帧的完整状态历史
   增量编码后只有 171 KB —— 全量保存没有任何成本压力。
2. **渲染器的"诊断"降级成目录。** 它是"值得看的地方"，不是"结论"。
   每条信号都带产生它的规则名，模型可以逐条推翻。
3. **每次查询只回一屏。** 一局 567~1175 帧，一次性倒进上下文只会让模型浮光掠影；
   按需取几帧才有注意力。

用法
----
    # 1. 存档（会跑一局并落盘）
    python3 scripts/episode_query.py capture --seed 30001 --level 7 --out /tmp/ep30001

    # 2. 目录：哪里值得看
    python3 scripts/episode_query.py index --archive /tmp/ep30001

    # 3. 按需查询
    python3 scripts/episode_query.py frame   --archive /tmp/ep30001 --tick 26760
    python3 scripts/episode_query.py strip   --archive /tmp/ep30001 --every 40
    python3 scripts/episode_query.py lane    --archive /tmp/ep30001 --row 3
    python3 scripts/episode_query.py events  --archive /tmp/ep30001 --kind 割草机启动
    python3 scripts/episode_query.py between --archive /tmp/ep30001 --wave 14 --to-wave 17
    python3 scripts/episode_query.py diff    --archive /tmp/ep30001 --from 26700 --to 27000
    python3 scripts/episode_query.py trace   --archive /tmp/ep30001 --zombie 67698688
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from render_episode import (  # noqa: E402
    DEFAULT_RESOURCE_DIR, GRID_ROWS, PLANT_GLYPH, PLANT_NAME, ZOMBIE_NAME,
    _board_ascii, _disp, _lane_causal, _lane_line, _lane_rows, _lane_table,
    _lane_warnings, _mower_fired, _plant_diff, _render_row, _signals,
    _wave_matrix, collect,
)

SCHEMA = "episode_archive_v1"


# ---------------------------------------------------------------- 存档读写


def capture(resource_dir: str, seed: int, level: int, policy: str, deck,
            out_dir: Path, max_actions: int) -> dict:
    rec = collect(resource_dir, seed, level, policy, max_actions, deck)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema": SCHEMA,
        "task": rec["task"],
        "outcome": rec["outcome"],
        "totals": rec["totals"],
        "frame_count": len(rec["frames"]),
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "frames.jsonl").open("w", encoding="utf-8") as fh:
        for fr in rec["frames"]:
            fh.write(json.dumps(fr, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {"meta": meta, "log": rec["log"]}


def load(archive: Path) -> tuple[dict, list]:
    meta = json.loads((archive / "meta.json").read_text(encoding="utf-8"))
    frames = [json.loads(line) for line in
              (archive / "frames.jsonl").read_text(encoding="utf-8").splitlines() if line]
    return meta, frames


# ---------------------------------------------------------------- 帧间变化
# 一帧单独看是没法解释的——必须知道"相比上一帧发生了什么"。
# 这是让按需取帧变得可读的关键。


def _change_records(prev: dict | None, cur: dict) -> list[dict]:
    """帧间变化的结构化形式。文本渲染和解说词都从这里派生，只有一份真相。"""
    if prev is None:
        return []
    out = []
    for r, c, t, _h in _plant_diff(cur["plants"], prev["plants"]):
        out.append({"kind": "plant_added", "row": r, "col": c, "plant": t})
    for r, c, t, _h in _plant_diff(prev["plants"], cur["plants"]):
        out.append({"kind": "plant_lost", "row": r, "col": c, "plant": t})
    for r in _mower_fired(prev, cur):
        out.append({"kind": "mower", "row": r})
    if cur["wave"] != prev["wave"]:
        out.append({"kind": "wave", "row": None, "wave": cur["wave"]})
    return out


def _changes(prev: dict | None, cur: dict) -> list[str]:
    if prev is None:
        return ["（起始帧）"]
    out = []
    for e in _change_records(prev, cur):
        if e["kind"] == "plant_added":
            out.append(f"新增 {PLANT_NAME.get(e['plant'], e['plant'])} @({e['row']},{e['col']})")
        elif e["kind"] == "plant_lost":
            out.append(f"失去 {PLANT_NAME.get(e['plant'], e['plant'])} @({e['row']},{e['col']})")
        elif e["kind"] == "mower":
            out.append(f"⚠ {e['row']} 号草坪割草机被用掉")
        elif e["kind"] == "wave":
            out.append(f"→ 进入第 {e['wave']} 波")
    return out or ["（无变化）"]


def _lane_events(frames: list, row: int) -> list[tuple[int, int, str]]:
    """单条路上的所有事件，从帧间差重建（不依赖事件计数器，便于离线查询）。"""
    out = []
    for i in range(1, len(frames)):
        prev, cur = frames[i - 1], frames[i]
        for r, c, t, _h in _plant_diff(cur["plants"], prev["plants"]):
            if r == row:
                out.append((cur["tick"], cur["wave"], f"种下 {PLANT_NAME.get(t, t)} @c{c}"))
        for r, c, t, _h in _plant_diff(prev["plants"], cur["plants"]):
            if r == row:
                out.append((cur["tick"], cur["wave"], f"失去 {PLANT_NAME.get(t, t)} @c{c}"))
        for r in _mower_fired(prev, cur):
            if r == row:
                out.append((cur["tick"], cur["wave"], "割草机被用掉"))
    return out


# ---------------------------------------------------------------- 目录（不是结论）
# 每条信号都带着产生它的规则名，模型可以逐条推翻。
# 这里的定位是"值得看的地方"，不是"输在哪"。


def _index(meta: dict, frames: list) -> str:
    L = []
    o = meta["outcome"]
    L.append(f"存档：第 {meta['task']['level']} 关 / seed {meta['task']['seed']} / "
             f"{meta['task']['policy']} / {o['reason']}")
    L.append(f"推进到第 {o['final_wave']}/{o['wave_count']} 波，"
             f"{o['actions']} 次决策，{o['final_tick']} tick，共 {len(frames)} 帧存档")
    L.append("")
    L.append("── 逐波态势（每波取僵尸最多那一帧；格内 火力/僵尸数）──")
    L.append(_wave_matrix({"frames": frames}))
    L.append("")
    L.append("── 候选信号（**这是目录，不是结论**；每条都带规则名，可逐条推翻）──")
    sigs = _signals({"frames": frames})
    if sigs:
        for s in sigs:
            L.append(f"  [规则 {s['rule']}] {s['text']}")
            if s.get("hint"):
                L.append(f"      → 建议查: {s['hint']}")
    else:
        L.append("  （没有触发任何机械信号）")

    L.append("")
    L.append("── 这些规则**盖不全**的失败方式（已知盲区，需要你亲自看）──")
    L.append("  · 植物类型选错（例如该用樱桃炸弹时用了豌豆）")
    L.append("  · 种植位置问题（火力放在 c5 而不是 c2，被僵尸先啃掉）")
    L.append("  · 时机问题（同一株植物，早 200 tick 种下去结局不同）")
    L.append("  · 资源花在错的地方（买了不需要的东西）")
    L.append("  · 整条路线的战略错误（例如全场只堆一种植物）")
    L.append("  ⇒ 建议用 strip 扫全貌，再用 frame 逐帧看可疑区间。")
    L.append("")
    L.append("── 但「位置/时机」这一类**能直接测**，不用推理 ──")
    L.append("  whatif --decision <帧号> --enumerate --row <路> --save-best <DIR>")
    L.append("  环境是确定性的（同 seed + 同动作序列 → 逐位相同），所以"
             "「如果那一步换个做法」是精确可算的。")
    L.append("  实测：同一株植物种在哪一列，就足以改变整局胜负 —— 这类差别"
             "规则推不出来，但回放能精确算出来。")
    return "\n".join(L)


# ---------------------------------------------------------------- 查询


def _fmt_legend(cur: dict) -> str:
    """本帧棋盘上出现的字符分别是什么意思。

    为什么必须逐帧给（2026-10-03 实测）
    ----------------------------------
    棋盘用**大小写**区分"这株血量满不满"（见 `render_episode._board_ascii`：
    `g if hp >= mx else g.lower()`），而这个约定**没写在任何模型能看到的地方**。
    模型看到 `p` 只能猜 —— 实测它猜成"小喷菇"或"刚种下还在蓄力"，都不对：
    那个 `p` 是**受伤的豌豆射手**（156/300）。

    更麻烦的是这个约定会撞车：**11 组字母大小写各自对应两个不同物种**
    （`P`=豌豆射手 / `p`=小喷菇，`S`=向日葵 / `s`=阳光菇，`T`=土豆雷 / `t`=火炬树桩…），
    所以"p 到底是受伤的豌豆还是健康的小喷菇"**靠字符本身分辨不出来**。
    字母不能改（人也在读这个棋盘），那就把答案放在棋盘下面。
    """
    seen: dict[str, list[str]] = {}
    has_lower = False
    for _r, _c, t, hp, mx in cur.get("plants") or []:
        g = PLANT_GLYPH.get(t, "?")
        injured = hp < mx
        ch = g.lower() if injured else g
        if injured:
            has_lower = True
        name = PLANT_NAME.get(t, f"type{t}")
        note = f"{name}(受伤 {hp}/{mx})" if injured else name
        bucket = seen.setdefault(ch, [])
        if note not in bucket:
            bucket.append(note)
    if not seen:
        return ""
    parts = "  ".join(f"{ch}={'. 或 '.join(v)}" for ch, v in sorted(seen.items()))
    if has_lower:
        return ("字符说明（**小写 = 这株血量不满**，不是另一个物种）：" + parts)
    return "字符说明：" + parts


def _fmt_frame(prev: dict | None, cur: dict, idx: int, total: int) -> str:
    L = [f"── 帧 {idx}/{total}  tick {cur['tick']}  第 {cur['wave']} 波  "
         f"阳光 {cur['sun']}  阳光收入 {cur.get('sun_income_rate')} ──"]
    L.append("植物：")
    L.append("  " + _board_ascii(cur, "plant").replace("\n", "\n  "))
    L.append("僵尸：")
    L.append("  " + _board_ascii(cur, "zombie").replace("\n", "\n  "))
    L.append("  " + _lane_table(cur).replace("\n", "\n  "))
    L.append("本帧变化：" + "；".join(_changes(prev, cur)))
    legend = _fmt_legend(cur)
    if legend:
        L.append(legend)
    return "\n".join(L)


def _fmt_strip(frames: list, every: int) -> str:
    cols = [("帧", 6), ("tick", 8), ("波", 4)] + \
           [(f"{r}号路", 8) for r in range(GRID_ROWS)] + \
           [("阳光", 6), ("剩割草机", 10), ("本帧变化", 0)]

    def fmt(cells) -> str:
        return _render_row(cols, cells)

    head = fmt([n for n, *_ in cols])
    L = [head, "  " + "-" * (_disp(head) - 2)]
    for i in range(0, len(frames), every):
        f = frames[i]
        lanes = _lane_rows(f)
        cells = [i, f["tick"], f["wave"]] + \
                [f"{lanes[r]['shooters']}/{lanes[r]['zombies']}" for r in range(GRID_ROWS)] + \
                [f["sun"], f"{len(f['mowers'])}/{GRID_ROWS}",
                 "；".join(_changes(frames[i - 1] if i else None, f))]
        L.append(fmt(cells))
    if (len(frames) - 1) % every:
        L.append(fmt([len(frames) - 1, frames[-1]["tick"], frames[-1]["wave"]] +
                     [f"{_lane_rows(frames[-1])[r]['shooters']}/"
                      f"{_lane_rows(frames[-1])[r]['zombies']}" for r in range(GRID_ROWS)] +
                     [frames[-1]["sun"], f"{len(frames[-1]['mowers'])}/{GRID_ROWS}", "（终局）"]))
    L.append("  格内写法：火力/僵尸数。每隔 %d 帧取一帧。" % every)
    return "\n".join(L)


def _fmt_lane(frames: list, row: int) -> str:
    cols = [("tick", 8), ("波", 4), ("火力", 6), ("僵尸", 6), ("僵尸血", 8),
            ("最前", 6), ("距割草机", 10, "<"), ("割草机", 8), ("判定", 0)]

    def fmt(cells) -> str:
        return _render_row(cols, cells)

    L = [f"── {row} 号草坪完整历史 ──"]
    head = fmt([n for n, *_ in cols])
    L.append(head)
    L.append("  " + "-" * (_disp(head) - 2))

    # 只在"这一路有值得看的变化"时打一行，否则 567 行没人读
    last = None
    for f in frames:
        lane = _lane_rows(f)[row]
        sig = (lane["shooters"], lane["zombies"], lane["mower"], lane["verdict"])
        if sig != last:
            L.append(fmt([f["tick"], f["wave"], lane["shooters"], lane["zombies"],
                          lane["zombie_hp"],
                          "—" if lane["front_col"] is None else f"c{lane['front_col']}",
                          lane["eta_label"] or "—", lane["mower"], lane["verdict"]]))
            last = sig

    ev = _lane_events(frames, row)
    L.append("")
    L.append(f"── {row} 号草坪事件（{len(ev)} 条）──")
    for tick, wave, text in ev:
        L.append(f"  tick {tick:>7}  第 {wave:>2} 波  {text}")

    ceiling = max((_lane_rows(f)[row]["shooters"] for f in frames), default=0)
    lost = sum(1 for _t, _w, text in ev if text.startswith("失去"))
    L.append("")
    L.append(f"  汇总：火力上限 {ceiling}，植物净损失 {lost} 株，事件 {len(ev)} 条。")
    return "\n".join(L)


def _fmt_events(frames: list, kind: str | None, wave: int | None) -> str:
    rows = []
    for i in range(1, len(frames)):
        prev, cur = frames[i - 1], frames[i]
        if wave is not None and cur["wave"] != wave:
            continue
        for text in _changes(prev, cur):
            if text == "（无变化）":
                continue
            if kind and kind not in text:
                continue
            rows.append((cur["tick"], cur["wave"], i, text))
    if not rows:
        return "（没有匹配的事件）"
    L = [f"共 {len(rows)} 条（tick / 波 / 帧号 / 事件）"]
    for tick, w, i, text in rows:
        L.append(f"  {tick:>7}  第 {w:>2} 波  帧{i:<5} {text}")
    return "\n".join(L)


def _fmt_between(frames: list, w0: int, w1: int) -> str:
    sel = [(i, f) for i, f in enumerate(frames) if w0 <= f["wave"] <= w1]
    if not sel:
        return f"（第 {w0}–{w1} 波之间没有帧）"
    L = [f"── 第 {w0}–{w1} 波，共 {len(sel)} 帧（tick {sel[0][1]['tick']}–"
         f"{sel[-1][1]['tick']}）──", ""]
    L.append(_fmt_frame(sel[0][1], sel[0][1], sel[0][0], len(frames)))
    L.append("")
    L.append(_fmt_frame(sel[len(sel) // 2][1], sel[len(sel) // 2][1],
                        sel[len(sel) // 2][0], len(frames)))
    L.append("")
    L.append(_fmt_frame(sel[-2][1] if len(sel) > 1 else None, sel[-1][1],
                        sel[-1][0], len(frames)))
    return "\n".join(L)


def _fmt_diff(frames: list, t0: int, t1: int) -> str:
    def nearest(t):
        return min(range(len(frames)), key=lambda i: abs(frames[i]["tick"] - t))
    i0, i1 = nearest(t0), nearest(t1)
    L = [f"── 从 tick {frames[i0]['tick']}（帧{i0}）到 tick {frames[i1]['tick']}（帧{i1}）"
         f"之间发生的事 ──", ""]
    rows = []
    for i in range(i0 + 1, i1 + 1):
        for text in _changes(frames[i - 1], frames[i]):
            if text != "（无变化）":
                rows.append((frames[i]["tick"], frames[i]["wave"], text))
    for tick, w, text in rows:
        L.append(f"  {tick:>7}  第 {w:>2} 波  {text}")
    L.append("")
    L.append(f"  合计 {len(rows)} 条变化。")
    return "\n".join(L)


def _fmt_trace(frames: list, zombie_id: int) -> str:
    hits = []
    for i, f in enumerate(frames):
        for z in f["zombies"]:
            if z[9] == zombie_id:  # 索引 9 是僵尸 id，不是类型
                hits.append((i, f, z))
    if not hits:
        return (f"（这个存档里没有 id={zombie_id} 的僵尸。"
                f"用 frame --tick T 看某一帧场上僵尸的 id。）")
    r, t = hits[0][2][0], hits[0][2][1]
    L = [f"── 僵尸 id={zombie_id}（{ZOMBIE_NAME.get(t, t)}，{r} 号草坪）"
         f"共出现在 {len(hits)} 帧 ──"]
    L.append("  tick     波   x像素   列  血量   状态")
    prev_col = None
    for i, f, z in hits:
        col = int((z[2] - 40) // 80)
        note = []
        if prev_col is not None and col != prev_col:
            note.append(f"进入 c{col}")
        if z[6]:
            note.append("正在啃食")
        if z[7]:
            note.append(f"减速{z[7]}")
        if not z[5]:
            note.append("尚未入场")
        L.append(f"  {f['tick']:>7}  {f['wave']:>2}  {z[2]:>7.1f}  c{col:<2}  "
                 f"{z[4]:>5}  {'、'.join(note) or '行进中'}")
        prev_col = col
    L.append("")
    L.append(f"  首次出现 tick {hits[0][1]['tick']}，最后出现 tick {hits[-1][1]['tick']}。")
    return "\n".join(L)


def _fmt_narrative(frames: list, w0: int | None, w1: int | None,
                   detail: int, meta: dict | None) -> str:
    """把整局写成逐波解说词 —— 这才是"LLM 友好的完整战局"。

    三条规矩：
    1. 不逐帧转储状态。每波一块，只有入场态势、值得说的事件、离场态势。
    2. **不做算术**。"1 火力 vs 3 僵尸共 1500 血"这种写法逼读者自己算够不够；
       这里直接给结论（✓/⚠/✗）和算好的差距倍数。读者只需要推因果。
    3. 事件按格子合并。一次绞肉战里同一格补种 6 次，读作一行而不是 12 行。
    """
    by_wave: dict[int, list[int]] = {}
    for i, f in enumerate(frames):
        by_wave.setdefault(f["wave"], []).append(i)
    waves = sorted(w for w in by_wave
                   if (w0 is None or w >= w0) and (w1 is None or w <= w1))

    L = []
    if meta:
        o, t = meta["outcome"], meta["task"]
        L.append(f"═══ 第 {t['level']} 关 / seed {t['seed']} / 策略 {t['policy']} / "
                 f"{o['reason']} ═══")
        L.append(f"推进到第 {o['final_wave']}/{o['wave_count']} 波，"
                 f"{o['actions']} 次决策，{o['final_tick']} tick。"
                 f"杀死 {meta['totals']['zombies_killed']} 只僵尸，"
                 f"损失 {meta['totals']['plants_eaten']} 株植物，"
                 f"割草机消耗 {meta['totals']['mower_triggered']} 台。")
        L.append("（每波两行态势。✓=这一路火力清得掉来犯僵尸；⚠=勉强或僵局；"
                 "✗=清不完。判定已算好，不需要自己换算。）")
        L.append("")

    for w in waves:
        idxs = by_wave[w]
        lo, hi = idxs[0], idxs[-1]
        entry, exit_ = frames[lo], frames[hi]
        L.append(f"【第 {w} 波】tick {entry['tick']} → {exit_['tick']}"
                 f"（{len(idxs)} 次决策）")
        L.append("  入场  " + _lane_line(entry))
        for r, sym, why in _lane_warnings(entry):
            L.append(f"    {sym} {r}路：{why}")

        if detail >= 3:
            L.append("  棋盘：")
            L.append("    " + _board_ascii(entry, "plant").replace("\n", "\n    "))

        recs = []
        for k in range(lo + 1, hi + 1):
            for e in _change_records(frames[k - 1], frames[k]):
                recs.append((frames[k]["tick"], k, e))

        lines: list[tuple[int, str]] = []
        for t, k, e in recs:
            if e["kind"] == "mower":
                sym, why = _lane_causal(frames[k], e["row"])
                lines.append((t, f"⚠ {e['row']} 号路割草机被用掉。当时该路：{why or '暂无威胁'}"
                                 f" —— 这台机器是这一路最后的缓冲，之后每株新种的植物都会"
                                 f"直接暴露在僵尸面前"))

        if detail >= 2:
            cells: dict[tuple[int, int], list] = {}
            for t, k, e in recs:
                if e["kind"] in ("plant_added", "plant_lost"):
                    cells.setdefault((e["row"], e["col"]), []).append((t, k, e))
            for (r, c), hist in sorted(cells.items()):
                if len(hist) >= 3:
                    first, last = hist[0][2]["plant"], hist[-1][2]["plant"]
                    lines.append((hist[0][0],
                                  f"({r},{c}) 整波被反复啃食与补种 {len(hist)} 次"
                                  f"（{PLANT_NAME.get(first, first)} → … → "
                                  f"{PLANT_NAME.get(last, last)}）"
                                  f"—— 说明种下去的活不过一波"))
                else:
                    for t, _k, e in hist:
                        verb = "种下" if e["kind"] == "plant_added" else "被吃掉"
                        lines.append((t, f"{verb} {PLANT_NAME.get(e['plant'], e['plant'])}"
                                         f" @({r},{c})"))

        lines.sort(key=lambda x: x[0])
        if detail == 1:
            lines = [ln for ln in lines if ln[1].startswith("⚠")]
        for t, text in lines:
            L.append(f"    · {text}")
        if not lines:
            L.append("    ·（本波无值得记录的事件）")

        L.append("  离场  " + _lane_line(exit_))
        for r, sym, why in _lane_warnings(exit_):
            L.append(f"    {sym} {r}路：{why}")
        L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------- 决策日志
# 存档里 frames[i]["action"] 就是第 i 步实际做了什么。这是复盘自己的选择用的，
# 不是看棋盘用的 —— 想知道"我哪一步选错了"，得先能看见"我当时选了什么"。


def _action_text(a: dict | None, deck: list) -> str:
    if not a:
        return "（开局）"
    kind = a.get("type")
    if kind == "plant":
        pk = a.get("packet")
        ty = deck[pk] if isinstance(pk, int) and 0 <= pk < len(deck) else pk
        return f"种下 {PLANT_NAME.get(ty, ty)} @({a.get('row')},{a.get('col')})"
    if kind == "shovel":
        return f"铲除 @({a.get('row')},{a.get('col')})"
    if kind == "wait":
        return f"等待 {a.get('ticks')} tick"
    return json.dumps(a, ensure_ascii=False)


def _action_consequence(frames: list, i: int, act: dict | None) -> str:
    """这一步之后发生了什么。种下的植物活了多久，是最直接的后果。"""
    if not act:
        return ""
    if act.get("type") == "plant":
        r, c = act.get("row"), act.get("col")
        for j in range(i + 1, len(frames)):
            if not any(p[0] == r and p[1] == c for p in frames[j]["plants"]):
                dur = frames[j]["tick"] - frames[i]["tick"]
                return f"这株只活了 {dur:,} tick 就被吃掉"
        return "这株活到了终局"
    if act.get("type") == "wait":
        return ""
    return ""


def _fmt_actions(frames: list, deck: list, lo: int, hi: int | None,
                 max_life: int | None) -> str:
    """决策日志。--max-life 只留"种下去没活多久"的那些决策。

    这是复盘自己选择用的：先看哪些决策的后果很差，再拿 whatif 去试替代方案。
    """
    hi = len(frames) - 1 if hi is None else min(hi, len(frames) - 1)
    lo = max(lo, 1)
    cols = [("帧", 6), ("tick", 8), ("波", 4), ("决策", 22, "<"), ("当时态势", 24, "<"),
            ("后果", 0)]

    def fmt(cells) -> str:
        return _render_row(cols, cells)

    head = fmt([n for n, *_ in cols])
    L = [head, "  " + "-" * (_disp(head) - 2)]
    n_shown = n_total = 0
    for i in range(lo, hi + 1):
        f = frames[i]
        act = f.get("action")
        cons = _action_consequence(frames, i, act)
        n_total += 1
        if max_life is not None:
            if not cons.startswith("这株只活了"):
                continue
            lived = int(cons.split("活了 ")[1].split(" tick")[0].replace(",", ""))
            if lived > max_life:
                continue
        sym = "｜".join(f"{r}路{_lane_causal(f, r)[0]}" for r in range(GRID_ROWS))
        L.append(fmt([i, f["tick"], f["wave"], _action_text(act, deck),
                      sym, cons]))
        n_shown += 1
    L.append(f"  显示 {n_shown}/{n_total} 条（范围 帧 {lo}–{hi}）。"
             f"「当时态势」的 ✓/⚠/✗ 是算出来的火力判定，见 narrative 命令。")
    L.append("  想看某一步换个做法会怎样："
             "whatif --archive <DIR> --decision <帧号> --enumerate")
    return "\n".join(L)


# ---------------------------------------------------------------- 反事实
# 环境是确定性的（同 task+seed+动作序列 → 逐位相同），所以"如果那一步换个做法"
# 是**精确可算**的，不是猜测。这是这个系统里唯一能产生真因果信号的东西 ——
# 任何硬编码规则都替代不了它。


def _run_variant(meta: dict, override: dict, resource_dir: str) -> dict:
    t = meta["task"]
    rec = collect(resource_dir, t["seed"], t["level"], t["policy"],
                  4000, t.get("deck"), override=override)
    return rec["outcome"]


def _whatif(archive: Path, meta: dict, frames: list, decision: int,
            raw_actions: list, resource_dir: str, enumerate_: bool,
            row_filter: int | None, limit: int, save_best: str | None) -> str:
    deck = meta["task"].get("deck") or []
    if decision < 1 or decision >= len(frames):
        return f"（决策序号要在 1–{len(frames) - 1} 之间）"

    base_act = frames[decision].get("action")
    base = meta["outcome"]
    L = [f"基线：第 {decision} 步（tick {frames[decision]['tick']}）做的是 "
         f"{_action_text(base_act, deck)} → 最终第 {base['final_wave']}/"
         f"{base['wave_count']} 波，{base['reason']}",
         "",
         # 重放语义必须写出来。2026-10-03 实测：模型在推理里反复追问
         # "whatif 是照抄原局录下的动作序列、只换一条，还是用策略从头重跑？"
         # —— 它猜对了，但**是猜的**，而这个区别直接决定结果怎么解读：
         # 照抄的话后面几步是"同一条动作撞上不同局面"，重跑的话后面几步
         # 是"策略对新局面重新决策"。我们的实现是后者。
         "重放方式：**同一套策略从头重跑**，只把第 "
         f"{decision} 步的动作换掉；之后的决策由策略看着新局面**重新做出**，"
         "不是照抄原局录下的那串动作。所以结果反映的是「改了这一步之后，"
         "策略会怎么接着打」，不是「只改这一步、后面硬按原剧本走」。",
         ""]

    cands: list[dict] = []
    for raw in raw_actions:
        try:
            cands.append(_parse_action(raw))
        except Exception as exc:  # noqa: BLE001 - 参数写错要给可读提示，不是 traceback
            # 之前这里直接抛，模型看到的是一坨 Python 栈 —— 它据此改不出对的东西。
            # 现在把"哪里错了 + 正确写法"直接回给它。
            return "\n".join(L + [
                f"✗ --try 的写法有问题：{raw!r}",
                f"  {exc}",
                "",
                "  正确写法：plant:<packet>:<row>:<col> / wait:<ticks> / shovel:<row>:<col>",
                "  packet 可以写数字、中文名（豌豆射手）或英文别名（peashooter）。",
                "  想省事就用 --enumerate：输出左列可以直接复制当 --try。",
            ])
    if enumerate_:
        probe = collect(resource_dir, meta["task"]["seed"], meta["task"]["level"],
                        meta["task"]["policy"], 4000, deck,
                        capture_legal_at=decision)
        legal = probe.get("legal_at") or {}
        for p in (legal.get("plants") or []):
            if row_filter is not None and p["row"] != row_filter:
                continue
            cands.append({"type": "plant", "packet": p["packet"],
                          "row": p["row"], "col": p["col"]})
        cands.append({"type": "wait", "ticks": 60})
        cands = cands[:limit]

    if not cands:
        return "\n".join(L + ["（没有候选动作。用 --try 给一个，或 --enumerate 枚举）"])

    L.append(f"试 {len(cands)} 个候选（每次都要从头重放，约 1–3 秒/个）：")
    L.append("")
    L.append("  左列**可以直接复制**当作 --try 用，不用自己翻译成数字。")
    L.append("")
    results = []
    for cand in cands:
        try:
            o = _run_variant(meta, {decision: cand}, resource_dir)
        except Exception as exc:  # noqa: BLE001 - 单个候选失败不该打断整批
            L.append(f"  {_action_text(cand, deck):<34} 重放失败：{exc}")
            continue
        delta = o["final_wave"] - base["final_wave"]
        results.append((delta, cand, o))

    results.sort(key=lambda x: (-x[0], x[2]["final_tick"]))
    for delta, cand, o in results:
        mark = "★" if delta > 0 else ("·" if delta == 0 else "↓")
        sign = f"+{delta}" if delta > 0 else str(delta)
        L.append(f"  {mark} {_action_cli(cand):<15} {_action_text(cand, deck):<28}"
                 f" → 第 {o['final_wave']:>2}/{o['wave_count']} 波  "
                 f"{sign:>4} 波  {o['reason']}")
    best = max((r[0] for r in results), default=0)
    L.append("")
    if best > 0:
        L.append(f"  最好的一步能多推 {best} 波。说明第 {decision} 步附近**确实有**"
                 f"可以改善的选择。")
        if save_best and results:
            top = results[0]
            out = Path(save_best)
            t = meta["task"]
            rec = collect(resource_dir, t["seed"], t["level"], t["policy"],
                          4000, deck, override={decision: top[1]})
            out.mkdir(parents=True, exist_ok=True)
            (out / "meta.json").write_text(json.dumps(
                {"schema": SCHEMA,
                 "task": {**t, "note": f"反事实：第 {decision} 步改为 "
                                       f"{_action_text(top[1], deck)}"},
                 "outcome": rec["outcome"], "totals": rec["totals"],
                 "frame_count": len(rec["frames"])},
                ensure_ascii=False, indent=2), encoding="utf-8")
            with (out / "frames.jsonl").open("w", encoding="utf-8") as fh:
                for fr in rec["frames"]:
                    fh.write(json.dumps(fr, ensure_ascii=False,
                                        separators=(",", ":")) + "\n")
            L.append(f"  最好的那个已存档到 {out} —— 可以接着用 lane / narrative / "
                     f"frame 查它，和原局对照。")
    else:
        L.append(f"  没有任何单个替换能多推一波。说明第 {decision} 步不是瓶颈 ——"
                 f"问题更可能是**系统性的**（整局的建线顺序、植物类型组合），"
                 f"而不是某一步选错。换个决策点再试。")
    return "\n".join(L)


# 常见植物的英文别名。模型很自然会写 `plant:peashooter:3:6` —— 与其让它
# 去猜数字，不如认下来。（2026-10-03：实测模型为找 packet id 浪费了 6 次调用，
# 还把 `plant:4:...` 当成豌豆射手用，得到了一个**静默错误**的结果。）
_PLANT_ALIAS = {
    "peashooter": 0, "sunflower": 1, "cherrybomb": 2, "cherry": 2,
    "wallnut": 3, "wall-nut": 3, "potatomine": 4, "potato": 4,
    "snowpea": 5, "chomper": 6, "repeater": 7, "puffshroom": 8,
    "sunshroom": 9, "fumeshroom": 10, "gravebuster": 11, "hypnoshroom": 12,
    "scaredyshroom": 13, "iceshroom": 14, "doomshroom": 15, "lilypad": 16,
    "squash": 17, "threepeater": 18, "tanglekelp": 19, "jalapeno": 20,
    "spikeweed": 21, "torchwood": 22, "tallnut": 23, "seashroom": 24,
    "plantern": 25, "cactus": 26, "blover": 27, "splitpea": 28, "starfruit": 29,
}


def _plant_id(token: str) -> int:
    """把 packet 参数解析成整数 id。接受数字、中文名、英文别名。

    为什么值得为"名字"写这么多代码：工具**输出**用的是名字
    （`种下 豌豆射手 @(3,6)`），工具**输入**却要数字 —— 中间那道映射
    模型只能靠猜。而猜错的代价不是报错，是**静默拿到另一个植物的结果**。
    与其训练模型背 id，不如让输入和输出说同一种语言。
    """
    token = token.strip()
    if token.isdigit():
        return int(token)
    for pid, name in PLANT_NAME.items():
        if token == name:
            return pid
    key = token.lower().replace(" ", "").replace("_", "").replace("-", "")
    if key in _PLANT_ALIAS:
        return _PLANT_ALIAS[key]
    # 报错要把**可用清单**一起给出来，否则模型只能继续猜。
    names = "、".join(f"{pid}={name}" for pid, name in sorted(PLANT_NAME.items()))
    raise ValueError(
        f"认不出这个植物：{token!r}。可以写数字 id、中文名，或英文别名。\n"
        f"当前可用：{names}"
    )


def _vocabulary() -> str:
    """动作写法速查。给模型看的"一页纸"。

    为什么单独做一个命令：实测模型为了搞清楚 packet id 是什么，连着调了
    `list_mcp_resources` / `list_mcp_resource_templates` 四次（都失败），
    再靠试数字（2 → 1 → 0）反推。工具**输出**用名字、**输入**要数字，
    中间那道映射没写在任何地方 —— 这个命令就是补上它。
    """
    L = ["── 动作写法 ──", ""]
    L.append("  plant:<packet>:<row>:<col>   在 row 行 col 列种下 packet 号植物")
    L.append("  wait:<ticks>                 等 ticks 个 tick（默认 60）")
    L.append("  shovel:<row>:<col>           铲掉 row 行 col 列的植物")
    L.append("")
    L.append("  <packet> 可以写**数字 id、中文名，或英文别名**，三种都认：")
    L.append("    plant:0:3:6 / plant:豌豆射手:3:6 / plant:peashooter:3:6   ← 等价")
    L.append("  也可以直接给原始 JSON。")
    L.append("")
    L.append("── 植物 id 表 ──")
    L.append("")
    items = sorted(PLANT_NAME.items())
    for i in range(0, len(items), 3):
        row = "   ".join(f"{pid:>2} = {name}" for pid, name in items[i:i + 3])
        L.append("  " + row)
    L.append("")
    # 别名清单**从 `_PLANT_ALIAS` 现场生成**，不手抄。
    # 手抄副本会漂移：加了别名忘改这里，工具就静默少报一个能用的写法。
    # （植物 id 表在下面也是从 PLANT_NAME 生成的 —— 同一原则。）
    L.append("  英文别名（全部，与解析器同一张表）：")
    alias_items = sorted(_PLANT_ALIAS.items(), key=lambda kv: (kv[1], kv[0]))
    for i in range(0, len(alias_items), 4):
        row = "  ".join(f"{name}={pid}" for name, pid in alias_items[i:i + 4])
        L.append("    " + row)
    L.append("")
    L.append("── 行列范围 ──")
    L.append(f"  row 0..{GRID_ROWS - 1}（路，0 是最上面一条）")
    L.append("  col 0..8（列，0 靠房子、8 靠僵尸来的方向；射手朝右打）")
    L.append("")
    L.append("── 棋盘字符（frame / strip / lane 里的棋盘）──")
    L.append("  植物格：一个字母 = 一株植物。")
    L.append("    **小写 = 这株血量不满**，不是另一个物种：`P`=健康的豌豆射手，")
    L.append("    `p`=受伤的豌豆射手。字母表里大小写确实各自对应不同物种")
    L.append("    （`p` 单独看也可以是 小喷菇），所以**别猜** —— ")
    L.append("    frame 输出末尾会给出「本帧出现的字符」对照，直接看那里。")
    L.append("  僵尸格：一个数字 = **这一格有几只僵尸**，不是种类。")
    L.append("")
    L.append("  植物字母表（type=字母 名称）：")
    glyph_items = sorted(PLANT_GLYPH.items())
    for i in range(0, len(glyph_items), 4):
        row = "  ".join(f"{t}={g} {PLANT_NAME.get(t, '?')}" for t, g in glyph_items[i:i + 4])
        L.append("    " + row)
    L.append("")
    L.append("── 省事的办法 ──")
    L.append("  用 whatif --enumerate 时，输出的**左列就是可以直接复制的 --try 字符串**，")
    L.append("  不用自己翻译。")
    return "\n".join(L)


def _parse_action(raw: str) -> dict:
    """支持简写 plant:packet:row:col / wait:ticks / shovel:row:col，也支持原始 JSON。

    `packet` 可以是数字 id、中文名（豌豆射手）或英文别名（peashooter）。
    """
    raw = raw.strip()
    if raw.startswith("{"):
        return json.loads(raw)
    parts = raw.split(":")
    if parts[0] == "plant" and len(parts) == 4:
        return {"type": "plant", "packet": _plant_id(parts[1]),
                "row": int(parts[2]), "col": int(parts[3])}
    if parts[0] == "wait":
        return {"type": "wait", "ticks": int(parts[1]) if len(parts) > 1 else 60}
    if parts[0] == "shovel" and len(parts) == 3:
        return {"type": "shovel", "row": int(parts[1]), "col": int(parts[2])}
    raise ValueError(f"看不懂的动作写法：{raw}（用 plant:packet:row:col / "
                     f"wait:ticks / shovel:row:col，或原始 JSON）")


def _action_cli(cand: dict) -> str:
    """把一个候选动作渲染成**可以直接粘回去**的 `--try` 字符串。

    这是修"输出用名字、输入要数字"那个坑的关键：模型不用再做任何翻译，
    复制粘贴即可。实测模型为了反推这个映射浪费了 6 次工具调用。
    """
    t = cand.get("type")
    if t == "plant":
        return f"plant:{cand['packet']}:{cand['row']}:{cand['col']}"
    if t == "wait":
        return f"wait:{cand.get('ticks', 60)}"
    if t == "shovel":
        return f"shovel:{cand['row']}:{cand['col']}"
    return json.dumps(cand, ensure_ascii=False)


# ---------------------------------------------------------------- CLI


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture", help="跑一局并落盘")
    c.add_argument("--resource-dir", default=DEFAULT_RESOURCE_DIR)
    c.add_argument("--seed", type=int, required=True)
    c.add_argument("--level", type=int, default=7)
    c.add_argument("--policy", choices=["scripted", "donothing"], default="scripted")
    c.add_argument("--deck", default=None)
    c.add_argument("--max-actions", type=int, default=4000)
    c.add_argument("--out", required=True)

    sub.add_parser("vocabulary", help="动作写法速查（不需要 --archive）")

    for name in ("index", "strip", "lane", "events", "between", "diff", "trace", "frame",
                 "narrative", "actions", "whatif"):
        p = sub.add_parser(name)
        p.add_argument("--archive", required=True)
        if name == "strip":
            p.add_argument("--every", type=int, default=40)
        if name == "lane":
            p.add_argument("--row", type=int, required=True)
        if name == "events":
            p.add_argument("--kind", default=None)
            p.add_argument("--wave", type=int, default=None)
        if name == "between":
            p.add_argument("--wave", type=int, required=True)
            p.add_argument("--to-wave", type=int, required=True)
        if name == "diff":
            p.add_argument("--from", dest="t0", type=int, required=True)
            p.add_argument("--to", dest="t1", type=int, required=True)
        if name == "trace":
            p.add_argument("--zombie", type=int, required=True)
        if name == "frame":
            p.add_argument("--tick", type=int, required=True)
        if name == "narrative":
            p.add_argument("--wave", type=int, default=None)
            p.add_argument("--to-wave", type=int, default=None)
            p.add_argument("--detail", type=int, choices=[1, 2, 3], default=2)
        if name == "actions":
            p.add_argument("--from", dest="lo", type=int, default=1)
            p.add_argument("--to", dest="hi", type=int, default=None)
            p.add_argument("--max-life", type=int, default=None,
                           help="只显示种下去活不过这么多 tick 的决策")
        if name == "whatif":
            p.add_argument("--decision", type=int, required=True)
            p.add_argument("--try", dest="raw", action="append", default=[],
                           metavar="ACTION",
                           help="plant:packet:row:col / wait:ticks / shovel:row:col "
                                "或原始 JSON。可重复。")
            p.add_argument("--enumerate", action="store_true",
                           help="枚举该决策点上所有合法动作（用 --row/--limit 收敛）")
            p.add_argument("--row", type=int, default=None)
            p.add_argument("--limit", type=int, default=12)
            p.add_argument("--save-best", default=None, metavar="DIR",
                           help="把最好的那个反事实整局存成新存档，"
                                "之后可以用 lane / narrative / frame 复查它")
            p.add_argument("--resource-dir", default=DEFAULT_RESOURCE_DIR)

    args = ap.parse_args()

    if args.cmd == "capture":
        deck = [int(v) for v in args.deck.split(",")] if args.deck else None
        res = capture(args.resource_dir, args.seed, args.level, args.policy, deck,
                      Path(args.out), args.max_actions)
        print(f"已存档到 {args.out}：{res['meta']['frame_count']} 帧，"
              f"{res['meta']['outcome']['reason']}")
        return

    if args.cmd == "vocabulary":
        # 不需要存档，所以要在 load() 之前返回。
        print(_vocabulary())
        return

    meta, frames = load(Path(args.archive))
    if args.cmd == "index":
        print(_index(meta, frames))
    elif args.cmd == "strip":
        print(_fmt_strip(frames, args.every))
    elif args.cmd == "lane":
        print(_fmt_lane(frames, args.row))
    elif args.cmd == "events":
        print(_fmt_events(frames, args.kind, args.wave))
    elif args.cmd == "between":
        print(_fmt_between(frames, args.wave, args.to_wave))
    elif args.cmd == "diff":
        print(_fmt_diff(frames, args.t0, args.t1))
    elif args.cmd == "trace":
        print(_fmt_trace(frames, args.zombie))
    elif args.cmd == "frame":
        i = min(range(len(frames)), key=lambda k: abs(frames[k]["tick"] - args.tick))
        print(_fmt_frame(frames[i - 1] if i else None, frames[i], i, len(frames)))
    elif args.cmd == "narrative":
        print(_fmt_narrative(frames, args.wave, args.to_wave, args.detail, meta))
    elif args.cmd == "actions":
        print(_fmt_actions(frames, meta["task"].get("deck") or [],
                           args.lo, args.hi, args.max_life))
    elif args.cmd == "whatif":
        print(_whatif(Path(args.archive), meta, frames, args.decision, args.raw,
                      args.resource_dir, args.enumerate, args.row, args.limit,
                      args.save_best))


if __name__ == "__main__":
    main()
