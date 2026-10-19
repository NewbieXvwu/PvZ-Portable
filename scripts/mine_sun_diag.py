"""Potato mines lost before arming, and early sunflower columns: student (executed) vs teacher (label), from DAgger archives.

    python3 scripts/mine_sun_diag.py artifacts/hard_mac/frames z17dag32b_fast_search
"""

import sys, json
from collections import Counter
from pathlib import Path
sys.path[:0] = ["scripts", "python"]
from pvz_frame_store import read_frames, frame_paths
from render_episode import _col_of

MINE, SUN = 4, 1
root, label = Path(sys.argv[1]), sys.argv[2]
paths = [p for p in frame_paths(root) if p.parent.name == label]

def gap(state, row, col):
    xs = [z[2] for z in state["zombies"] if z[0] == row and z[5]]
    return None if not xs else _col_of(min(xs)) - col

stu = Counter(); tea = Counter(); lost = Counter(); n = Counter()
sun_s, sun_t = Counter(), Counter(); agree = Counter()
gaps_lost, gaps_ok, gaps_teacher = Counter(), Counter(), Counter()
for p in paths:
    f = read_frames(p)
    mines = {}   # (row,col) -> [plant index, last countdown, gap at planting]
    for i in range(1, len(f)):
        s, a, lab = f[i - 1], f[i].get("action") or {}, f[i].get("label") or {}
        if a.get("type") == "plant" and not f[i].get("action_rejected"):
            kind = s["packets"][a["packet"]][1]
            if kind == MINE:
                mines[(a["row"], a["col"])] = [i, None, gap(s, a["row"], a["col"])]
                n["student_mines"] += 1
                same = lab.get("type") == "plant" and (lab.get("row"), lab.get("col"), lab.get("packet")) == (a["row"], a["col"], a["packet"])
                lab_kind = s["packets"][lab["packet"]][1] if lab.get("type") == "plant" else None
                agree["same cell+card" if same else ("teacher: mine elsewhere" if lab_kind == MINE else
                       f"teacher: {lab.get('type')}" + (f" card {lab_kind}" if lab_kind is not None else ""))] += 1
            if kind == SUN and s["wave"] <= 3:
                sun_s[a["col"]] += 1
        if lab.get("type") == "plant":
            lk = s["packets"][lab["packet"]][1]
            if lk == SUN and s["wave"] <= 3:
                sun_t[lab["col"]] += 1
            if lk == MINE:
                g = gap(s, lab["row"], lab["col"]); gaps_teacher[g] += 1
        here = {(q[0], q[1]): q for q in f[i]["v2"]["plants"] if q[2] == MINE}
        for cell, m in list(mines.items()):
            if cell in here:
                m[1] = here[cell][4]
            elif m[1] is not None or i > m[0]:
                armed = m[1] == 0
                (gaps_ok if armed else gaps_lost)[m[2]] += 1
                n["armed_then_gone" if armed else "lost_unarmed"] += 1
                del mines[cell]
print(label, len(paths), "games")
print(dict(n))
print("student mine plantings, what the teacher labelled there:", agree.most_common(6))
fmt = lambda c: sorted(c.items(), key=lambda kv: (kv[0] is None, kv[0] if kv[0] is not None else 0))
print("gap (zombie col - mine col) at planting, lost unarmed:", fmt(gaps_lost))
print("gap, armed:", fmt(gaps_ok))
print("gap, teacher-labelled mines:", fmt(gaps_teacher))
print("sunflower col (waves 0-3) student:", sorted(sun_s.items()), " teacher:", sorted(sun_t.items()))
