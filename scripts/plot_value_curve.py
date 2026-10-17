"""Draw the value network's live win probability through whole games (DESIGN.md §13, stage 1).

Writes one self-contained HTML page: small multiples, one panel per game, each a single line
P(win) over game time, with wave starts marked, a 0.5 reference line, a hover crosshair and a
table view. Pass archive directories (``.../seed_X/<checkpoint>``) or let ``--pick`` choose
contrasting games (an early loss, a late loss, a close win, a comfortable win) from a data dir.

Usage:
  python scripts/plot_value_curve.py --value artifacts/value_net/v2/model.pt \
      --pick /tmp/vnet/data --checkpoint v3_1m --out learn/01-value-curves.html
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import torch  # noqa: E402

from pvz_value_net import ValueNet, featurize  # noqa: E402

W, H = 560, 220
PAD_L, PAD_R, PAD_T, PAD_B = 44, 16, 14, 30


def load(archive: Path) -> tuple[dict, list[dict]]:
    from pvz_frame_store import frame_path, read_frames
    meta = json.loads((archive / "meta.json").read_text())
    frames = read_frames(frame_path(archive))
    return meta, frames


def curve(vnet: ValueNet, frames: list[dict]) -> list[dict]:
    live = [f for f in frames if not f.get("terminal")]
    boards, scalars = zip(*(featurize(f) for f in live))
    with torch.inference_mode():
        p = torch.sigmoid(vnet(torch.stack(boards), torch.stack(scalars))).tolist()
    return [{"tick": f["tick"], "p": round(v, 4), "wave": f["wave"], "sun": f["sun"],
             "plants": len(f["plants"]), "zombies": len(f["zombies"])} for f, v in zip(live, p)]


def pick(vnet: ValueNet, data: Path, checkpoint: str) -> list[Path]:
    games = []
    for meta_path in sorted(data.glob(f"*/{checkpoint}/meta.json")):
        meta = json.loads(meta_path.read_text())
        games.append((meta_path.parent, meta["outcome"]))
    losses = sorted((g for g in games if not g[1]["won"]), key=lambda g: g[1]["final_wave"])
    wins = [g for g in games if g[1]["won"]]
    chosen = [losses[0][0], losses[-1][0]] if losses else []
    if wins:
        lows = [(min(pt["p"] for pt in curve(vnet, load(d)[1])), d) for d, _o in wins[:40]]
        lows.sort()
        chosen += [lows[0][1], lows[-1][1]]
    return chosen


def panel(index: int, meta: dict, points: list[dict]) -> str:
    out = meta["outcome"]
    last_tick = max(points[-1]["tick"], 1)
    x = lambda t: PAD_L + (W - PAD_L - PAD_R) * t / last_tick
    y = lambda p: PAD_T + (H - PAD_T - PAD_B) * (1 - p)
    path = "M" + " L".join(f"{x(pt['tick']):.1f},{y(pt['p']):.1f}" for pt in points)
    waves, seen = [], set()
    for pt in points:
        if pt["wave"] >= 1 and pt["wave"] not in seen:
            seen.add(pt["wave"]); waves.append((pt["wave"], pt["tick"]))
    wave_marks = "".join(
        f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{H - PAD_B}" y2="{H - PAD_B + (6 if w % 5 == 0 else 3)}" class="tick"/>'
        + (f'<text x="{x(t):.1f}" y="{H - 8}" class="axis" text-anchor="middle">{w}</text>' if w % 5 == 0 else "")
        for w, t in waves)
    grid = "".join(
        f'<line x1="{PAD_L}" x2="{W - PAD_R}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="{"ref" if v == 0.5 else "grid"}"/>'
        f'<text x="{PAD_L - 6}" y="{y(v) + 4:.1f}" class="axis" text-anchor="end">{int(v * 100)}%</text>'
        for v in (0.0, 0.25, 0.5, 0.75, 1.0))
    verdict = "赢" if out["won"] else f"输（死于第 {out['final_wave']} 波）"
    title = f"种子 {meta['task']['seed']} · {verdict}"
    rows = "".join(f"<tr><td>{w}</td><td>{t}</td><td>{next(pt['p'] for pt in points if pt['tick'] >= t):.0%}</td></tr>"
                   for w, t in waves)
    return f"""
<figure class="panel" data-points='{html.escape(json.dumps(points))}' data-last="{last_tick}">
  <figcaption>{html.escape(title)}</figcaption>
  <svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}：胜率随时间变化">
    {grid}
    <line x1="{PAD_L}" x2="{W - PAD_R}" y1="{H - PAD_B}" y2="{H - PAD_B}" class="baseline"/>
    {wave_marks}
    <text x="{PAD_L - 6}" y="{H - 8}" class="axis" text-anchor="end">波次</text>
    <path d="{path}" class="series"/>
    <line class="cross" y1="{PAD_T}" y2="{H - PAD_B}" x1="-10" x2="-10"/>
    <circle class="dot" r="4" cx="-10" cy="-10"/>
    <rect class="hit" x="{PAD_L}" y="{PAD_T}" width="{W - PAD_L - PAD_R}" height="{H - PAD_T - PAD_B}"/>
  </svg>
  <div class="tip" hidden></div>
  <details><summary>表格视图（每波开始时的胜率）</summary>
    <table><thead><tr><th>波次</th><th>tick</th><th>P(赢)</th></tr></thead><tbody>{rows}</tbody></table>
  </details>
</figure>"""


PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>价值网络：对局中的实时胜率</title>
<style>
.viz-root {{ color-scheme: light; --surface-1:#fcfcfb; --text-primary:#0b0b0b; --text-secondary:#52514e;
  --muted:#8a8984; --grid:#e4e3df; --series-1:#2a78d6; }}
@media (prefers-color-scheme: dark) {{ :root:where(:not([data-theme="light"])) .viz-root {{ color-scheme: dark;
  --surface-1:#1a1a19; --text-primary:#ffffff; --text-secondary:#c3c2b7; --muted:#8f8e86; --grid:#33332f; --series-1:#3987e5; }} }}
:root[data-theme="dark"] .viz-root {{ color-scheme: dark; --surface-1:#1a1a19; --text-primary:#ffffff;
  --text-secondary:#c3c2b7; --muted:#8f8e86; --grid:#33332f; --series-1:#3987e5; }}
body.viz-root {{ background:var(--surface-1); color:var(--text-primary); font:14px/1.5 system-ui,sans-serif; margin:24px; }}
h1 {{ font-size:18px; margin:0 0 4px; }} p.lede {{ color:var(--text-secondary); margin:0 0 16px; max-width:70ch; }}
.grid2 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(520px,1fr)); gap:20px; }}
.panel {{ margin:0; position:relative; }} figcaption {{ font-weight:600; margin-bottom:4px; }}
svg {{ width:100%; height:auto; display:block; }}
.series {{ fill:none; stroke:var(--series-1); stroke-width:2; stroke-linejoin:round; }}
.grid {{ stroke:var(--grid); stroke-width:1; }} .ref {{ stroke:var(--muted); stroke-width:1; stroke-dasharray:4 4; }}
.baseline {{ stroke:var(--muted); stroke-width:1; }} .tick {{ stroke:var(--muted); stroke-width:1; }}
.axis {{ fill:var(--text-secondary); font-size:11px; }}
.cross {{ stroke:var(--text-secondary); stroke-width:1; }} .dot {{ fill:var(--series-1); stroke:var(--surface-1); stroke-width:2; }}
.hit {{ fill:transparent; cursor:crosshair; }}
.tip {{ position:absolute; pointer-events:none; background:var(--surface-1); color:var(--text-primary);
  border:1px solid var(--grid); border-radius:6px; padding:6px 8px; font-size:12px; box-shadow:0 2px 8px rgba(0,0,0,.12); }}
details {{ color:var(--text-secondary); font-size:12px; }} table {{ border-collapse:collapse; }}
td,th {{ padding:1px 10px 1px 0; text-align:right; }}
</style></head>
<body class="viz-root">
<h1>价值网络：对局中的实时胜率</h1>
<p class="lede">{lede}</p>
<div class="grid2">{panels}</div>
<script>
for (const fig of document.querySelectorAll('.panel')) {{
  const pts = JSON.parse(fig.dataset.points), last = +fig.dataset.last;
  const svg = fig.querySelector('svg'), hit = fig.querySelector('.hit'), tip = fig.querySelector('.tip');
  const cross = fig.querySelector('.cross'), dot = fig.querySelector('.dot');
  const [L, R, T, B, W, H] = [{pl}, {pr}, {pt}, {pb}, {w}, {h}];
  hit.addEventListener('mousemove', e => {{
    const box = svg.getBoundingClientRect(), sx = (e.clientX - box.left) * W / box.width;
    const tick = (sx - L) / (W - L - R) * last;
    let best = pts[0]; for (const p of pts) if (Math.abs(p.tick - tick) < Math.abs(best.tick - tick)) best = p;
    const x = L + (W - L - R) * best.tick / last, y = T + (H - T - B) * (1 - best.p);
    cross.setAttribute('x1', x); cross.setAttribute('x2', x); dot.setAttribute('cx', x); dot.setAttribute('cy', y);
    tip.hidden = false;
    tip.innerHTML = `第 ${{best.wave}} 波 · tick ${{best.tick}}<br><b>P(赢) ${{(best.p * 100).toFixed(0)}}%</b><br>阳光 ${{best.sun}} · 植物 ${{best.plants}} · 僵尸 ${{best.zombies}}`;
    const fx = x / W * box.width, fy = y / H * box.height;
    tip.style.left = Math.min(fx + 12, box.width - 170) + 'px'; tip.style.top = (fy + 24) + 'px';
  }});
  hit.addEventListener('mouseleave', () => {{ tip.hidden = true; cross.setAttribute('x1', -10); cross.setAttribute('x2', -10); dot.setAttribute('cx', -10); }});
}}
</script></body></html>"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--value", required=True)
    p.add_argument("archives", nargs="*", type=Path)
    p.add_argument("--pick", type=Path, help="choose contrasting games from this data dir")
    p.add_argument("--checkpoint", default="v3_1m")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    vnet = ValueNet()
    vnet.load_state_dict(torch.load(args.value, map_location="cpu", weights_only=False)["state_dict"])
    vnet.eval()
    archives = list(args.archives) or pick(vnet, args.pick, args.checkpoint)
    panels = []
    for archive in archives:
        meta, frames = load(archive)
        panels.append(panel(len(panels), meta, curve(vnet, frames)))
    lede = (f"价值网络 {html.escape(str(args.value))} 在 {len(archives)} 局对局（检查点 "
            f"{html.escape(args.checkpoint)}，评估种子，训练时从未见过）里，对每一个决策时刻给出的“从这里打下去赢的概率”。"
            "横轴是游戏时间，底部刻度标出每一波开始的时刻（数字是第 5、10、15… 波）。虚线是 50%。鼠标悬停可看具体数值。")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(PAGE.format(lede=lede, panels="".join(panels), pl=PAD_L, pr=PAD_R, pt=PAD_T,
                                    pb=PAD_B, w=W, h=H))
    print(f"wrote {args.out} ({len(archives)} panels: {[str(a) for a in archives]})")


if __name__ == "__main__":
    main()
