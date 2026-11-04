"""Compare two task236 diagnosis runs leaf-vs-simulation on the same drills.

`shovel_only` is "铲掉不补", `replant_same` is "白铲", `human_replace` is "人类救场".
For every drill and horizon the paired deltas are plan - wait, so the leaf error is
(leaf delta) - (simulation delta).  Sign disagreement is when the two deltas point in
opposite directions; the gate only cares about that, not about magnitude.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PLANS = ("shovel_only", "replant_same", "human_replace")


def pairs(rows: list[dict], plan: str, horizon: int) -> list[dict]:
    out = []
    for row in rows:
        wait, candidate = row.get("plans", {}).get("wait"), row.get("plans", {}).get(plan)
        if not wait or not candidate or not candidate.get("applicable") or not wait.get("applicable"):
            continue
        sim_key = f"sim_{horizon}"
        if sim_key not in wait or sim_key not in candidate:
            continue
        d_sim = float(candidate[sim_key]) - float(wait[sim_key])
        d_leaf = float(candidate["leaf"]) - float(wait["leaf"])
        out.append(dict(id=row["id"], terrain=row["terrain"], urgent=bool(row.get("urgent")),
                        sim=d_sim, leaf=d_leaf, error=d_leaf - d_sim))
    return out


def block(items: list[dict]) -> dict:
    if not items:
        return dict(n=0)
    sim = np.asarray([x["sim"] for x in items])
    leaf = np.asarray([x["leaf"] for x in items])
    err = leaf - sim
    opposite = (leaf * sim) < 0
    return dict(n=len(items), mean_sim_delta=float(sim.mean()), mean_leaf_delta=float(leaf.mean()),
                bias=float(err.mean()), mae=float(np.abs(err).mean()), rmse=float(np.sqrt((err ** 2).mean())),
                sign_disagree=int(opposite.sum()),
                sign_disagree_rate=float(opposite.mean()),
                false_positive=int(((leaf > 0) & (sim < 0)).sum()),
                false_negative=int(((leaf < 0) & (sim > 0)).sum()),
                max_abs_error=float(np.abs(err).max()))


def summarize(rows: list[dict], horizons=(600, 1200)) -> dict:
    out = {}
    for plan in PLANS:
        out[plan] = {}
        for horizon in horizons:
            items = pairs(rows, plan, horizon)
            entry = block(items)
            entry["by_terrain"] = {}
            for terrain in sorted({item["terrain"] for item in items}):
                entry["by_terrain"][terrain] = block([item for item in items if item["terrain"] == terrain])
            entry["by_urgency"] = {
                name: block([item for item in items if item["urgent"] == urgent])
                for name, urgent in (("urgent", True), ("economic", False))}
            out[plan][f"h{horizon}"] = entry
    return out


def bootstrap_difference(left: list[dict], right: list[dict], repetitions: int, seed: int) -> dict:
    """Paired drill bootstrap of (left - right) MAE and sign-disagreement rate."""
    index = {item["id"]: item for item in right}
    shared = [item for item in left if item["id"] in index]
    if not shared:
        return dict(n=0)
    rng = np.random.default_rng(seed)
    mae, rate = [], []
    for _ in range(repetitions):
        draw = rng.choice(len(shared), size=len(shared), replace=True)
        a = [shared[i] for i in draw]
        b = [index[shared[i]["id"]] for i in draw]
        ea = np.abs([x["leaf"] - x["sim"] for x in a])
        eb = np.abs([x["leaf"] - x["sim"] for x in b])
        mae.append(float(ea.mean() - eb.mean()))
        ra = np.mean([(x["leaf"] * x["sim"]) < 0 for x in a])
        rb = np.mean([(x["leaf"] * x["sim"]) < 0 for x in b])
        rate.append(float(ra - rb))
    return dict(n=len(shared), mae_difference=dict(mean=float(np.mean(mae)),
                                                   low=float(np.percentile(mae, 2.5)),
                                                   high=float(np.percentile(mae, 97.5))),
                sign_rate_difference=dict(mean=float(np.mean(rate)),
                                          low=float(np.percentile(rate, 2.5)),
                                          high=float(np.percentile(rate, 97.5))))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v5", type=Path, required=True)
    parser.add_argument("--v6", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    rows5 = json.loads((args.v5 / "rows.json").read_text())
    rows6 = json.loads((args.v6 / "rows.json").read_text())
    report = dict(v5=str(args.v5), v6=str(args.v6), drills5=len(rows5), drills6=len(rows6),
                  summaries={}, bootstrap={})
    for name, rows in (("v5", rows5), ("v6", rows6)):
        report["summaries"][name] = summarize(rows)
    for plan in PLANS:
        report["bootstrap"][plan] = {}
        for horizon in (600, 1200):
            left = pairs(rows5, plan, horizon)
            right = pairs(rows6, plan, horizon)
            report["bootstrap"][plan][f"h{horizon}"] = bootstrap_difference(left, right, args.bootstrap, 239)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    for plan in PLANS:
        for horizon in ("h600", "h1200"):
            a = report["summaries"]["v5"][plan][horizon]
            b = report["summaries"]["v6"][plan][horizon]
            if not a.get("n") or not b.get("n"):
                continue
            print(f"{plan:13s} {horizon}  n={a['n']:2d}  "
                  f"MAE {a['mae']:.4f}->{b['mae']:.4f}  "
                  f"|err|max {a['max_abs_error']:.3f}->{b['max_abs_error']:.3f}  "
                  f"opposite {a['sign_disagree']}->{b['sign_disagree']} "
                  f"({a['sign_disagree_rate']:.0%}->{b['sign_disagree_rate']:.0%})")
            boot = report["bootstrap"][plan][horizon]
            if boot.get("n"):
                print(f"    paired bootstrap MAE diff {boot['mae_difference']['mean']:+.4f} "
                      f"[{boot['mae_difference']['low']:+.4f},{boot['mae_difference']['high']:+.4f}], "
                      f"sign-rate diff {boot['sign_rate_difference']['mean']:+.3f} "
                      f"[{boot['sign_rate_difference']['low']:+.3f},{boot['sign_rate_difference']['high']:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
