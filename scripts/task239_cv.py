"""task239 pilot: seed-grouped 5-fold comparison of v5 and v6 leaves.

One split cannot separate a real feature gain from which six games landed in the holdout, and
frames inside a game are heavily correlated.  This runs the same small conv net on five folds
(each held-out seed is predicted exactly once), then bootstraps whole games for the paired
v6-v5 difference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts")]
from task239_train import TERRAINS, load_rows, metrics, run_arm  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--warm-start", action="store_true", help="initialise v6 from the fold's v5")
    args = parser.parse_args()
    data = load_rows(args.data)
    keys = sorted({(terrain, int(seed)) for terrain, rows in data.items() for seed in rows["seed"]})
    # Stable row metadata in the same order task239_train concatenates the terrains.
    terrain_of = np.concatenate([[terrain] * len(data[terrain]["y"]) for terrain in TERRAINS if terrain in data])
    game_of = np.concatenate([data[terrain]["game"] for terrain in TERRAINS if terrain in data])
    label = np.concatenate([data[terrain]["y"] for terrain in TERRAINS if terrain in data]).astype(np.float32)
    torch_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"rows {len(label)}, seeds {len(keys)}, device {torch_device}", flush=True)

    predictions = {name: np.full(len(label), np.nan, dtype=np.float32) for name in ("v5", "v6")}
    folds = []
    for fold in range(args.folds):
        holdout = {key for index, key in enumerate(keys) if index % args.folds == fold}
        train = set(keys) - holdout
        fold_report = dict(fold=fold, holdout_seeds=len(holdout), train_seeds=len(train))
        v5_state = None
        for name, encoder in (("v5", 2), ("v6", 6)):
            print(f"== fold {fold} {name}", flush=True)
            report, checkpoint, prob, mask = run_arm(
                data, encoder, args.width, args.seed + fold, args.epochs, torch_device,
                train, holdout, lr=2e-4, dropout=args.dropout, weight_decay=args.weight_decay,
                init_state=v5_state if encoder == 6 and args.warm_start else None)
            if encoder == 2:
                v5_state = checkpoint["state_dict"]
            predictions[name][mask] = prob[mask]
            fold_report[name] = dict(best_epoch=report["best_epoch"], best_validation=report["best_validation"],
                                     holdout=report["all_holdout"])
        folds.append(fold_report)

    reports = {}
    for name, prob in predictions.items():
        if np.isnan(prob).any():
            raise RuntimeError(f"{name}: {int(np.isnan(prob).sum())} rows were never held out")
        reports[name] = dict(all=metrics(prob, label), by_terrain={
            terrain: metrics(prob[terrain_of == terrain], label[terrain_of == terrain]) for terrain in TERRAINS
            if (terrain_of == terrain).any()})

    # Paired game-level bootstrap of the v6-v5 difference.
    game_keys = np.asarray([f"{terrain}:{game}" for terrain, game in zip(terrain_of, game_of)])
    unique = sorted(set(game_keys))
    index_of = {key: np.flatnonzero(game_keys == key) for key in unique}
    rng = np.random.default_rng(args.seed)
    deltas = {"log_loss": [], "brier": [], "auc": []}
    for _ in range(args.bootstrap):
        draw = rng.choice(len(unique), size=len(unique), replace=True)
        rows = np.concatenate([index_of[unique[i]] for i in draw])
        v5p, v6p = predictions["v5"][rows], predictions["v6"][rows]
        y = label[rows]
        m5, m6 = metrics(v5p, y), metrics(v6p, y)
        deltas["log_loss"].append(m5["log_loss"] - m6["log_loss"])
        deltas["brier"].append(m5["brier"] - m6["brier"])
        deltas["auc"].append(m6["auc"] - m5["auc"])
    summary = {name: dict(mean=float(np.mean(values)), low=float(np.percentile(values, 2.5)),
                          high=float(np.percentile(values, 97.5)))
               for name, values in deltas.items()}
    report = dict(folds=args.folds, width=args.width, epochs=args.epochs, dropout=args.dropout,
                  weight_decay=args.weight_decay, seed=args.seed, rows=len(label), games=len(unique),
                  metric_direction="positive means v6 better",
                  reports=reports, paired_bootstrap=summary, fold_details=folds)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cv_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    np.savez_compressed(args.out / "cv_predictions.npz", v5=predictions["v5"], v6=predictions["v6"],
                        y=label, terrain=terrain_of, game=game_of)
    for name in ("v5", "v6"):
        print(name, "all", reports[name]["all"])
        for terrain, value in reports[name]["by_terrain"].items():
            print(" ", terrain, value)
    print("paired bootstrap (v6 better if positive)", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
