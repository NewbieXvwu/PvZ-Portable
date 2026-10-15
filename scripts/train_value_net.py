"""Train and evaluate the value network V(s) = P(win | public board) (DESIGN.md §13, stage 1).

Data: episode archives (``<seed>/<checkpoint>/frames.jsonl`` + ``meta.json``) as written by
``scripts/episode_query.py capture``. Every frame is labelled with its game's final result.

Evaluation follows the A6 protocol (MEMORY/2026-10-04.md): **one sample per game per wave**
(the first frame of that wave), AUC computed separately per wave, so "died early -> short
game" cannot leak into the score. Splits are by environment seed: the same seed played by
three checkpoints shares its zombie schedule, so all three must sit on the same side.

Baselines on the very same samples:
  * phi      -- the PPO shaping potential (sun, wave progress, mean plant health)
  * shooters -- simply counting shooter plants on the board

Usage:
  python scripts/train_value_net.py --data /tmp/vnet/data --out artifacts/value_net/v1
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from pvz_value_net import ValueNet, featurize, potential_phi  # noqa: E402

SHOOTER_TYPES = {0, 5, 7, 40}
FRAME_STRIDE = 3


def load_games(data_dir: Path) -> list[dict]:
    games = []
    for meta_path in sorted(data_dir.glob("*/*/meta.json")):
        meta = json.loads(meta_path.read_text())
        frames = [json.loads(line) for line in (meta_path.parent / "frames.jsonl").open()]
        starts = wave_starts(frames)
        # Keep only the frames that become samples; whole games as dicts do not fit in memory
        # (2048 night games are ~4 GB of JSON).
        needed = set(range(0, len(frames) - 1, FRAME_STRIDE)) | set(starts.values())
        games.append({
            "seed": meta["task"]["seed"],
            "checkpoint": meta_path.parent.name,
            "won": bool(meta["outcome"]["won"]),
            "frames": [f if i in needed else None for i, f in enumerate(frames)],
            "wave_starts": starts,
        })
    return games


def wave_starts(frames: list[dict]) -> dict[int, int]:
    """Index of the first frame of each wave (wave >= 1)."""
    first = {}
    for i, frame in enumerate(frames):
        w = frame["wave"]
        if w >= 1 and w not in first and not frame.get("terminal"):
            first[w] = i
    return first


def build_tensors(games: list[dict]) -> dict:
    # Preallocated: a list of per-row tensors plus torch.stack would need twice the memory.
    capacity = sum(len(range(0, len(g["frames"]) - 1, FRAME_STRIDE)) for g in games)
    boards = scalars = None
    labels, weights, game_of = [], [], []
    eval_rows = []  # (game index, wave, row in eval tensors)
    eval_boards, eval_scalars = [], []
    n = 0
    for g_idx, game in enumerate(games):
        frames = game["frames"]
        picked = list(range(0, len(frames) - 1, FRAME_STRIDE))  # drop the terminal frame
        for i in picked:
            b, s = featurize(frames[i])
            if boards is None:
                boards = torch.empty((capacity, *b.shape), dtype=torch.half)
                scalars = torch.empty((capacity, *s.shape), dtype=s.dtype)
            boards[n] = b; scalars[n] = s; n += 1
            labels.append(float(game["won"]))
            weights.append(1.0 / len(picked)); game_of.append(g_idx)
        for wave, i in game["wave_starts"].items():
            b, s = featurize(frames[i])
            eval_boards.append(b); eval_scalars.append(s)
            eval_rows.append((g_idx, wave, i))
    return {
        "boards": boards[:n], "scalars": scalars[:n],
        "labels": torch.tensor(labels), "weights": torch.tensor(weights),
        "game_of": torch.tensor(game_of),
        "eval_boards": torch.stack(eval_boards), "eval_scalars": torch.stack(eval_scalars),
        "eval_rows": eval_rows,
    }


def auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Mann-Whitney AUC with average ranks for ties."""
    pos, neg = labels.sum(), len(labels) - labels.sum()
    if pos == 0 or neg == 0:
        return None
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def train_model(t: dict, train_games: set[int], device: str, epochs: int, seed: int) -> ValueNet:
    """Train on ``train_games``; 15% of their seeds are held out for early stopping."""
    torch.manual_seed(seed); random.seed(seed)
    game_of = t["game_of"].numpy()
    seeds_of_game = t["seed_of_game"]
    train_seeds = sorted({seeds_of_game[g] for g in train_games})
    random.shuffle(train_seeds)
    stop_seeds = set(train_seeds[: max(1, len(train_seeds) * 15 // 100)])
    fit_mask = np.array([g in train_games and seeds_of_game[g] not in stop_seeds for g in game_of])
    stop_mask = np.array([g in train_games and seeds_of_game[g] in stop_seeds for g in game_of])
    fit_idx, stop_idx = np.nonzero(fit_mask)[0], np.nonzero(stop_mask)[0]

    model = ValueNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, patience = float("inf"), None, 0

    def weighted_loss(idx: np.ndarray, train: bool) -> float:
        total, wsum = 0.0, 0.0
        order = np.random.permutation(idx) if train else idx
        for start in range(0, len(order), 1024):
            batch = torch.from_numpy(order[start:start + 1024])
            board = t["boards"][batch].float().to(device)
            scal = t["scalars"][batch].to(device)
            y = t["labels"][batch].to(device)
            w = t["weights"][batch].to(device)
            with torch.set_grad_enabled(train):
                logits = model(board, scal)
                loss = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * w).sum() / w.sum()
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * float(w.sum()); wsum += float(w.sum())
        return total / wsum

    for epoch in range(epochs):
        model.train(); weighted_loss(fit_idx, True)
        model.eval(); stop_loss = weighted_loss(stop_idx, False)
        if stop_loss < best - 1e-4:
            best, patience = stop_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 3:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def predict(model: ValueNet, boards: torch.Tensor, scalars: torch.Tensor, device: str) -> np.ndarray:
    out = []
    for start in range(0, len(boards), 2048):
        logits = model(boards[start:start + 2048].float().to(device), scalars[start:start + 2048].to(device))
        out.append(torch.sigmoid(logits).cpu())
    return torch.cat(out).numpy()


def per_wave_report(games, rows, scores_by_name: dict[str, np.ndarray], boot: int, rng) -> list[dict]:
    by_wave = defaultdict(list)
    for k, (g, wave, _i) in enumerate(rows):
        by_wave[wave].append(k)
    report = []
    for wave in sorted(by_wave):
        ks = np.array(by_wave[wave])
        labels = np.array([games[rows[k][0]]["won"] for k in ks], dtype=float)
        entry = {"wave": wave, "games": len(ks), "losses": int((labels == 0).sum())}
        for name, scores in scores_by_name.items():
            entry[f"auc_{name}"] = auc(scores[ks], labels)
        if entry["losses"] >= 5 and entry["games"] - entry["losses"] >= 5:
            diffs = []
            for _ in range(boot):
                pick = rng.integers(0, len(ks), len(ks))
                a = auc(scores_by_name["value"][ks][pick], labels[pick])
                b = auc(scores_by_name["phi"][ks][pick], labels[pick])
                if a is not None and b is not None:
                    diffs.append(a - b)
            if diffs:
                entry["value_minus_phi_ci95"] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
        report.append(entry)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--test-data", type=Path, help="external test archives (unseen seeds)")
    args = p.parse_args()
    started = time.time()

    games = load_games(args.data)
    t = build_tensors(games)
    t["seed_of_game"] = [g["seed"] for g in games]
    print(f"{len(games)} games, {len(t['labels'])} training frames, {len(t['eval_rows'])} wave-start samples", flush=True)

    seeds = sorted({g["seed"] for g in games})
    random.Random(0).shuffle(seeds)
    fold_of_seed = {s: i % args.folds for i, s in enumerate(seeds)}
    oof = np.full(len(t["eval_rows"]), np.nan)
    for fold in range(args.folds):
        train_games = {i for i, g in enumerate(games) if fold_of_seed[g["seed"]] != fold}
        model = train_model(t, train_games, args.device, args.epochs, seed=fold)
        held = [k for k, (g, _w, _i) in enumerate(t["eval_rows"]) if g not in train_games]
        oof[held] = predict(model, t["eval_boards"][held], t["eval_scalars"][held], args.device)
        print(f"fold {fold} done ({time.time() - started:.0f}s)", flush=True)

    phi = np.array([potential_phi(games[g]["frames"][i]) for g, _w, i in t["eval_rows"]])
    shooters = np.array([sum(1 for p in games[g]["frames"][i]["plants"] if p[2] in SHOOTER_TYPES)
                         for g, _w, i in t["eval_rows"]], dtype=float)
    scores = {"value": oof, "phi": phi, "shooters": shooters}
    rng = np.random.default_rng(0)
    waves = per_wave_report(games, t["eval_rows"], scores, args.bootstrap, rng)

    by_checkpoint = {}
    for ck in sorted({g["checkpoint"] for g in games}):
        ks = [k for k, (g, _w, _i) in enumerate(t["eval_rows"]) if games[g]["checkpoint"] == ck]
        sub_rows = [t["eval_rows"][k] for k in ks]
        by_checkpoint[ck] = per_wave_report(games, sub_rows, {n: s[ks] for n, s in scores.items()}, 0, rng)

    # Final model on every seed, for later use by search.
    final = train_model(t, set(range(len(games))), args.device, args.epochs, seed=99)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": final.state_dict(), "encoder": "pvz_value_net.featurize v1",
                "trained_on": sorted({g["seed"] for g in games}), "games": len(games)},
               args.out / "model.pt")
    oof_rows = [{"seed": games[g]["seed"], "checkpoint": games[g]["checkpoint"], "won": games[g]["won"],
                 "wave": w, "frame": i, "value": float(oof[k]), "phi": float(phi[k]), "shooters": shooters[k]}
                for k, (g, w, i) in enumerate(t["eval_rows"])]
    (args.out / "oof_predictions.json").write_text(json.dumps(oof_rows))

    external = None
    if args.test_data:
        test_games = load_games(args.test_data)
        test_rows, test_boards, test_scalars = [], [], []
        for g_idx, game in enumerate(test_games):
            for wave, i in game["wave_starts"].items():
                b, s = featurize(game["frames"][i])
                test_rows.append((g_idx, wave, i)); test_boards.append(b); test_scalars.append(s)
        test_value = predict(final, torch.stack(test_boards), torch.stack(test_scalars), args.device)
        test_scores = {
            "value": test_value,
            "phi": np.array([potential_phi(test_games[g]["frames"][i]) for g, _w, i in test_rows]),
            "shooters": np.array([sum(1 for p in test_games[g]["frames"][i]["plants"] if p[2] in SHOOTER_TYPES)
                                  for g, _w, i in test_rows], dtype=float),
        }
        test_ck = {}
        for ck in sorted({g["checkpoint"] for g in test_games}):
            ks = [k for k, (g, _w, _i) in enumerate(test_rows) if test_games[g]["checkpoint"] == ck]
            test_ck[ck] = per_wave_report(test_games, [test_rows[k] for k in ks],
                                          {n: s[ks] for n, s in test_scores.items()}, 0, rng)
        seed_overlap = sorted({g["seed"] for g in test_games} & set(seeds))
        external = {"data": str(args.test_data), "games": len(test_games), "seed_overlap": seed_overlap,
                    "per_wave": per_wave_report(test_games, test_rows, test_scores, args.bootstrap, rng),
                    "per_checkpoint": test_ck}
        print("external test (final model, unseen seeds):")
        _print_waves(external["per_wave"])

    result = {
        "schema": "value_net_eval_v1",
        "data": str(args.data), "games": len(games), "seeds": len(seeds), "folds": args.folds,
        "frame_stride": FRAME_STRIDE, "device": args.device,
        "protocol": "one sample per game per wave (first frame of the wave); out-of-fold by env seed",
        "per_wave": waves, "per_checkpoint": by_checkpoint, "external_test": external,
        "seconds": round(time.time() - started, 1),
    }
    (args.out / "metrics.json").write_text(json.dumps(result, indent=1))
    print(f"wrote {args.out} ({result['seconds']}s)")
    print("cross-validation (out-of-fold):")
    _print_waves(waves)


def _print_waves(waves: list[dict]) -> None:
    fmt = lambda v: "  -  " if v is None else f"{v:.3f}"
    for e in waves:
        if e["wave"] in (1, 2, 3, 5, 8, 10, 15, 20, 25):
            print(f"wave {e['wave']:2d}  n={e['games']:4d} losses={e['losses']:4d}  "
                  f"V {fmt(e['auc_value'])}  phi {fmt(e['auc_phi'])}  shooters {fmt(e['auc_shooters'])}  "
                  f"V-phi CI {e.get('value_minus_phi_ci95')}")


if __name__ == "__main__":
    main()
