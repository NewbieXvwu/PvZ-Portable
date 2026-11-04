"""Full replay-corpus training and episode-held-out comparison for task243.

Feature encoding and training use the existing C++ exporter and task239 trainer.
Only small batches move from the float16 memmaps to MPS.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "python")]
from task239_train import TERRAINS, load_rows, metrics, run_arm, torch_view
from pvz_value_net import load_value_net


def pack(source, target):
    target.mkdir(parents=True, exist_ok=True)
    for terrain in TERRAINS:
        paths = sorted(source.glob(f"{terrain}-*.npz"))
        sizes = []
        for path in paths:
            with np.load(path) as shard:
                sizes.append(len(shard["y"]))
        if not sizes:
            continue
        directory = target / terrain
        directory.mkdir(exist_ok=True)
        with np.load(paths[0]) as first:
            arrays = {key: np.lib.format.open_memmap(directory / f"{key}.npy", mode="w+",
                      dtype=first[key].dtype, shape=(sum(sizes), *first[key].shape[1:]))
                      for key in first.files}
        offset = 0
        for path, size in zip(paths, sizes):
            with np.load(path) as shard:
                for key, array in arrays.items():
                    array[offset:offset + size] = shard[key]
            offset += size
        for array in arrays.values():
            array.flush()
        print(f"packed {terrain}: {offset} rows", flush=True)


def split(data, seed, exposed_seeds=()):
    train, val, test = set(), set(), set()
    exposed = set(exposed_seeds)
    rng = random.Random(seed)
    for terrain in TERRAINS:
        rows = data[terrain]
        for outcome in (0, 1):
            seeds = sorted(set(map(int, rows["seed"][rows["y"] == outcome])))
            rng.shuffle(seeds)
            ntrain, nval = int(len(seeds) * .7), int(len(seeds) * .15)
            eligible = [value for value in seeds if value not in exposed]
            val_seeds = set(eligible[:nval])
            test_seeds = set(eligible[nval:len(seeds) - ntrain])
            for value in seeds:
                part = val if value in val_seeds else test if value in test_seeds else train
                part.add((terrain, value))
    return train, val, test


def predict(data, paths, device):
    predictions = []
    for terrain in TERRAINS:
        rows = data[terrain]
        net = load_value_net(paths[terrain]).to(device)
        prob = np.empty(len(rows["y"]), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(prob), 512):
                stop = min(start + 512, len(prob))
                batch = {key: rows[key][start:stop] for key in ("board", "scalars", "context", "y")}
                b, s, c, _ = torch_view(batch, net.encoder_version)
                prob[start:stop] = torch.sigmoid(net(b.to(device), torch.cat([s, c], 1).to(device))).cpu().numpy()
        predictions.append(prob)
        del net
    return np.concatenate(predictions)


def comparison(predictions, labels, games, repetitions, seed):
    """Whole-game paired bootstrap; fixed score orders avoid sorting every draw."""
    unique, inverse = np.unique(games, return_inverse=True)
    prepared = {}
    for name, probability in predictions.items():
        p = np.clip(probability.astype(np.float64), 1e-6, 1 - 1e-6)
        order = np.argsort(p, kind="stable")
        ends = np.r_[np.flatnonzero(np.diff(p[order])), len(p) - 1]
        prepared[name] = (-(labels * np.log(p) + (1 - labels) * np.log(1 - p)),
                          (p - labels) ** 2, order, ends)
    pairs = [(a, b) for a, b in (("v6", "v5"), ("v6", "production"), ("v5", "production"))]
    draws = {f"{a}_minus_{b}": {k: [] for k in ("log_loss", "brier", "auc")} for a, b in pairs}
    rng = np.random.default_rng(seed)
    for _ in range(repetitions):
        counts = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))
        weight = counts[inverse]
        results = {}
        for name, (loss, brier, order, ends) in prepared.items():
            pos = np.cumsum((weight * labels)[order])[ends]
            neg = np.cumsum((weight * (1 - labels))[order])[ends]
            pos_group, neg_group = np.diff(np.r_[0, pos]), np.diff(np.r_[0, neg])
            area = np.sum(pos_group * (neg - .5 * neg_group)) / (pos[-1] * neg[-1]) if pos[-1] * neg[-1] else np.nan
            results[name] = dict(log_loss=float(np.dot(weight, loss) / weight.sum()),
                                 brier=float(np.dot(weight, brier) / weight.sum()), auc=float(area))
        for a, b in pairs:
            for key in draws[f"{a}_minus_{b}"]:
                draws[f"{a}_minus_{b}"][key].append(results[a][key] - results[b][key])
    return {pair: {key: dict(mean=float(np.nanmean(v)), low=float(np.nanpercentile(v, 2.5)),
                            high=float(np.nanpercentile(v, 97.5))) for key, v in values.items()}
            for pair, values in draws.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--packed", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--production-value", type=Path, required=True)
    parser.add_argument("--production-roof", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=243)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    started = time.time()
    torch.set_num_threads(1)
    pack(args.features, args.packed)
    data = load_rows(args.packed)
    exposed = set()
    for path in (args.production_value, args.production_roof):
        exposed.update(torch.load(path, map_location="cpu", weights_only=False).get("trained_on", []))
    train, val, test = split(data, args.seed, exposed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "split.json").write_text(json.dumps({k: sorted(v) for k, v in
        (("train", train), ("validation", val), ("test", test))}, indent=1))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    terrain = np.concatenate([[t] * len(data[t]["y"]) for t in TERRAINS])
    seeds = np.concatenate([data[t]["seed"] for t in TERRAINS])
    games = np.concatenate([data[t]["game"] for t in TERRAINS])
    labels = np.concatenate([data[t]["y"] for t in TERRAINS])
    test_mask = np.asarray([(t, int(s)) in test for t, s in zip(terrain, seeds)])
    predictions, training = {}, {}
    for name, encoder in (("v5", 2), ("v6", 6)):
        print(f"training {name} on {device}: train/val/test {len(train)}/{len(val)}/{len(test)} games", flush=True)
        report, checkpoint, prob, _ = run_arm(data, encoder, 64, args.seed, args.epochs, device,
            train, val, lr=1e-4, dropout=0.0, weight_decay=1e-4, batch_size=512)
        checkpoint["state_dict"] = {key: value.cpu() for key, value in checkpoint["state_dict"].items()}
        checkpoint["task"] = "task243"
        checkpoint["trained_on"] = sorted({seed for _, seed in train | val})
        checkpoint["validation_seeds"] = sorted({seed for _, seed in val})
        torch.save(checkpoint, args.out / f"{name}.pt")
        predictions[name], training[name] = prob[test_mask], report
        del checkpoint
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
    paths = {t: args.production_roof if t == "roof" else args.production_value for t in TERRAINS}
    predictions["production"] = predict(data, paths, device)[test_mask]
    y, g, t = labels[test_mask], games[test_mask], terrain[test_mask]
    report = dict(task="task243", device=str(device), width=64, epochs=args.epochs,
        lr=1e-4, weight_decay=1e-4, dropout=0.0, batch_size=512, seed=args.seed,
        split_games=dict(train=len(train), validation=len(val), test=len(test)),
        known_production_training_seeds=len(exposed),
        production_training_seeds_in_new_train=sum(seed in exposed for _, seed in train),
        production_training_seeds_in_new_validation_or_test=sum(seed in exposed for _, seed in val | test),
        production="v5 outside roof; v5roof on roof (current production routing)",
        roof="unified models, no roof fine-tune: this corpus has episode labels, not task105 scan continuation labels",
        training=training, metrics={}, bootstrap={}, bootstrap_repetitions=args.bootstrap,
        data={terrain: dict(rows=len(rows["y"]), games=len(np.unique(rows["game"])),
                            wins=len(np.unique(rows["game"][rows["y"] == 1])))
              for terrain, rows in data.items()},
        delta_direction="candidate minus control; loss/Brier negative is better, AUC positive is better")
    for scope in ("all", *TERRAINS):
        mask = np.ones(len(y), dtype=bool) if scope == "all" else t == scope
        p = {name: value[mask] for name, value in predictions.items()}
        report["metrics"][scope] = {name: metrics(value, y[mask]) for name, value in p.items()}
        report["bootstrap"][scope] = comparison(p, y[mask], g[mask], args.bootstrap, args.seed)
        print(scope, report["metrics"][scope], flush=True)
    report["seconds"] = time.time() - started
    (args.out / "offline.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    np.savez_compressed(args.out / "test_predictions.npz", **predictions, y=y, game=g, terrain=t)


if __name__ == "__main__":
    main()
