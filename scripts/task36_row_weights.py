"""task36 step 3: build --row-weights manifests for the two mix controls.

Plan A: every zone carries the same total weight (day / night / pool), evenly over its rows;
        inside night the n10 rows carry twice the weight of the n9 rows.
Plan B: plan A, then the day early-level rows (L1/L3/L4/L6, family day_early_d7) get another x2.

The multiplier multiplies the trainer's existing per-row weight (plant/teacher rows 1.0, kept
wait rows 1/--wait-keep, times --label-weight), so the internal structure is preserved and only
the mix between groups changes. Validation rows must stay exactly 1.0 (RowWeights.finish).

  python3 scripts/task36_row_weights.py --plan A --out artifacts/task36/row_weights_A.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from pvz_frame_store import frame_paths  # noqa: E402
from pvz_training_rows import update_identity  # noqa: E402

WAIT_KEEP = 0.1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, help="comma-separated: A and/or B")
    parser.add_argument("--data", type=Path, default=ROOT / "artifacts/task27/training_frames")
    parser.add_argument("--command", type=Path,
                        default=ROOT / "artifacts/task27/training_s1.command.json")
    parser.add_argument("--ledger", type=Path, default=ROOT / "artifacts/task36/data_ledger.json")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    command = json.loads(args.command.read_text())["command"]
    labels = set(command[command.index("--labels") + 1].split(","))
    label_weight = {kv.split("=")[0]: float(kv.split("=")[1])
                    for kv in command[command.index("--label-weight") + 1].split(",") if kv}
    ledger = json.loads(args.ledger.read_text())
    attribution = {row["label"]: (row["zone"], row["family"]) for row in ledger["dw5i"]["labels"]}

    paths = [p for p in frame_paths(args.data) if p.parent.name in labels]
    capacity = 0
    rng = random.Random(0)
    masks, row_arrays = [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as stored:
            hints = np.asarray(stored["hints"], dtype=bool)
            rows = np.asarray(stored["rows"], dtype=np.int64)
        draws = np.fromiter((rng.random() for _ in range(int((~hints).sum()))),
                            dtype=np.float64, count=int((~hints).sum()))
        mask = hints.copy()
        mask[~hints] = draws < WAIT_KEEP
        masks.append(mask)
        row_arrays.append(rows)
        capacity += int(mask[rows].sum())

    seeds = sorted({int(p.parent.parent.name.split("_")[-1]) for p in paths})
    shuffled = list(seeds)
    random.Random(0).shuffle(shuffled)
    val_seeds = set(shuffled[: len(shuffled) // 10])

    base = np.empty(capacity, np.float32)
    group = np.empty(capacity, np.int32)
    is_val = np.zeros(capacity, bool)
    digest = hashlib.sha256()
    n = 0
    group_names = []
    group_index = {}
    for path, mask, rows in zip(paths, masks, row_arrays):
        label = path.parent.name
        seed = int(path.parent.parent.name.split("_")[-1])
        with np.load(path, allow_pickle=False) as stored:
            selected = mask[rows]
            index = rows[selected]
            plant = np.asarray(stored["plant"], dtype=np.float32)[selected]
            values = np.asarray(stored["values"], dtype=np.float32)[selected]
        update_identity(digest, str(path.relative_to(args.data)), index)
        zone, family = attribution[label]
        key = (zone, family)
        if key not in group_index:
            group_index[key] = len(group_names)
            group_names.append(key)
        count = len(index)
        scale = label_weight.get(label, 1.0)
        keep = (plant != 0) | (~np.isnan(values)).any(1)
        base[n:n + count] = np.where(keep, 1.0, 1.0 / WAIT_KEEP) * scale
        group[n:n + count] = group_index[key]
        is_val[n:n + count] = seed in val_seeds
        n += count
    assert n == capacity

    totals = np.zeros(len(group_names), np.float64)
    for g in range(len(group_names)):
        totals[g] = base[(group == g) & ~is_val].sum()
    zones = sorted({name[0] for name in group_names})
    zone_total = {z: sum(totals[i] for i, name in enumerate(group_names) if name[0] == z) for z in zones}

    plans = args.plan.split(",")
    outputs = {}
    for plan in plans:
        multiplier = np.ones(capacity, np.float32)
        factors = {}
        for z in zones:
            members = [i for i, name in enumerate(group_names) if name[0] == z]
            weight_sum = sum(totals[i] for i in members)
            if z == "night":
                # zone share 1/3, split inside the night zone n10:n9 = 2:1
                for i in members:
                    if not totals[i]:
                        continue
                    share = {"night_n10": 2.0 / 3.0, "night_n9": 1.0 / 3.0}.get(group_names[i][1], 0.0)
                    factors[group_names[i]] = (share / len(zones)) / totals[i]
            else:
                for i in members:
                    if not totals[i]:
                        continue
                    factors[group_names[i]] = (1.0 / len(zones)) / weight_sum
            for i in members:
                if group_names[i] in factors:
                    multiplier[group == i] = factors[group_names[i]]
        if plan == "B":
            for i, name in enumerate(group_names):
                if name[1] == "day_early_d7":
                    factors[name] *= 2.0
                    multiplier[group == i] *= 2.0
        # keep the overall loss scale where it was: mean multiplier over training rows == 1
        training = ~is_val
        mean = float((multiplier[training] * base[training]).sum() / base[training].sum())
        multiplier /= mean
        for name in factors:
            factors[name] /= mean
        multiplier[is_val] = 1.0
        outputs[plan] = (multiplier, mean)

    summary = {}
    for plan, (multiplier, mean) in outputs.items():
        result = {}
        for i, name in enumerate(group_names):
            sel = (group == i) & ~is_val
            result["%s/%s" % name] = dict(
                base_weight=float(base[sel].sum()),
                final_weight=float((base[sel] * multiplier[sel]).sum()),
                rows=int(sel.sum()),
            )
        total_final = sum(v["final_weight"] for v in result.values())
        for key, value in result.items():
            value["share"] = value["final_weight"] / total_final if total_final else 0.0
        summary[plan] = dict(plan=plan, rows=int(capacity), training_rows=int((~is_val).sum()),
                             validation_rows=int(is_val.sum()), groups=result,
                             zone_share={z: sum(v["final_weight"] for k, v in result.items()
                                                 if k.startswith(z + "/")) / total_final for z in zones},
                             normalisation=mean)
        print(json.dumps(summary[plan], indent=2))
        if args.dry_run:
            continue
        out = (args.out.parent / ("row_weights_%s.json" % plan)) if args.out else \
            ROOT / ("artifacts/task36/row_weights_%s.json" % plan)
        out.parent.mkdir(parents=True, exist_ok=True)
        values_path = out.parent / ("row_weights_%s.npy" % plan)
        np.save(values_path, multiplier.astype(np.float32))
        spec = dict(
            name="task36_plan_%s" % plan,
            multipliers=values_path.name,
            rows=int(capacity),
            multipliers_sha256=hashlib.sha256(multiplier.astype(np.float32).tobytes()).hexdigest(),
            row_identity_sha256=digest.hexdigest(),
            split_sha256=hashlib.sha256(np.packbits(is_val).tobytes()).hexdigest(),
            plan=plan,
            definition=("zone-equal total weight over day/night/pool; night n10:n9 = 2:1"
                        + ("; day early L1/L3/L4/L6 x2" if plan == "B" else "")),
            group_shares={k: round(v["share"], 6) for k, v in result.items()},
            normalisation=mean,
        )
        out.write_text(json.dumps(spec, indent=2) + "\n")
        print("wrote", out, values_path)


if __name__ == "__main__":
    main()
