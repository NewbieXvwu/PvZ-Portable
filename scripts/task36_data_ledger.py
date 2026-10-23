"""task36 step 2: data-mix ledger for the dw4i and dw5i training sets.

Replays the trainer's exact wait-row sampling (random.Random(0), one draw per non-hint
frame, archives in frame_paths order) so the retained-row counts per label are the ones the
run actually trained on, then attributes every label to
  zone x deck family x collection machine x collecting student.

  python3 scripts/task36_data_ledger.py --out artifacts/task36/data_ledger.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from pvz_frame_store import frame_paths  # noqa: E402

WAIT_KEEP = 0.1
ZONE = {
    "BACKGROUND_1_DAY": "day", "BACKGROUND_2_NIGHT": "night",
    "BACKGROUND_3_POOL": "pool", "BACKGROUND_4_FOG": "fog",
    "BACKGROUND_5_ROOF": "roof",
}
FAMILY_RULES = [
    (re.compile(r"^n\d+t\d+_fast_search$"), "night_n10"),
    (re.compile(r"^n\d+dag\d+m?_fast_search$"), "night_n9"),
    (re.compile(r"^p\d+dag\d+_fast_search$"), "pool_p9k"),
    (re.compile(r"^d\d+orig\d+_fast_search$"), "day_orig_conveyor"),
    (re.compile(r"^d\d+p\d+[dm]_fast_search$"), "day_d7p"),
    (re.compile(r"^d(?:1|3|4|6)dag\d+d_fast_search$"), "day_early_d7"),
    (re.compile(r"^d\d+dag\d+b?_fast_search$"), "day_d7"),
    (re.compile(r"^z\d+dag\d+.*_fast_search$"), "identity_base"),
]


def family_of(label: str) -> str:
    for pattern, name in FAMILY_RULES:
        if pattern.match(label):
            return name
    return "unclassified"


def zone_of(level: int) -> str:
    from pvz_constants import background_for_level
    return ZONE[background_for_level(level)]


def read_meta(path: Path) -> dict:
    meta = json.loads((path.parent / "meta.json").read_text())
    return {
        "archive": str(path.parent.relative_to(path.parents[2])),
        "label": path.parent.name,
        "seed": int(path.parent.parent.name.split("_")[-1]),
        "level": int(meta["context"]["level"]),
        "policy": meta["task"].get("policy", "?"),
        "won": bool(meta.get("outcome", {}).get("won", False)),
    }


def replay(data: Path, labels: set[str]) -> dict:
    """Per-label retained rows and validation rows, exactly as the trainer samples them."""
    paths = [p for p in frame_paths(data) if p.parent.name in labels]
    rng = random.Random(0)
    rows = Counter()
    val = Counter()
    seeds_by_label = defaultdict(set)
    archive_seeds = []
    for path in paths:
        with np.load(path, allow_pickle=False) as stored:
            hints = np.asarray(stored["hints"], dtype=bool)
            row_index = np.asarray(stored["rows"], dtype=np.int64)
        draws = np.fromiter((rng.random() for _ in range(int((~hints).sum()))),
                            dtype=np.float64, count=int((~hints).sum()))
        mask = hints.copy()
        mask[~hints] = draws < WAIT_KEEP
        selected = mask[row_index]
        label = path.parent.name
        seed = int(path.parent.parent.name.split("_")[-1])
        rows[label] += int(selected.sum())
        seeds_by_label[label].add(seed)
        archive_seeds.append((label, seed, int(selected.sum())))
    seeds = sorted({seed for _, seed, _ in archive_seeds})
    shuffled = list(seeds)
    random.Random(0).shuffle(shuffled)
    val_seeds = set(shuffled[: len(shuffled) // 10])
    for label, seed, count in archive_seeds:
        if seed in val_seeds:
            val[label] += count
    return dict(paths=len(paths), rows=dict(rows), validation_rows=dict(val),
                seeds=sorted(val_seeds))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path,
                        default=ROOT / "artifacts/task27/training_frames")
    parser.add_argument("--inventory", type=Path,
                        default=ROOT / "artifacts/task27/dataset_inventory.json")
    parser.add_argument("--mac-journal", type=Path,
                        default=ROOT / "artifacts/task27/mac_import.json")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/task36/data_ledger.json")
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()

    inventory = json.loads(args.inventory.read_text())
    groups = defaultdict(set)
    for record in inventory:
        groups[record["group"]].add(record["label"])
    dw5i_labels = set().union(*groups.values())
    dw4i_labels = set(groups["dw4i_frozen"])
    journal = json.loads(args.mac_journal.read_text())
    mac_labels = {r["label"] + "_fast_search" for r in journal["rounds"]}

    paths = [p for p in frame_paths(args.data) if p.parent.name in dw5i_labels]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        metas = list(pool.map(read_meta, paths, chunksize=64))

    by_label = defaultdict(lambda: dict(archives=0, wins=0, levels=Counter(), policies=Counter(),
                                        zones=Counter(), seeds=set()))
    for meta in metas:
        record = by_label[meta["label"]]
        record["archives"] += 1
        record["wins"] += int(meta["won"])
        record["levels"][meta["level"]] += 1
        record["policies"][meta["policy"]] += 1
        record["zones"][zone_of(meta["level"])] += 1
        record["seeds"].add(meta["seed"])

    dw5i = replay(args.data, dw5i_labels)
    dw4i = replay(args.data, dw4i_labels)

    def attribute(rows: dict, val: dict, label_set: set[str]) -> list[dict]:
        out = []
        for label in sorted(label_set):
            info = by_label[label]
            source = "mac" if (label in mac_labels or label.endswith("m_fast_search")
                               or label.startswith("z15")) else "desktop"
            zone = info["zones"].most_common(1)[0][0] if info["zones"] else "?"
            out.append(dict(
                label=label, zone=zone, family=family_of(label), source=source,
                policies={k: v for k, v in info["policies"].items()},
                levels={str(k): v for k, v in sorted(info["levels"].items())},
                archives=info["archives"], wins=info["wins"],
                rows=rows.get(label, 0), validation_rows=val.get(label, 0),
                zones={k: v for k, v in info["zones"].items()},
            ))
        return out

    dw5i_rows = attribute(dw5i["rows"], dw5i["validation_rows"], dw5i_labels)
    dw4i_rows = attribute(dw4i["rows"], dw4i["validation_rows"], dw4i_labels)

    def aggregate(rows: list[dict]) -> dict:
        total = sum(r["rows"] for r in rows)
        cells = defaultdict(lambda: dict(rows=0, archives=0, labels=0, validation_rows=0))
        for r in rows:
            for key in [(r["zone"], r["family"], r["source"], "+".join(sorted(r["policies"]))),
                        (r["zone"], r["family"], r["source"], "*"),
                        (r["zone"], "*", "*", "*")]:
                cell = cells[key]
                cell["rows"] += r["rows"]
                cell["archives"] += r["archives"]
                cell["labels"] += 1
                cell["validation_rows"] += r["validation_rows"]
        out = []
        for key, cell in cells.items():
            zone_rows = cells[(key[0], "*", "*", "*")]["rows"]
            out.append(dict(zone=key[0], family=key[1], source=key[2], collector=key[3],
                            rows=cell["rows"], archives=cell["archives"], labels=cell["labels"],
                            validation_rows=cell["validation_rows"],
                            share_of_total=cell["rows"] / total if total else 0.0,
                            share_of_zone=cell["rows"] / zone_rows if zone_rows else 0.0))
        out.sort(key=lambda r: (-r["rows"], r["zone"], r["family"], r["source"]))
        return dict(total_rows=total, total_archives=sum(r["archives"] for r in rows),
                    cells=out)

    report = dict(
        schema_version=1,
        data=str(args.data),
        wait_keep=WAIT_KEEP,
        dw4i=dict(labels=dw4i_rows, summary=aggregate(dw4i_rows),
                  replay_rows=sum(dw4i["rows"].values()),
                  replay_archives=dw4i["paths"], validation_seeds=dw4i["seeds"]),
        dw5i=dict(labels=dw5i_rows, summary=aggregate(dw5i_rows),
                  replay_rows=sum(dw5i["rows"].values()),
                  replay_archives=dw5i["paths"], validation_seeds=dw5i["seeds"]),
        source_rule=("mac = label in artifacts/task27/mac_import.json rounds, or _m suffix, "
                     "or z15* prefix (documented Mac rounds); everything else desktop"),
        zone_rule="scripts/pvz_constants.background_for_level(level) from each archive meta.json",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(dw4i_rows=report["dw4i"]["replay_rows"],
                          dw5i_rows=report["dw5i"]["replay_rows"],
                          dw4i_archives=report["dw4i"]["replay_archives"],
                          dw5i_archives=report["dw5i"]["replay_archives"]), indent=1))


if __name__ == "__main__":
    main()
