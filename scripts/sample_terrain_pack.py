#!/usr/bin/env python3
"""Stratified terrain + deck sample of a merged curriculum pack (task158).

Read-only on the corpus. Selects whole episodes per terrain (episode integrity keeps the
train/validation split clean), copies their rows out of the immutable segments into one merged
npz that ``train_rollout_policy.read_pack`` already understands, and adds per-row terrain,
episode, level and deck columns for the arm study.

Deck = unordered plant set of the seed bank, read from ``packet_col0`` (constant inside a game).
A per-terrain deck holdout is chosen from the sample the way task153 split seen/unseen decks:
whole decks (unordered plant sets) are excluded from training and used only for the unseen-deck
metric. The terrain's largest deck stays in training so the sample keeps its data.

The level -> terrain mapping is frozen from ``pvz_constants.background_for_level`` (which parses
Board::PickBackground, including the L35 ScaryPotter night override).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

TERRAINS = ("day", "night", "pool", "fog", "roof")
TERRAIN_ID = {name: index for index, name in enumerate(TERRAINS)}


def terrain_table() -> dict[int, str]:
    from pvz_constants import background_for_level

    zones = {"BACKGROUND_1_DAY": "day", "BACKGROUND_2_NIGHT": "night",
             "BACKGROUND_3_POOL": "pool", "BACKGROUND_4_FOG": "fog",
             "BACKGROUND_5_ROOF": "roof", "BACKGROUND_6_BOSS": "roof"}
    return {level: zones[background_for_level(level)] for level in range(1, 51)}


def archive_seed(directory: str) -> int:
    tag = directory.removeprefix("seed_").rsplit("_f", 1)[-1]
    return int(tag)


def segments_of(index: Path):
    with np.load(index, allow_pickle=False) as stored:
        return stored["segments"].tolist(), int(stored["rows"]), json.loads(str(stored["meta"]))


def scan(index: Path, table: dict[int, str]) -> list[dict]:
    """One record per archive: where its rows sit and which terrain it belongs to."""
    segments, total, _meta = segments_of(index)
    records = []
    for name in segments:
        with np.load(index.parent / name, allow_pickle=False) as stored:
            sources = [str(value) for value in stored["source"]]
            archive = np.asarray(stored["source_archive"])
        if len(archive) and np.any(np.diff(archive) < 0):
            raise SystemExit(f"{name}: archive rows are not contiguous")
        counts = np.bincount(archive, minlength=len(sources))
        start = 0
        for source, count in zip(sources, counts):
            count = int(count)
            if count:
                meta = json.loads((index.parent.parent / Path(source).parent / "meta.json").read_text())
                level = int(meta["context"]["level"])
                records.append(dict(segment=name, start=start, stop=start + count, rows=count,
                                    source=source, level=level, terrain=table[level],
                                    seed=archive_seed(Path(source).parent.parent.name)))
            start += count
    if sum(record["rows"] for record in records) != total:
        raise SystemExit(f"scan covered {sum(r['rows'] for r in records)} of {total} rows")
    return records


def select(records: list[dict], caps: dict[str, int], seed: int) -> list[dict]:
    chosen, rng = [], random.Random(seed)
    for terrain in TERRAINS:
        pool = [record for record in records if record["terrain"] == terrain]
        rng.shuffle(pool)
        rows = 0
        for record in pool:
            if rows >= caps[terrain]:
                break
            chosen.append(record)
            rows += record["rows"]
    chosen.sort(key=lambda record: (record["segment"], record["start"]))
    return chosen


def choose_deck_holdout(deck_column, terrain_column, deck_names, seed, fraction=0.12):
    """Whole unseen decks per terrain (task153 split); the terrain's largest deck trains."""
    holdout: set[int] = set()
    for terrain in range(len(TERRAINS)):
        mask = terrain_column == terrain
        if not mask.any():
            continue
        totals = np.bincount(deck_column[mask], minlength=len(deck_names))
        order = [int(index) for index in np.argsort(-totals) if totals[index]]
        if len(order) <= 1:
            continue
        target = fraction * int(mask.sum())
        held = sum(int(totals[index]) for index in holdout)
        candidates = [index for index in order[1:]
                      if index not in holdout and totals[index] <= max(1.0, 0.5 * target)]
        random.Random(seed + terrain).shuffle(candidates)
        for index in candidates:
            if held >= target:
                break
            holdout.add(index)
            held += int(totals[index])
    return holdout


class CSRWriter:
    """Same layout CompactBoards.from_arrays reads: CSR plus one hoisted plane table."""

    def __init__(self, name: str):
        self.name = name
        self.indices = bytearray()
        self.values = bytearray()
        self.offsets = bytearray(np.zeros(1, dtype=np.int64).tobytes())
        self.plane_ids = bytearray()
        self.planes: list[tuple[np.ndarray, np.ndarray]] = []
        self._plane_index: dict[bytes, int] = {}
        self.nnz = 0
        self.shape = None
        self.dtype = None
        self.index_dtype = None

    def append(self, stored, start: int, stop: int):
        prefix = self.name
        offsets = np.asarray(stored[prefix + "_offsets"])
        lo, hi = int(offsets[start]), int(offsets[stop])
        indices = np.asarray(stored[prefix + "_indices"])[lo:hi]
        values = np.asarray(stored[prefix + "_values"])[lo:hi]
        if self.shape is None:
            self.shape = tuple(int(n) for n in stored[prefix + "_shape"])
            self.dtype, self.index_dtype = values.dtype, indices.dtype
        self.indices.extend(indices.tobytes())
        self.values.extend(values.tobytes())
        counts = offsets[start + 1:stop + 1] - offsets[start:stop]
        self.offsets.extend((counts.cumsum() + self.nnz).astype(np.int64).tobytes())
        self.nnz += hi - lo
        plane_of = np.asarray(stored[prefix + "_plane_of"])[start:stop]
        cells = np.asarray(stored[prefix + "_plane_cells"])
        values_plane = np.asarray(stored[prefix + "_plane_values"])
        plane_offsets = np.asarray(stored[prefix + "_plane_offsets"])
        for identity in plane_of:
            if identity < 0:
                self.plane_ids.extend(np.int32(-1).tobytes())
                continue
            plane = (cells[plane_offsets[identity]:plane_offsets[identity + 1]],
                     values_plane[plane_offsets[identity]:plane_offsets[identity + 1]])
            key = plane[0].tobytes() + plane[1].tobytes()
            found = self._plane_index.get(key)
            if found is None:
                found = len(self.planes)
                self._plane_index[key] = found
                self.planes.append((plane[0].copy(), plane[1].copy()))
            self.plane_ids.extend(np.int32(found).tobytes())

    def arrays(self) -> dict:
        if self.planes:
            cells = np.concatenate([plane[0] for plane in self.planes])
            values = np.concatenate([plane[1] for plane in self.planes])
            offsets = np.zeros(len(self.planes) + 1, dtype=np.int64)
            np.cumsum([len(plane[0]) for plane in self.planes], out=offsets[1:])
        else:
            cells, values, offsets = (np.zeros(0, np.int32), np.zeros(0, self.dtype),
                                      np.zeros(1, np.int64))
        return {self.name + "_indices": np.frombuffer(self.indices, self.index_dtype),
                self.name + "_values": np.frombuffer(self.values, self.dtype),
                self.name + "_offsets": np.frombuffer(self.offsets, np.int64),
                self.name + "_plane_of": np.frombuffer(self.plane_ids, np.int32),
                self.name + "_plane_cells": cells, self.name + "_plane_values": values,
                self.name + "_plane_offsets": offsets,
                self.name + "_shape": np.asarray(self.shape, np.int64),
                self.name + "_dtype": np.asarray(str(self.dtype))}


class Columns:
    """Append-only row columns; a 2-D input (the packet columns) keeps its trailing extent."""

    def __init__(self, dtype):
        self.data = bytearray()
        self.dtype = np.dtype(dtype)
        self.width = None

    def extend(self, array):
        array = np.ascontiguousarray(array, dtype=self.dtype)
        if self.width is None and array.ndim == 2:
            self.width = array.shape[1]
        if self.width is not None:
            array = array.reshape(-1, self.width)
        self.data.extend(array.tobytes())

    def array(self):
        flat = np.frombuffer(self.data, self.dtype)
        return flat if self.width is None else flat.reshape(-1, self.width)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--caps", default="day=500000,night=500000,pool=500000,fog=500000,roof=500000")
    parser.add_argument("--holdout-fraction", type=float, default=0.12)
    parser.add_argument("--select-seed", type=int, default=20261029)
    parser.add_argument("--table", type=Path, help="level -> terrain json, instead of parsing pvz_constants")
    parser.add_argument("--copy-segments", type=int, help="smoke test: only copy these leading segments")
    parser.add_argument("--scan", action="store_true", help="only report the terrain census")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    table = ({int(level): terrain for level, terrain in json.loads(args.table.read_text()).items()}
             if args.table else terrain_table())
    started = time.time()
    records = scan(args.index, table)
    census = {terrain: sum(r["rows"] for r in records if r["terrain"] == terrain) for terrain in TERRAINS}
    print(json.dumps({"archives": len(records), "rows": sum(census.values()),
                      "terrain_rows": census, "seconds": round(time.time() - started, 1)}), flush=True)
    if args.scan:
        return
    if args.out is None:
        raise SystemExit("--out is required unless --scan")

    caps = {key: int(value) for key, value in (item.split("=") for item in args.caps.split(","))}
    chosen = select(records, caps, args.select_seed)
    selected_rows = Counter()
    for record in chosen:
        selected_rows[record["terrain"]] += record["rows"]
    print(json.dumps({"chosen_archives": len(chosen), "chosen_rows": dict(selected_rows),
                      "total_rows": sum(selected_rows.values())}), flush=True)

    by_segment = defaultdict(list)
    for record in chosen:
        by_segment[record["segment"]].append(record)

    board, scalar = CSRWriter("board"), CSRWriter("scalar")
    plant, packet, cell, weight, seed = (Columns(np.uint8), Columns(np.int16), Columns(np.int16),
                                        Columns(np.float32), Columns(np.int32))
    terrain_column, episode_column, level_column = Columns(np.uint8), Columns(np.int32), Columns(np.int16)
    deck_column = Columns(np.uint16)
    packet_columns = [Columns(dtype) for dtype in (np.int8, np.uint8, np.float32, np.uint8)]
    value_counts, value_slots, value_values = Columns(np.uint8), Columns(np.uint8), Columns(np.float32)
    cand_counts, cand_entries = Columns(np.uint8), Columns(np.float16)
    has_cands = True
    episodes: list[dict] = []
    deck_names: list[str] = []
    deck_index: dict[str, int] = {}
    written = 0

    for segment in segments_of(args.index)[0][:args.copy_segments]:
        parts = by_segment.get(segment)
        if not parts:
            continue
        with np.load(args.index.parent / segment, allow_pickle=False) as stored:
            has_cands = "cand_counts" in stored.files
            value_offsets = np.concatenate([[0], np.cumsum(np.asarray(stored["value_counts"]).astype(np.int64))])
            cand_offsets = (np.concatenate([[0], np.cumsum(np.asarray(stored["cand_counts"]).astype(np.int64))])
                            if has_cands else None)
            kinds = np.asarray(stored["packet_col0"])
            for record in parts:
                start, stop = record["start"], record["stop"]
                deck = ",".join(str(kind) for kind in sorted({int(k) for k in kinds[start] if k >= 0}))
                if deck not in deck_index:
                    deck_index[deck] = len(deck_names)
                    deck_names.append(deck)
                record.update(episode=len(episodes), deck=deck)
                episodes.append(record)
                board.append(stored, start, stop)
                scalar.append(stored, start, stop)
                plant.extend(np.asarray(stored["plant"])[start:stop])
                packet.extend(np.asarray(stored["packet"])[start:stop])
                cell.extend(np.asarray(stored["cell"])[start:stop])
                weight.extend(np.asarray(stored["weight"])[start:stop])
                seed.extend(np.asarray(stored["seed"])[start:stop])
                for column, name in zip(packet_columns, ("packet_col0", "packet_col1", "packet_col2", "packet_col3")):
                    column.extend(np.asarray(stored[name])[start:stop])
                count = stop - start
                terrain_column.extend(np.full(count, TERRAIN_ID[record["terrain"]], np.uint8))
                episode_column.extend(np.full(count, len(episodes) - 1, np.int32))
                level_column.extend(np.full(count, record["level"], np.int16))
                deck_column.extend(np.full(count, deck_index[deck], np.uint16))
                value_counts.extend(np.asarray(stored["value_counts"])[start:stop])
                value_slots.extend(np.asarray(stored["value_slots"])[value_offsets[start]:value_offsets[stop]])
                value_values.extend(np.asarray(stored["value_values"])[value_offsets[start]:value_offsets[stop]])
                if has_cands:
                    cand_counts.extend(np.asarray(stored["cand_counts"])[start:stop])
                    cand_entries.extend(np.asarray(stored["cand_entries"])[cand_offsets[start]:cand_offsets[stop]])
                written += count
        print(f"{segment}: {len(parts)} archives, {written} rows, {time.time() - started:.0f}s", flush=True)

    terrain_rows = terrain_column.array()
    deck_rows = deck_column.array()
    holdout = choose_deck_holdout(deck_rows, terrain_rows, deck_names, args.select_seed,
                                  args.holdout_fraction)
    holdout_column = np.isin(deck_rows, np.asarray(sorted(holdout), np.uint16)).astype(np.uint8)
    unseen = {name: int(holdout_column[terrain_rows == index].sum()) for index, name in enumerate(TERRAINS)}
    print(json.dumps({"holdout_decks": len(holdout), "holdout_rows": unseen}), flush=True)

    output = dict(board.arrays(), **scalar.arrays(),
                  rows=np.int64(written),
                  plant=plant.array(), packet=packet.array(), cell=cell.array(),
                  weight=weight.array(), seed=seed.array(),
                  value_counts=value_counts.array(), value_slots=value_slots.array(),
                  value_values=value_values.array(),
                  **{f"packet_col{i}": column.array() for i, column in enumerate(packet_columns)},
                  terrain=terrain_rows, episode=episode_column.array(), level=level_column.array(),
                  deck=deck_rows, holdout_deck=holdout_column,
                  deck_names=np.asarray(deck_names),
                  episode_source=np.asarray([record["source"] for record in episodes]),
                  episode_seed=np.asarray([record["seed"] for record in episodes], np.int32),
                  episode_terrain=np.asarray([TERRAIN_ID[record["terrain"]] for record in episodes], np.uint8),
                  episode_level=np.asarray([record["level"] for record in episodes], np.int16),
                  episode_deck=np.asarray([record["deck"] for record in episodes]),
                  meta=np.asarray(json.dumps({
                      "sampler": "scripts/sample_terrain_pack.py",
                      "index": str(args.index),
                      "index_sha256": hashlib.sha256(args.index.read_bytes()).hexdigest(),
                      "index_meta": segments_of(args.index)[2],
                      "select_seed": args.select_seed, "caps": caps,
                      "terrain_rows": census, "chosen_rows": dict(selected_rows),
                      "holdout_fraction": args.holdout_fraction,
                      "holdout_decks": sorted(deck_names[index] for index in holdout),
                      "holdout_rows": unseen, "terrains": list(TERRAINS),
                      "level_terrain": {str(k): v for k, v in table.items()},
                      "rows": written, "episodes": len(episodes)})))
    if has_cands:
        output.update(cand_counts=cand_counts.array(),
                      cand_entries=cand_entries.array().reshape(-1, 3))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **output)
    print(json.dumps({"out": str(args.out), "rows": written, "episodes": len(episodes),
                      "decks": len(deck_names), "bytes": args.out.stat().st_size,
                      "seconds": round(time.time() - started, 1)}), flush=True)


if __name__ == "__main__":
    main()
