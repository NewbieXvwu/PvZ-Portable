"""Pack the training rows of one label set into a single sequential file (task33 item 2).

Why: a training run used to decompress 25 M rows x 13.8 KB (345 GB) out of 21 056 per-game
frames.npz files and compress every row into CSR again -- on every run, because neither the
decompressed rows nor the CSR were kept. Measured 2026-10-23: 48 598 rows/s with --cache-dir and 8
precompute workers (1.28 M rows in 32 s, 13.6 M rows ~280 s), and the packed file reads back in
0.17 s. This writes what the trainer wants, once: the CSR boards and scalars, the labels, the
search targets, the per-row seed and the per-row archive position, in archive order.

The trainer (train_rollout_policy.load) reads only packs. The archives are read here and nowhere
else; that is the LEGACY side of the pair -- an archive written before packs still works, through
this tool.

  python3 scripts/pack_training_rows.py --data DIR --labels a,b
  python3 scripts/pack_training_rows.py --data DIR --labels a,b --verify DIR/packs/x.npz

Options that change the rows (--wait-keep, --obs, --cells, --aux-horizon, --packet-head, --context,
--label-weight) are part of the pack key: the trainer looks for the pack of exactly its own options
and prints the command to run when it is missing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import random  # noqa: E402

import torch  # noqa: E402

from pvz_frame_store import (CELL_CANDIDATES, cache_aux, cache_candidates, cache_episode,  # noqa: E402
                             cache_identity_scene, cache_packets, candidate_cells, frame_bytes,
                             frame_paths, read_frames, row_damage_labels, target_values)
from pvz_rollout_policy import (LAWN_COLS, PACKETS, features, identity_scalars,  # noqa: E402
                                packet_features)
from train_rollout_policy import (CandidateStore, CompactBoards, PacketStore, ValueStore,  # noqa: E402
                                  _episode_context, constant_plane)

from pvz_training_pack import PACK_VERSION, archive_fingerprint, pack_options, pack_path  # noqa: E402


def _store_arrays(store) -> dict:
    planes = store.planes
    return {"indices": store.indices, "values": store.values, "offsets": store.offsets,
            "plane_of": store.plane_of if store.plane_of is not None else np.zeros(0, np.int32),
            "plane_cells": np.concatenate([p[0] for p in planes]) if planes else np.zeros(0, np.int32),
            "plane_values": (np.concatenate([p[1] for p in planes]) if planes
                             else np.zeros(0, store.dtype)),
            "plane_offsets": np.concatenate([[0], np.cumsum([len(p[0]) for p in planes])]).astype(np.int64),
            "shape": np.asarray(store.shape[1:], np.int64), "dtype": np.asarray(store.dtype.str)}
def build_from_archives(data: Path, labels: set[str], wait_keep: float = 1.0,
                        label_weight: dict[str, float] | None = None, cache_dir: Path | None = None,
                        with_context: bool = False, obs: str = "v1", precompute_workers: int = 1,
                        cells: bool = False, aux_horizon: int = 0, row_observer=None,
                        packet_head: str = 'slot', read_cache_dir: Path | None = None):
    """Rows: frame i-1 -> action at frame i. With ``wait_keep`` < 1 only that fraction of the wait rows
    (~94% of all rows) is kept, each weighted 1 / wait_keep, so more games fit in memory.
    ``label_weight`` multiplies the weight of every row of a label (a small, newer teacher among many older games).

    This is the only place that reads per-game archives. Training reads the pack this writes
    (train_rollout_policy.load), so the decompression and the CSR build happen once per label set
    instead of once per training run (task33).

    Rows are stored compactly (task32): bit-packed keep masks, uint8/int16 labels, a split packet block, uint8
    scalar indices and per-archive constant planes taken out of the board CSR. Every value a batch sees is
    bit-identical to the dense arrays this replaced."""
    paths = [p for p in frame_paths(data) if p.parent.name in labels]
    # Decide the kept rows first and preallocate exactly: a list of per-row tensors plus torch.stack needs
    # twice the memory, which on 887k rows (~10 GB of boards) took down the 23 GB WSL VM (2026-10-15).
    rng = random.Random(0)
    keep = {}
    cached = {}
    cell_cached = {}
    aux_cached = {}
    packet_cached = {}
    scene_cached = {}
    if cache_dir is not None and precompute_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        from functools import partial
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=precompute_workers, mp_context=mp.get_context("spawn")) as pool:
            cached = dict(zip(paths, pool.map(partial(cache_episode, directory=cache_dir, obs=obs,
                                                    read_directory=read_cache_dir), paths, chunksize=8)))
            if cells:
                cell_cached = dict(zip(paths, pool.map(partial(cache_candidates, directory=cache_dir), paths, chunksize=8)))
            if aux_horizon:
                aux_cached = dict(zip(paths, pool.map(partial(cache_aux, directory=cache_dir, horizon=aux_horizon), paths, chunksize=8)))
            if packet_head == 'identity':
                packet_cached = dict(zip(paths, pool.map(partial(cache_packets, directory=cache_dir), paths, chunksize=8)))
                scene_cached = dict(zip(paths, pool.map(partial(cache_identity_scene, directory=cache_dir), paths, chunksize=8)))
    for path in paths:
        mask = []
        if cache_dir is not None:
            if path not in cached:
                cached[path] = cache_episode(path, cache_dir, obs, read_directory=read_cache_dir)
            with np.load(cached[path], allow_pickle=False) as stored:
                mask = [bool(hint) or rng.random() < wait_keep for hint in stored["hints"]]
            # Bit-packed: a python list of bools is 8 bytes a frame, and at wait_keep .1 there are ~3.5 frames
            # behind every kept row (task32).
            keep[path] = np.packbits(mask)
            continue
        for line in frame_bytes(path).decode().splitlines():
            # The target is the last field: the DAgger label if there is one, else the executed action.
            tail = line[max(line.rfind('"label":'), line.rfind('"action":')):]
            mask.append('"type":"plant"' in tail or '"targets":' in line or rng.random() < wait_keep)
        keep[path] = np.packbits(mask)
    capacity = sum(int(np.unpackbits(packed).sum()) for packed in keep.values())
    boards = scalars = None
    # Per-row labels go straight into preallocated arrays: as Python lists (8 floats a row for `values`) they cost
    # ~2 GB of objects on 6.5M rows and the freed arenas stayed in the training process (13 GB RSS, 2026-10-19).
    plant, weight = np.empty(capacity, np.uint8), np.empty(capacity, np.float32)
    packet = np.empty(capacity, np.int16)
    cell = np.empty(capacity, np.int16)
    values = ValueStore()
    seed_of = []
    # Where each row came from: the pack has to be able to answer a per-archive row plan.
    source_archive = np.zeros(capacity, np.int32)
    source_row = np.zeros(capacity, np.int32)
    cands = CandidateStore() if cells else None
    auxiliary = np.empty((capacity, 6), dtype=np.float32) if aux_horizon else None
    packets = PacketStore(capacity, PACKETS) if packet_head == 'identity' else None
    n = 0
    for archive_index, path in enumerate(paths):
        seed = int(path.parent.parent.name.split("_")[-1])
        scale = (label_weight or {}).get(path.parent.name, 1.0)
        if path in cached:
            with np.load(cached[path], allow_pickle=False) as stored:
                mask = np.unpackbits(keep[path])[:len(stored["hints"])].astype(bool)
                selected = np.flatnonzero(mask[stored["rows"]])
                count = len(selected)
                if packets is not None:
                    if path not in packet_cached:
                        packet_cached[path] = cache_packets(path, cache_dir)
                    with np.load(packet_cached[path], allow_pickle=False) as side:
                        assert np.array_equal(side['rows'], stored['rows']), path
                        packets.set(n, n + count, side['packets'][selected])
                if row_observer is not None:
                    row_observer(path, stored["rows"][selected])
                rows_scalars = stored["scalars"][selected]
                if packets is not None:
                    if path not in scene_cached:
                        scene_cached[path] = cache_identity_scene(path, cache_dir)
                    with np.load(scene_cached[path], allow_pickle=False) as side:
                        assert np.array_equal(side['rows'], stored['rows']), path
                        rows_scalars[:, 4] = side['wave_scalar'][selected]
                if with_context:
                    ctx = _episode_context(path)
                    context_rows = np.broadcast_to(ctx, (count, len(ctx)))
                    rows_scalars = np.concatenate([rows_scalars[:, :115], context_rows, rows_scalars[:, 115:]], axis=1)
                rows_boards = stored["boards"]
                if boards is None:
                    boards = CompactBoards(rows_boards.shape[1:])
                    scalars = CompactBoards(rows_scalars.shape[1:], np.float32)
                if count:
                    plane, covered = constant_plane(rows_boards[selected])
                    boards.append(rows_boards[selected], covered=covered, plane=plane)
                    scalars.append(rows_scalars)
                types = stored["plant"][selected]
                plant[n:n + count] = types
                packet[n:n + count] = stored["packet"][selected]
                if packets is not None:
                    planted_indices = np.flatnonzero(types > .5)
                    if len(planted_indices):
                        planted_rows = torch.from_numpy(n + planted_indices)
                        planted_slots = torch.from_numpy(packet[n + planted_indices]).long()
                        target = packets.kinds(planted_rows, planted_slots)
                        if (target < 0).any():
                            raise ValueError(f'identity labelled plant has an empty packet slot: {path}')
                cell[n:n + count] = stored["cell"][selected]
                vals = stored["values"][selected]
                if cells:
                    if path not in cell_cached:
                        cell_cached[path] = cache_candidates(path, cache_dir)
                    with np.load(cell_cached[path], allow_pickle=False) as side:
                        assert np.array_equal(side["rows"], stored["rows"]), path
                        for row in side["cands"][selected]:
                            cands.append(row)
                for row in vals:
                    values.append(row)
                if aux_horizon:
                    if path not in aux_cached:
                        aux_cached[path] = cache_aux(path, cache_dir, aux_horizon)
                    with np.load(aux_cached[path], allow_pickle=False) as side:
                        assert np.array_equal(side["rows"], stored["rows"]), path
                        auxiliary[n:n + count] = side["aux"][selected]
                weight[n:n + count] = np.where((types != 0) | (~np.isnan(vals)).any(1), 1.0, 1.0 / wait_keep) * scale
                seed_of.extend([seed] * count)
                source_archive[n:n + count] = archive_index
                source_row[n:n + count] = stored["rows"][selected]
                n += count
            continue
        frames = read_frames(path)
        future_damage = row_damage_labels(frames, aux_horizon) if aux_horizon else None
        ctx = _episode_context(path) if with_context else None
        if obs != "v1":
            from pvz_student_observation import StudentHistory
            history = StudentHistory(obs)
        kept = np.unpackbits(keep[path])[:len(frames)].astype(bool)
        # Collected per archive, not per row, so this path hoists the same constant plane the cached
        # path does: a pack must not depend on whether a feature cache happened to exist.
        archive_boards, archive_scalars = [], []
        for i in range(1, len(frames)):
            if obs != "v1":
                b, s = history.features(frames[i - 1], frames[i - 2] if i >= 2 else None, ctx)
            action = frames[i].get("label") or frames[i].get("action")
            if not action or frames[i].get("action_rejected") or not kept[i]:
                continue
            if obs == "v1":
                b, s = features(frames[i - 1], frames[i - 2] if i >= 2 else None, ctx)
            if packets is not None:
                s = identity_scalars(frames[i - 1], s)
            if boards is None:
                boards = CompactBoards(b.shape)
                scalars = CompactBoards(s.shape, np.float32)
            archive_boards.append(b.numpy())
            archive_scalars.append(s.numpy())
            if packets is not None:
                packets.set(n, n + 1, packet_features(frames[i - 1])[None])
            n += 1
            seed_of.append(seed)
            source_archive[n - 1] = archive_index
            source_row[n - 1] = i
            is_plant = action["type"] == "plant"
            plant[n - 1] = float(is_plant)
            weight[n - 1] = scale * (1.0 if is_plant or frames[i].get("targets") else 1.0 / wait_keep)
            packet[n - 1] = action["packet"] if is_plant else -1
            if packets is not None and is_plant and packets.columns[0][n - 1, packet[n - 1]] < 0:
                raise ValueError(f'identity labelled plant has an empty packet slot: {path}, row {i}')
            cell[n - 1] = action["row"] * LAWN_COLS + action["col"] if is_plant else -1
            values.append(target_values(frames[i].get("targets")))
            if aux_horizon:
                auxiliary[n - 1] = future_damage[i - 1]
            if cells:
                cands.append(candidate_cells(frames[i].get("targets")))
        if archive_boards:
            stacked = np.stack(archive_boards)
            plane, covered = constant_plane(stacked)
            boards.append(stacked, covered=covered, plane=plane)
            scalars.append(np.stack(archive_scalars))
    result = (boards.finish(), scalars.finish(), *(torch.from_numpy(a[:n]) for a in (plant, packet, cell, weight)),
              seed_of, values.finish(),
              None if cands is None else cands.finish())
    if aux_horizon:
        result = (*result, torch.from_numpy(auxiliary[:n]))
    if packets is not None:
        result = (*result, packets.finish(n))
    sources = [str(path.relative_to(data)) for path in paths]
    return result, sources, source_archive[:n], source_row[:n]




def pack(data: Path, options: dict, cache_dir: Path | None = None, precompute_workers: int = 1,
         read_cache_dir: Path | None = None, out: Path | None = None, quiet: bool = False) -> Path:
    """Build the pack for these options from the archives under ``data``; returns its path."""
    started = time.perf_counter()
    loaded, sources, source_archive, source_row = build_from_archives(
                                 data, set(options["labels"]), options["wait_keep"],
                                 options["label_weight"], cache_dir, options["with_context"],
                                 options["obs"], precompute_workers, cells=options["cells"],
                                 aux_horizon=options["aux_horizon"], packet_head=options["packet_head"],
                                 read_cache_dir=read_cache_dir)
    boards, scalars, plant, packet, cell, weight, seed_of, values, cands = loaded[:9]
    auxiliary = loaded[9] if options["aux_horizon"] else None
    packets = loaded[9 + (auxiliary is not None)] if options["packet_head"] == "identity" else None
    build_seconds = time.perf_counter() - started
    arrays = {"meta": np.asarray(json.dumps(options)), "rows": np.int64(len(plant))}
    arrays.update({"board_" + key: value for key, value in _store_arrays(boards).items()})
    arrays.update({"scalar_" + key: value for key, value in _store_arrays(scalars).items()})
    arrays.update({"plant": np.asarray(plant), "packet": np.asarray(packet), "cell": np.asarray(cell),
                   "weight": np.asarray(weight), "seed": np.asarray(seed_of, np.int32)})
    arrays.update({"source": np.asarray(sources), "source_archive": source_archive,
                   "source_row": source_row})
    arrays.update({"value_counts": values.counts, "value_slots": values.slots,
                   "value_values": values.values})
    if cands is not None:
        arrays.update({"cand_counts": cands.counts, "cand_entries": cands.entries})
    if auxiliary is not None:
        arrays["aux"] = np.asarray(auxiliary)
    if packets is not None:
        for index, column in enumerate(packets.columns):
            arrays["packet_col%d" % index] = np.asarray(column)
    target = Path(out) if out is not None else pack_path(data, options)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_started = time.perf_counter()
    np.savez(target, **arrays)
    if not quiet:
        print(json.dumps({"pack": str(target), "rows": int(len(plant)), "labels": options["labels"],
                          "build_s": round(build_seconds, 1),
                          "write_s": round(time.perf_counter() - write_started, 1),
                          "bytes": target.stat().st_size}), flush=True)
    return target


def prune(data: Path, labels: set[str], keep: Path) -> list[Path]:
    """Delete the packs of a label subset that ``keep`` supersedes; returns what was removed.

    The collector packs the growing label set after every round (task33), and each round's pack
    already holds every row of the previous one -- at 13.6 M rows that is ~3.5 GB a round. A pack of
    a set that is *not* a subset (another training corpus in the same directory) is left alone.
    """
    removed = []
    for path in sorted((data / "packs").glob("*.npz")):
        if path == keep:
            continue
        try:
            with np.load(path, allow_pickle=False) as stored:
                meta = json.loads(str(stored["meta"]))
        except Exception:          # a pack another process is still writing; try again next round
            continue
        if set(meta["labels"]) <= labels:
            path.unlink()
            removed.append(path)
    return removed


def packed_load(data, labels, wait_keep: float = 1.0, label_weight=None, with_context: bool = False,
                obs: str = "v1", cells: bool = False, aux_horizon: int = 0, packet_head: str = "slot",
                row_observer=None, cache_dir=None, rebuild: bool = False):
    """Build the pack if this exact input has none yet, then read it the way the trainer does.

    rebuild forces the build even when a pack exists: a changed archive of the same size would
    otherwise keep its old pack, and that is exactly the case a test wants to look at.
    """
    from train_rollout_policy import read_pack
    options = pack_options(labels, wait_keep, obs, cells, aux_horizon, packet_head, with_context, label_weight,
                           sources=archive_fingerprint(data, set(labels)))
    path = pack_path(data, options)
    if rebuild or not path.is_file():
        pack(data, options, cache_dir, 1, None, path, quiet=True)
    return read_pack(path, data, row_observer)


def verify(data: Path, options: dict, out: Path) -> None:
    """Compare every array of the pack with a fresh build from the archives."""
    loaded, _sources, source_archive, source_row = build_from_archives(
                                 data, set(options["labels"]), options["wait_keep"], options["label_weight"],
                                 None, options["with_context"], options["obs"], 1, cells=options["cells"],
                                 aux_horizon=options["aux_horizon"], packet_head=options["packet_head"])
    boards, scalars, plant, packet, cell, weight, seed_of, values, cands = loaded[:9]
    with np.load(out, allow_pickle=False) as stored:
        read = {name: stored[name] for name in stored.files}
    same = {
        "meta": json.loads(str(read["meta"])) == options,
        "rows": int(read["rows"]) == len(plant),
        "board": _same_store(read, "board", boards),
        "scalar": _same_store(read, "scalar", scalars),
        "plant": np.array_equal(read["plant"], np.asarray(plant)),
        "packet": np.array_equal(read["packet"], np.asarray(packet)),
        "cell": np.array_equal(read["cell"], np.asarray(cell)),
        "weight": np.array_equal(read["weight"], np.asarray(weight)),
        "seed": np.array_equal(read["seed"], np.asarray(seed_of, np.int32)),
        "source": np.array_equal(read["source_archive"], source_archive),
        "values": (np.array_equal(read["value_counts"], values.counts)
                   and np.array_equal(read["value_slots"], values.slots)
                   and np.array_equal(read["value_values"], values.values)),
    }
    if cands is not None:
        same["cands"] = (np.array_equal(read["cand_counts"], cands.counts)
                         and np.array_equal(read["cand_entries"], cands.entries))
    for name, ok in same.items():
        print("  %-8s %s" % (name, "same" if ok else "DIFFERENT"), flush=True)
    if not all(same.values()):
        raise SystemExit("pack does not reproduce the archives")


def _same_store(read: dict, prefix: str, store) -> bool:
    return all(np.array_equal(read[prefix + "_" + key], value)
               for key, value in _store_arrays(store).items())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True, help="frames root (seed_*/<label>_<arm>/ dirs)")
    p.add_argument("--labels", required=True, help="comma separated <label> directory names")
    p.add_argument("--out", type=Path, help="pack path (default: <data>/packs/<key>.npz)")
    p.add_argument("--wait-keep", type=float, default=1.0, help="fraction of wait rows kept (reweighted)")
    p.add_argument("--obs", choices=["v1", "v2lite", "v2"], default="v1")
    p.add_argument("--cells", action="store_true", help="store the search cell candidates")
    p.add_argument("--aux-horizon", type=int, default=0, help="row damage labels over this many decisions")
    p.add_argument("--packet-head", choices=["slot", "identity"], default="slot")
    p.add_argument("--context", action="store_true", help="append the task context to the scalars")
    p.add_argument("--label-weight", default="", help="label=factor,... row-weight multipliers per label")
    p.add_argument("--cache-dir", type=Path, help="per-episode feature cache (default: <data>/.feature_cache)")
    p.add_argument("--precompute-workers", type=int, default=1)
    p.add_argument("--verify", action="store_true", help="compare the pack with a fresh build")
    p.add_argument("--prune", action="store_true",
                   help="delete the packs of a label subset this one supersedes")
    args = p.parse_args()
    label_weight = {k: float(v) for k, v in (kv.split("=") for kv in args.label_weight.split(",") if kv)}
    options = pack_options(args.labels.split(","), args.wait_keep, args.obs, args.cells, args.aux_horizon,
                           args.packet_head, args.context, label_weight,
                           sources=archive_fingerprint(args.data, set(args.labels.split(","))))
    if args.verify:
        if args.out is None:
            p.error("--verify needs --out")
        verify(args.data, options, args.out)
        return
    cache_dir = args.cache_dir or (args.data / ".feature_cache")
    target = pack(args.data, options, cache_dir, args.precompute_workers, None, args.out)
    if args.prune:
        removed = prune(args.data, set(options["labels"]), target)
        print(json.dumps({"pruned": [str(path) for path in removed]}), flush=True)


if __name__ == "__main__":
    main()
