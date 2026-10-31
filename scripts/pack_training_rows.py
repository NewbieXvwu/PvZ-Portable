"""Immutable per-label training packs with an atomic corpus index (task49).

Only new/changed label segments are built; row options and source fingerprints identify
segments. --precompute-workers builds labels in parallel under the memory cap. The trainer
reads the small index and restores global archive order from compact segments, without
reading game frames. Version 2 uses independent seed=0 wait-sampling streams per label;
adding another label never changes an existing segment's rows. This intentionally changes
v1's cross-label sampling; v1 packs are never reused as v2.

pack_monolithic is the full-build reference for verification, with identical row semantics.
The production CLI always writes segments; there is no alternate packing mode.

  python3 /absolute/repo/scripts/pack_training_rows.py --data /absolute/frames --labels a,b
  python3 /absolute/repo/scripts/pack_training_rows.py --data /absolute/frames --labels a,b --out /absolute/pack.npz
"""
from __future__ import annotations

import argparse
import os
import tempfile
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import random  # noqa: E402

import torch  # noqa: E402

from pvz_frame_store import (CELL_CANDIDATES, archive_seed, cache_aux, cache_candidates, cache_episode,  # noqa: E402
                             cache_identity_scene, cache_packets, candidate_cells, frame_bytes,
                             frame_paths, read_frames, read_episode_cache, row_damage_labels, target_values)
from pvz_rollout_policy import (LAWN_COLS, PACKETS, features, identity_scalars,  # noqa: E402
                                packet_features)
from train_rollout_policy import (CandidateStore, CompactBoards, PacketStore, ValueStore,  # noqa: E402
                                  _episode_context, _member_array, _member_header, constant_plane)

from pvz_training_pack import (PACK_VERSION, archive_fingerprint, fingerprint_paths, pack_options,
                               pack_path, segment_options, segment_path, training_rows, keep_archive)  # noqa: E402


def _store_arrays(store) -> dict:
    planes = store.planes
    plane_of = (store.plane_of if store.plane_of is not None
                else np.full(store.shape[0], -1, np.int32))
    if plane_of.shape != (store.shape[0],):
        raise ValueError('packed plane ids must have exactly one entry per row')
    return {"indices": store.indices, "values": store.values, "offsets": store.offsets,
            "plane_of": plane_of,
            "plane_cells": np.concatenate([p[0] for p in planes]) if planes else np.zeros(0, np.int32),
            "plane_values": (np.concatenate([p[1] for p in planes]) if planes
                             else np.zeros(0, store.dtype)),
            "plane_offsets": np.concatenate([[0], np.cumsum([len(p[0]) for p in planes])]).astype(np.int64),
            "shape": np.asarray(store.shape[1:], np.int64), "dtype": np.asarray(store.dtype.str)}
def build_from_archives(data: Path, labels: set[str], wait_keep: float = 1.0,
                        label_weight: dict[str, float] | None = None, cache_dir: Path | None = None,
                        with_context: bool = False, obs: str = "v1", precompute_workers: int = 1,
                        cells: bool = False, aux_horizon: int = 0, row_observer=None,
                        packet_head: str = 'slot', read_cache_dir: Path | None = None, archive_paths=None,
                        label_keep=None):
    """Rows: frame i-1 -> action at frame i. With ``wait_keep`` < 1 only that fraction of the wait rows
    (~94% of all rows) is kept, each weighted 1 / wait_keep, so more games fit in memory.
    ``label_weight`` multiplies the weight of every row of a label (a small, newer teacher among many older games).

    This is the only place that reads per-game archives. Training reads the pack this writes
    (train_rollout_policy.load), so the decompression and the CSR build happen once per label set
    instead of once per training run (task33).

    Rows are stored compactly (task32): bit-packed keep masks, uint8/int16 labels, a split packet block, uint8
    scalar indices and per-archive constant planes taken out of the board CSR. Every value a batch sees is
    bit-identical to the dense arrays this replaced."""
    paths = ([p for p in frame_paths(data) if p.parent.name in labels]
             if archive_paths is None else list(archive_paths))
    if label_keep:
        paths = [path for path in paths if keep_archive(data, path, label_keep)]
    if not paths:
        raise ValueError('no archives retained for training')
    # Decide the kept rows first and preallocate exactly: a list of per-row tensors plus torch.stack needs
    # twice the memory, which on 887k rows (~10 GB of boards) took down the 23 GB WSL VM (2026-10-15).
    # Independent streams make row selection invariant to other labels entering the corpus.
    # Version 2 deliberately differs from v1 across labels; single-label selection is unchanged.
    label_rng = {}
    keep = {}
    cached = {}
    cell_cached = {}
    aux_cached = {}
    packet_cached = {}
    scene_cached = {}
    explicit_rows = {path: training_rows(path) for path in paths}

    def selected_mask(path, mask):
        selected = explicit_rows[path]
        if selected is None:
            return mask
        if selected and selected[-1] >= len(mask):
            raise ValueError(f'{path}: training_rows outside archive')
        wanted = set(selected)
        return [i in wanted for i in range(len(mask))]
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
        rng = label_rng.setdefault(path.parent.name, random.Random(0))
        mask = []
        if cache_dir is not None:
            if path not in cached:
                cached[path] = cache_episode(path, cache_dir, obs, read_directory=read_cache_dir)
            with read_episode_cache(cached[path]) as stored:
                mask = [bool(hint) or rng.random() < wait_keep for hint in stored["hints"]]
            # Bit-packed: a python list of bools is 8 bytes a frame, and at wait_keep .1 there are ~3.5 frames
            # behind every kept row (task32).
            keep[path] = np.packbits(selected_mask(path, mask))
            continue
        for line in frame_bytes(path).decode().splitlines():
            # The target is the last field: the DAgger label if there is one, else the executed action.
            tail = line[max(line.rfind('"label":'), line.rfind('"action":')):]
            mask.append('"type":"plant"' in tail or '"targets":' in line or rng.random() < wait_keep)
        keep[path] = np.packbits(selected_mask(path, mask))
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
        seed = archive_seed(path.parent.parent.name)
        scale = (label_weight or {}).get(path.parent.name, 1.0)
        retention = wait_keep if explicit_rows[path] is None else 1.0
        if path in cached:
            with read_episode_cache(cached[path]) as stored:
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
                if explicit_rows[path] is not None and count != len(explicit_rows[path]):
                    raise ValueError(f'{path}: requested training_rows lack valid labels')
                weight[n:n + count] = np.where((types != 0) | (~np.isnan(vals)).any(1), 1.0, 1.0 / retention) * scale
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
            weight[n - 1] = scale * (1.0 if is_plant or frames[i].get("targets") else 1.0 / retention)
            packet[n - 1] = action["packet"] if is_plant else -1
            if packets is not None and is_plant and packets.columns[0][n - 1, packet[n - 1]] < 0:
                raise ValueError(f'identity labelled plant has an empty packet slot: {path}, row {i}')
            cell[n - 1] = action["row"] * LAWN_COLS + action["col"] if is_plant else -1
            values.append(target_values(frames[i].get("targets")))
            if aux_horizon:
                auxiliary[n - 1] = future_damage[i - 1]
            if cells:
                cands.append(candidate_cells(frames[i].get("targets")))
        if explicit_rows[path] is not None and len(archive_boards) != len(explicit_rows[path]):
            raise ValueError(f'{path}: requested training_rows lack valid labels')
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




def pack_monolithic(data: Path, options: dict, cache_dir: Path | None = None, precompute_workers: int = 1,
         read_cache_dir: Path | None = None, out: Path | None = None, quiet: bool = False, archive_paths=None) -> Path:
    """Build the pack for these options from the archives under ``data``; returns its path."""
    started = time.perf_counter()
    if archive_paths is None and options.get('llm_demo_import'):
        from pvz_training_pack import training_paths
        archive_paths = training_paths(data, options['labels'], options['llm_demo_import'],
                                       options['llm_demo_results'])
    loaded, sources, source_archive, source_row = build_from_archives(
                                 data, set(options["labels"]), options["wait_keep"],
                                 options["label_weight"], cache_dir, options["with_context"],
                                 options["obs"], precompute_workers, cells=options["cells"],
                                 aux_horizon=options["aux_horizon"], packet_head=options["packet_head"],
                                 read_cache_dir=read_cache_dir, archive_paths=archive_paths,
                                 label_keep=options.get('label_keep'))
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
    atomic_savez(target, arrays)
    if not quiet:
        print(json.dumps({"pack": str(target), "rows": int(len(plant)), "labels": options["labels"],
                          "build_s": round(build_seconds, 1),
                          "write_s": round(time.perf_counter() - write_started, 1),
                          "bytes": target.stat().st_size}), flush=True)
    return target


def atomic_savez(target: Path, arrays: dict) -> None:
    """Readers see a complete generation, including when two packers build the same label."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=target.parent, suffix='.building')
    try:
        with os.fdopen(fd, 'wb') as handle:
            np.savez(handle, **arrays)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _build_segment(job):
    data, options, cache, read_cache, paths = job
    torch.set_num_threads(1)
    target = segment_path(data, options)
    return pack_monolithic(data, options, cache, 1, read_cache, target, True, paths)


def pack_memory_limit(available: int, jobs: list[tuple]) -> int:
    """并发建造者的内存上限：每人 2 GiB（Python/torch 与解压缓冲）加该批最大的标签原始档案
    体积，并给整 机留 6 GiB（curriculum_queue.build_plan 用同一单建造者估算）。"""
    heavy = max((sum(path.stat().st_size for path in paths) for *_, paths in jobs), default=0)
    return max(1, (available - 6 * 1024**3) // (2 * 1024**3 + heavy))


def segment_plane_shapes_valid(path: Path, rows: int) -> bool:
    """Do not reuse the old short-archive bug; inspect headers, not large CSR data.

    Empty plane ids remain valid for legacy stores with no planes. New writers
    emit an explicit -1 per row even in that case. Sampling/version stays v2.
    """
    with zipfile.ZipFile(path) as archive:
        for prefix in ('board', 'scalar'):
            shape, _ = _member_header(archive, prefix + '_plane_of')
            offsets, _ = _member_header(archive, prefix + '_plane_offsets')
            _, plane_dtype = _member_header(archive, prefix + '_plane_values')
            expected_dtype = np.dtype(str(_member_array(archive, prefix + '_dtype')))
            if plane_dtype != expected_dtype:
                return False
            if shape != (rows,) and not (shape == (0,) and offsets == (1,)):
                return False
    return True


def pack(data: Path, options: dict, cache_dir: Path | None = None, precompute_workers: int = 1,
         read_cache_dir: Path | None = None, out: Path | None = None, quiet: bool = False,
         rebuild: bool = False) -> Path:
    """Publish an index of immutable per-label segments; only missing segments are built.

    The whole archive catalog is scanned once. Parallelism is across labels, never a nested
    feature pool. A training snapshot pins its segment names, so later rounds cannot change it.
    """
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    started = time.perf_counter()
    catalog = {label: [] for label in options['labels']}
    from pvz_training_pack import training_paths
    for path in training_paths(data, options['labels'], options.get('llm_demo_import'),
                               options.get('llm_demo_results', [1])):
        catalog.setdefault(path.parent.name, []).append(path)
    jobs, segments = [], []
    for label, paths in sorted(catalog.items()):
        if not paths:
            raise ValueError(f'no archives for label {label!r} in {data}')
        contract = segment_options(options, label, fingerprint_paths(data, paths))
        if contract.get('label_keep') and not any(keep_archive(data, path, contract['label_keep']) for path in paths):
            continue
        target = segment_path(data, contract)
        segments.append(target)
        valid = False
        if target.is_file() and not rebuild:
            with np.load(target, allow_pickle=False) as stored:
                valid = (json.loads(str(stored['meta'])) == contract
                         and segment_plane_shapes_valid(target, int(stored['rows'])))
        if not valid:
            jobs.append((data, contract, cache_dir, read_cache_dir, paths))
    if not segments:
        raise ValueError('no archives retained for training')
    # Each label worker includes Python/torch and a label's compact rows; its peak grows with
    # the label's raw archives. Same 6 GiB reserve as collection (pack_memory_limit docstring).
    try:
        available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))
        memory_limit = pack_memory_limit(available, jobs)
    except (OSError, StopIteration):
        memory_limit = 1
    workers = max(1, min(precompute_workers, len(jobs) or 1, memory_limit))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
            list(pool.map(_build_segment, jobs))
    else:
        for job in jobs:
            _build_segment(job)
    target = Path(out) if out is not None else pack_path(data, options)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    for segment in segments:
        with np.load(segment, allow_pickle=False) as stored:
            rows += int(stored['rows'])
    atomic_savez(target, {'meta': np.asarray(json.dumps(options)), 'rows': np.int64(rows),
                          'segments': np.asarray([os.path.relpath(path, target.parent) for path in segments])})
    if not quiet:
        print(json.dumps({'pack': str(target), 'rows': rows, 'built': len(jobs),
                          'reused': len(segments) - len(jobs), 'workers': workers,
                          'seconds': round(time.perf_counter() - started, 3)}), flush=True)
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
        # A different recipe or a v1 corpus is not superseded by this index.
        with np.load(keep, allow_pickle=False) as stored:
            current = json.loads(str(stored['meta']))
        keys = set(current) - {'labels', 'sources', 'label_weight'}
        if (set(meta['labels']) <= labels and all(meta.get(k) == current[k] for k in keys)
                and all(meta['label_weight'].get(label, 1.) == current['label_weight'].get(label, 1.)
                        for label in meta['labels'])):
            path.unlink()
            removed.append(path)
    return removed


def packed_load(data, labels, wait_keep: float = 1.0, label_weight=None, with_context: bool = False,
                obs: str = "v1", cells: bool = False, aux_horizon: int = 0, packet_head: str = "slot",
                row_observer=None, cache_dir=None, rebuild: bool = False, label_keep=None,
                llm_demo_import=None, llm_demo_results=(1,)):
    """Build the pack if this exact input has none yet, then read it the way the trainer does.

    rebuild forces the build even when a pack exists: a changed archive of the same size would
    otherwise keep its old pack, and that is exactly the case a test wants to look at.
    """
    from train_rollout_policy import read_pack
    options = pack_options(labels, wait_keep, obs, cells, aux_horizon, packet_head, with_context, label_weight,
                           sources=archive_fingerprint(data, set(labels), llm_demo_import, llm_demo_results),
                           label_keep=label_keep, llm_demo_import=llm_demo_import, llm_demo_results=llm_demo_results)
    path = pack_path(data, options)
    if rebuild or not path.is_file():
        pack(data, options, cache_dir, 1, None, path, quiet=True, rebuild=rebuild)
    return read_pack(path, data, row_observer)


def verify(data: Path, options: dict, out: Path) -> None:
    """Compare every array of the pack with a fresh build from the archives."""
    loaded, _sources, source_archive, source_row = build_from_archives(
                                 data, set(options["labels"]), options["wait_keep"], options["label_weight"],
                                 None, options["with_context"], options["obs"], 1, cells=options["cells"],
                                 aux_horizon=options["aux_horizon"], packet_head=options["packet_head"],
                                 label_keep=options.get('label_keep'))
    from train_rollout_policy import read_pack
    observed = []
    actual = read_pack(out, data, lambda path, rows: observed.append((str(path.relative_to(data)), rows)))
    assert_payload_equal(loaded, actual)
    expected = []
    for index, source in enumerate(_sources):
        rows = source_row[source_archive == index]
        if len(rows):
            expected.append((source, rows))
    if len(expected) != len(observed) or any(a != b or not np.array_equal(x, y)
                                              for (a, x), (b, y) in zip(expected, observed)):
        raise AssertionError('source archive / row order differs')
    print('all tensors and source rows bitwise identical', flush=True)


def assert_payload_equal(expected, actual, batch_size=64):
    """Compare dense batch bytes (including signed zeros and NaNs), not CSR layout."""
    if len(expected) != len(actual):
        raise AssertionError('payload columns differ')
    rows = len(expected[2])
    for column, (left, right) in enumerate(zip(expected, actual)):
        if left is None or right is None:
            assert left is right, column
            continue
        for start in range(0, rows, batch_size):
            idx = torch.arange(start, min(start + batch_size, rows))
            def block(value):
                if isinstance(value, CompactBoards):
                    return value.batch(idx, 'cpu').numpy()
                if isinstance(value, (ValueStore, CandidateStore, PacketStore)):
                    return value[idx].numpy()
                return np.asarray(value)[idx.numpy()]
            a, b = block(left), block(right)
            if column == 6:  # the archive builder returns Python seed ints; packs store int32
                a, b = a.astype(np.int32), b.astype(np.int32)
            if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
                raise AssertionError(f'column {column}, rows {start}:{start + batch_size} differ')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True, help="frames root (seed_*/<label>_<arm>/ dirs)")
    p.add_argument("--labels", required=True, help="comma separated <label> directory names")
    p.add_argument("--out", type=Path, help="pack path (default: <data>/packs/<key>.npz)")
    p.add_argument("--wait-keep", type=float, default=1.0, help="fraction of wait rows kept (reweighted)")
    p.add_argument("--obs", choices=["v1", "v2lite", "v2", "v2fog", "v2boss", "v2bungee", "v2sleep",
                                     "v2timer"], default="v1")
    p.add_argument("--cells", action="store_true", help="store the search cell candidates")
    p.add_argument("--aux-horizon", type=int, default=0, help="row damage labels over this many decisions")
    p.add_argument("--packet-head", choices=["slot", "identity"], default="slot")
    p.add_argument("--context", action="store_true", help="append the task context to the scalars")
    p.add_argument("--label-weight", default="", help="label=factor,... row-weight multipliers per label")
    p.add_argument("--label-keep", type=json.loads, default={}, help='JSON {label_regex: fraction}; keep whole archives')
    p.add_argument('--llm-demo-import', type=Path, help='task105 import or preconverted GPT demo data')
    p.add_argument('--llm-demo-results', default='1')
    p.add_argument("--cache-dir", type=Path, help="per-episode feature cache (default: <data>/.feature_cache)")
    p.add_argument("--precompute-workers", type=int, default=1, help="parallel label builders (memory capped)")
    p.add_argument("--prune", action="store_true",
                   help="delete the packs of a label subset this one supersedes")
    args = p.parse_args()
    label_weight = {k: float(v) for k, v in (kv.split("=") for kv in args.label_weight.split(",") if kv)}
    results = [int(value) for value in args.llm_demo_results.split(',')]
    options = pack_options(args.labels.split(","), args.wait_keep, args.obs, args.cells, args.aux_horizon,
                           args.packet_head, args.context, label_weight,
                           sources=archive_fingerprint(args.data, set(args.labels.split(",")), args.llm_demo_import, results),
                           label_keep=args.label_keep, llm_demo_import=args.llm_demo_import, llm_demo_results=results)
    cache_dir = args.cache_dir or (args.data / ".feature_cache")
    target = pack(args.data, options, cache_dir, args.precompute_workers, None, args.out)
    if args.prune:
        removed = prune(args.data, set(options["labels"]), target)
        print(json.dumps({"pruned": [str(path) for path in removed]}), flush=True)


if __name__ == "__main__":
    main()
