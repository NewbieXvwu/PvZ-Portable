"""Train the fast rollout policy on pi's own games (see python/pvz_rollout_policy.py).

Data: episode archives whose frames carry the executed action (``frames[i]["action"]`` produced
frame i). Input is frame i-1, target is the action at frame i. Split by environment seed.

Usage:
  python /absolute/path/scripts/train_rollout_policy.py --recipe /absolute/path/recipes/dw4i.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import zipfile
import random
import sys
import time
import signal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from pvz_recipe import training_args, training_row_options, write_recipe  # noqa: E402
from pvz_metrics import Metrics  # noqa: E402
from pvz_common import OOM_SCORE_TRAINING, set_oom_score_adj  # noqa: E402
from pvz_rollout_policy import LAWN_COLS, PACKETS, RolloutNet, create_student, features, packet_features, identity_scalars  # noqa: E402
from pvz_frame_store import CELL_CANDIDATES  # noqa: E402
from pvz_training_pack import archive_fingerprint, pack_flags, pack_options, pack_path  # noqa: E402


def _episode_context(path: Path) -> np.ndarray:
    """The task context of one archive (meta.json, written by search_play or scripts/backfill_frame_context.py)."""
    from pvz_value_net import context_vector
    meta = json.loads((path.parent / "meta.json").read_text())
    if "context" not in meta:
        raise SystemExit(f"{path.parent} has no task context: run scripts/backfill_frame_context.py first")
    return context_vector(meta["context"]["multiplier"], meta["context"]["roster"])


def _spilled_array(directory: Path, name: str, dtype, shape):
    """Writable file-backed array that leaves the directory at once: the mapping keeps the pages alive,
    and the kernel can drop them under pressure instead of holding them as anonymous memory (task32)."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}_{os.getpid()}.npy"
    store = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    path.unlink()
    return store


def constant_plane(rows):
    """Cells that hold the same value in every row of one archive, and the flat mask that covers them.

    The playable-lane channel is 50 of the ~134 nonzeros a row and never changes inside a game, so hoisting
    it into one plane per archive drops a third of the CSR. Exact: the plane writes back the very float16
    values the CSR would have stored (task32).
    """
    if len(rows) < 4:
        return None, None
    flat = np.asarray(rows).reshape(len(rows), -1)
    first = flat[0]
    covered = np.all(flat == first, axis=0) & (first.view(f"u{first.dtype.itemsize}") != 0)
    positions = np.flatnonzero(covered)
    if len(positions) < 8:          # a handful of cells is not worth a plane id on every row
        return None, None
    return (positions.astype(np.int32), first[positions].copy()), covered


class CompactBoards:
    """CSR rows: element indices, values (float16 boards, float32 scalars -- lossless either way), int64 row offsets.
    Scalars are 84% zeros, so 488 bytes a row become ~120 (2026-10-19).

    Three more things are exact and together take the row from 536 to 330 bytes (task32): the index width
    follows the cell count (uint8 for a 92x6x10 board), the cells that are constant inside an archive are
    hoisted into one plane per archive plus an int32 plane id a row, and -0.0 stays a stored zero because
    nonzero() is asked of the bit pattern, not of the value.
    """

    def __init__(self, shape, dtype=np.float16):
        self.shape = (0, *shape)
        self.dtype = np.dtype(dtype)
        self.cells = int(np.prod(shape))
        self.index_dtype = np.uint8 if self.cells <= 256 else np.uint16 if self.cells <= 65536 else np.uint32
        self.nnz = 0
        self.planes, self._plane_ids, self._plane_index = [], bytearray(), {}
        self.plane_of = None
        self._plane_cells = None            # lazy flatten of planes, see _plane_tables
        self._plane_device = {}             # device -> (cells, values) resident tensors
        self.indices = bytearray()
        self.values = bytearray()
        self.offsets = bytearray(np.zeros(1, dtype=np.int64).tobytes())

    def append(self, boards, covered=None, plane=None):
        flat = np.asarray(boards, dtype=self.dtype).reshape(len(boards), -1)
        rows, columns = np.nonzero(flat.view(f"u{self.dtype.itemsize}"))  # bit test: keeps -0.0
        if covered is not None:                 # these cells are carried by the plane instead
            keep = ~covered[columns]
            rows, columns = rows[keep], columns[keep]
        counts = np.bincount(rows, minlength=len(boards))
        chunks = (columns.astype(self.index_dtype).tobytes(), flat[rows, columns].tobytes(),
                  (counts.cumsum(dtype=np.int64) + self.nnz).tobytes())
        for target, chunk in zip((self.indices, self.values, self.offsets), chunks):
            target.extend(chunk)
        if plane is not None:
            # A plane replaces CSR values, so it must use the same precision.
            # The uncached featurizer supplies float32 boards; keeping those
            # plane values would silently disagree with cached float16 rows.
            plane = (np.asarray(plane[0], dtype=np.int32), np.asarray(plane[1], dtype=self.dtype))
            if not self.planes:
                # Short leading archives may have no hoistable plane. Once a
                # later archive creates one, their already-stored CSR rows still
                # need an explicit no-plane entry to keep row identities aligned.
                self._plane_ids.extend(np.full(self.shape[0], -1, dtype=np.int32).tobytes())
            key = plane[0].tobytes() + plane[1].tobytes()
            identity = self._plane_index.setdefault(key, len(self.planes))
            if identity == len(self.planes):
                self.planes.append(plane)
            self._plane_ids.extend(np.full(len(boards), identity, dtype=np.int32).tobytes())
        elif self.planes:
            self._plane_ids.extend(np.full(len(boards), -1, dtype=np.int32).tobytes())
        self.nnz += len(columns)
        self.shape = (self.shape[0] + len(boards), *self.shape[1:])

    def finish(self):
        if self.planes and len(self._plane_ids) != self.shape[0] * np.dtype(np.int32).itemsize:
            raise ValueError('compact plane ids must cover every stored row')
        self.indices = np.frombuffer(self.indices, dtype=self.index_dtype)
        self.values = np.frombuffer(self.values, dtype=self.dtype)
        self.offsets = np.frombuffer(self.offsets, dtype=np.int64)
        self.plane_of = np.frombuffer(self._plane_ids, dtype=np.int32) if self.planes else None
        self._plane_ids = bytearray()
        return self

    @classmethod
    def from_arrays(cls, stored, prefix: str):
        """Rebuild the store from a pack: the arrays are already CSR, so nothing is recomputed (task33)."""
        store = cls(tuple(int(n) for n in stored[prefix + "_shape"]), np.dtype(str(stored[prefix + "_dtype"])))
        store.index_dtype = np.dtype(stored[prefix + "_indices"].dtype)
        store.indices = stored[prefix + "_indices"]
        store.values = stored[prefix + "_values"]
        store.offsets = stored[prefix + "_offsets"]
        plane_of = stored[prefix + "_plane_of"]
        store.plane_of = plane_of if len(plane_of) else None
        cells, values = stored[prefix + "_plane_cells"], stored[prefix + "_plane_values"]
        offsets = stored[prefix + "_plane_offsets"]
        store.planes = [(cells[offsets[index]:offsets[index + 1]], values[offsets[index]:offsets[index + 1]])
                        for index in range(len(offsets) - 1)]
        store.nnz = len(store.values)
        store.shape = (int(stored["rows"]), *store.shape[1:])
        return store

    @property
    def nbytes(self):
        return (self.indices.nbytes + self.values.nbytes + self.offsets.nbytes
                + (0 if self.plane_of is None else self.plane_of.nbytes))

    def _plane_tables(self):
        """Flatten the per-archive planes once: (cells, values, offsets) with offsets[nplanes+1]."""
        if self._plane_cells is None:
            if self.planes:
                cells = np.concatenate([p[0] for p in self.planes])
                values = np.concatenate([p[1] for p in self.planes])
                offsets = np.zeros(len(self.planes) + 1, dtype=np.int64)
                np.cumsum([len(p[0]) for p in self.planes], out=offsets[1:])
            else:
                cells, values, offsets = np.zeros(0, dtype=np.int64), np.zeros(0, dtype=self.dtype), np.zeros(1, dtype=np.int64)
            self._plane_cells, self._plane_values, self._plane_offsets = cells, values, offsets
        return self._plane_cells, self._plane_values, self._plane_offsets

    def _plane_device_tables(self, device):
        """The flattened planes resident on one device (round-2 pack: 1.5 MiB total)."""
        tables = self._plane_device.get(device)
        if tables is None:
            cells, values, _ = self._plane_tables()
            tables = (torch.from_numpy(cells).to(device).long(),
                      torch.from_numpy(values).to(device=device, dtype=torch.float32))
            self._plane_device[device] = tables
        return tables

    def batch(self, idx, device):
        selected = np.asarray(idx)
        starts = self.offsets[selected]
        counts = self.offsets[selected + 1] - starts
        ends = counts.cumsum()
        positions = np.arange(ends[-1] if len(ends) else 0) + np.repeat(starts - (ends - counts), counts)
        size = self.cells
        destinations = np.repeat(np.arange(len(selected)) * size, counts) + self.indices[positions]
        output = torch.zeros(len(selected) * size, dtype=torch.float32, device=device)
        output.index_put_((torch.from_numpy(destinations).to(device),),
                          torch.from_numpy(self.values[positions]).to(device=device, dtype=torch.float32))
        if self.plane_of is not None:
            identities = self.plane_of[selected]
            mask = identities >= 0
            if mask.any():
                # One fused scatter instead of one kernel launch per distinct plane: on the
                # round-2 pack (5155 planes, ~1400 distinct in a 2048-row batch) the loop was
                # 218 ms of a 222 ms batch build; gathers below are ~1 ms (task61).
                cells, values, offsets = self._plane_tables()
                ids = identities[mask]
                lens = offsets[ids + 1] - offsets[ids]
                total = int(lens.sum())
                row_of = np.repeat(np.flatnonzero(mask), lens)
                positions = (np.repeat(offsets[ids], lens) + np.arange(total)
                             - np.repeat(np.cumsum(lens) - lens, lens))
                plane_cells, plane_values = self._plane_device_tables(device)
                gathered = torch.from_numpy(positions).to(device)
                output.view(len(selected), size)[
                    torch.from_numpy(row_of).to(device), plane_cells[gathered]] = plane_values[gathered]
        return output.reshape(len(selected), *self.shape[1:])


class PacketStore:
    """(rows, PACKETS, 4) float32 packet features split by column: kind int8, active uint8, progress float32,
    affordable uint8. 160 bytes a row become 70, and every widening back to float32 is exact (task32)."""

    def __init__(self, capacity, slots):
        self.slots = slots
        dtypes = (np.int8, np.uint8, np.float32, np.uint8)
        self.columns = [np.empty((capacity, slots), dtype) for dtype in dtypes]

    def set(self, start, stop, block):
        block = np.asarray(block, dtype=np.float32)
        for column, values in zip(self.columns, (block[:, :, 0], block[:, :, 1], block[:, :, 2], block[:, :, 3])):
            column[start:stop] = values

    def finish(self, n):
        self.columns = [column[:n] for column in self.columns]
        return self

    def __getitem__(self, idx):
        if isinstance(idx, tuple) and len(idx) == 3:        # (row, slot, component): one column, one cell
            row, slot, component = idx
            return torch.tensor(np.asarray(self.columns[component])[row, slot])
        kind, active, progress, affordable = (torch.from_numpy(np.asarray(column)) for column in self.columns)
        return torch.stack([kind[idx].float(), active[idx].float(), progress[idx], affordable[idx].float()], dim=-1)

    def kinds(self, rows, slots):
        return torch.from_numpy(np.asarray(self.columns[0])[np.asarray(rows), np.asarray(slots)].astype(np.int64))

    @classmethod
    def from_arrays(cls, stored, rows: int):
        store = cls(rows, int(stored["packet_col0"].shape[1]))
        store.columns = [np.asarray(stored[f"packet_col{index}"]) for index in range(4)]
        return store

    @property
    def nbytes(self):
        return sum(column.nbytes for column in self.columns)


class CandidateStore:
    """The search's cell candidates, stored sparse.

    candidate_cells() ranks at most CELL_CANDIDATES entries and pads the rest with [-1, -1, nan], and
    only the rows whose frame had search targets have any at all -- but the dense block costs 12*3*2 = 72
    bytes on every row. A uint8 count plus the ranked entries is 6 bytes a candidate instead, and
    getitem() rebuilds exactly the padded block the cell soft target reads (task32).
    """

    def __init__(self):
        self.counts = bytearray()
        self.entries = bytearray()

    def append(self, block):
        block = np.asarray(block, dtype=np.float16).reshape(-1, 3)
        padding = np.isnan(block[:, 2])
        count = int(np.argmax(padding)) if padding.any() else len(block)
        self.counts.append(count)
        self.entries.extend(block[:count].tobytes())

    def finish(self):
        self.counts = np.frombuffer(self.counts, dtype=np.uint8)
        self.entries = np.frombuffer(self.entries, dtype=np.float16).reshape(-1, 3)
        self.offsets = np.zeros(len(self.counts) + 1, dtype=np.int32)
        np.cumsum(self.counts, out=self.offsets[1:])
        return self

    def __getitem__(self, idx):
        selected = np.asarray(idx)
        counts = self.counts[selected].astype(np.int64)
        output = np.full((len(selected), CELL_CANDIDATES, 3), np.nan, dtype=np.float16)
        output[:, :, :2] = -1  # match candidate_cells padding; cell indices must stay finite
        if counts.sum():
            starts = self.offsets[selected]
            positions = np.arange(counts.sum()) + np.repeat(starts - (np.cumsum(counts) - counts), counts)
            slots = (np.repeat(np.arange(len(selected)), counts) * CELL_CANDIDATES
                     + np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts))
            output.reshape(-1, 3)[slots] = self.entries[positions]
        return torch.from_numpy(output)

    @classmethod
    def from_arrays(cls, stored):
        store = cls()
        store.counts = np.asarray(stored["cand_counts"])
        store.entries = np.asarray(stored["cand_entries"]).reshape(-1, 3)
        store.offsets = np.zeros(len(store.counts) + 1, dtype=np.int32)
        np.cumsum(store.counts, out=store.offsets[1:])
        return store

    @property
    def nbytes(self):
        return self.counts.nbytes + self.entries.nbytes + self.offsets.nbytes


class ValueStore:
    """The search's per-candidate win probabilities, stored sparse.

    target_values() returns PACKETS+1 floats with NaN for the candidates the search did not compare, and
    most rows have no targets at all -- the dense block costs 44 bytes on every row. A uint8 slot plus the
    float32 value is 5 bytes a known candidate, and getitem() rebuilds exactly the NaN-padded row the soft
    target reads (task32).
    """

    def __init__(self):
        self.counts = bytearray()
        self.slots = bytearray()
        self.values = bytearray()

    def append(self, row):
        row = np.asarray(row, dtype=np.float32)
        known = np.flatnonzero(~np.isnan(row))
        self.counts.append(len(known))
        self.slots.extend(known.astype(np.uint8).tobytes())
        self.values.extend(row[known].tobytes())

    def finish(self):
        self.counts = np.frombuffer(self.counts, dtype=np.uint8)
        self.slots = np.frombuffer(self.slots, dtype=np.uint8)
        self.values = np.frombuffer(self.values, dtype=np.float32)
        self.offsets = np.zeros(len(self.counts) + 1, dtype=np.int32)
        np.cumsum(self.counts, out=self.offsets[1:])
        return self

    def __getitem__(self, idx):
        selected = np.asarray(idx)
        counts = self.counts[selected].astype(np.int64)
        output = np.full((len(selected), PACKETS + 1), np.nan, dtype=np.float32)
        if counts.sum():
            starts = self.offsets[selected]
            positions = np.arange(counts.sum()) + np.repeat(starts - (np.cumsum(counts) - counts), counts)
            output[np.repeat(np.arange(len(selected)), counts), self.slots[positions]] = self.values[positions]
        return torch.from_numpy(output)

    def spread(self, eligible=None) -> tuple[np.ndarray, np.ndarray]:
        """(rows with at least two known values, max - min of them): the spread weighting, without a dense row."""
        counts = self.counts.astype(np.int64)
        rows = np.flatnonzero(counts >= 2)
        if eligible is not None:
            rows = rows[np.asarray(eligible)[rows]]
        spreads = np.empty(len(rows), dtype=np.float32)
        for index, row in enumerate(rows):
            block = self.values[self.offsets[row]:self.offsets[row + 1]]
            spreads[index] = block.max() - block.min()
        return rows, spreads

    @classmethod
    def from_arrays(cls, stored):
        store = cls()
        store.counts = np.asarray(stored["value_counts"])
        store.slots = np.asarray(stored["value_slots"])
        store.values = np.asarray(stored["value_values"])
        store.offsets = np.zeros(len(store.counts) + 1, dtype=np.int32)
        np.cumsum(store.counts, out=store.offsets[1:])
        return store

    @property
    def nbytes(self):
        return self.counts.nbytes + self.slots.nbytes + self.values.nbytes + self.offsets.nbytes


def _pack_flags(options: dict) -> str:
    import shlex
    return shlex.join(pack_flags(options))


def load(data: Path, labels: set[str], wait_keep: float = 1.0, label_weight: dict[str, float] | None = None,
         with_context: bool = False, obs: str = "v1", cells: bool = False, aux_horizon: int = 0,
         row_observer=None, packet_head: str = "slot", source_weight: dict[str, float] | None = None):
    """The training rows of this option tuple, read from its pack (scripts/pack_training_rows.py).

    Read the corpus index and immutable label segments, merging CSR slices in archive order.
    The collector writes the index; a label set without one says which command to run.
    """
    options = pack_options(labels, wait_keep, obs, cells, aux_horizon, packet_head, with_context, label_weight,
                           sources=archive_fingerprint(data, labels))
    path = pack_path(data, options)
    if not path.is_file():
        raise SystemExit(f"no training pack for these options: {path} -- run: "
                         f"python3 scripts/pack_training_rows.py --data {data} "
                         f"--labels {','.join(sorted(labels))} {_pack_flags(options)}")
    return read_pack(path, data, row_observer, source_weight)


def read_pack(path: Path, data: Path | None = None, row_observer=None, source_weight=None, merge_dir=None):
    """The tuple load has always returned, from one packed file."""
    stored = np.load(path, mmap_mode="r", allow_pickle=False)
    if 'segments' in stored.files:
        names = stored['segments'].tolist()
        stored.close()
        # source_weight is applied while merging: the segments carry the per-row source columns,
        # the concatenated pack does not.
        return read_segments([path.parent / name for name in names], data, row_observer, source_weight, merge_dir)
    # npz arrays are materialized (np.load's mmap_mode does not mmap ZIP members).
    # Close the archive after reading; every returned array owns its bytes.
    rows = int(stored["rows"])
    boards = CompactBoards.from_arrays(stored, "board")
    scalars = CompactBoards.from_arrays(stored, "scalar")
    plant = torch.from_numpy(np.asarray(stored["plant"]))
    packet = torch.from_numpy(np.asarray(stored["packet"]))
    cell = torch.from_numpy(np.asarray(stored["cell"]))
    weight = torch.from_numpy(np.asarray(stored["weight"]))
    if source_weight:
        weight = weight.clone()
    seed_of = np.asarray(stored["seed"])
    values = ValueStore.from_arrays(stored)
    cands = CandidateStore.from_arrays(stored) if "cand_counts" in stored.files else None
    if (row_observer is not None or source_weight) and data is not None:
        archive, row = np.asarray(stored["source_archive"]), np.asarray(stored["source_row"])
        sources = [str(name) for name in np.asarray(stored["source"])]
        counts = np.bincount(archive, minlength=len(sources))
        start = 0
        for index, count in enumerate(counts):
            if count:
                source = Path(data) / sources[index]
                if row_observer is not None:
                    row_observer(source, row[start:start + count])
                if source_weight:
                    meta_path = source.parent / 'meta.json'
                    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
                    weight[start:start + count] *= source_weight.get(meta.get('label_source', 'teacher'), 1.0)
            start += count
    result = (boards, scalars, plant, packet, cell, weight, seed_of, values, cands)
    if "aux" in stored.files:
        result = (*result, torch.from_numpy(np.asarray(stored["aux"])))
    if "packet_col0" in stored.files:
        result = (*result, PacketStore.from_arrays(stored, rows))
    stored.close()
    return result


def _member_header(archive, name):
    """One npz member's (shape, dtype) from its .npy header, without reading the data.

    Segments are uncompressed np.savez archives, so sizing a big CSR array costs a header
    parse, not a decompress (task61).
    """
    with archive.open(name if name.endswith(".npy") else name + ".npy") as handle:
        version = np.lib.format.read_magic(handle)
        read_header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                       else np.lib.format.read_array_header_2_0)
        shape, _, dtype = read_header(handle)
    return shape, np.dtype(dtype)


def _member_array(archive, name):
    """Read one npz member fully -- only used for the tiny per-store descriptors."""
    with archive.open(name + ".npy") as handle:
        return np.lib.format.read_array(handle)


def read_segments(paths: list[Path], data: Path | None = None, row_observer=None, source_weight=None, merge_dir=None):
    """Share one merged generation between trainers, keeping their row weights private."""
    if data is None:
        return _read_segments(paths, data, row_observer, source_weight)
    import fcntl
    import hashlib
    directory = Path(merge_dir) if merge_dir else Path(data) / '.merge' / 'shared-v1'
    directory.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(json.dumps([
        (str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns)
        for path in paths]).encode()).hexdigest()
    with (directory.parent / 'shared.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = directory / 'complete'
        reuse = marker.is_file() and marker.read_text() == signature
        if not reuse:
            for previous in directory.iterdir():
                previous.unlink()
        result = _read_segments(paths, data, row_observer, source_weight, directory, reuse)
        marker.write_text(signature)
    return result


def _read_segments(paths, data, row_observer, source_weight, spill=None, reuse=False):
    """Merge compact slices in global archive order, without decompressing game frames.

    Labels interleave by seed in frame_paths. Simply concatenating label segments would change
    the split, shuffle and losses. Copy CSR slices in that original order instead; no dense
    boards are created and the trainer keeps its ordinary compact stores and batch code.

    task61: two passes keep the peak near one segment instead of the whole corpus twice. Pass
    one reads only per-row metadata and every member's npy header (no big array is decompressed),
    lays the chunks out in global archive order and preallocates the targets -- shared file-backed
    memmaps when a spill directory is supplied, plain arrays otherwise.
    Pass two loads one segment at a time, copies each chunk
    to its precomputed position and releases it. The result is bitwise identical to the previous
    concatenate-everything implementation (tests compare every column, plus hashes over a real
    round-2 pack subset).
    """
    allocated = []
    def allocate(name, dtype, shape):
        if spill is not None:
            path = spill / f'{name}.npy'
            if reuse:
                return np.load(path, mmap_mode='c', allow_pickle=False)
            store = np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape)
            allocated.append(store)
            return store
        return np.zeros(shape, dtype=np.dtype(dtype))

    # Pass one: the chunk layout comes from per-row metadata and the (small) offsets/counts
    # arrays, which also give every chunk its source entry range per stream. Nothing big is
    # held. source_row (int32, 4 bytes a row) is kept because row_observer needs it at the
    # end, exactly like the old implementation kept it inside chunks.
    layouts, chunks = [], []
    for index, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as stored:
            sources = stored["source"].tolist()
            archive = np.asarray(stored["source_archive"])
            rows = np.asarray(stored["source_row"])
            has_cands = "cand_counts" in stored.files
            streams = ("board", "scalar", "value") + (("cand",) if has_cands else ())
            # CSR streams carry an offsets member; the sparse stores derive theirs from
            # counts (uint8 per row, cumsum like ValueStore.from_arrays does at load).
            boundaries = {"board": np.asarray(stored["board_offsets"]),
                          "scalar": np.asarray(stored["scalar_offsets"]),
                          "value": np.concatenate(
                              [[0], np.cumsum(np.asarray(stored["value_counts"]).astype(np.int64))])}
            if has_cands:
                boundaries["cand"] = np.concatenate(
                    [[0], np.cumsum(np.asarray(stored["cand_counts"]).astype(np.int64))])
        counts = np.bincount(archive, minlength=len(sources))
        start = 0
        for source, count in zip(sources, counts):
            stop = start + int(count)
            ranges = {name: (int(boundaries[name][start]), int(boundaries[name][stop]))
                      for name in streams}
            chunks.append((source, index, start, stop, rows[start:stop], ranges))
            start = stop
        layouts.append(dict(path=path, rows=len(rows), source_row=rows))
    chunks.sort(key=lambda chunk: str(Path(chunk[0]).parent))

    # Column composition is fixed by the shared row options; the first segment describes
    # every segment, and the copy loop asserts each store's descriptors as it goes.
    with zipfile.ZipFile(layouts[0]["path"]) as zfile:
        members = set(name[:-4] for name in zfile.namelist())
        dense_names = ("plant", "packet", "cell", "weight")
        has_aux = "aux" in members
        has_packets = "packet_col0" in members
        has_cands = "cand_counts" in members
        streams = ("board", "scalar", "value") + (("cand",) if has_cands else ())
        headers = {name: _member_header(zfile, name) for name in
                   dense_names + (("aux",) if has_aux else ()) + ("seed",)
                   + ("value_counts", "value_slots", "value_values")
                   + (("cand_counts", "cand_entries") if has_cands else ())
                   + (("packet_col0", "packet_col1", "packet_col2", "packet_col3") if has_packets else ())}
        descriptors = {"board": dict(shape=tuple(int(n) for n in _member_array(zfile, "board_shape")),
                                     dtype=np.dtype(str(_member_array(zfile, "board_dtype"))),
                                     index_dtype=_member_header(zfile, "board_indices")[1]),
                       "scalar": dict(shape=tuple(int(n) for n in _member_array(zfile, "scalar_shape")),
                                      dtype=np.dtype(str(_member_array(zfile, "scalar_dtype"))),
                                      index_dtype=_member_header(zfile, "scalar_indices")[1])}

    # Target positions in global archive order: the row position and, per stream, the entry
    # position -- a running prefix over the sorted chunks, exactly what np.concatenate
    # produced in the old implementation.
    placed, position = [], 0
    cursors = {name: 0 for name in streams}
    for source, index, start, stop, rows, ranges in chunks:
        entry = dict(source=source, index=index, start=start, stop=stop, rows=rows, pos=position)
        position += stop - start
        for name in streams:
            lo, hi = ranges[name]
            entry[name] = (cursors[name], lo, hi)
            cursors[name] += hi - lo
        placed.append(entry)
    total_rows = position
    entries_by_index = {index: [] for index in range(len(layouts))}
    for entry in placed:
        entries_by_index[entry["index"]].append(entry)

    totals = dict(cursors)      # the running prefixes end at each stream's full size
    plane_counts = {"board": [], "scalar": []}
    for layout in layouts:
        with zipfile.ZipFile(layout["path"]) as zfile:
            for name in ("board", "scalar"):
                plane_counts[name].append(int(_member_header(zfile, name + "_plane_offsets")[0][0]) - 1)
    plane_bases = {name: np.cumsum([0] + plane_counts[name]) for name in ("board", "scalar")}

    # Targets, preallocated once: file backed memmaps in the trainer (data given), plain
    # arrays otherwise. The whole merged corpus is written exactly once, position by position.
    stores = []
    for name in ("board", "scalar"):
        descriptor = descriptors[name]
        store = CompactBoards(descriptor["shape"], descriptor["dtype"])
        store.index_dtype = descriptor["index_dtype"]
        store.indices = allocate(name + "_indices", store.index_dtype, (totals[name],))
        store.values = allocate(name + "_values", descriptor["dtype"], (totals[name],))
        store.offsets = allocate(name + "_offsets", np.int64, (total_rows + 1,))
        store.offsets[0] = 0
        store.plane_of = (allocate(name + "_plane_of", np.int32, (total_rows,))
                          if plane_bases[name][-1] else None)
        store.planes = []
        store.nnz = totals[name]
        store.shape = (total_rows, *descriptor["shape"])
        stores.append(store)
    dense_targets = {name: allocate(name, headers[name][1], (total_rows, *headers[name][0][1:]))
                     for name in dense_names + (("aux",) if has_aux else ())}
    seed_target = allocate("seed", headers["seed"][1], (total_rows,))
    value_arrays = {"value_counts": allocate("value_counts", headers["value_counts"][1], (total_rows,)),
                    "value_slots": allocate("value_slots", headers["value_slots"][1], (totals["value"],)),
                    "value_values": allocate("value_values", headers["value_values"][1], (totals["value"],))}
    cand_arrays = None
    if has_cands:
        cand_arrays = {"cand_counts": allocate("cand_counts", headers["cand_counts"][1], (total_rows,)),
                       "cand_entries": allocate("cand_entries", headers["cand_entries"][1],
                                                (totals["cand"], *headers["cand_entries"][0][1:]))}
    packets = PacketStore(total_rows, headers["packet_col0"][0][1]) if has_packets else None
    if packets is not None:
        packets.columns = [allocate(f'packet_col{index}', column.dtype, column.shape)
                           for index, column in enumerate(packets.columns)]

    # Pass two: one segment in memory at a time; every chunk goes to its precomputed position.
    for index, layout in enumerate(layouts):
        if reuse:
            with np.load(layout['path'], allow_pickle=False) as stored:
                for store, name in zip(stores, ('board', 'scalar')):
                    cells = stored[name + '_plane_cells']
                    values = stored[name + '_plane_values']
                    offsets = stored[name + '_plane_offsets']
                    store.planes.extend((cells[lo:hi], values[lo:hi])
                                        for lo, hi in zip(offsets[:-1], offsets[1:]))
            continue
        part = read_pack(layout["path"])
        for store, name, part_store in ((stores[0], "board", part[0]), (stores[1], "scalar", part[1])):
            descriptor = descriptors[name]
            if (part_store.indices.dtype != descriptor["index_dtype"]
                    or part_store.dtype != descriptor["dtype"]
                    or tuple(part_store.shape[1:]) != descriptor["shape"]):
                raise ValueError(f'{layout["path"]}: {name} descriptors differ from the first segment')
            store.planes.extend(part_store.planes)
        for entry in entries_by_index[index]:
            start, stop, pos = entry["start"], entry["stop"], entry["pos"]
            span = stop - start
            for store, part_store, name, plane_base in (
                    (stores[0], part[0], "board", plane_bases["board"][index]),
                    (stores[1], part[1], "scalar", plane_bases["scalar"][index])):
                nnz_at, lo, hi = entry[name]
                store.offsets[pos + 1:pos + span + 1] = nnz_at + np.cumsum(
                    np.diff(part_store.offsets[start:stop + 1]), dtype=np.int64)
                store.indices[nnz_at:nnz_at + (hi - lo)] = part_store.indices[lo:hi]
                store.values[nnz_at:nnz_at + (hi - lo)] = part_store.values[lo:hi]
                if store.plane_of is not None:
                    # A segment whose plane ids do not cover every row (10-26: a demo-start label whose
                    # games start 8 decisions before the end, 72 ids for 84 rows) is read without planes
                    # instead of stopping training; pack_training_rows is fixed separately.
                    ids = (np.full(span, -1, np.int32)
                           if part_store.plane_of is None
                           or len(part_store.plane_of) != len(part_store.offsets) - 1
                           else part_store.plane_of[start:stop])
                    store.plane_of[pos:pos + span] = np.where(
                        ids >= 0, ids + plane_base, -1).astype(np.int32)
            for name, part_store, arrays, fields in (
                    ("value", part[7], value_arrays, ("slots", "values")),
                    ("cand", part[8], cand_arrays, ("entries",))):
                if arrays is None:
                    continue
                entry_at, lo, hi = entry[name]
                arrays[name + "_counts"][pos:pos + span] = part_store.counts[start:stop]
                for field in fields:
                    arrays[name + "_" + field][entry_at:entry_at + (hi - lo)] = \
                        getattr(part_store, field)[lo:hi]
            for name, part_index in (("plant", 2), ("packet", 3), ("cell", 4), ("weight", 5),
                                     *((("aux", 9),) if has_aux else ())):
                dense_targets[name][pos:pos + span] = np.asarray(part[part_index])[start:stop]
            seed_target[pos:pos + span] = np.asarray(part[6])[start:stop]
            if packets is not None:
                for column, target_column in zip(part[9 + has_aux].columns, packets.columns):
                    target_column[pos:pos + span] = np.asarray(column)[start:stop]
        del part

    for store in allocated:
        store.flush()
    if spill is not None:
        dense_targets['weight'] = dense_targets['weight'].copy()
    result = (stores[0], stores[1],
              *(torch.from_numpy(dense_targets[name]) for name in dense_names),
              seed_target, ValueStore.from_arrays(value_arrays),
              CandidateStore.from_arrays(cand_arrays) if cand_arrays is not None else None)
    if has_aux:
        result = (*result, torch.from_numpy(dense_targets["aux"]))
    if has_packets:
        result = (*result, packets)
    if row_observer is not None and data is not None:
        for entry in placed:
            if len(entry["rows"]):
                row_observer(Path(data) / entry["source"], entry["rows"])
    if source_weight and data is not None:
        # Same rule as the single-file branch: a row's weight is scaled by its archive's
        # label_source. task61's segmented packs merged the weights but dropped this step, so
        # human/demonstration rows kept full weight in the production trainer.
        for entry in placed:
            span = entry["stop"] - entry["start"]
            if not span:
                continue
            meta_path = (Path(data) / entry["source"]).parent / "meta.json"
            meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
            factor = source_weight.get(meta.get("label_source", "teacher"), 1.0)
            if factor != 1.0:
                dense_targets["weight"][entry["pos"]:entry["pos"] + span] *= factor
    return result

def main(*, step_observer=None) -> None:
    # task34: train after collection, but before the controller and the WSL access path.
    set_oom_score_adj(OOM_SCORE_TRAINING)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--resume', type=Path, help='resume an interruption checkpoint (model + optimizer + RNG)')
    p.add_argument('--checkpoint-seconds', type=float, default=0, help='opt in to resumable checkpoints and SIGUSR1 yielding (exit 75)')
    p.add_argument('--skip-validation', action='store_true', help='defer row evaluation to the CPU worker')
    p.add_argument('--resident-training-rows', action='store_true', help='cache selected training inputs in RAM; use with --skip-validation')
    p.add_argument('--pack', type=Path, help='use an existing frozen pack index without rescanning the corpus')
    p.add_argument('--merge-dir', type=Path, help='private merged pack cache; never lock the production merge')
    p.add_argument('--deck-holdout', type=Path, help='common boolean row mask excluding complete heldout decks')
    p.add_argument('--tokens', type=Path, help='sparse public token sidecar for the same frozen rows')
    p.add_argument('--training-budget', type=Path, help='common measured epochs/row budget, immutable after the GPU probe')
    p.add_argument('--budget-arm', help='architecture key for the measured safe micro batch')
    args = training_args(p)
    if args.training_budget:
        import hashlib
        budget_bytes = args.training_budget.read_bytes()
        budget = json.loads(budget_bytes)
        args.epochs = budget['epochs']
        args.train_row_fraction = budget['train_row_fraction']
        args.micro_batch_size = budget['micro_batches'][args.budget_arm]
        args.resolved_recipe.update(epochs=args.epochs, train_row_fraction=args.train_row_fraction,
                                   micro_batch_size=args.micro_batch_size)
        args.resolved_recipe['provenance']['notes'] = args.resolved_recipe['provenance'].get('notes','') + ' budget_sha256=' + hashlib.sha256(budget_bytes).hexdigest()
    yielding = args.checkpoint_seconds > 0 or args.resume is not None
    yield_requested = False
    resume_ready = False
    def request_yield(_signal, _frame):
        nonlocal yield_requested
        if not resume_ready:
            raise SystemExit(75)  # Loading has no optimizer progress to save.
        yield_requested = True
    if yielding:
        signal.signal(signal.SIGUSR1, request_yield)
        signal.signal(signal.SIGTERM, request_yield)
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    write_recipe(args.out / "recipe.resolved.json", args.resolved_recipe)
    metrics = Metrics("train_rollout_policy", training=True).start()
    metrics.update(step=0, rows=0, phase="loading")
    row_plan = None
    if args.row_weights:
        from pvz_training_rows import RowWeights
        row_plan = RowWeights(args.row_weights, args.data)
    source_counts = None
    observer = row_plan.observe if row_plan else None
    level_cap = args.source_weight.get('level_balance')
    level_power = args.source_weight.get('level_balance_power', 1.0)
    level_max_multiplier = args.source_weight.get('level_balance_max_multiplier')
    if args.source_row_report or level_cap:
        from pvz_training_rows import SourceRowCounts
        source_counts = SourceRowCounts(balance=bool(level_cap))
        def observer(path, rows):
            source_counts.observe(path, rows)
            if row_plan: row_plan.observe(path, rows)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    started = time.time()
    label_weight = {k: float(v) for k, v in (kv.split("=") for kv in args.label_weight.split(",") if kv)}
    loaded = (read_pack(args.pack, args.data, observer, args.source_weight, args.merge_dir) if args.pack else
              load(args.data, **training_row_options(args.resolved_recipe), row_observer=observer))
    if yield_requested:
        metrics.close()
        raise SystemExit(75)  # No optimizer work yet; restart loading in the next window.
    boards, scalars, plant, packet, cell, weight, seed_of, values, cands = loaded[:9]
    auxiliary = loaded[9] if args.aux_weight else None
    packet_info = loaded[9 + bool(args.aux_weight)] if args.packet_head == 'identity' else None
    load_seconds = time.time() - started
    seeds = sorted(set(seed_of)); random.Random(args.validation["seed"]).shuffle(seeds)
    val_seeds = set(seeds[: int(len(seeds) * args.validation["fraction"])])
    is_val = torch.from_numpy(np.fromiter((s in val_seeds for s in seed_of), dtype=bool, count=len(seed_of)))
    if args.deck_holdout:
        from pvz_research_data import extra_validation
        is_val |= torch.from_numpy(extra_validation(args.deck_holdout, len(seed_of)))
    training_eligible = ~is_val
    if args.train_row_fraction < 1:
        from pvz_research_data import keep_training_rows
        for begin in range(0, len(seed_of), 65536):
            end = min(len(seed_of), begin + 65536)
            training_eligible[begin:end] &= torch.from_numpy(keep_training_rows(np.arange(begin, end), args.train_row_fraction))
    token_store = None
    if args.tokens:
        from pvz_research_data import SparseTokens
        token_store = SparseTokens(args.tokens)
        if token_store.meta['rows'] != len(seed_of):
            raise ValueError('token sidecar does not cover the frozen r42 rows')
    if row_plan:
        row_plan.finish(len(plant), is_val.numpy())
    if source_counts and args.source_row_report:
        source_counts.finish(len(plant), val_seeds, args.source_row_report)
    print(f"{len(plant)} decisions ({int(plant.sum())} plants), {len(seeds)} seeds, "
          f"{int(is_val.sum())} validation rows", flush=True)
    memory = {"rows": len(plant), "board_bytes": boards.nbytes,
              "scalar_bytes": scalars.nbytes, "cand_bytes": 0 if cands is None else cands.nbytes,
              "value_bytes": values.nbytes,
              "tensor_bytes": sum(tensor.numel() * tensor.element_size()
                                  for tensor in (plant, packet, cell, weight, is_val))}
    if packet_info is not None:
        memory['packet_bytes'] = (packet_info.nbytes if isinstance(packet_info, PacketStore)
                                  else packet_info.numel() * packet_info.element_size())
    print(json.dumps({"data_memory": memory}), flush=True)
    if args.packet_balance:
        planted_rows = packet >= 0
        if args.deck_holdout or args.train_row_fraction < 1:
            planted_rows &= training_eligible
        balanced_packet = (packet.long() if packet_info is None else
                           packet_info.kinds(torch.arange(len(packet)), packet.clamp(min=0)))
        freq = torch.bincount(balanced_packet[planted_rows], minlength=PACKETS if packet_info is None else 54).float() / int(planted_rows.sum())
        scale = freq.clamp(min=1e-6) ** -args.packet_balance
        scale = (scale / scale[balanced_packet[planted_rows]].mean()).clamp(max=20.0)  # mean plant-row weight ~1
        weight[planted_rows] *= scale[balanced_packet[planted_rows]]
        print("packet weights", {k: round(float(scale[k]), 2) for k in range(len(freq)) if freq[k] > 0}, flush=True)
    if args.spread_weight:
        # 2026-10-17: only ~3% of rows are the teacher overriding the student, and those carry what the student lacks;
        # plain cross-entropy weighs a trivial wait as much as a decision worth 10 points of win rate.
        compared_rows, spread = values.spread(training_eligible.numpy() if args.train_row_fraction < 1 else None)
        compared = torch.from_numpy(compared_rows).long()
        spread = torch.from_numpy(spread)
        # capped: a few spreads near 1 are games decided inside the horizon, not choices 30x more important
        spread_mean = spread[training_eligible[compared]].mean() if args.deck_holdout or args.train_row_fraction < 1 else spread.mean()
        weight[compared] *= ((1 + args.spread_weight * spread / spread_mean) / (1 + args.spread_weight)).clamp(max=args.spread_cap)
        print(f"spread weights: {len(compared)} compared rows, mean spread {float(spread.mean()):.4f}, "
              f"weight range {float(weight[compared].min()):.2f}-{float(weight[compared].max()):.2f}", flush=True)
    if row_plan:
        row_plan.apply(weight)
        print(json.dumps({"row_intervention":row_plan.metadata(),
                          "positive_weight_rows":int((weight>0).sum())}),flush=True)
    if level_cap:
        shares = source_counts.rebalance(weight, is_val, level_cap, level_power, level_max_multiplier)
        (args.out / 'level_shares.json').write_text(json.dumps(shares, indent=2) + '\n')
        print(json.dumps({'level_balance': shares}), flush=True)
    net = create_student(args.arch, width=args.width, pool=args.pool, context=args.context, obs=args.obs, row_pool=args.row_pool,
                     aux=bool(args.aux_weight), packet_head=args.packet_head,
                     **({'card_attributes': True} if args.card_attributes else {}))
    if args.init:
        from pvz_rollout_policy import load_student, warm_start_student
        warm_start_student(net, load_student(args.init))
    net = net.to(args.device)
    opt = torch.optim.AdamW(net.parameters(), **{k: v for k, v in args.optimizer.items() if k != "name"})
    if args.deck_holdout or args.train_row_fraction < 1:
        eligible = training_eligible
        waits, plants = float(weight[(plant < .5) & eligible].sum()), float(plant[eligible].sum())
    else:
        waits, plants = float(weight[plant < .5].sum()), float(plant.sum())
    pos_weight = torch.tensor([waits / max(1.0, plants)], device=args.device).sqrt()

    def batch_loss(idx, train, normalizers=None):
        b = boards.batch(idx, args.device)
        s = scalars.batch(idx, args.device)
        y = plant[idx].to(args.device).float()          # uint8 in the compact store
        pk = packet[idx].to(args.device).long()         # int16 in the compact store
        c = cell[idx].to(args.device).long()
        w = weight[idx].to(args.device)
        planted = y > 0.5
        packet_args = {'packet_info': packet_info[idx].to(args.device)} if packet_info is not None else {}
        if token_store is not None:
            packet_args['tokens'] = token_store.batch(idx, args.device)
        out = net(b, s, pk.clamp(min=0), auxiliary=bool(args.aux_weight), **packet_args)
        denom = normalizers['main'] if normalizers else w.sum().clamp_min(1e-30) if row_plan else w.sum()
        loss = (F.binary_cross_entropy_with_logits(out["plant"], y, pos_weight=pos_weight, reduction="none") * w
                ).sum() / denom
        if args.aux_weight:
            row_loss = F.binary_cross_entropy_with_logits(out["aux"], auxiliary[idx].to(args.device), reduction="none").mean(1)
            loss = loss + args.aux_weight * (row_loss * w).sum() / denom
        if args.cell_soft_weight and planted.any():
            cd = cands[idx].to(args.device).float()                       # (B, K, 3): packet, cell, value
            same = (cd[:, :, 0] == pk[:, None].float()) & ~torch.isnan(cd[:, :, 2]) & planted[:, None]
            rows = same.sum(1) >= 2                                       # >= 2 cells of the labelled card compared
            if rows.any():
                m = same[rows]
                # empty candidate slots are NaN; NaN -> long is 0 on ARM but INT64_MIN on x86, so zero them first (masked by m anyway)
                logits = out["cell"][rows].gather(1, cd[rows][:, :, 1].nan_to_num(0).clamp(min=0).long())
                student = torch.log_softmax(logits.masked_fill(~m, -1e9), 1)
                teacher = torch.softmax((cd[rows][:, :, 2].nan_to_num(0) / args.soft_temperature).masked_fill(~m, -1e9), 1)
                kl = (teacher * (torch.log(teacher.clamp(min=1e-12)) - student)).masked_fill(~m, 0).sum(1)
                wc = w[rows]
                cell_denom=normalizers['cell'] if normalizers else wc.sum().clamp_min(1e-30) if row_plan else wc.sum()
                loss = loss + args.cell_soft_weight * (kl * wc).sum() / cell_denom
        if planted.any():
            wp = w[planted]                               # all 1 unless --label-weight is used
            loss = loss + ((F.cross_entropy(out["packet"][planted], pk[planted], reduction="none") +
                            F.cross_entropy(out["cell"][planted], c[planted], reduction="none")) * wp).sum() / (normalizers['plant'] if normalizers else wp.sum().clamp_min(1e-30) if row_plan else wp.sum())
        if train and getattr(net, 'routing_loss', None) is not None:
            loss = loss + .01 * net.routing_loss * (len(idx) / normalizers['rows'] if normalizers else 1)
        stats = {}
        if not train:
            with torch.no_grad():
                pred_plant = out["plant"] > 0
                stats = {"type_acc": float(((pred_plant == planted).float() * w).sum() / w.sum()),
                         "plant_recall": float(pred_plant[planted].float().mean()) if planted.any() else 0.0,
                         "packet_acc": float((out["packet"][planted].argmax(1) == pk[planted]).float().mean()) if planted.any() else 0.0,
                         "cell_acc": float((out["cell"][planted].argmax(1) == c[planted]).float().mean()) if planted.any() else 0.0}
        return loss, stats

    if not is_val.any() or is_val.all():
        raise ValueError("recipe split must produce nonempty training and validation sets")
    train_idx = torch.nonzero(training_eligible).squeeze(1).to(torch.int32)
    val_idx = torch.nonzero(is_val).squeeze(1).to(torch.int32)
    if args.resident_training_rows and args.skip_validation:
        from pvz_research_data import ResidentRows
        boards, scalars = (ResidentRows(store, train_idx) for store in (boards, scalars))
        if cands is not None:
            cands = ResidentRows(cands, train_idx)
        if packet_info is not None:
            packet_info = ResidentRows(packet_info, train_idx)
        print(json.dumps(dict(resident_training_rows=len(train_idx))), flush=True)
    steps = args.epochs * -(-len(train_idx) // args.batch_size)
    if args.max_steps is not None:
        if args.max_steps>steps:
            p.error("--max-steps exceeds the requested epochs")
        steps=args.max_steps
    schedule = (torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.5 * (1 + math.cos(math.pi * min(t, steps) / steps)))
                if args.lr_schedule == "cosine" else None)
    history = []

    def checkpoint() -> dict:
        return {"state_dict": {k: v.cpu() for k, v in net.state_dict().items()}, "labels": args.labels, "width": args.width, "arch": args.arch,
                "label_weight": label_weight, "pool": args.pool, "packet_balance": args.packet_balance,
                "row_pool": args.row_pool, "aux_weight": args.aux_weight, "aux_horizon": args.aux_horizon,
                "packet_head": args.packet_head, "plant_embedding_dim": net.plant_embedding_dim,
                **({"card_attributes": True} if args.card_attributes else {}),
                "context": args.context, "obs": args.obs, "spread_weight": args.spread_weight, "spread_cap": args.spread_cap, "lr_schedule": args.lr_schedule, "cell_soft_weight": args.cell_soft_weight,
                # Historical reports compare these metadata keys; both experiments are retired.
                "soft_weight": 0.0, "ema": 0.0,
                "history": list(history), "recipe": args.resolved_recipe, "recipe_sha256": args.recipe_sha256,
                **({"row_intervention":row_plan.metadata()} if row_plan else {}),
                **({"max_steps":args.max_steps,"optimizer_steps":completed_steps} if args.max_steps else {})}

    completed_steps=0
    completed_rows=0
    next_epoch, next_start, epoch_rng, train_loss = 1, 0, None, 0.0
    def save_resume():
        target = args.out / 'resume.pt'
        payload = dict(checkpoint(), optimizer=opt.state_dict(), scheduler=None if schedule is None else schedule.state_dict(),
                       completed_steps=completed_steps, completed_rows=completed_rows, next_epoch=next_epoch,
                       next_start=next_start, epoch_rng=epoch_rng, train_loss=train_loss,
                       rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else [],
                       pack=str(args.pack.resolve()) if args.pack else None, pack_size=args.pack.stat().st_size if args.pack else None,
                       training_rows=len(train_idx), validation_rows=len(val_idx))
        temporary = target.with_suffix('.building')
        torch.save(payload, temporary)
        temporary.replace(target)
        (args.out / 'progress.json').write_text(json.dumps(dict(steps=completed_steps, rows=completed_rows,
            epoch=next_epoch, batch_start=next_start, saved=time.time())) + '\n')
    if args.resume:
        saved = torch.load(args.resume, map_location=args.device, weights_only=False)
        # Constant-size contract checks only; never replay or rehash the training corpus.
        if (saved['recipe'] != args.resolved_recipe or saved['training_rows'] != len(train_idx)
                or saved['validation_rows'] != len(val_idx) or saved['pack'] != (str(args.pack.resolve()) if args.pack else None)
                or saved['pack_size'] != (args.pack.stat().st_size if args.pack else None)):
            raise ValueError('resume recipe/pack/split differs from the saved run')
        net.load_state_dict(saved['state_dict']); opt.load_state_dict(saved['optimizer'])
        if schedule is not None:
            schedule.load_state_dict(saved['scheduler'])
        completed_steps, completed_rows = saved['completed_steps'], saved['completed_rows']
        next_epoch, next_start, epoch_rng = saved['next_epoch'], saved['next_start'], saved['epoch_rng'].cpu()
        train_loss, history = saved['train_loss'], saved['history']
        torch.set_rng_state(saved['rng'].cpu())
        if saved['cuda_rng']:
            torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_rng']])
    last_save = time.monotonic()
    resume_ready = True
    def maybe_yield():
        nonlocal last_save
        if yielding and (yield_requested or time.monotonic() - last_save >= (args.checkpoint_seconds or 30)):
            save_resume()
            last_save = time.monotonic()
        if yield_requested:
            metrics.close()
            raise SystemExit(75)
    metrics.update(phase="training")
    for epoch in range(next_epoch, args.epochs + 1):
        if completed_steps >= steps:
            break
        epoch_started = time.perf_counter()
        metrics.update(phase="training")
        net.train()
        if not next_start:
            epoch_rng = torch.get_rng_state()
            train_loss = 0.0
        current_rng = torch.get_rng_state()
        torch.set_rng_state(epoch_rng)
        perm = train_idx[torch.randperm(len(train_idx))]
        if next_start:
            torch.set_rng_state(current_rng)
        for start in range(next_start, len(perm), args.batch_size):
            maybe_yield()
            batch = perm[start:start + args.batch_size]
            batch_rng = torch.get_rng_state()
            batch_cuda_rng = torch.cuda.get_rng_state_all() if args.device.startswith('cuda') and args.micro_batch_size else None
            opt.zero_grad()
            if args.micro_batch_size and args.micro_batch_size < len(batch):
                ww = weight[batch]
                pp = plant[batch] > .5
                denominators = dict(main=ww.sum().clamp_min(1e-30).to(args.device),
                    plant=ww[pp].sum().clamp_min(1e-30).to(args.device), rows=len(batch))
                if args.cell_soft_weight:
                    cd = cands[batch].float()
                    same = (cd[:, :, 0] == packet[batch, None].float()) & ~torch.isnan(cd[:, :, 2]) & pp[:, None]
                    denominators['cell'] = ww[same.sum(1) >= 2].sum().clamp_min(1e-30).to(args.device)
                loss_value = 0.0
                for micro in batch.split(args.micro_batch_size):
                    partial, _ = batch_loss(micro, True, denominators)
                    partial.backward(); loss_value += float(partial.detach())
                    if yield_requested:
                        # Discard the incomplete logical batch; resume repeats it with the saved RNG.
                        torch.set_rng_state(batch_rng)
                        if batch_cuda_rng is not None:
                            torch.cuda.set_rng_state_all(batch_cuda_rng)
                        opt.zero_grad(); maybe_yield()
                loss = torch.tensor(loss_value)
            else:
                loss, _ = batch_loss(batch, True)
                loss.backward()
            if step_observer is not None:
                step_observer(completed_steps + 1, float(loss.detach()))
            train_loss += float(loss.detach()) * len(perm[start:start + args.batch_size])
            opt.step()
            if schedule is not None:
                schedule.step()
            completed_steps+=1
            completed_rows += len(perm[start:start + args.batch_size])
            next_epoch, next_start = epoch, start + args.batch_size
            metrics.update(step=completed_steps, rows=completed_rows, epoch=epoch)
            if completed_steps>=steps:
                break
        maybe_yield()
        metrics.update(phase="validation")
        net.eval()
        # Validate in chunks: one ~95k-row batch is a >2 GB tensor, which MPS silently gets wrong
        # (2026-10-15: garbage val metrics while the model itself was fine on the CPU). 1024 rather than
        # 4096: no gradients are kept, so the chunk only has to hold one forward's activations, and the
        # cell head's (chunk, 320, 6, 10) input is what made 4096 the tallest part of a whole run (task32).
        parts = []
        if not args.skip_validation:
            with torch.no_grad():
                for chunk in val_idx.split(args.validation_batch_size):
                    maybe_yield()
                    parts.append((len(chunk), *batch_loss(chunk, False)))
        total = sum(n for n, _l, _s in parts)
        vl = sum(float(l) * n for n, l, _s in parts) / total if total else None
        vs = {k: sum(st[k] * n for n, _l, st in parts) / total for k in parts[0][2]} if parts else {}
        history.append({"epoch": epoch, "train_loss": train_loss / len(train_idx), "val_loss": vl,
                        **vs, "seconds": time.perf_counter() - epoch_started})
        print(json.dumps(history[-1]), flush=True)
        if args.save_epochs and epoch > args.epochs - args.save_epochs:
            args.out.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint(), args.out / f"epoch_{epoch:02d}.pt")
        next_epoch, next_start = epoch + 1, 0
        if completed_steps>=steps:
            break
    args.out.mkdir(parents=True, exist_ok=True)
    if yielding:
        temporary = args.out / 'model.building'
        torch.save(checkpoint(), temporary)
        temporary.replace(args.out / 'model.pt')
    else:
        torch.save(checkpoint(), args.out / "model.pt")
    (args.out / "metrics.json").write_text(json.dumps({"history": history, "seconds": time.time() - started,
                                                       "load_seconds": load_seconds, "data_memory": memory}, indent=1))
    metrics.update(phase="complete")
    metrics.close()
    print(f"wrote {args.out} ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
