"""Train the fast rollout policy on pi's own games (see python/pvz_rollout_policy.py).

Data: episode archives whose frames carry the executed action (``frames[i]["action"]`` produced
frame i). Input is frame i-1, target is the action at frame i. Split by environment seed.

Usage:
  python scripts/train_rollout_policy.py --data artifacts/value_net/episodes --labels u84_greedy,u84_sampled \
      --out artifacts/rollout_policy/v1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from pvz_rollout_policy import LAWN_COLS, PACKETS, RolloutNet, features, packet_features, identity_scalars  # noqa: E402
from pvz_frame_store import (CELL_CANDIDATES, cache_aux, cache_candidates, cache_episode, cache_packets, cache_identity_scene, candidate_cells, frame_bytes,  # noqa: E402
                              frame_paths, read_frames, row_damage_labels, target_values)


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
        self.indices = np.frombuffer(self.indices, dtype=self.index_dtype)
        self.values = np.frombuffer(self.values, dtype=self.dtype)
        self.offsets = np.frombuffer(self.offsets, dtype=np.int64)
        self.plane_of = np.frombuffer(self._plane_ids, dtype=np.int32) if self.planes else None
        self._plane_ids = bytearray()
        return self

    @property
    def nbytes(self):
        return (self.indices.nbytes + self.values.nbytes + self.offsets.nbytes
                + (0 if self.plane_of is None else self.plane_of.nbytes))

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
            if (identities >= 0).any():
                view = output.view(len(selected), size)
                for identity in np.unique(identities[identities >= 0]):
                    rows = torch.from_numpy(np.flatnonzero(identities == identity)).to(device)
                    cells, values = self.planes[identity]
                    view[rows.unsqueeze(1), torch.from_numpy(cells).long().to(device)] = torch.from_numpy(values).to(device=device, dtype=torch.float32)
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
        if counts.sum():
            starts = self.offsets[selected]
            positions = np.arange(counts.sum()) + np.repeat(starts - (np.cumsum(counts) - counts), counts)
            slots = (np.repeat(np.arange(len(selected)), counts) * CELL_CANDIDATES
                     + np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts))
            output.reshape(-1, 3)[slots] = self.entries[positions]
        return torch.from_numpy(output)

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

    def spread(self) -> tuple[np.ndarray, np.ndarray]:
        """(rows with at least two known values, max - min of them): the spread weighting, without a dense row."""
        counts = self.counts.astype(np.int64)
        rows = np.flatnonzero(counts >= 2)
        spreads = np.empty(len(rows), dtype=np.float32)
        for index, row in enumerate(rows):
            block = self.values[self.offsets[row]:self.offsets[row + 1]]
            spreads[index] = block.max() - block.min()
        return rows, spreads

    @property
    def nbytes(self):
        return self.counts.nbytes + self.slots.nbytes + self.values.nbytes + self.offsets.nbytes


def load(data: Path, labels: set[str], wait_keep: float = 1.0, label_weight: dict[str, float] | None = None,
         cache_dir: Path | None = None, with_context: bool = False, obs: str = "v1", precompute_workers: int = 1,
         cells: bool = False, aux_horizon: int = 0, row_observer=None,
         packet_head: str = 'slot', read_cache_dir: Path | None = None):
    """Rows: frame i-1 -> action at frame i. With ``wait_keep`` < 1 only that fraction of the wait rows
    (~94% of all rows) is kept, each weighted 1 / wait_keep, so more games fit in memory.
    ``label_weight`` multiplies the weight of every row of a label (a small, newer teacher among many older games).

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
    cands = CandidateStore() if cells else None
    auxiliary = np.empty((capacity, 6), dtype=np.float32) if aux_horizon else None
    packets = PacketStore(capacity, PACKETS) if packet_head == 'identity' else None
    n = 0
    for path in paths:
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
                n += count
            continue
        frames = read_frames(path)
        future_damage = row_damage_labels(frames, aux_horizon) if aux_horizon else None
        ctx = _episode_context(path) if with_context else None
        if obs != "v1":
            from pvz_student_observation import StudentHistory
            history = StudentHistory(obs)
        kept = np.unpackbits(keep[path])[:len(frames)].astype(bool)
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
            boards.append(b.numpy()[None])
            scalars.append(s.numpy()[None])
            if packets is not None:
                packets.set(n, n + 1, packet_features(frames[i - 1])[None])
            n += 1
            seed_of.append(seed)
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
    result = (boards.finish(), scalars.finish(), *(torch.from_numpy(a[:n]) for a in (plant, packet, cell, weight)),
              seed_of, values.finish(),
              None if cands is None else cands.finish())
    if aux_horizon:
        result = (*result, torch.from_numpy(auxiliary[:n]))
    if packets is not None:
        result = (*result, packets.finish(n))
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--labels", default="u84_greedy,u84_sampled")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--wait-keep", type=float, default=1.0, help="fraction of wait rows kept (reweighted)")
    p.add_argument("--seed", type=int, default=None, help="torch seed for init and shuffling")
    p.add_argument("--width", type=int, default=64, help="channels of the net (smaller = faster rollouts)")
    p.add_argument("--label-weight", default="", help="label=factor,... row-weight multipliers per label")
    p.add_argument("--packet-balance", type=float, default=0.0,
                   help="weight plant rows by (packet frequency)^-x, so rare cards (cherry 1.5%%) are not drowned out")
    p.add_argument("--pool", choices=["mean", "meanmax"], default="mean", help="global pooling of the trunk")
    p.add_argument("--row-pool", action="store_true", help="broadcast learned per-row mean/max features")
    p.add_argument('--packet-head', choices=['slot', 'identity'], default='slot')
    p.add_argument("--aux-weight", type=float, default=0.0, help="row damage prediction BCE weight (0 = off)")
    p.add_argument("--aux-horizon", type=int, default=8, help="future decision points for row damage labels")
    p.add_argument("--context", action="store_true",
                   help="the student also reads the task context (zombie-count multiplier, roster) from meta.json")
    p.add_argument("--obs", choices=["v1", "v2lite", "v2"], default="v1", help="student feature version")
    p.add_argument("--soft-temperature", type=float, default=0.05,
                   help="temperature of the cell-level soft labels (--cell-soft-weight), in win-probability units")
    p.add_argument("--cell-soft-weight", type=float, default=0.0,
                   help="cell-level soft labels: on planted rows whose search compared >= 2 cells of the labelled card, "
                        "add KL(softmax(cell values / --soft-temperature) || student cell distribution over those cells). "
                        "The hard label teaches only the chosen cell; this also teaches how much worse the other cells were")
    p.add_argument("--spread-weight", type=float, default=0.0,
                   help="cost-weighted imitation: a searched row (>= 2 candidates) weighs (1 + x * spread / mean spread) / (1 + x) (at most --spread-cap), "
                        "spread = best minus worst candidate value, so decisions where the choice matters count more")
    p.add_argument("--spread-cap", type=float, default=5.0, help="upper bound of the --spread-weight factor")
    p.add_argument("--save-epochs", type=int, default=0,
                   help="also keep the last N epochs as epoch_XX.pt (candidates for scripts/select_student.py)")
    p.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant",
                   help="cosine: learning rate falls from 2e-3 to 0 over the run (per batch), so the final weights are not "
                        "a snapshot of a still-bouncing optimiser; constant = the old behaviour")
    p.add_argument("--init", help="start from this student / fast_ppo checkpoint (same width and pool) instead of random")
    p.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True, help="cache exact per-episode features (--no-cache reads JSON directly)")
    p.add_argument("--cache-dir", type=Path, help="shared feature cache (default: .feature_cache beside the output directory)")
    p.add_argument("--precompute-workers", type=int, default=1, help="parallel per-episode feature cache workers")
    p.add_argument('--read-cache-dir', type=Path, help='optional existing feature cache to read only; misses go to --cache-dir')
    p.add_argument("--row-weights", type=Path, help="optional checksummed per-row experiment manifest; cached rows required, default off")
    p.add_argument('--source-row-report', type=Path, help='optional actual retained row counts by source; cached rows required')
    p.add_argument("--max-steps", type=int, help="optional optimizer-step budget and cosine length; default all epochs")
    args = p.parse_args()
    if args.aux_weight < 0 or args.aux_horizon < 1:
        p.error("--aux-weight must be nonnegative and --aux-horizon must be positive")
    if args.row_weights and not args.cache:
        p.error("--row-weights requires --cache")
    if args.source_row_report and not args.cache:
        p.error('--source-row-report requires --cache')
    if args.max_steps is not None and args.max_steps<=0:
        p.error("--max-steps must be positive")
    row_plan = None
    if args.row_weights:
        from pvz_training_rows import RowWeights
        row_plan = RowWeights(args.row_weights, args.data)
    source_counts = None
    observer = row_plan.observe if row_plan else None
    if args.source_row_report:
        from pvz_training_rows import SourceRowCounts
        source_counts = SourceRowCounts()
        def observer(path, rows):
            source_counts.observe(path, rows)
            if row_plan: row_plan.observe(path, rows)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    started = time.time()
    label_weight = {k: float(v) for k, v in (kv.split("=") for kv in args.label_weight.split(",") if kv)}
    loaded = load(args.data, set(args.labels.split(",")),
                                                                 args.wait_keep, label_weight,
                                                                 (args.cache_dir or args.out.parent / ".feature_cache") if args.cache else None,
                                                                 args.context, args.obs, args.precompute_workers,
                                                                 cells=bool(args.cell_soft_weight),
                                                                 aux_horizon=args.aux_horizon if args.aux_weight else 0,
                                                                 row_observer=observer,
                                                                 packet_head=args.packet_head, read_cache_dir=args.read_cache_dir)
    boards, scalars, plant, packet, cell, weight, seed_of, values, cands = loaded[:9]
    auxiliary = loaded[9] if args.aux_weight else None
    packet_info = loaded[9 + bool(args.aux_weight)] if args.packet_head == 'identity' else None
    load_seconds = time.time() - started
    seeds = sorted(set(seed_of)); random.Random(0).shuffle(seeds)
    val_seeds = set(seeds[: len(seeds) // 10])
    # np.fromiter skips the ~8 bytes a row that a python list of bools costs on the way to the tensor (task32).
    is_val = torch.from_numpy(np.fromiter((s in val_seeds for s in seed_of), dtype=bool, count=len(seed_of)))
    if row_plan:
        row_plan.finish(len(plant), is_val.numpy())
    if source_counts:
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
        compared_rows, spread = values.spread()
        compared = torch.from_numpy(compared_rows).long()
        spread = torch.from_numpy(spread)
        # capped: a few spreads near 1 are games decided inside the horizon, not choices 30x more important
        weight[compared] *= ((1 + args.spread_weight * spread / spread.mean()) / (1 + args.spread_weight)).clamp(max=args.spread_cap)
        print(f"spread weights: {len(compared)} compared rows, mean spread {float(spread.mean()):.4f}, "
              f"weight range {float(weight[compared].min()):.2f}-{float(weight[compared].max()):.2f}", flush=True)
    if row_plan:
        row_plan.apply(weight)
        print(json.dumps({"row_intervention":row_plan.metadata(),
                          "positive_weight_rows":int((weight>0).sum())}),flush=True)
    net = RolloutNet(args.width, args.pool, args.context, args.obs, args.row_pool,
                     aux=bool(args.aux_weight), packet_head=args.packet_head)
    if args.init:
        from pvz_rollout_policy import load_student
        initial = load_student(args.init)
        weights = initial.state_dict()
        if net.aux is not None:
            weights.update({k: v for k, v in net.state_dict().items() if k.startswith("aux.")})
        if args.obs == "v2" and initial.obs == "v1" and args.context == initial.context:
            weights["trunk.0.weight"] = F.pad(weights["trunk.0.weight"], (0, 0, 0, 0, 0, 9))
            weights["scalar.0.weight"] = F.pad(weights["scalar.0.weight"], (0, 7))
        net.load_state_dict(weights)
    net = net.to(args.device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    waits = float(weight[plant < 0.5].sum())
    pos_weight = torch.tensor([waits / max(1.0, float(plant.sum()))], device=args.device).sqrt()

    def batch_loss(idx, train):
        b = boards.batch(idx, args.device)
        s = scalars.batch(idx, args.device)
        y = plant[idx].to(args.device).float()          # uint8 in the compact store
        pk = packet[idx].to(args.device).long()         # int16 in the compact store
        c = cell[idx].to(args.device).long()
        w = weight[idx].to(args.device)
        planted = y > 0.5
        packet_args = {'packet_info': packet_info[idx].to(args.device)} if packet_info is not None else {}
        out = net(b, s, pk.clamp(min=0), auxiliary=bool(args.aux_weight), **packet_args)
        denom = w.sum().clamp_min(1e-30) if row_plan else w.sum()
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
                logits = out["cell"][rows].gather(1, cd[rows][:, :, 1].clamp(min=0).long())
                student = torch.log_softmax(logits.masked_fill(~m, -1e9), 1)
                teacher = torch.softmax((cd[rows][:, :, 2].nan_to_num(0) / args.soft_temperature).masked_fill(~m, -1e9), 1)
                kl = (teacher * (torch.log(teacher.clamp(min=1e-12)) - student)).masked_fill(~m, 0).sum(1)
                wc = w[rows]
                cell_denom=wc.sum().clamp_min(1e-30) if row_plan else wc.sum()
                loss = loss + args.cell_soft_weight * (kl * wc).sum() / cell_denom
        if planted.any():
            wp = w[planted]                               # all 1 unless --label-weight is used
            loss = loss + ((F.cross_entropy(out["packet"][planted], pk[planted], reduction="none") +
                            F.cross_entropy(out["cell"][planted], c[planted], reduction="none")) * wp).sum() / (wp.sum().clamp_min(1e-30) if row_plan else wp.sum())
        stats = {}
        if not train:
            with torch.no_grad():
                pred_plant = out["plant"] > 0
                stats = {"type_acc": float(((pred_plant == planted).float() * w).sum() / w.sum()),
                         "plant_recall": float(pred_plant[planted].float().mean()) if planted.any() else 0.0,
                         "packet_acc": float((out["packet"][planted].argmax(1) == pk[planted]).float().mean()) if planted.any() else 0.0,
                         "cell_acc": float((out["cell"][planted].argmax(1) == c[planted]).float().mean()) if planted.any() else 0.0}
        return loss, stats

    # int32 halves the two index tensors a row (perm is a second copy of train_idx); every use is a gather (task32).
    train_idx = torch.nonzero(~is_val).squeeze(1).to(torch.int32)
    val_idx = torch.nonzero(is_val).squeeze(1).to(torch.int32)
    steps = args.epochs * -(-len(train_idx) // 2048)
    if args.max_steps is not None:
        if args.max_steps>steps:
            p.error("--max-steps exceeds the requested epochs")
        steps=args.max_steps
    schedule = (torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.5 * (1 + math.cos(math.pi * min(t, steps) / steps)))
                if args.lr_schedule == "cosine" else None)
    history = []

    def checkpoint() -> dict:
        return {"state_dict": {k: v.cpu() for k, v in net.state_dict().items()}, "labels": args.labels, "width": args.width,
                "label_weight": label_weight, "pool": args.pool, "packet_balance": args.packet_balance,
                "row_pool": args.row_pool, "aux_weight": args.aux_weight, "aux_horizon": args.aux_horizon,
                "packet_head": args.packet_head, "plant_embedding_dim": net.plant_embedding_dim,
                "context": args.context, "obs": args.obs, "spread_weight": args.spread_weight, "spread_cap": args.spread_cap, "lr_schedule": args.lr_schedule, "cell_soft_weight": args.cell_soft_weight,
                "history": list(history),
                **({"row_intervention":row_plan.metadata()} if row_plan else {}),
                **({"max_steps":args.max_steps,"optimizer_steps":completed_steps} if args.max_steps else {})}

    completed_steps=0
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        net.train()
        perm = train_idx[torch.randperm(len(train_idx))]
        train_loss = 0.0
        for start in range(0, len(perm), 2048):
            loss, _ = batch_loss(perm[start:start + 2048], True)
            train_loss += float(loss.detach()) * len(perm[start:start + 2048])
            opt.zero_grad(); loss.backward(); opt.step()
            if schedule is not None:
                schedule.step()
            completed_steps+=1
            if completed_steps>=steps:
                break
        net.eval()
        # Validate in chunks: one ~95k-row batch is a >2 GB tensor, which MPS silently gets wrong
        # (2026-10-15: garbage val metrics while the model itself was fine on the CPU). 1024 rather than
        # 4096: no gradients are kept, so the chunk only has to hold one forward's activations, and the
        # cell head's (chunk, 320, 6, 10) input is what made 4096 the tallest part of a whole run (task32).
        with torch.no_grad():
            parts = [(len(chunk), *batch_loss(chunk, False)) for chunk in val_idx.split(1024)]
        total = sum(n for n, _l, _s in parts)
        vl = sum(float(l) * n for n, l, _s in parts) / total
        vs = {k: sum(st[k] * n for n, _l, st in parts) / total for k in parts[0][2]}
        history.append({"epoch": epoch, "train_loss": train_loss / len(train_idx), "val_loss": float(vl),
                        **vs, "seconds": time.perf_counter() - epoch_started})
        print(json.dumps(history[-1]), flush=True)
        if args.save_epochs and epoch > args.epochs - args.save_epochs:
            args.out.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint(), args.out / f"epoch_{epoch:02d}.pt")
        if completed_steps>=steps:
            break
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint(), args.out / "model.pt")
    (args.out / "metrics.json").write_text(json.dumps({"history": history, "seconds": time.time() - started,
                                                       "load_seconds": load_seconds, "data_memory": memory}, indent=1))
    print(f"wrote {args.out} ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
