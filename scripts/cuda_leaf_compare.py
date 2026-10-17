"""Phase-1 leaf comparison: the *original* simulator's continuations vs the CUDA ones.

The GPU and CPU rollouts deliberately run different RNG streams (the brief says random-number order
does not matter, only that the mechanics are faithful), so a per-(root, future) leaf is NOT expected
to match bit for bit.  What has to match is the *distribution* of the leaf statistics the search reads
-- terminal outcome, waves reached, per-row zombie advance, plant loss -- within the spread the
original itself shows when you resample its futures.  A systematic gap is a mechanic bug; a gap inside
the resample spread is just the RNG the two paths disagree about on purpose.

  python3 scripts/cuda_leaf_compare.py \
      --cpu artifacts/cuda/corpus_l7_z15/corpus.jsonl \
      --gpu /tmp/gpu_z15_leaves.bin

``--cpu`` is the CPU reference the corpus exporter wrote (one jsonl line per (root, future));
``--gpu`` is the raw RolloutResult stream from ``build/rollout_cuda --out`` (layout cuda/CudaRollout.h,
33 little-endian int32 per record).

Two more things this has to say something about, because a distribution is a weak instrument and the
leaf features are strong ones:

  python3 scripts/cuda_leaf_compare.py \
      --root-features /tmp/root_feat.bin --roots artifacts/cuda/corpus_l7_z15/roots.cudaroot

``--root-features`` is ``build/sim_host --root-features``: the value net's encoding of every root as
the CUDA core holds it *before any tick has run*.  Rebuilding the same vector from the ``.cudaroot`` in
Python goes through the observation's own semantics (``python/pvz_cuda_root.py::value_frame``), so the
two sides see the identical position and there is no random stream between them -- a single differing
float is a bug in one encoder.  This is the check that found the truncated cooldown channel.

  python3 scripts/cuda_leaf_compare.py --cpu ... --gpu ... --pairs-out /tmp/pairs.txt \\
      --gpu-features /tmp/gpu_leaf.bin --value-net ~/PvZ-search/artifacts/value_net/v2/model.pt

``--pairs-out`` writes the exact (root, future) workload the reference ran, which is what ``--pairs`` on
both drivers consumes; ``--gpu-features`` scores the GPU's own leaves with the teacher's value net
(terminal leaves take 1/0, the rest sigmoid V, exactly as the exporter's ``_leaf_value`` does) and puts
that distribution next to the reference's ``leaf_value`` column.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _directory in (ROOT / "python", ROOT / "scripts"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

INT32_MIN = -2147483648
# RolloutResult, in cuda/CudaRollout.h declaration order: 17 scalars, plantedByType[10], rowMinZombieX[6].
FIELDS = ("rootIndex", "decisions", "netCalls", "ticks", "endTick", "result", "sun", "wave",
          "zombiesAlive", "zombiesKilled", "plantsAlive", "plantsEaten", "plantsPlanted",
          "mowersLeft", "mowersTriggered", "sunProduced", "errorFlags")
RECORD = struct.Struct("<" + "i" * (len(FIELDS) + 10 + 6))
ROWS = 6


def read_gpu(path: Path) -> list[dict]:
    blob = path.read_bytes()
    if len(blob) % RECORD.size:
        raise SystemExit(f"{path}: size {len(blob)} is not a whole number of {RECORD.size}-byte records")
    out = []
    for offset in range(0, len(blob), RECORD.size):
        words = RECORD.unpack_from(blob, offset)
        record = dict(zip(FIELDS, words[:len(FIELDS)]))
        record["plantedByType"] = list(words[len(FIELDS):len(FIELDS) + 10])
        record["rowMinZombieX"] = list(words[len(FIELDS) + 10:])
        out.append(record)
    return out


def read_cpu(path: Path) -> list[dict]:
    records = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        # Normalise the CPU keys onto the RolloutResult names so one table prints both sides.
        records.append({
            "rootIndex": row["root"],
            "decisions": row["decisions"],
            "netCalls": row["net_calls"],
            "ticks": row["ticks"],
            "endTick": row["end_tick"],
            "result": row["result"],
            "sun": row["sun"],
            "wave": row["wave"],
            "zombiesAlive": row["zombies"],
            "plantsAlive": row["plants"],
            "plantsPlanted": row["plants_planted"],
            "mowersLeft": row["mowers_left"],
            "plantedByType": row["planted_by_type"],
            "rowMinZombieX": row["row_min_zombie_x"],
            # Not statistics columns but the two things the later stages need: --pairs-out has to name
            # the future so the GPU replays exactly this continuation, and the leaf-value check reads
            # the value net's own score of this leaf.
            "future": row["future"],
            "leaf_value": row.get("leaf_value"),
        })
    return records


def _summary(values: list[float]) -> str:
    if not values:
        return "n=0"
    values = sorted(values)
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = var ** 0.5

    def q(frac: float) -> float:
        index = min(n - 1, max(0, int(round(frac * (n - 1)))))
        return values[index]

    return (f"n={n:5d} mean={mean:9.3f} sd={std:8.3f} "
            f"p10={q(0.10):8.2f} p50={q(0.50):8.2f} p90={q(0.90):8.2f}")


def _outcome(records: list[dict]) -> str:
    n = len(records) or 1
    won = sum(1 for r in records if r["result"] > 0)
    lost = sum(1 for r in records if r["result"] < 0)
    ongoing = n - won - lost
    return f"won {won / n * 100:5.2f}%  lost {lost / n * 100:5.2f}%  ongoing {ongoing / n * 100:5.2f}%"


def _row_advance(records: list[dict]) -> list[float]:
    """Closest-to-house zombie x per row (fixed-point / 4 = pixels), averaged over rollouts where the
    row is not clear.  This is the phase-1 "各行僵尸推进" measure."""
    means = []
    for row in range(ROWS):
        samples = [r["rowMinZombieX"][row] / 4.0 for r in records
                   if row < len(r["rowMinZombieX"]) and r["rowMinZombieX"][row] != INT32_MIN]
        means.append(sum(samples) / len(samples) if samples else float("nan"))
    return means


# ---------------------------------------------------------------------------
# The exported value encoding
# ---------------------------------------------------------------------------

VALUE_MAGIC = b"PVZLEAF1"
PLANT_TYPES, ZOMBIE_TYPES = 54, 34
BOARD_CELLS = 92 * 60
VALUE_SCALARS, VALUE_CONTEXT = 113, 35


def read_features(path: Path) -> list:
    """The PVZLEAF1 block stream both drivers write: ``[record count][floats per record]`` then the
    blocks in lane order.  Returns one numpy row per record."""
    import numpy as np

    blob = path.read_bytes()
    if blob[:8] != VALUE_MAGIC:
        raise SystemExit(f"{path}: not a PVZLEAF1 dump (found {blob[:8]!r})")
    count, floats = int.from_bytes(blob[8:12], "little"), int.from_bytes(blob[12:16], "little")
    if floats != BOARD_CELLS + VALUE_SCALARS + VALUE_CONTEXT:
        raise SystemExit(f"{path}: {floats} floats per record, expected "
                         f"{BOARD_CELLS + VALUE_SCALARS + VALUE_CONTEXT}")
    payload = np.frombuffer(blob[16:], dtype="<f4")
    if payload.size != count * floats:
        raise SystemExit(f"{path}: header says {count} records but the file holds "
                         f"{payload.size // max(1, floats)}")
    return [payload[index * floats:(index + 1) * floats] for index in range(count)]


def describe(index: int) -> str:
    """What one float of the value vector is, so a difference reads as a game fact and not an offset."""
    if index < BOARD_CELLS:
        channel, cell = divmod(index, 60)
        row, col = divmod(cell, 10)
        if channel < PLANT_TYPES:
            return f"board plant type {channel} at row {row} col {col} (health ratio)"
        if channel < PLANT_TYPES + ZOMBIE_TYPES:
            return f"board zombie type {channel - PLANT_TYPES} at row {row} col {col} (health / 200)"
        names = {PLANT_TYPES + ZOMBIE_TYPES: "eating", PLANT_TYPES + ZOMBIE_TYPES + 1: "chilled",
                 90: "mower ready", 91: "playable lane"}
        return f"board {names.get(channel, channel)} at row {row} col {col}"
    index -= BOARD_CELLS
    if index < VALUE_SCALARS:
        if index < 5:
            return ["sun / 1000", "wave / wave count", "tick / 60000", "sun income rate / 100",
                    "log1p(wave count)"][index]
        if index < 5 + PLANT_TYPES:
            return f"packet of plant type {index - 5} affordable and off cooldown"
        return f"packet of plant type {index - 5 - PLANT_TYPES} cooldown progress"
    index -= VALUE_SCALARS
    return "task context: zombie count multiplier / 3" if index == 0 \
        else f"task context: roster allows zombie type {index - 1}"


def check_encoder(roots_path: Path, features_path: Path, show: int) -> None:
    """Element-for-element: the CUDA core's encoding of each root vs the Python value encoder's."""
    import numpy as np
    import pvz_cuda_root
    from pvz_value_net import context_vector, featurize_numpy

    corpus = pvz_cuda_root.read_cudaroot(roots_path)
    blocks = read_features(features_path)
    if len(blocks) != len(corpus["roots"]):
        raise SystemExit(f"{features_path}: {len(blocks)} blocks for "
                         f"{len(corpus['roots'])} roots")
    print(f"encoder check: {len(blocks)} roots, {len(blocks[0])} floats each "
          f"({BOARD_CELLS} board + {VALUE_SCALARS} scalars + {VALUE_CONTEXT} context)")
    bad_roots = compared = worst = 0
    first_differences: list[str] = []
    for index, record in enumerate(corpus["roots"]):
        if not record["supported"]:
            continue
        board, scalars = featurize_numpy(pvz_cuda_root.value_frame(record, corpus["level"]))
        multiplier, roster = pvz_cuda_root.value_context(corpus["level"])
        want = np.concatenate([board.reshape(-1), scalars,
                               context_vector(multiplier, roster)]).astype(np.float32)
        got = np.asarray(blocks[index], dtype=np.float32)
        compared += 1
        differing = np.nonzero(want != got)[0]
        if differing.size == 0:
            continue
        bad_roots += 1
        worst = max(worst, int(differing.size))
        if len(first_differences) < show:
            for where in differing[:4]:
                first_differences.append(f"root {index}: {describe(int(where))} -- "
                                         f"python {float(want[where]):.7g}, core {float(got[where]):.7g}")
    print(f"  roots compared {compared}, exact {compared - bad_roots}, differing {bad_roots}"
          f" (worst root {worst} floats)")
    for line in first_differences:
        print(f"  {line}")
    if bad_roots:
        print("  -> the two encoders disagree; this is a bug in one of them, not a different future")
        raise SystemExit(1)
    else:
        print("  -> every supported root encodes identically: the core's StoreLeafValueFeatures and "
              "pvz_value_net.featurize_numpy are the same function")


def write_pairs(cpu: list[dict], path: Path) -> None:
    """The exact workload the CPU reference ran, so the GPU can be pointed at the same pairs."""
    with path.open("w", encoding="utf-8") as stream:
        for record in cpu:
            stream.write(f"{record['rootIndex']} {record['future']}\n")
    print(f"wrote {len(cpu)} root/future pairs to {path}")


def score_gpu_leaves(cpu: list[dict], gpu: list[dict], features_path: Path, net_path: Path) -> None:
    """Score the GPU's leaf features with the teacher's own value net and compare the distribution to
    the CPU reference's ``leaf_value`` column.  Per pair the two sides deliberately differ -- the
    continuations use different random streams -- so only the distribution can be the gate."""
    import numpy as np
    import torch

    from pvz_value_net import load_value_net

    net = load_value_net(net_path)
    # The exported vector is [92*60 board][113 scalars][35 task context].  A net trained without the
    # context (value_net/v2) reads only the 113 scalars, which is exactly what the corpus exporter's
    # _leaf_value appends -- so both sides of the comparison feed the tensor width the net was built for.
    scalars_wanted = VALUE_SCALARS + (VALUE_CONTEXT if net.context else 0)
    blocks = read_features(features_path)
    if len(blocks) != len(gpu):
        raise SystemExit(f"{features_path}: {len(blocks)} blocks for {len(gpu)} result records")
    scored = [0.0] * len(gpu)
    pending, boards, rests = [], [], []
    for index, (record, block) in enumerate(zip(gpu, blocks)):
        if record["result"] != 0:
            scored[index] = 1.0 if record["result"] > 0 else 0.0
            continue
        pending.append(index)
        boards.append(block[:BOARD_CELLS].reshape(92, ROWS, 10))
        rests.append(block[BOARD_CELLS:BOARD_CELLS + scalars_wanted])
    if pending:
        board = torch.from_numpy(np.stack(boards).astype(np.float32, copy=False))
        rest = torch.from_numpy(np.stack(rests).astype(np.float32, copy=False))
        with torch.inference_mode():
            values = torch.sigmoid(net(board, rest)).tolist()
        for index, value in zip(pending, values):
            scored[index] = float(value)
    want = [r["leaf_value"] for r in cpu if r["leaf_value"] is not None]
    print(f"leaf value: scored {len(scored)} GPU leaves with {net_path}")
    print(f"  GPU  {_summary(scored)}")
    print(f"  CPU  {_summary(want)}")
    if len(scored) == len(want):
        delta = [g - c for g, c in zip(scored, want)]
        print(f"  per-pair GPU-CPU  {_summary(delta)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", type=Path)
    ap.add_argument("--gpu", type=Path)
    ap.add_argument("--metrics", default="ticks,decisions,netCalls,wave,sun,plantsAlive,"
                                        "zombiesAlive,plantsPlanted,mowersLeft")
    # The encoder cross-check: ``sim_host --root-features`` dumps what the CUDA core writes for a
    # position it has just imported; ``pvz_value_net.featurize_numpy`` builds the same vector from the
    # .cudaroot through the observation's own semantics.  Same position, two encoders, no RNG between
    # them, so any differing float is a bug rather than a different future.
    ap.add_argument("--root-features", type=Path)
    ap.add_argument("--roots", type=Path)
    ap.add_argument("--gpu-features", type=Path, help="PVZLEAF1 dump from --leaf-out, scored by "
                                                     "--value-net and compared to leaf_value")
    ap.add_argument("--value-net", type=Path,
                    default=Path.home() / "PvZ-search" / "artifacts" / "value_net" / "v2" / "model.pt")
    ap.add_argument("--pairs-out", type=Path, help="write <root> <future> per corpus line for --pairs")
    ap.add_argument("--show", type=int, default=8, help="how many differing entries to print")
    args = ap.parse_args()

    if args.root_features:
        if not args.roots:
            raise SystemExit("--root-features needs --roots FILE.cudaroot")
        check_encoder(args.roots, args.root_features, args.show)
        if not (args.cpu and args.gpu):
            return
    if args.pairs_out:
        if not args.cpu:
            raise SystemExit("--pairs-out needs --cpu FILE.jsonl to take the futures from")
        write_pairs(read_cpu(args.cpu), args.pairs_out)
        if not args.gpu:
            return
    if args.gpu_features:
        if not (args.cpu and args.gpu):
            raise SystemExit("--gpu-features needs --cpu and --gpu to name the leaf each block belongs to")
        score_gpu_leaves(read_cpu(args.cpu), read_gpu(args.gpu), args.gpu_features, args.value_net)
        return
    if not (args.cpu and args.gpu):
        raise SystemExit("needs --cpu and --gpu (or --root-features with --roots)")

    cpu = read_cpu(args.cpu)
    gpu = read_gpu(args.gpu)
    print(f"CPU records {len(cpu)} (roots {len(set(r['rootIndex'] for r in cpu))}), "
          f"GPU records {len(gpu)} (roots {len(set(r['rootIndex'] for r in gpu))})")
    print()
    print("terminal outcome")
    print(f"  CPU  {_outcome(cpu)}")
    print(f"  GPU  {_outcome(gpu)}")
    print()
    print("per-row zombie advance (mean nearest-x pixels; lower = closer to the house)")
    cpu_row, gpu_row = _row_advance(cpu), _row_advance(gpu)
    for row in range(ROWS):
        print(f"  row {row}:  CPU {cpu_row[row]:8.1f}   GPU {gpu_row[row]:8.1f}")
    print()
    print("distribution by metric")
    for metric in args.metrics.split(","):
        print(f"  {metric:14s} CPU  {_summary([float(r[metric]) for r in cpu if metric in r])}")
        print(f"  {'':14s} GPU  {_summary([float(r[metric]) for r in gpu])}")
    print()
    print("planted-by-packet-slot (total plants of each deck slot over the batch)")
    for slot in range(10):
        cpu_total = sum(r["plantedByType"][slot] for r in cpu if slot < len(r["plantedByType"]))
        gpu_total = sum(r["plantedByType"][slot] for r in gpu)
        print(f"  slot {slot}:  CPU {cpu_total:6d}   GPU {gpu_total:6d}")


if __name__ == "__main__":
    main()
