"""Is the packed token block worth delta-coding across decisions?

``tokens`` is 75% of the rollout payload -- 312.5 KiB of 415 KiB per episode, ~570 MiB
for a 2000-episode batch -- so it is the only field where a big memory win is even
possible.  The idea under test ("P2, cross-decision incremental encoding") is to store
the first decision in full and then only what changed.

That idea only pays if the block is mostly *stable* between decisions.  ``pack_tokens``
lays tokens out in a deterministic order (global, profile, cells in observation order,
then entities), so a position-by-position comparison is meaningful -- but every plant
or zombie that appears, dies or moves shifts every later index.  This probe measures
how bad that is, and prices the alternative (just zlib the block) before anyone writes
a delta codec.

    python scripts/token_delta_probe.py --episodes 8
"""

from __future__ import annotations

import argparse
import zlib
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from pvz_agent_model import GameplayModelV1, configure_torch_threads  # noqa: E402
from pvz_env import PvZEnv  # noqa: E402
from train_pvz_ppo import collect_task_episode  # noqa: E402
import train_pvz_ppo_task_family as trainer  # noqa: E402

DEFAULT_RESOURCES = Path("/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN")
ROLLOUT_EPISODES = 2000
FIELDS = ("ids", "features", "cell_index", "packet_ids", "packet_index")


def _bits(array: np.ndarray) -> np.ndarray:
    """Reinterpret as unsigned ints so NaN and -0.0 compare bitwise, not numerically."""
    if array.dtype == np.float16:
        return array.view(np.uint16)
    if array.dtype == np.int8:
        return array.view(np.uint8)
    return array


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--resource-dir", type=Path, default=DEFAULT_RESOURCES)
    args = parser.parse_args()

    resource_dir = args.resource_dir.expanduser()
    configure_torch_threads(1)
    torch.manual_seed(0)
    model = GameplayModelV1().eval()
    train, _ = trainer._task_family()
    cap1 = trainer._curriculum_tasks(train["tasks"], "cap1")
    task = cap1[0]

    episodes: list[dict[str, Any]] = []
    with PvZEnv(resource_dir) as env:
        for index in range(args.episodes):
            episodes.append(collect_task_episode(model, env, task, task["seeds"][index], index, 4000))

    blocks = [[t["tokens"] for t in episode["transitions"]] for episode in episodes]
    pairs = sum(len(block) - 1 for block in blocks)
    decisions = sum(len(block) for block in blocks)
    print(f"episodes={len(blocks)} decisions={decisions} consecutive pairs={pairs}\n")

    # ------------------------------------------------------ 1. element churn
    print("--- 1. how much of the block changes from one decision to the next ---")
    print(f"  {'field':16s} {'dtype':9s} {'shape':12s} {'shape chg':>10s}"
          f" {'elements changed':>18s}")
    for field in FIELDS:
        changed = total = 0
        shape_changes = 0
        shapes: set[tuple[int, ...]] = set()
        dtype = "?"
        for block in blocks:
            for before, after in zip(block, block[1:]):
                left, right = before[field], after[field]
                shapes.add(left.shape)
                dtype = str(left.dtype)
                if left.shape != right.shape:
                    shape_changes += 1
                    continue
                changed += int(np.count_nonzero(_bits(left) != _bits(right)))
                total += left.size
        span = f"{min(s[0] for s in shapes)}-{max(s[0] for s in shapes)}"
        share = changed / total * 100 if total else 0.0
        print(f"  {field:16s} {dtype:9s} {span:12s} "
              f"{shape_changes:6d}/{pairs:<4d} {changed:9d}/{total:<8d} {share:5.1f}%")

    # ------------------------------------------------------ 2. row-level stability
    print("\n--- 2. can whole token rows be reused? (ids+features travel together) ---")
    for field in ("ids", "features"):
        rows_same = rows_total = 0
        for block in blocks:
            for before, after in zip(block, block[1:]):
                left, right = before[field], after[field]
                if left.shape != right.shape:
                    rows_total += max(left.shape[0], right.shape[0])
                    continue
                same = np.all(_bits(left) == _bits(right), axis=1)
                rows_same += int(np.count_nonzero(same))
                rows_total += left.shape[0]
        share = rows_same / rows_total * 100 if rows_total else 0.0
        print(f"  {field:16s} identical rows {rows_same:7d}/{rows_total:<7d} {share:5.1f}%")

    # ------------------------------------------- 3. price it against plain zlib
    print("\n--- 3. price the alternatives on one decision's block ---")
    raws, zlibs = [], []
    for block in blocks:
        for decision in block:
            raw = sum(decision[field].nbytes for field in FIELDS)
            packed = b"".join(decision[field].tobytes() for field in FIELDS)
            raws.append(raw)
            zlibs.append(len(zlib.compress(packed, 6)))
    raw = sum(raws) / len(raws)
    compressed = sum(zlibs) / len(zlibs)
    print(f"  raw bytes for one decision (mean)          {raw:9.0f}")
    print(f"  zlib(level 6) the same block (mean)        {compressed:9.0f}"
          f"  ({raw / compressed:.2f}x)")

    # a row-granular delta: keep changed rows only, 2-byte index + row payload
    deltas = []
    for block in blocks:
        for before, after in zip(block, block[1:]):
            if before["ids"].shape != after["ids"].shape:
                deltas.append(1.0)
                continue
            changed_rows = int(np.count_nonzero(
                np.any(_bits(before["ids"]) != _bits(after["ids"]), axis=1)
                | np.any(_bits(before["features"]) != _bits(after["features"]), axis=1)))
            deltas.append(changed_rows / max(after["ids"].shape[0], 1))
    mean_share = sum(deltas) / len(deltas)
    row_bytes = int(sum(sample_row.nbytes for sample_row in (blocks[0][0]["ids"][0],
                                                             blocks[0][0]["features"][0])))
    print(f"  changed-row share (ids or features differ) {mean_share * 100:8.1f}%"
          f"  -> delta block ~{mean_share * raw:.0f} B + index overhead")
    print(f"  row size (ids+features, one token)         {row_bytes:9d} B")
    print(f"  delta index cost (2 B per changed row)     "
          f"{mean_share * blocks[0][0]['ids'].shape[0] * 2:9.0f} B")

    # -------------------------------------------------- 4. what it would save
    print("\n--- 4. projected effect on the rollout payload ---")
    per_episode_tokens = sum(
        sum(t["tokens"][field].nbytes for field in FIELDS)
        for episode in episodes for t in episode["transitions"]
    ) / len(episodes)
    print(f"  tokens per episode (array bytes)           {per_episode_tokens / 1024:8.1f} KiB")
    print(f"  projected for {ROLLOUT_EPISODES} episodes            "
          f"{per_episode_tokens * ROLLOUT_EPISODES / 1024**2:8.1f} MiB")
    delta_mib = per_episode_tokens * mean_share * ROLLOUT_EPISODES / 1024**2
    zlib_mib = per_episode_tokens * (compressed / raw) * ROLLOUT_EPISODES / 1024**2
    print(f"  cross-decision delta coding                {delta_mib:8.1f} MiB"
          f"   ({raw / (mean_share * raw):.2f}x on the block)")
    print(f"  plain zlib on each block, no cross-decision {zlib_mib:8.1f} MiB"
          f"   ({raw / compressed:.2f}x on the block)")
    print(f"  zlib is {delta_mib / zlib_mib:.1f}x smaller than the delta scheme"
          f" and needs no reconstruction state")


if __name__ == "__main__":
    main()
