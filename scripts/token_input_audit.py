"""Audit what the network actually consumes, not what the protocol sends.

``PROTOCOL_OBSERVABILITY_AUDIT.md`` priced the wire.  This prices the *input
tensor*: how many tokens a decision produces, which kinds dominate, how much of
each token's 32-slot feature vector is padding, and where the forward pass's
arithmetic actually goes.

The distinction matters because the two layers disagree.  ``legal_actions`` is
43.6% of the observation bytes but produces **zero** tokens -- it is consumed by
``legal_summary`` as an action mask.  Conversely ``cells`` is 36% of the bytes
*and* 54 tokens, which is most of the sequence.

Run with the mise interpreter::

    /Users/newbiexvwu/.local/share/mise/installs/python/3.14/bin/python3 \
        scripts/token_input_audit.py --episodes 3
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import torch  # noqa: E402

from pvz_agent_model import (  # noqa: E402
    FEATURE_COUNT, MODEL_CONFIG, TOKEN_KINDS, GameplayModelV1,
    configure_torch_threads, unpack_tokens,
)
from pvz_env import PvZEnv  # noqa: E402
from train_pvz_ppo import _task_spec, collect_task_episode  # noqa: E402
import train_pvz_ppo_task_family as trainer  # noqa: E402

LOCAL_RESOURCES = Path("/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN")

# How many slots each ``add(...)`` call actually fills.  Read off
# ``observation_tokens``; kept here as data so the padding share is visible.
FILLED_SLOTS = {
    "global": 18,
    "profile": 8,
    "cell": 8,
    "lane": 4,
    "plant": 16,
    "zombie": 20,
    "projectile": 12,
    "defense": 4,
    "grid_item": 7,
    "seed_packet": 5,
    "zombie_roster": 1,
}

KIND_NAMES = {index: name for name, index in TOKEN_KINDS.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resource-dir", type=Path, default=LOCAL_RESOURCES)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    resource_dir = args.resource_dir.expanduser()
    if not resource_dir.is_dir():
        raise SystemExit(f"resource dir not found: {resource_dir}")

    configure_torch_threads(1)
    torch.manual_seed(0)
    model = GameplayModelV1().eval()

    # ------------------------------------------------------------------ collect
    train, _heldout = trainer._task_family()
    tasks = trainer._curriculum_tasks(train["tasks"], "cap1")
    episodes: list[dict[str, Any]] = []
    with PvZEnv(resource_dir) as env:
        for index in range(args.episodes):
            task = tasks[index % len(tasks)]
            episodes.append(collect_task_episode(
                model, env, task, task["seeds"][index % 64], index, 4000))

    decisions = sum(len(e["transitions"]) for e in episodes)
    print(f"episodes={len(episodes)}  decisions={decisions}  "
          f"({decisions / len(episodes):.1f} decisions/episode)\n")

    # ------------------------------------------------------- token composition
    counts_per_decision: list[int] = []
    kind_totals: Counter[str] = Counter()
    kind_per_decision: dict[str, list[int]] = defaultdict(list)
    for episode in episodes:
        for transition in episode["transitions"]:
            kinds = transition["tokens"]["ids"][:, 0]
            counts_per_decision.append(len(kinds))
            local = Counter(KIND_NAMES[int(k)] for k in kinds)
            kind_totals.update(local)
            for name in TOKEN_KINDS:
                kind_per_decision[name].append(local.get(name, 0))

    print("--- tokens per decision ---")
    print(f"  mean {statistics.mean(counts_per_decision):.1f}   "
          f"min {min(counts_per_decision)}   max {max(counts_per_decision)}")
    print(f"  total tokens over {len(episodes)} episodes: {sum(counts_per_decision):,}")
    print()
    print("--- token kinds (share of the sequence the encoder pays for) ---")
    print(f"  {'kind':16s} {'total':>7s} {'share':>7s} {'mean/decision':>14s}  filled/32")
    total_tokens = sum(kind_totals.values())
    for name, count in kind_totals.most_common():
        filled = FILLED_SLOTS[name]
        mean = statistics.mean(kind_per_decision[name])
        print(f"  {name:16s} {count:7d} {count / total_tokens * 100:6.1f}% "
              f"{mean:14.1f}  {filled:2d}/32 ({filled / FEATURE_COUNT * 100:.0f}%)")

    # --------------------------------------------------------- feature padding
    print()
    print("--- feature padding: slots filled vs slots projected ---")
    padded = 0
    for name, count in kind_totals.items():
        padded += count * (FEATURE_COUNT - FILLED_SLOTS[name])
    slots = total_tokens * FEATURE_COUNT
    print(f"  feature slots projected (tokens x 32): {slots:,}")
    print(f"  of which never written (stay 0.0):     {padded:,} ({padded / slots * 100:.1f}%)")
    print("  NOTE: a 0.0 slot still costs a multiply in feature_projection, but that")
    print("  layer is ~1.7% of the forward pass -- padding is a clarity problem here,")
    print("  not a speed problem.  Section 'where the arithmetic goes' has the numbers.")

    # --------------------------------------------------- duplicated encodings
    print()
    print("--- information carried in more than one channel ---")
    print("  the same value reaches the encoder through several inputs:")
    for line in (
        "    cell.terrain      category embedding + feature[2]",
        "    cell.row_type     variant embedding  + feature[3]",
        "    cell.row / col    row/col embedding  + feature[0..1]",
        "    plant.row / col   row/col embedding  + feature[0..1]",
        "    zombie.row / col  row/col embedding  + feature[0..1]",
        "    projectile.motion variant embedding  + feature[8]",
        "    seed_packet.index feature[0]         + metadata packet_tokens",
        "    lane.*            aggregated from the plant/zombie tokens also present",
        "    zombie_roster     category only; feature vector is literally (1.0, 0, ...)",
    ):
        print(line)

    # --------------------------------------------- where the arithmetic goes
    print()
    print("--- where the arithmetic goes (measured, FlopCounterMode) ---")
    sample = episodes[0]["transitions"][0]
    tensors, metadata = unpack_tokens(sample["tokens"], torch.device("cpu"))
    print(f"  sample decision: {tensors['kinds'].shape[0]} tokens")

    from torch.utils.flop_counter import FlopCounterMode

    by_module: dict[str, int] = defaultdict(int)
    with FlopCounterMode(display=False) as counter:
        model.step_tokens(tensors, metadata, 0, None, None, 0, {})
    for module, operations in counter.get_flop_counts().items():
        name = str(module)
        parts = name.split(".")
        if len(parts) > 2 and parts[0] == "encoder":
            owner = f"encoder[].{parts[2]}"
        elif len(parts) > 1:
            owner = ".".join(parts[:2])
        else:
            owner = parts[0]
        by_module[owner] += sum(int(value) for value in operations.values())
    total_flops = sum(by_module.values())
    print(f"  {'module':28s} {'GFLOP':>9s} {'share':>7s}")
    for name, value in sorted(by_module.items(), key=lambda kv: -kv[1])[:14]:
        print(f"  {name:28s} {value / 1e9:9.3f} {value / total_flops * 100:6.1f}%")
    print(f"  {'TOTAL':28s} {total_flops / 1e9:9.3f}  (one decision, batch 1)")

    parameters = sum(p.numel() for p in model.parameters())
    print()
    print(f"  model parameters: {parameters:,}  ({parameters * 4 / 1024**2:.2f} MiB fp32)")
    print(f"  config: {MODEL_CONFIG}")

    # --------------------------------------------------------- constant tokens
    print()
    print("--- how much of each kind is a per-decision constant ---")
    print("  a (kind, slot) is CONSTANT if it holds the same value at every decision")
    print("  of the episode; those slots are recomputed and re-attended for nothing.")
    values: dict[tuple[str, int], set[float]] = defaultdict(set)
    slot_constant: Counter[str] = Counter()
    for episode in episodes:
        for transition in episode["transitions"]:
            ids = transition["tokens"]["ids"]
            features = transition["tokens"]["features"].astype("float32")
            for row in range(len(ids)):
                name = KIND_NAMES[int(ids[row, 0])]
                for slot in range(FEATURE_COUNT):
                    values[(name, slot)].add(round(float(features[row, slot]), 6))
    print(f"  {'kind':16s} {'tokens':>8s} {'slots ever written':>19s} "
          f"{'of which constant':>18s}")
    for name, count in kind_totals.most_common():
        written = FILLED_SLOTS[name]
        constant = sum(1 for slot in range(written) if len(values[(name, slot)]) == 1)
        print(f"  {name:16s} {count:8d} {written:19d} {constant:18d}")
        slot_constant[name] = constant
    total_written = sum(kind_totals[name] * FILLED_SLOTS[name] for name in kind_totals)
    total_constant = sum(kind_totals[name] * slot_constant[name] for name in kind_totals)
    print(f"  written feature slots: {total_written:,}; "
          f"constant across the episode: {total_constant:,} "
          f"({total_constant / total_written * 100:.1f}%)")

    # ------------------------------------------- unchanged between decisions
    print()
    print("--- tokens that are byte-identical to the previous decision ---")
    print("  token order is deterministic (global, profile, 54 cells, lanes, ...), so")
    print("  index i at decision t and t+1 normally denote the same entity.")
    identical = 0
    compared = 0
    identical_by_kind: Counter[str] = Counter()
    total_by_kind: Counter[str] = Counter()
    for episode in episodes:
        transitions = episode["transitions"]
        for previous, current in zip(transitions, transitions[1:]):
            before, after = previous["tokens"], current["tokens"]
            if before["ids"].shape != after["ids"].shape:
                continue
            same = ((before["ids"] == after["ids"]).all(axis=1)
                    & (before["features"] == after["features"]).all(axis=1))
            for row in range(len(same)):
                name = KIND_NAMES[int(after["ids"][row, 0])]
                total_by_kind[name] += 1
                compared += 1
                if same[row]:
                    identical += 1
                    identical_by_kind[name] += 1
    print(f"  overall: {identical:,}/{compared:,} tokens unchanged "
          f"({identical / compared * 100:.1f}%)")
    print(f"  {'kind':16s} {'unchanged':>10s} {'total':>8s} {'share':>7s}")
    for name, count in total_by_kind.most_common():
        share = identical_by_kind[name] / count * 100
        print(f"  {name:16s} {identical_by_kind[name]:10d} {count:8d} {share:6.1f}%")

    # ------------------------------------------------- empty cells in the sequence
    print()
    print("--- cells: how many carry anything beyond their own terrain ---")
    empty_cells = 0
    all_cells = 0
    for episode in episodes:
        for transition in episode["transitions"]:
            ids = transition["tokens"]["ids"]
            features = transition["tokens"]["features"].astype("float32")
            for row in range(len(ids)):
                if KIND_NAMES[int(ids[row, 0])] != "cell":
                    continue
                all_cells += 1
                # feature[4] = plant count, feature[5..7] = grid-item flags
                if features[row, 4] == 0.0 and features[row, 5:8].sum() == 0.0:
                    empty_cells += 1
    print(f"  cell tokens: {all_cells:,}; with no plant and no grid item: "
          f"{empty_cells:,} ({empty_cells / all_cells * 100:.1f}%)")
    print("  those still occupy 8/32 feature slots and a full attention slot each;")
    print("  what they contribute is their (row, col, terrain, row_type), which is")
    print("  fixed at reset.")

    # ------------------------------------------------------- what never arrives
    print()
    print("--- observation fields with no token at all ---")
    print("  these are read by something, but observation_tokens never touches them:")
    for name, note in (
        ("legal_actions", "consumed as a mask by legal_summary(); 43.6% of the wire, 0 tokens"),
        ("level", "no token; the model cannot tell which level it is playing"),
        ("terrain", "no token; only cells[].terrain reaches the encoder"),
        ("enemy_zombies_on_screen", "no token; only scripted_baseline.py reads it"),
        ("coins", "no reader anywhere"),
        ("grid", "no reader anywhere"),
    ):
        print(f"    {name:26s} {note}")

    # --------------------------------------------------------- packed storage
    packed_bytes = 0
    for episode in episodes:
        for transition in episode["transitions"]:
            packed = transition["tokens"]
            packed_bytes += packed["ids"].nbytes + packed["features"].nbytes
    per_episode = packed_bytes / len(episodes)
    print()
    print(f"--- packed storage ---")
    print(f"  per token: {5} B ids + {FEATURE_COUNT * 2} B float16 features = "
          f"{5 + FEATURE_COUNT * 2} B")
    print(f"  per episode: {per_episode / 1024:.1f} KiB"
          f"  ({per_episode / decisions:.0f} B per decision)")
    print(f"  project 2000 episodes: {per_episode * 2000 / 1024**2:.1f} MiB")


def _unused() -> None:
    """Placeholder kept so the module imports cleanly during development."""


if __name__ == "__main__":
    main()
