"""Compare the snapshot bytes two builds produce, via the hashes the protocol reports.

``BRANCH_SNAPSHOT_FAST`` returns a ``state_hash`` computed over the serialised board
image.  That makes it an exact, cheap fingerprint of the save-game writer: if an
optimisation to ``LawnSaveGameToMemory`` changed a single byte, every hash would move.

This matters because the hash is the search's transposition key.  Changing it would
silently change which branches the search considers duplicates -- a behaviour change that
a state-digest comparison cannot see, since the *game state* would be identical.

Both builds are driven through the same deterministic rollout, sampling the hash at
several points so the comparison covers many board configurations rather than one.

    python scripts/snapshot_hash_compare.py <exe-a> <exe-b> --resource-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from pvz_common import canonical_digest  # noqa: E402
from pvz_env import PvZEnv, branch_action_token, training_task  # noqa: E402

DECK = (0, 1, 2, 3, 4, 5)
PLANT_LIMIT = 12
BATCH = 8


def sample_hashes(executable: Path | None, resource_dir: Path, level: int, seed: int,
                  ticks: int, stride: int) -> dict[str, Any]:
    """Hashes of the child states branched from several points of one rollout."""
    task = training_task(seed, level)
    samples: list[dict[str, Any]] = []
    planted = 0
    error = None
    try:
        with PvZEnv(resource_dir, executable, headless=True) as env:
            observation = env.reset(deck=DECK, task=task)[0]
            for step in range(ticks):
                if observation["terminal"]:
                    break
                placements = observation["legal_actions"]["plants"]
                if placements and planted < PLANT_LIMIT:
                    observation = env.step({"type": "plant", **placements[0]})[0]
                    planted += 1
                else:
                    observation = env.step({"type": "wait", "ticks": 1})[0]

                if step % stride:
                    continue
                actions = [{"type": "plant", **placement} for placement in placements[:BATCH]]
                actions += [{"type": "wait", "ticks": 1}] * (BATCH - len(actions))
                tokens = " ".join(branch_action_token(action) for action in actions)
                parent = env._command("SNAPSHOT_FAST")["snapshot_id"]
                response = env._command(f"BRANCH_SNAPSHOT_FAST {parent} {len(actions)} {tokens}")
                if not response.get("ok"):
                    raise RuntimeError(f"branch failed at step {step}: {response}")
                samples.append({
                    "step": step,
                    "tick": observation["tick"],
                    "plants": len(observation["plants"]),
                    "zombies": len(observation["zombies"]),
                    "reanimations": len(env.privileged_state().get("hidden", {}).get("reanimations", [])),
                    "digest": canonical_digest(env.privileged_state()),
                    "hashes": [str(item.get("state_hash", "")) for item in response["branches"]],
                })
                for item in response["branches"]:
                    if item.get("snapshot_id") is not None:
                        env._command(f"DROP_SNAPSHOT_FAST {item['snapshot_id']}")
                # Branching leaves the simulator at the last branch's state, so the
                # rollout only stays deterministic because the parent is restored here.
                env._command(f"RESTORE_FAST {parent}")
                env._command(f"DROP_SNAPSHOT_FAST {parent}")
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    return {"samples": samples, "error": error, "planted": planted}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("exe_a", type=Path)
    parser.add_argument("exe_b", type=Path)
    parser.add_argument("--resource-dir", type=Path, required=True)
    parser.add_argument("--level", type=int, default=8)
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument("--ticks", type=int, default=1500)
    parser.add_argument("--stride", type=int, default=50)
    args = parser.parse_args()

    left = sample_hashes(args.exe_a, args.resource_dir, args.level, args.seed, args.ticks, args.stride)
    right = sample_hashes(args.exe_b, args.resource_dir, args.level, args.seed, args.ticks, args.stride)

    left_hashes = [sample["hashes"] for sample in left["samples"]]
    right_hashes = [sample["hashes"] for sample in right["samples"]]

    first_difference = None
    for index, (one, other) in enumerate(zip(left_hashes, right_hashes)):
        if one != other:
            first_difference = index
            break

    # The board metadata is a control: if it differs as well, the two runs played
    # different rollouts and the hash comparison means nothing.
    meta = [{key: sample[key] for key in ("step", "tick", "plants", "zombies", "reanimations", "digest")}
            for sample in left["samples"]]
    other_meta = [{key: sample[key] for key in ("step", "tick", "plants", "zombies", "reanimations", "digest")}
                  for sample in right["samples"]]
    first_meta_difference = next(
        (index for index, (one, other) in enumerate(zip(meta, other_meta)) if one != other), None)

    identical = (first_difference is None
                 and len(left_hashes) == len(right_hashes)
                 and left["error"] == right["error"] is None
                 and bool(left_hashes))

    print(json.dumps({
        "a": str(args.exe_a), "b": str(args.exe_b),
        "level": args.level, "seed": args.seed,
        "samples": len(left_hashes),
        "hashes_compared": sum(len(item) for item in left_hashes),
        "errors": [left["error"], right["error"]],
        "first_differing_sample": first_difference,
        "first_differing_board": first_meta_difference,
        "board_a": meta[first_meta_difference] if first_meta_difference is not None else None,
        "board_b": other_meta[first_meta_difference] if first_meta_difference is not None else None,
        "sample_a": left_hashes[first_difference] if first_difference is not None else None,
        "sample_b": right_hashes[first_difference] if first_difference is not None else None,
        "snapshot_bytes_identical": identical,
    }, indent=2))


if __name__ == "__main__":
    main()
