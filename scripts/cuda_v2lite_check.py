"""Bitwise check of the CUDA core's *student* encoder (v2lite: 96 channels + 116 scalars).

The value-net encoder check (cuda_leaf_compare.py --root-features) compares 92 channels + 113
scalars + task context.  A v2lite student reads four more board channels and one more scalar, and
those depend on history the *student* carries (plant birth ticks, wave start tick).  That history
lives in Python during search and in the root record on the GPU, so it has its own check:

  1. play the task with the v2lite student, keeping a StudentHistory exactly as search_play does;
  2. at every decision with a legal plant, export the position (CUDA_DUMP ... V2H) -- the same call
     the GPU search arm makes -- and build the vector Python's own encoder produces for it;
  3. ask the CUDA host core (sim_host --student-features) for the vector it builds after importing
     the same record;
  4. compare element for element.  Any differing float is an encoder bug: both sides see the same
     position and the same history, and no random stream runs between them.

Usage:
  python3 scripts/cuda_v2lite_check.py --manifest artifacts/stage4/day_l7_hard.json \
      --task-id day_l7_z15_train --rollout-policy ~/PvZ-search/artifacts/student/v20/model.pt \
      --net artifacts/cuda/student/v20.pvzrn --seeds 4 --roots-per-game 8
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))


def _read_student_blocks(path: Path) -> tuple[int, int, list]:
    import struct
    blob = path.read_bytes()
    magic, count, width = struct.unpack_from("<8sii", blob, 0)
    if magic != b"PVZSTU01":
        raise SystemExit(f"{path}: bad magic {magic!r}")
    offset = 16
    blocks = [struct.unpack_from(f"<{width}f", blob, offset + index * width * 4)
              for index in range(count)]
    return count, width, blocks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--rollout-policy", type=Path, required=True)
    parser.add_argument("--net", type=Path, required=True, help="exported .pvzrn for the same student")
    parser.add_argument("--sim-host", type=Path, default=ROOT / "build" / "sim_host")
    parser.add_argument("--resource-dir", type=Path,
                        default=Path.home() / ".cache" / "pvz-research-resources")
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--roots-per-game", type=int, default=8)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "cuda" / "v2lite_check")
    parser.add_argument("--show", type=int, default=12)
    args = parser.parse_args()

    import torch
    torch.set_num_threads(1)
    from pvz_env import TaskSpec
    from pvz_event_env import EventWaitEnv
    from pvz_rollout_policy import act, load_student
    from pvz_student_observation import StudentHistory
    from render_episode import _frame, load_training_task, training_task_kwargs
    from scripted_baseline import profile_for_deck

    net = load_student(args.rollout_policy)
    if net.obs != "v2lite":
        raise SystemExit(f"{args.rollout_policy} is a {net.obs} student; this check is v2lite only")

    args.out.mkdir(parents=True, exist_ok=True)
    roots_path = args.out / "roots.cudaroot"
    features_path = args.out / "student_features.bin"

    kw = load_training_task(args.manifest, args.task_id)
    deck = tuple(kw["deck"])
    seeds = kw["seeds"][: args.seeds]
    expected: list = []
    metadata: list[dict] = []

    env = EventWaitEnv(args.resource_dir, headless=True)
    try:
        for seed in seeds:
            task = TaskSpec(level=kw["level"], seed=seed, playthrough=2,
                            profile=profile_for_deck(deck),
                            **{k: v for k, v in training_task_kwargs(kw, seed)["task_extra"].items()
                               if v is not None})
            obs, _ = env.reset(deck=deck, task=task)
            env.set_lean_observation(True)
            history = StudentHistory()
            previous_frame = None
            previous_action = None
            roots = 0
            while not obs["terminal"] and roots < args.roots_per_game:
                frame = _frame(obs)
                frame["action"] = previous_action
                legal = (obs.get("legal_actions") or {}).get("plants") or []
                if legal:
                    # The history as it stands *before* this frame is observed, which is exactly
                    # what search_play passes to CUDA_DUMP (its copy was advanced at the previous
                    # decision, not at this position).
                    response = env.cuda_dump(roots_path, write_level=not expected,
                                             student_history=history)
                    if response.get("supported"):
                        # previous=None: the exported root carries no prior action (the GPU's first
                        # decision gets one from the forced first action, not from this position).
                        board, scalars = copy.deepcopy(history).features(frame, None)
                        expected.append(torch.cat([board.reshape(-1), scalars]).numpy())
                        metadata.append({"seed": seed, "tick": obs["tick"], "wave": obs["wave"],
                                         "plants": len(obs.get("plants") or []),
                                         "zombies": len(obs.get("zombies") or [])})
                        roots += 1
                action = act(net, frame, previous_frame, legal, history=history)
                previous_frame, previous_action = frame, action
                obs, _reward, done, _truncated, info = env.step(action)
                if not info.get("ok"):
                    obs, _reward, done, _truncated, info = env.step({"type": "wait", "ticks": 60})
                if done:
                    break
            env.set_lean_observation(False)
    finally:
        env.close()

    if not expected:
        raise SystemExit("no supported roots were exported")
    print(f"exported {len(expected)} roots from {len(seeds)} games into {roots_path}")

    import subprocess
    command = [str(args.sim_host), "--roots", str(roots_path), "--net", str(args.net),
               "--student-features", str(features_path)]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    print(completed.stdout.strip())

    count, width, blocks = _read_student_blocks(features_path)
    if count != len(expected):
        raise SystemExit(f"{features_path}: {count} blocks for {len(expected)} roots")
    if width != expected[0].size:
        raise SystemExit(f"{features_path}: {width} floats per root, python built {expected[0].size}")

    import numpy as np
    bad = 0
    worst = 0
    shown = 0
    for index, (want, block) in enumerate(zip(expected, blocks)):
        got = np.asarray(block, dtype=np.float32)
        differing = np.nonzero(want != got)[0]
        if differing.size == 0:
            continue
        bad += 1
        worst = max(worst, int(differing.size))
        if shown < args.show:
            for where in differing[:3]:
                if where < 96 * 60:
                    channel, cell = divmod(int(where), 60)
                    where_text = f"channel {channel} cell {cell}"
                else:
                    where_text = f"scalar {int(where) - 96 * 60}"
                print(f"  root {index} ({metadata[index]}): index {int(where)} {where_text}"
                      f" -- python {float(want[where]):.7g}, core {float(got[where]):.7g}")
                shown += 1
    print(f"roots compared {len(expected)}, exact {len(expected) - bad}, differing {bad} "
          f"(worst root {worst} floats of {width})")
    if bad:
        print("  -> the two student encoders disagree; this is a bug in one of them")
        raise SystemExit(1)
    print("  -> every root encodes identically: the core's BuildFeatures and "
          "StudentHistory.features are the same function")


if __name__ == "__main__":
    main()
