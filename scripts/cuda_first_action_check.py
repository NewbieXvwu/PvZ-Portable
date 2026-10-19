"""The forced first action, checked against the original with no random stream in the way.

cuda/spec/host.md finding 1: the protocol's own words are `isFirst = decisions == 0 && first != "A"`
(src/main.cpp:437) -- the candidate the teacher is scoring runs *before* the student is asked, without a
forward, but it still counts as a decision and still sets the previous-action context the next forward
reads.  The CUDA core had no way to express it at all, which made every continuation measure the
student's policy instead of the value of the action under test.  It is implemented now (Rollout.inc's
`RolloutBegin`, fed by the five `S_FIRST_*` lane scalars that `--pairs` lines can name), and this is the
check that it does what the original does.

Why this check is worth a script when a distribution comparison already exists: a *forced plant with no
ticks advanced* is deterministic.  Both sides start from the identical position, apply one named action,
and stop.  There is no future to disagree about, so any differing float of the value encoding is a bug in
one of the two action paths -- unlike the phase-1 leaf comparison, where the two sides deliberately run
different random streams and only a distribution can be the gate.

Three things get compared, because they fail for three different reasons:

  1. core leaf vs original child   -- the semantics of ActionPlant vs ExecuteEnvironmentAction("PLANT")
  2. device leaf vs host leaf      -- the plumbing of a --pairs line into a lane's five scalars
                                     (a difference here is a GPU bug by construction: same code twice)
  3. root block vs its own leaf    -- what the action changed at all, which is how a candidate that
                                     landed in the wrong cell or silently double-planted reads as a
                                     game fact ("a sunflower at (2,4)") instead of as offset 1934

Usage:
  python3 scripts/cuda_first_action_check.py --manifest ~/PvZ-search/artifacts/stage4/day_l7_hard.json \
      --task-id day_l7_z15_eval --seeds 2 --roots-per-game 2 \
      --rollout-policy ~/PvZ-search/artifacts/student/v14/model.pt \
      --value ~/PvZ-search/artifacts/value_net/v2/model.pt --work /tmp/forced

Exits non-zero on any differing float, and prints what each differing index *is* (the naming lives in
scripts/cuda_leaf_compare.py::describe, shared with the encoder check).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _directory in (ROOT / "python", ROOT / "scripts"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import cuda_leaf_compare as leaf  # noqa: E402  (describe/read_features are the shared encoder vocabulary)

BATCH_LIMIT = 128  # src/main.cpp caps BRANCH_OBSERVE at 128 specs per command


def _candidates(obs: dict, per_root: int) -> list[dict]:
    """Legal plants to test, spread over the packets rather than taken from the first one.

    `legal_actions.plants` is ordered packet-major, so a plain prefix would test two packets' lawns and
    never touch the rest of the deck -- and the packet columns (`F_PACKET_ACTIVE` / cooldown / cost) are
    exactly the part of the state a plant action mutates.
    """
    plants = obs.get("legal_actions", {}).get("plants") or []
    by_packet: dict[int, list[dict]] = {}
    for action in plants:
        # The observation names a cell and nothing else; BRANCH_OBSERVE's spec token wants a typed
        # action (python/pvz_env.py::branch_action_token), and `plant` is the only kind a legal cell
        # can be.
        typed = {"type": "plant", "packet": int(action["packet"]),
                 "row": int(action["row"]), "col": int(action["col"])}
        by_packet.setdefault(typed["packet"], []).append(typed)
    if not by_packet:
        return []
    per_packet = max(1, per_root // len(by_packet))
    out: list[dict] = []
    for packet in sorted(by_packet):
        cell = by_packet[packet]
        stride = max(1, len(cell) // per_packet)
        out.extend(cell[index] for index in range(0, len(cell), stride)[:per_packet])
    return out


def _changed(want, got, show: int) -> list[str]:
    """Every differing entry, named -- used both for a mismatch and for the 'what did the action change'
    reading, which is the same function applied to the root block and the leaf block."""
    lines = []
    for index in (i for i in range(len(want)) if want[i] != got[i]):
        if len(lines) < show:
            lines.append(f"      {leaf.describe(index)}: {want[index]:.7g} -> {got[index]:.7g}")
        elif len(lines) == show:
            lines.append("      ...")
            break
    return lines


def _branch(env, root: int, actions: list[dict], context) -> list:
    """The original's child state for each action, as the value net's own vector."""
    import numpy as np
    from pvz_value_net import featurize_numpy
    from render_episode import _frame

    children = []
    for start in range(0, len(actions), BATCH_LIMIT):
        chunk = actions[start:start + BATCH_LIMIT]
        branches = env.branch_snapshot(root, chunk, retain_snapshots=False)
        for spec, branch in zip(chunk, branches):
            if not branch.get("ok"):
                raise SystemExit(f"original rejected its own legal action {spec}: {branch}")
            child = branch["observation"]
            board, scalars = featurize_numpy(_frame(child))
            children.append((child, np.concatenate([board.reshape(-1), scalars,
                                                    context.reshape(-1)]).astype(np.float32)))
    return children


def _run(driver: Path, arguments: list[str], label: str) -> None:
    completed = subprocess.run([str(driver)] + arguments, capture_output=True, text=True,
                               env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
    if completed.returncode != 0:
        print(completed.stdout[-4000:])
        print(completed.stderr[-4000:])
        raise SystemExit(f"{label} exited {completed.returncode}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--roots-per-game", type=int, default=2)
    ap.add_argument("--root-start", type=int, default=900)
    ap.add_argument("--root-stride", type=int, default=600)
    ap.add_argument("--candidates-per-root", type=int, default=24)
    ap.add_argument("--rollout-policy", required=True)
    ap.add_argument("--value", required=True)
    ap.add_argument("--net-export", type=Path, default=Path("/tmp/forced_student.pvzrn"))
    ap.add_argument("--work", type=Path, default=Path("/tmp/forced"))
    ap.add_argument("--host", type=Path, default=ROOT / "build" / "sim_host")
    ap.add_argument("--device", type=Path, default=ROOT / "build" / "rollout_cuda")
    ap.add_argument("--executable", default=str(ROOT / "build" / "pvz-portable"))
    ap.add_argument("--resource-dir", default=os.path.expanduser("~/.cache/pvz-research-resources"))
    ap.add_argument("--skip-device", action="store_true", help="compare the host core only")
    ap.add_argument("--show", type=int, default=6)
    args = ap.parse_args()
    args.manifest = os.path.expanduser(args.manifest)
    args.task_id = os.path.expanduser(args.task_id)
    args.rollout_policy = os.path.expanduser(args.rollout_policy)
    args.value = os.path.expanduser(args.value)
    args.resource_dir = os.path.expanduser(args.resource_dir)
    args.work.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    import numpy as np
    from pvz_env import PvZEnv, TaskSpec
    from pvz_rollout_policy import export_student, load_student
    from pvz_value_net import load_value_net, observation_context
    from render_episode import load_training_task, training_task_kwargs
    from pvz_tasks import profile_for_deck

    kw = load_training_task(Path(args.manifest), args.task_id)
    deck = tuple(kw["deck"])
    seeds = kw["seeds"][:args.seeds]
    export_student(load_student(args.rollout_policy), str(args.net_export))
    vnet = load_value_net(args.value)

    env = PvZEnv(args.resource_dir, executable=args.executable, headless=True)
    env.load_rollout_net(str(args.net_export))

    cudaroot = args.work / "roots.cudaroot"
    pairs = args.work / "pairs.txt"
    host_leaf = args.work / "host_leaf.bin"
    host_root = args.work / "host_root.bin"
    device_leaf = args.work / "device_leaf.bin"

    rows, originals = [], []
    for seed in seeds:
        task = TaskSpec(level=kw["level"], seed=seed, playthrough=2, profile=profile_for_deck(deck),
                        **{k: v for k, v in training_task_kwargs(kw, seed)["task_extra"].items()
                           if v is not None})
        obs, _ = env.reset(deck=deck, task=task)
        env.set_lean_observation(True)
        env.set_sim_skip(0)
        context = torch.from_numpy(observation_context(obs))
        root_tick, kept = args.root_start, 0
        while not obs["terminal"] and kept < args.roots_per_game and obs["tick"] < 24000:
            obs = env.rollout(None, root_tick, max_actions=3000)
            if obs["terminal"]:
                break
            root_tick = obs["tick"] + args.root_stride
            candidates = _candidates(obs, args.candidates_per_root)
            if not candidates:
                continue
            kept += 1
            handle = env.snapshot()
            env.restore(handle)
            dumped = env.cuda_dump(cudaroot, write_level=not rows)
            if not dumped["supported"]:
                raise SystemExit(f"root not importable by the core: {dumped['reason']}")
            root_index = len(rows)
            children = _branch(env, handle, candidates, context)
            env.restore(handle)
            env.release_snapshot(handle)
            rows.append({"root": root_index, "seed": seed, "tick": int(dumped["tick"]),
                        "candidates": candidates})
            originals.append(children)
        env.set_lean_observation(False)
    if not rows:
        raise SystemExit("no root with a legal plant in this sample; raise --seeds or --roots-per-game")

    with pairs.open("w", encoding="utf-8") as stream:
        for row in rows:
            for action in row["candidates"]:
                stream.write(f"{row['root']} {row['seed'] * 1000 + 7} 1 {action['packet']} "
                             f"{action['row']} {action['col']}\n")
    count = sum(len(row["candidates"]) for row in rows)
    print(f"{len(rows)} roots, {count} forced plants; one tick advanced per lane: 0 (max-actions 1)")

    base = ["--roots", str(cudaroot), "--net", str(args.net_export), "--pairs", str(pairs),
            "--horizon", "1", "--max-actions", "1"]
    _run(args.host, base + ["--root-features", str(host_root)], "sim_host --root-features")
    _run(args.host, base + ["--leaf-out", str(host_leaf)], "sim_host --leaf-out")
    if not args.skip_device:
        _run(args.device, base + ["--lanes", str(min(count, 4096)), "--out",
                                  str(args.work / "device_results.bin"),
                                  "--leaf-out", str(device_leaf)], "rollout_cuda")

    root_blocks = leaf.read_features(host_root)
    host_blocks = leaf.read_features(host_leaf)
    device_blocks = leaf.read_features(device_leaf) if not args.skip_device else []
    cursor = 0
    bad_original = bad_device = 0
    for row, children in zip(rows, originals):
        shown = 0
        for action, (child, want) in zip(row["candidates"], children):
            got = host_blocks[cursor]
            block = cursor
            cursor += 1
            differences = _changed(want, got, args.show)
            if differences and shown < 3:
                shown += 1
                bad_original += 1
                print(f"  root {row['root']}@{row['tick']} plant packet {action['packet']} "
                      f"at (row {action['row']}, col {action['col']}): "
                      f"{len([i for i in range(len(want)) if want[i] != got[i]])} floats differ "
                      f"(core tick {int(child['tick'])})")
                for line in differences:
                    print(line)
            elif differences:
                bad_original += 1
            if device_blocks and not np.array_equal(device_blocks[block], got):
                if shown < 3:
                    print(f"  root {row['root']} lane {block}: DEVICE leaf != HOST leaf")
                    for line in _changed(device_blocks[block], got, args.show):
                        print(line)
                bad_device += 1
        # What the action changed relative to the root itself: the reading that catches a candidate
        # planted in the wrong cell, or planted twice.
        if row["candidates"]:
            probe = row["candidates"][0]
            delta = _changed(root_blocks[row["root"]], host_blocks[cursor - len(row["candidates"])],
                             99)
            print(f"root {row['root']}@{row['tick']}: candidate 0 "
                  f"(packet {probe['packet']}, row {probe['row']}, col {probe['col']}) changed "
                  f"{len(delta)} entries of the encoding")
            for line in delta[:8]:
                print(line)
            if len(delta) > 8:
                print(f"      ... {len(delta) - 8} more")
    print(f"forced-action agreement: {count} candidates, {bad_original} differ from the original, "
          f"{bad_device} differ between device and host")
    if bad_original or bad_device:
        raise SystemExit(1)
    print("  -> every forced plant lands exactly where the original puts it, on both cores")
    print(json.dumps({"roots": len(rows), "candidates": count,
                      "differ_from_original": bad_original, "device_vs_host": bad_device}))


if __name__ == "__main__":
    main()
