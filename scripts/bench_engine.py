#!/usr/bin/env python3
"""Engine CPU benchmark on real decision points.

The teacher's cost is dominated by C++ rollouts: every decision branches into
"candidate actions x verification samples" continuations, each of which steps the
engine tick by tick with the student network (EnvironmentRollout) choosing the
actions.  This script pins that workload down so every optimisation can be measured
on the same positions.

manifest  walks the frame archives (artifacts/student/frames_v2) and picks real
          decision points from four scene families -- day L7, original conveyor
          L1-10, night L12-14, pool L24 -- and writes them to a JSON list (archive,
          task spec, frame indices).  The task spec is rebuilt from the archive's
          meta.json so the episode can be replayed.
bench     replays each archive once, snapshots the chosen decision points, and runs a
          fixed number of C++ rollouts from each root.  It reports simulated
          ticks/second, seconds per rollout, and the student-network share of the
          time (both measured inside the simulator).
replay    the correctness fixture: replays whole episodes with the recorded action
          sequence and compares every frame with the archive, and checks that a
          rollout is reproducible from the same root and seed.
search    the whole-game metric: search_play with the queue's parameters, parsed into
          seconds per decision.

Only one simulator process is used at a time (the queue on this machine owns the
rest); --workers exists for the search metric only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

FRAMES_ROOT = Path.home() / "PvZ-search" / "artifacts" / "student" / "frames_v2"
STUDENT = Path.home() / "PvZ-search" / "artifacts" / "student" / "dw4i_s1" / "model.pt"
RESOURCES = Path.home() / ".cache" / "pvz-research-resources"
EXECUTABLE = ROOT / "build" / "pvz-portable"

# The queue (~/pvz-logs/queue103.sh) runs search_play with these; the benchmark has to
# measure the same workload or the numbers do not transfer.
QUEUE_ARGS = [
    "--verify-samples", "9", "--student-candidates", "2",
    "--static-value", str(Path.home() / "PvZ-search" / "artifacts" / "hard" / "value_branch1" / "model.pt"),
    "--value", str(Path.home() / "PvZ-search" / "artifacts" / "value_net" / "v3" / "model.pt"),
    "--dagger-beta", "0",
]


def scene_family(level: int, conveyor_original: bool) -> str | None:
    import pvz_constants
    if conveyor_original and level <= 10:
        return "original_1_10"
    background = pvz_constants.background_for_level(level)
    if background == "BACKGROUND_1_DAY" and level == 7:
        return "day_l7"
    if background == "BACKGROUND_2_NIGHT" and 12 <= level <= 14:
        return "night_12_14"
    if background == "BACKGROUND_3_POOL" and level == 24:
        return "pool_24"
    return None


LABEL_KIND = re.compile(r"^(?P<kind>[dnp])(?P<level>\d+)")
LABEL_ORIGINAL = re.compile(r"d(?P<level>\d+)orig")


def label_family(name: str) -> str | None:
    """Cheap family guess from the archive label, so the scan only reads matching metas.

    Labels are task ids with a run suffix: d7dag2 (day L7), n13dag2 (night L13), p24dag1
    (pool L24), c1d10origm125 (original conveyor L10).  scene_family still confirms the
    level and the background from the meta file.
    """
    original = LABEL_ORIGINAL.search(name)
    if original is not None and int(original.group("level")) <= 10:
        return "original_1_10"
    match = LABEL_KIND.match(name)
    if match is None:
        return None
    kind, level = match.group("kind"), int(match.group("level"))
    if kind == "d" and level == 7:
        return "day_l7"
    if kind == "n" and 12 <= level <= 14:
        return "night_12_14"
    if kind == "p" and level == 24:
        return "pool_24"
    return None


def _archive_rows(path: Path) -> list[dict]:
    from pvz_frame_store import read_frames
    return read_frames(path)


def _archive_spec(meta: dict, rows: list[dict]) -> dict | None:
    task = meta.get("task") or {}
    context = meta.get("context") or {}
    level = task.get("level", context.get("level"))
    seed = task.get("seed", context.get("seed"))
    if level is None or seed is None or not rows:
        return None
    conveyor = task.get("conveyor_start")
    # The deck is the seed bank the episode started with.  context.roster is NOT it (it is the
    # subset the archive recorded per frame), so read the types out of the first frame packets;
    # the original conveyor start has no task deck at all.
    deck = [] if conveyor == "original" else [int(packet[1]) for packet in rows[0].get("packets") or []]
    multiplier = task.get("zombie_count_multiplier", context.get("multiplier", 1.0))
    return {
        "level": int(level), "seed": int(seed), "deck": deck,
        "multiplier": float(multiplier),
        "first_aid_owned": bool(task.get("first_aid_owned", False)),
        "conveyor_start": conveyor,
        "task_id": task.get("task_id"),
    }


def build_manifest(args) -> dict:
    """Pick --archives episodes per family and --decisions points in each."""
    families: dict[str, list] = {}
    seeds = sorted(path for path in args.frames_root.glob("seed_*") if path.is_dir())
    # The archives span seed_1600000 (day L7) to seed_5083975 (original conveyor), so a scan
    # that stops at the first N directories only ever sees day L7.  Sample evenly instead.
    stride = max(1, len(seeds) // max(1, args.scan_seeds))
    seeds = seeds[::stride]
    scanned = 0
    for seed_dir in seeds:
        if len(families) == 4 and all(len(v) >= args.archives for v in families.values()):
            break
        scanned += 1
        for archive in sorted(seed_dir.iterdir()):
            if not archive.is_dir():
                continue
            candidate = label_family(archive.name)
            if candidate is None or len(families.get(candidate, [])) >= args.archives:
                continue
            meta_path = archive / "meta.json"
            if not meta_path.is_file():
                continue
            try:
                meta = json.loads(meta_path.read_text())
            except (OSError, ValueError):
                continue
            try:
                rows = _archive_rows(archive / ("frames.npz" if (archive / "frames.npz").is_file()
                                                else "frames.jsonl"))
            except Exception:  # a truncated archive must not abort the scan
                continue
            spec = _archive_spec(meta, rows)
            if spec is None or len(rows) < args.min_frames:
                continue
            family = scene_family(spec["level"], spec["conveyor_start"] == "original")
            if family != candidate or len(families.get(family, [])) >= args.archives:
                continue
            if spec["conveyor_start"] != "original" and not 6 <= len(spec["deck"]) <= 10:
                continue
            points = []
            for fraction in args.fractions:
                index = max(1, min(len(rows) - 1, int(round(fraction * (len(rows) - 1)))))
                if index in points:
                    continue
                row = rows[index]
                if row.get("terminal") or not row.get("action") or int(row.get("tick") or 0) <= 0:
                    continue
                points.append(index)
            if len(points) < args.decisions:
                continue
            families.setdefault(family, []).append({
                "archive": str(archive),
                "frames": len(rows),
                "decisions": points[: args.decisions],
                **spec,
            })
    missing = [name for name in ("day_l7", "original_1_10", "night_12_14", "pool_24")
               if name not in families]
    if missing:
        raise SystemExit(f"manifest scan found no archives for: {missing} (scanned {scanned} seeds)")
    manifest = {"frames_root": str(args.frames_root), "scanned_seeds": scanned, "families": families}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=1) + "\n")
    total = sum(len(v) for v in families.values())
    print(f"manifest: {total} archives, {total * args.decisions} decision points -> {args.out}")
    for name, entries in families.items():
        print(f"  {name}: {len(entries)} archives x {len(entries[0]['decisions'])} points")
    return manifest


def _task_spec(spec: dict):
    from pvz_env import TaskSpec
    from pvz_tasks import profile_for_deck
    deck = tuple(spec["deck"])
    return TaskSpec(level=spec["level"], seed=spec["seed"], playthrough=2,
                    profile=profile_for_deck(deck, spec.get("first_aid_owned", False)),
                    zombie_count_multiplier=spec["multiplier"],
                    conveyor_start=spec.get("conveyor_start"))


def _recorded_action(action: dict) -> dict:
    """The archive stores the action exactly as the search executed it.

    Waits carry until=timeout and genuinely stop early when the event fires (measured: the
    archives contain 2/3/4/28/33/46/49/53-tick waits, not only 60), so replay has to go through
    the same event-wait path (EventWaitEnv) instead of flattening them to a fixed wait.
    """
    if not action:
        raise ValueError("frame has no action")
    return dict(action)


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


COMPARE_KEYS = ("tick", "wave", "sun", "plants", "zombies", "mowers", "packets", "graves",
                "lanes", "row_terrain", "scene")


def _frame(observation: dict) -> dict:
    from render_episode import _frame as compact
    return compact(observation)


def _state_of(frame: dict) -> dict:
    return {key: frame.get(key) for key in COMPARE_KEYS if key in frame}


ROLLOUT_STAT_KEYS = ("estimated_result", "decisions", "mowers_left", "planted", "net_calls",
                     "ticks", "substeps")


def _rollout_digest(observation: dict, response: dict) -> str:
    # Timers vary between runs; the leaf, outcome and every logical statistic must agree.
    return _digest({"state": observation,
                    "stats": {key: response[key] for key in ROLLOUT_STAT_KEYS}})


def _require_identical(a: dict, b: dict) -> None:
    if not a["families"] or not b["families"]:
        raise RuntimeError("correctness gate: empty benchmark")
    if a["families"].keys() != b["families"].keys():
        raise RuntimeError("correctness gate: benchmark families differ")
    for family, left in a["families"].items():
        right = b["families"][family]
        for value in (left, right):
            if (value["rollouts"] <= 0 or value["replay_mismatches"] or
                    len(value["rollout_digests"]) != value["rollouts"]):
                raise RuntimeError(f"correctness gate: incomplete or invalid evidence for {family}")
        if left["rollout_digests"] != right["rollout_digests"]:
            raise RuntimeError(f"correctness gate: rollout state/statistics differ for {family}")


def _export_student(path: Path) -> None:
    from pvz_rollout_policy import export_student, load_student
    export_student(load_student(str(STUDENT)), path)


def replay_episode(env, spec: dict, rows: list[dict], decisions: list[int]):
    """Replay the recorded actions, snapshotting the requested decision points.

    Returns (snapshots, mismatches) where snapshots maps frame index -> snapshot id.
    """
    # The deck travels as reset()'s own argument, not inside TaskSpec: that is how every
    # other caller (search_play, sim_skip_agreement) reconstructs an episode.
    env.reset(deck=tuple(spec["deck"]), task=_task_spec(spec))
    snapshots = {}
    mismatches = []
    last = max(decisions)
    for index in range(1, last + 1):
        if index in decisions:
            snapshots[index] = env.snapshot()
        observation, _reward, terminal, _truncated, _info = env.step(_recorded_action(rows[index].get("action")))
        expected = _state_of(rows[index])
        # The archive rows are compact frames (render_episode._frame); the live observation is
        # the full protocol payload.  Compare through the same projection.
        actual = _state_of(_frame(observation))
        for key in expected:
            # Archives are JSON (lists); _frame returns tuples.  Compare the JSON form.
            if _digest(actual.get(key)) != _digest(expected[key]):
                mismatches.append({"frame": index, "key": key,
                                   "expected": expected[key], "actual": actual.get(key)})
                break
        if terminal:
            break
    return snapshots, mismatches


def _arm(env, on: bool) -> None:
    """Tell the SIGPROF sampler which phase to count (scripts/prof_sample.c).

    Without LD_PRELOAD the simulator has no handler and the signal kills it, so only send
    it when the sampler was requested.
    """
    process = getattr(env, "_process", None)
    if process is None or process.poll() is not None or not os.environ.get("PVZ_PROF_ARM_ONLY"):
        return
    try:
        os.kill(process.pid, signal.SIGUSR1 if on else signal.SIGUSR2)
    except OSError:
        pass


def _process_cpu(pid: int) -> float:
    """CPU seconds the simulator has burned (utime+stime from /proc/<pid>/stat).

    The in-simulator timers are steady_clock, so on a machine that is also running the
    collection queue they inherit every preemption.  Process CPU time is the number that
    survives a noisy neighbour, and it is what this benchmark compares across builds.
    """
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return 0.0
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


def _open_env(executable: str):
    from pvz_event_env import EventWaitEnv
    return EventWaitEnv(str(RESOURCES), executable=executable, headless=True)


def bench(args) -> dict:
    manifest = json.loads(args.manifest.read_text())
    with tempfile.TemporaryDirectory(prefix="pvz-bench-engine-") as directory:
        net_path = Path(directory) / "student.bin"
        _export_student(net_path)
        env = _open_env(str(args.executable))
        try:
            env.load_rollout_net(net_path)
            # The replay phase is not the workload being measured, and it must not skew the
            # profile: run every replay first, then arm the sampler for the rollouts only.
            env.set_sim_skip(args.sim_skip)
            report = {"benchmark_version": 2, "executable": str(args.executable), "student": str(STUDENT),
                      "sim_skip": args.sim_skip, "horizon": args.horizon,
                      "rollouts_per_root": args.rollouts, "families": {}}
            replay_failures = 0
            for family, entries in manifest["families"].items():
                ticks = simulated = network = wall = process_cpu = 0.0
                rollouts = decisions = roots = 0
                leaves, rollout_digests = [], []
                for entry in entries:
                    archive = Path(entry["archive"])
                    rows = _archive_rows(archive / ("frames.npz" if (archive / "frames.npz").is_file()
                                                    else "frames.jsonl"))
                    # RESET_V2 clears the snapshot table, so every archive is replayed and
                    # measured before the next one starts.
                    snapshots, mismatches = replay_episode(env, entry, rows, entry["decisions"])
                    if mismatches:
                        replay_failures += len(mismatches)
                        raise RuntimeError(f"correctness gate: replay mismatch {archive}: {mismatches[0]}")
                    if len(snapshots) != len(entry["decisions"]):
                        raise RuntimeError(f"correctness gate: missing decision snapshots for {archive}")
                    _arm(env, True)
                    cpu_before = _process_cpu(env._process.pid)
                    for index, snapshot in snapshots.items():
                        roots += 1
                        for repeat in range(args.rollouts):
                            # The snapshot precedes rows[index]'s recorded action;
                            # that row's tick is the state *after* the action.
                            start_tick = int(env.restore(snapshot)["tick"])
                            started = time.perf_counter()
                            observation = env.rollout(
                                None, start_tick + args.horizon, sim_skip=args.sim_skip,
                                max_actions=args.max_actions, snapshot_id=snapshot,
                                seed=args.seed_base + repeat)
                            wall += time.perf_counter() - started
                            last = env._last_response
                            ticks += last.get("ticks", 0)
                            simulated += last.get("simulation_seconds", 0.0)
                            network += last.get("network_seconds", 0.0)
                            rollouts += 1
                            decisions += last.get("decisions", 0)
                            leaves.append(_digest([_state_of(observation), observation.get("tick")]))
                            rollout_digests.append({"archive": str(archive), "frame": index,
                                                    "seed": args.seed_base + repeat,
                                                    "digest": _rollout_digest(observation, last)})
                        env.release_snapshot(snapshot)
                    process_cpu += _process_cpu(env._process.pid) - cpu_before
                    _arm(env, False)
                cpu = simulated + network
                report["families"][family] = {
                    "archives": len(entries), "roots": roots, "rollouts": rollouts,
                    "ticks": ticks, "decisions": decisions,
                    "simulation_seconds": round(simulated, 4), "network_seconds": round(network, 4),
                    "wall_seconds": round(wall, 4),
                    "ticks_per_second": round(ticks / cpu, 1) if cpu else None,
                    "process_cpu_seconds": round(process_cpu, 3),
                    "ticks_per_cpu_second": round(ticks / process_cpu, 1) if process_cpu else None,
                    "seconds_per_rollout": round(wall / rollouts, 4) if rollouts else None,
                    "network_share": round(network / cpu, 4) if cpu else None,
                    "decisions_per_second": round(decisions / cpu, 1) if cpu else None,
                    "replay_mismatches": replay_failures,
                    "leaf_digests": leaves,
                    "rollout_digests": rollout_digests,
                }
                print(f"{family}: " + json.dumps({key: value for key, value in report['families'][family].items()
                                                 if key not in ('leaf_digests', 'rollout_digests')}), flush=True)
            aggregate = {"ticks": sum(v["ticks"] for v in report["families"].values()),
                         "cpu_seconds": sum(v["simulation_seconds"] + v["network_seconds"]
                                            for v in report["families"].values()),
                         "network_seconds": sum(v["network_seconds"] for v in report["families"].values()),
                         "wall_seconds": sum(v["wall_seconds"] for v in report["families"].values()),
                         "rollouts": sum(v["rollouts"] for v in report["families"].values()),
                         "decisions": sum(v["decisions"] for v in report["families"].values()),
                         "process_cpu_seconds": sum(v["process_cpu_seconds"]
                                                    for v in report["families"].values())}
            aggregate["ticks_per_second"] = round(aggregate["ticks"] / aggregate["cpu_seconds"], 1)
            aggregate["ticks_per_cpu_second"] = round(
                aggregate["ticks"] / aggregate["process_cpu_seconds"], 1)
            aggregate["seconds_per_rollout_cpu"] = round(
                aggregate["process_cpu_seconds"] / aggregate["rollouts"], 4)
            aggregate["seconds_per_rollout"] = round(aggregate["wall_seconds"] / aggregate["rollouts"], 4)
            aggregate["network_share"] = round(aggregate["network_seconds"] / aggregate["cpu_seconds"], 4)
            report["total"] = aggregate
            print("total: " + json.dumps(aggregate), flush=True)
        finally:
            env.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"-> {args.out}")
    return report


def compare(args) -> dict:
    """Paired A/B measurement: alternate the two binaries so machine load cancels out.

    This machine runs the collection queue at the same time (13 simulators, load ~13 on 20
    cores), so a sequential "before, then after" comparison drifts by several percent.  Every
    round here runs both binaries back to back and the result is the median of the ratios.
    """
    import statistics
    if args.executable.resolve() == args.compare.resolve():
        raise ValueError("compare requires two different executable paths")
    ratios, per_binary = [], {str(args.executable): [], str(args.compare): []}
    evidence = args.out.parent / (args.out.stem + "_rounds")
    reference = None
    for _round in range(args.rounds):
        round_values = {}
        # Alternate which binary runs first to reduce ordering bias.
        order = (args.executable, args.compare) if _round % 2 == 0 else (args.compare, args.executable)
        for binary in order:
            arm = "a" if binary == args.executable else "b"
            namespace = argparse.Namespace(**{**vars(args), "executable": binary,
                                              "out": evidence / f"round_{_round + 1}_{arm}.json"})
            report = bench(namespace)
            _require_identical(report if reference is None else reference, report)
            if reference is None:
                reference = report
            value = report["total"]["ticks_per_cpu_second"]
            per_binary[str(binary)].append(value)
            round_values[str(binary)] = value
        ratios.append(round_values[str(args.compare)] / round_values[str(args.executable)])
        print(f"round {_round + 1}: {ratios[-1]:.4f}x", flush=True)
    summary = {"a": str(args.executable), "b": str(args.compare),
               "a_median": statistics.median(per_binary[str(args.executable)]),
               "b_median": statistics.median(per_binary[str(args.compare)]),
               "ratios": ratios, "ratio_median": statistics.median(ratios),
               "correctness": "all rollout states and logical statistics identical",
               "evidence": str(evidence)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return summary


def replay(args) -> dict:
    """Correctness fixture: whole-episode replay vs the archive, and rollout reproducibility."""
    manifest = json.loads(args.manifest.read_text())
    report = {"executable": str(args.executable), "episodes": [], "determinism": []}
    env = _open_env(str(args.executable))
    try:
        with tempfile.TemporaryDirectory(prefix="pvz-bench-replay-") as directory:
            net_path = Path(directory) / "student.bin"
            _export_student(net_path)
            env.load_rollout_net(net_path)
            for family, entries in manifest["families"].items():
                for entry in entries[: args.episodes]:
                    archive = Path(entry["archive"])
                    rows = _archive_rows(archive / ("frames.npz" if (archive / "frames.npz").is_file()
                                                    else "frames.jsonl"))
                    frames = [index for index in range(1, len(rows)) if rows[index].get("action")]
                    snapshots, mismatches = replay_episode(env, entry, rows, frames[-1:] if frames else [1])
                    report["episodes"].append({
                        "family": family, "archive": entry["archive"], "frames": len(frames),
                        "mismatches": len(mismatches), "first_mismatch": mismatches[:1]})
                    print(f"replay {family} {archive.parent.name}: {len(frames)} frames, "
                          f"{len(mismatches)} mismatches", flush=True)
                    for snapshot in snapshots.values():
                        env.release_snapshot(snapshot)
                    # Rollout reproducibility: same root, same seed, twice.
                    snapshots, _ = replay_episode(env, entry, rows, entry["decisions"][:1])
                    for index, snapshot in snapshots.items():
                        digests = []
                        for _ in range(2):
                            start_tick = int(env.restore(snapshot)["tick"])
                            observation = env.rollout(None, start_tick + args.horizon,
                                                      sim_skip=args.sim_skip, max_actions=args.max_actions,
                                                      snapshot_id=snapshot, seed=args.seed_base)
                            digests.append(_rollout_digest(observation, env._last_response))
                        env.release_snapshot(snapshot)
                        report["determinism"].append({"family": family, "archive": entry["archive"],
                                                      "frame": index, "identical": digests[0] == digests[1],
                                                      "digest": digests[0]})
                        print(f"  rollout determinism frame {index}: {digests[0] == digests[1]}", flush=True)
    finally:
        env.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"-> {args.out}")
    if any(entry["mismatches"] for entry in report["episodes"]) or any(
            not entry["identical"] for entry in report["determinism"]):
        raise RuntimeError(f"correctness gate failed: {args.out}")
    return report


def summarize_games(rows: list[dict], expected_games: int) -> dict:
    if not rows or len(rows) != expected_games or len({row["seed"] for row in rows}) != expected_games:
        raise RuntimeError(f"incomplete whole-game benchmark: {len(rows)} rows for {expected_games} games")
    report = {"games": len(rows), "won": sum(1 for row in rows if row.get("won")),
              "mean_seconds": sum(row["seconds"] for row in rows) / len(rows),
              "mean_decisions": sum(row["decisions"] for row in rows) / len(rows)}
    for key in ("simulation_seconds", "network_seconds"):
        report["mean_" + key] = sum(row[key] for row in rows) / len(rows) if all(key in row for row in rows) else None
    report["milliseconds_per_decision"] = round(
        1000 * report["mean_seconds"] / report["mean_decisions"], 3) if report["mean_decisions"] else None
    report["games_per_hour"] = round(3600 * len(rows) / sum(row["seconds"] for row in rows), 2)
    return report


def search(args) -> dict:
    """Whole-game metric: the queue's search_play parameters, seconds per decision."""
    if not 1 <= args.workers <= 2:
        raise ValueError("task35 permits at most two game workers")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(ROOT / "scripts" / "search_play.py"),
               "--rollout-policy", str(STUDENT), *QUEUE_ARGS,
               "--manifest", str(args.manifest), "--task-id", args.task_id,
               "--seed-offset", str(args.seed_offset), "--seeds", str(args.games),
               "--executable", str(args.executable), "--workers", str(args.workers),
               "--resource-dir", str(RESOURCES), "--out", str(out)]
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=str(ROOT))
    rows = [json.loads(line) for line in (out / "games.jsonl").read_text().splitlines() if line.strip()]
    report = summarize_games(rows, args.games)
    (out / "engine_search.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("manifest", help="pick decision points from the frame archives")
    p.add_argument("--frames-root", type=Path, default=FRAMES_ROOT)
    p.add_argument("--out", type=Path, default=ROOT / "artifacts" / "perf" / "engine_roots.json")
    p.add_argument("--archives", type=int, default=4)
    p.add_argument("--decisions", type=int, default=4)
    p.add_argument("--fractions", type=float, nargs="+", default=[0.15, 0.4, 0.65, 0.85])
    p.add_argument("--min-frames", type=int, default=200)
    p.add_argument("--scan-seeds", type=int, default=4000)

    p = sub.add_parser("compare", help="paired A/B of two binaries (machine load cancels)")
    p.add_argument("--manifest", type=Path, default=ROOT / "artifacts" / "perf" / "engine_roots.json")
    p.add_argument("--executable", type=Path, default=EXECUTABLE)
    p.add_argument("--compare", type=Path, required=True)
    p.add_argument("--out", type=Path, default=ROOT / "artifacts" / "perf" / "engine_compare.json")
    p.add_argument("--horizon", type=int, default=2500)
    p.add_argument("--max-actions", type=int, default=1000)
    p.add_argument("--sim-skip", type=int, default=61)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--rollouts", type=int, default=3)
    p.add_argument("--rounds", type=int, default=4)

    for name in ("bench", "replay"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", type=Path, default=ROOT / "artifacts" / "perf" / "engine_roots.json")
        p.add_argument("--executable", type=Path, default=EXECUTABLE)
        p.add_argument("--out", type=Path,
                       default=ROOT / "artifacts" / "perf" / f"engine_{name}.json")
        p.add_argument("--horizon", type=int, default=2500)
        p.add_argument("--max-actions", type=int, default=1000)
        p.add_argument("--sim-skip", type=int, default=61)
        p.add_argument("--seed-base", type=int, default=0)
        if name == "bench":
            p.add_argument("--rollouts", type=int, default=3)
        else:
            p.add_argument("--episodes", type=int, default=1)

    p = sub.add_parser("search", help="whole-game search_play timing")
    p.add_argument("--manifest", type=Path,
                   default=Path.home() / "PvZ-search" / "artifacts" / "stage4" / "day_panel.json")
    p.add_argument("--task-id", default="day_l7_eval")
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--games", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--executable", type=Path, default=EXECUTABLE)
    p.add_argument("--out", type=Path, default=ROOT / "artifacts" / "perf" / "engine_search")

    args = parser.parse_args()
    {"manifest": build_manifest, "bench": bench, "replay": replay, "search": search,
     "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
