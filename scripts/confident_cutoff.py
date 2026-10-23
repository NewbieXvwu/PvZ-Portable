#!/usr/bin/env python3
"""task40 step 1: how much can a rollout stop early when the value net is already certain?

The unit is a *root*: a real decision point taken from a frames_v2 archive (the same four
scene families task35 pinned down in scripts/bench_engine.py).  From every root the greedy
student is rolled out with the teacher's parameters (sim skip 61, at most MAX_ACTIONS
actions, nine futures).  The same root/seed/horizon is evaluated at t in {500, 1000, 1500,
2000} and at the full horizon 2500, and every leaf is scored exactly the way
search_play._leaf_values scores it (terminal result first, otherwise the value net with the
game's multiplier/roster context).

The output is per-rollout JSONL; `report` turns it into the tables of
MEMORY/docs/task40-confident-cutoff.md.  `prefix` checks the assumption that makes the
comparison meaningful: the t-tick leaf is the state the 2500-tick rollout passes through at
t (same root, same seed), compared through the engine's own student trace.

Reuses the manifest/archive-replay code from task35's scripts/bench_engine.py (perf
worktree); the format is deliberately the same so the roots are comparable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

FRAMES_ROOT = Path.home() / "PvZ-search" / "artifacts" / "student" / "frames_v2"
STUDENT = Path.home() / "PvZ-search" / "artifacts" / "student" / "dw4i_s1" / "model.pt"
VALUE = Path.home() / "PvZ-search" / "artifacts" / "value_net" / "v3" / "model.pt"
STATIC_VALUE = Path.home() / "PvZ-search" / "artifacts" / "hard" / "value_branch1" / "model.pt"
RESOURCES = Path.home() / ".cache" / "pvz-research-resources"
EXECUTABLE = Path.home() / "PvZ-search" / ".claude" / "worktrees" / "cq" / "build" / "pvz-portable"
HORIZONS = (500, 1000, 1500, 2000, 2500)
FULL_HORIZON = 2500
ROLLOUT_SEEDS = tuple(range(9))
SIM_SKIP = 61
MAX_ACTIONS = 3000
BATCH = 32

LABEL_KIND = re.compile(r"^(?P<kind>[dnp])(?P<level>\d+)")
LABEL_ORIGINAL = re.compile(r"d(?P<level>\d+)orig")


def scene_family(level: int, conveyor_original: bool) -> str | None:
    import pvz_constants
    if conveyor_original and level <= 10:
        return "original_1_10"
    background = pvz_constants.background_for_level(level)
    if background == "BACKGROUND_1_DAY" and level == 7:
        return "day_l7"
    if background == "BACKGROUND_2_NIGHT" and level == 14:
        return "night_l14"
    if background == "BACKGROUND_3_POOL" and level == 24:
        return "pool_24"
    return None


def label_family(name: str) -> str | None:
    original = LABEL_ORIGINAL.search(name)
    if original is not None and int(original.group("level")) <= 10:
        return "original_1_10"
    match = LABEL_KIND.match(name)
    if match is None:
        return None
    kind, level = match.group("kind"), int(match.group("level"))
    if kind == "d" and level == 7:
        return "day_l7"
    if kind == "n" and level == 14:
        return "night_l14"
    if kind == "p" and level == 24:
        return "pool_24"
    return None


def archive_rows(path: Path) -> list[dict]:
    from pvz_frame_store import read_frames
    return read_frames(path)


def archive_spec(meta: dict, rows: list[dict]) -> dict | None:
    task = meta.get("task") or {}
    context = meta.get("context") or {}
    level = task.get("level", context.get("level"))
    seed = task.get("seed", context.get("seed"))
    if level is None or seed is None or not rows:
        return None
    conveyor = task.get("conveyor_start")
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
    families: dict[str, list] = {}
    seeds = sorted(path for path in args.frames_root.glob("seed_*") if path.is_dir())
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
                rows = archive_rows(archive / ("frames.npz" if (archive / "frames.npz").is_file()
                                               else "frames.jsonl"))
            except Exception:
                continue
            spec = archive_spec(meta, rows)
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
    missing = [name for name in ("day_l7", "original_1_10", "night_l14", "pool_24")
               if name not in families]
    if missing:
        raise SystemExit(f"manifest scan found no archives for: {missing} (scanned {scanned} seeds)")
    manifest = {"frames_root": str(args.frames_root), "scanned_seeds": scanned, "families": families}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=1) + "\n")
    total = sum(len(v) for v in families.values())
    print(f"manifest: {total} archives, {total * args.decisions} roots -> {args.out}")
    for name, entries in families.items():
        print(f"  {name}: {len(entries)} archives x {len(entries[0]['decisions'])} points")
    return manifest


def task_spec(spec: dict):
    from pvz_env import TaskSpec
    from pvz_tasks import profile_for_deck
    deck = tuple(spec["deck"])
    return TaskSpec(level=spec["level"], seed=spec["seed"], playthrough=2,
                    profile=profile_for_deck(deck, spec.get("first_aid_owned", False)),
                    zombie_count_multiplier=spec["multiplier"],
                    conveyor_start=spec.get("conveyor_start"))


def replay_episode(env, spec: dict, rows: list[dict], decisions: list[int]):
    env.reset(deck=tuple(spec["deck"]), task=task_spec(spec))
    snapshots = {}
    last = max(decisions)
    for index in range(1, last + 1):
        if index in decisions:
            snapshots[index] = env.snapshot()
        observation, _reward, terminal, _truncated, _info = env.step(dict(rows[index].get("action")))
        if terminal:
            break
    return snapshots


def export_student(path: Path) -> None:
    from pvz_rollout_policy import export_student as export, load_student
    export(load_student(str(STUDENT)), path)


def leaf_value(leaf: dict) -> float | None:
    """search_play._score_terminal_or_value, without the value net."""
    if "rollout_result" in leaf:
        return float(leaf["rollout_result"])
    if leaf.get("terminal"):
        return 1.0 if leaf.get("result") == 1 else 0.0
    return None


def score_frames(frames: list[dict], net, context) -> list[float]:
    import numpy as np
    import torch
    from pvz_value_net import featurize
    values: list[float] = []
    for start in range(0, len(frames), BATCH):
        part = frames[start:start + BATCH]
        boards, scalars = zip(*(featurize(frame) for frame in part))
        scalars = torch.stack(scalars)
        if net.context:
            if context is None:
                raise RuntimeError("value net needs the task context but none was recorded")
            scalars = torch.cat([scalars, torch.from_numpy(np.asarray(context)).expand(len(part), -1)], dim=1)
        with torch.inference_mode():
            values.extend(torch.sigmoid(net(torch.stack(boards), scalars)).tolist())
    return values


def measure(args) -> dict:
    import torch
    torch.set_num_threads(1)
    from pvz_value_net import load_value_net, observation_context
    from pvz_event_env import EventWaitEnv
    from render_episode import _frame
    manifest = json.loads(args.manifest.read_text())
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    vnet, snet = load_value_net(str(VALUE)), load_value_net(str(STATIC_VALUE))
    env = EventWaitEnv(str(RESOURCES), executable=str(args.executable), headless=True)
    rows_written = 0
    try:
        with tempfile.TemporaryDirectory(prefix="pvz-cutoff-") as directory:
            net_path = Path(directory) / "student.bin"
            export_student(net_path)
            env.load_rollout_net(net_path)
            for family, entries in manifest["families"].items():
                for entry in entries:
                    archive = Path(entry["archive"])
                    rows = archive_rows(archive / ("frames.npz" if (archive / "frames.npz").is_file()
                                                   else "frames.jsonl"))
                    snapshots = replay_episode(env, entry, rows, entry["decisions"])
                    for index, snapshot in snapshots.items():
                        root = env.restore(snapshot)
                        context = observation_context(root)
                        start_tick = int(root["tick"])
                        results = []
                        for horizon in HORIZONS:
                            for seed in ROLLOUT_SEEDS:
                                leaf = env.rollout(None, start_tick + horizon, sim_skip=SIM_SKIP,
                                                   max_actions=MAX_ACTIONS, snapshot_id=snapshot,
                                                   seed=seed)
                                terminal = leaf_value(leaf)
                                frame = None if terminal is not None else _frame(leaf)
                                results.append({"horizon": horizon, "seed": seed, "leaf": leaf,
                                                "terminal": terminal, "frame": frame,
                                                "decisions": int(env._last_response["decisions"])})
                        pending = [item for item in results if item["terminal"] is None]
                        fixed = {id(item): item["terminal"] for item in results if item["terminal"] is not None}
                        if pending:
                            v3 = score_frames([item["frame"] for item in pending], vnet, context)
                            branch = score_frames([item["frame"] for item in pending], snet, context)
                        for item, a, b in zip([i for i in results if i["terminal"] is None],
                                              v3 if pending else [], branch if pending else []):
                            fixed[id(item)] = (a, b)
                        with out.open("a") as handle:
                            for item in results:
                                leaf = item["leaf"]
                                value = fixed[id(item)]
                                if isinstance(value, tuple):
                                    v3v, branchv = value
                                else:
                                    v3v = branchv = value
                                row = {
                                    "family": family, "archive": str(archive), "level": entry["level"],
                                    "game_seed": entry["seed"], "frame": index, "root_tick": start_tick,
                                    "multiplier": entry["multiplier"], "rollout_seed": item["seed"],
                                    "horizon": item["horizon"], "leaf_tick": int(leaf["tick"]),
                                    "ticks": int(leaf["tick"]) - start_tick,
                                    "decisions": item["decisions"],
                                    "terminal": bool(leaf.get("terminal")),
                                    "result": leaf.get("result"),
                                    "v3": float(v3v), "branch1": float(branchv),
                                }
                                handle.write(json.dumps(row) + "\n")
                                rows_written += 1
                        print(f"{family} {archive.parent.name} frame {index} tick {start_tick}: "
                              f"{len(results)} rollouts", flush=True)
                        env.release_snapshot(snapshot)
        env.close()
    finally:
        try:
            env.close()
        except Exception:
            pass
    print(f"{rows_written} rows -> {out}", flush=True)
    return {"rows": rows_written, "out": str(out)}


def trace_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def observation_payload(observation: dict) -> dict:
    """The engine's observation without the two fields the Python wrapper adds."""
    return {key: value for key, value in observation.items()
            if key not in ("sun_income_rate", "rollout_result")}


def same_observation(left: dict, right: dict) -> bool:
    return json.dumps(observation_payload(left), sort_keys=True) == \
        json.dumps(observation_payload(right), sort_keys=True)


def prefix(args) -> dict:
    import torch
    torch.set_num_threads(1)
    from pvz_event_env import EventWaitEnv
    manifest = json.loads(args.manifest.read_text())
    env = EventWaitEnv(str(RESOURCES), executable=str(args.executable), headless=True)
    checks = []
    try:
        with tempfile.TemporaryDirectory(prefix="pvz-prefix-") as directory:
            net_path = Path(directory) / "student.bin"
            export_student(net_path)
            env.load_rollout_net(net_path)
            for family, entries in manifest["families"].items():
                entry = entries[0]
                archive = Path(entry["archive"])
                rows = archive_rows(archive / ("frames.npz" if (archive / "frames.npz").is_file()
                                               else "frames.jsonl"))
                snapshots = replay_episode(env, entry, rows, entry["decisions"][:1])
                for index, snapshot in snapshots.items():
                    start_tick = int(env.restore(snapshot)["tick"])
                    for seed in range(args.seeds):
                        full_trace = Path(directory) / f"full_{family}_{seed}.jsonl"
                        leaf_full = env.rollout(None, start_tick + FULL_HORIZON, sim_skip=SIM_SKIP,
                                                max_actions=MAX_ACTIONS, snapshot_id=snapshot,
                                                seed=seed, trace_path=str(full_trace))
                        full = trace_lines(full_trace)
                        for horizon in args.horizons:
                            short_trace = Path(directory) / f"t{horizon}_{family}_{seed}.jsonl"
                            leaf_short = env.rollout(None, start_tick + horizon, sim_skip=SIM_SKIP,
                                                     max_actions=MAX_ACTIONS, snapshot_id=snapshot,
                                                     seed=seed, trace_path=str(short_trace))
                            short = trace_lines(short_trace)
                            common = min(len(short), len(full))
                            trace_prefix = all(
                                json.dumps(short[i], sort_keys=True) == json.dumps(full[i], sort_keys=True)
                                for i in range(common))
                            leaf_tick = int(leaf_short["tick"])
                            if leaf_short.get("terminal"):
                                state_match = bool(leaf_full.get("terminal")) and \
                                    same_observation(leaf_short, leaf_full)
                            else:
                                # The leaf is the state after the short run's last action; the full
                                # run traces that same state as its next decision.  Several decisions
                                # can share a tick (plants cost no ticks), so the index is the anchor,
                                # not the tick.
                                matched = full[len(short)] if len(short) < len(full) else None
                                state_match = matched is not None and \
                                    int(matched["observation"]["tick"]) == leaf_tick and \
                                    same_observation(leaf_short, matched["observation"])
                            checks.append({"family": family, "archive": str(archive), "frame": index,
                                           "seed": seed, "horizon": horizon, "start_tick": start_tick,
                                           "leaf_tick": leaf_tick, "short_decisions": len(short),
                                           "full_decisions": len(full), "trace_prefix": trace_prefix,
                                           "state_at_t": state_match,
                                           "short_terminal": bool(leaf_short.get("terminal"))})
                            print(json.dumps(checks[-1]), flush=True)
    finally:
        env.close()
    summary = {"checks": len(checks), "trace_prefix_all": all(c["trace_prefix"] for c in checks),
               "state_at_t_all": all(c["state_at_t"] for c in checks), "rows": checks}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({key: summary[key] for key in ("checks", "trace_prefix_all", "state_at_t_all")}, indent=1))
    return summary


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def rounded(value: float | None, digits: int = 5) -> float | None:
    return None if value is None else round(value, digits)


def report(args) -> dict:
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    by_key: dict[tuple, dict] = {}
    for row in rows:
        key = (row["family"], row["archive"], row["frame"], row["rollout_seed"])
        by_key.setdefault(key, {})[row["horizon"]] = row
    thresholds = (0.90, 0.95, 0.98, 0.99)
    table = {}
    for family in sorted({row["family"] for row in rows}):
        family_rows = {key: value for key, value in by_key.items() if key[0] == family}
        table[family] = {}
        for horizon in HORIZONS[:-1]:
            full_ticks = sum(value[FULL_HORIZON]["ticks"] for value in family_rows.values())
            table[family][horizon] = {}
            for theta in thresholds:
                confident = []
                for key, value in family_rows.items():
                    short = value.get(horizon)
                    full = value.get(FULL_HORIZON)
                    if short is None or full is None:
                        continue
                    if short["v3"] >= theta or short["v3"] <= 1.0 - theta:
                        confident.append((short, full))
                diffs = [full["v3"] - short["v3"] for short, full in confident]
                abs_diffs = [abs(diff) for diff in diffs]
                flips = sum(1 for short, full in confident
                            if (short["v3"] >= 0.5) != (full["v3"] >= 0.5))
                stopped = sum(full["ticks"] - short["ticks"] for short, full in confident)
                saved_fraction = stopped / full_ticks if full_ticks else None
                table[family][horizon][theta] = {
                    "rollouts": len(family_rows), "confident": len(confident),
                    "confident_share": round(len(confident) / len(family_rows), 4) if family_rows else None,
                    "diff_mean": round(sum(diffs) / len(diffs), 5) if diffs else None,
                    "abs_diff_mean": round(sum(abs_diffs) / len(abs_diffs), 5) if abs_diffs else None,
                    "abs_diff_p95": rounded(percentile(abs_diffs, 0.95)),
                    "abs_diff_max": rounded(max(abs_diffs)) if abs_diffs else None,
                    "flips": flips, "flip_rate": round(flips / len(confident), 4) if confident else None,
                    "ticks_saved": stopped, "tick_saved_fraction": rounded(saved_fraction, 4),
                }
    # Aggregate over all families (rollout-weighted) for the recommendation.
    aggregate = {}
    for horizon in HORIZONS[:-1]:
        aggregate[horizon] = {}
        for theta in thresholds:
            cells = [table[family][horizon][theta] for family in table]
            confident = sum(cell["confident"] for cell in cells)
            rollouts = sum(cell["rollouts"] for cell in cells)
            flips = sum(cell["flips"] for cell in cells)
            saved = sum(cell["ticks_saved"] for cell in cells)
            full_ticks = sum(value[FULL_HORIZON]["ticks"] for value in by_key.values())
            diffs = [value[FULL_HORIZON]["v3"] - value[horizon]["v3"] for key, value in by_key.items()
                     if horizon in value and FULL_HORIZON in value and
                     (value[horizon]["v3"] >= theta or value[horizon]["v3"] <= 1.0 - theta)]
            aggregate[horizon][theta] = {
                "rollouts": rollouts, "confident": confident,
                "confident_share": round(confident / rollouts, 4) if rollouts else None,
                "diff_mean": round(sum(diffs) / len(diffs), 5) if diffs else None,
                "abs_diff_p95": rounded(percentile([abs(diff) for diff in diffs], 0.95)),
                "flips": flips, "flip_rate": round(flips / confident, 4) if confident else None,
                "tick_saved_fraction": round(saved / full_ticks, 4) if full_ticks else None,
            }
    summary = {"input": str(args.input), "thresholds": thresholds, "families": table,
               "aggregate": aggregate,
               "input_sha256": {str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                for path in (STUDENT, VALUE, STATIC_VALUE)}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1) + "\n")
    print_markdown(summary)
    print(f"-> {args.out}")
    return summary


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def _num(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def print_markdown(summary: dict) -> None:
    for family, horizons in summary["families"].items():
        print(f"\n### {family}")
        print("| t | θ | 自信占比 | 价值差均值 | |差| p95 | 翻转率 | 省 tick |")
        print("|---|---|---|---|---|---|---|")
        for horizon, cells in horizons.items():
            for theta, cell in cells.items():
                print(f"| {horizon} | {theta:.2f} | {_pct(cell['confident_share'])} | "
                      f"{_num(cell['diff_mean'])} | {_num(cell['abs_diff_p95'])} | "
                      f"{_pct(cell['flip_rate'])} | {_pct(cell['tick_saved_fraction'])} |")
    print("\n### all families")
    print("| t | θ | 自信占比 | 价值差均值 | |差| p95 | 翻转率 | 省 tick |")
    print("|---|---|---|---|---|---|---|")
    for horizon, cells in summary["aggregate"].items():
        for theta, cell in cells.items():
            print(f"| {horizon} | {theta:.2f} | {_pct(cell['confident_share'])} | "
                  f"{_num(cell['diff_mean'])} | {_num(cell['abs_diff_p95'])} | "
                  f"{_pct(cell['flip_rate'])} | {_pct(cell['tick_saved_fraction'])} |")


def main() -> None:
    parent = argparse.ArgumentParser(description=__doc__)
    sub = parent.add_subparsers(dest="command", required=True)

    p = sub.add_parser("manifest")
    p.add_argument("--frames-root", type=Path, default=FRAMES_ROOT)
    p.add_argument("--out", type=Path, default=ROOT / "artifacts" / "task40" / "roots.json")
    p.add_argument("--archives", type=int, default=16)
    p.add_argument("--decisions", type=int, default=4)
    p.add_argument("--fractions", type=float, nargs="+", default=[0.15, 0.4, 0.65, 0.85])
    p.add_argument("--min-frames", type=int, default=200)
    p.add_argument("--scan-seeds", type=int, default=4000)

    for name in ("prefix", "measure"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", type=Path, default=ROOT / "artifacts" / "task40" / "roots.json")
        p.add_argument("--executable", type=Path, default=EXECUTABLE)
        if name == "prefix":
            p.add_argument("--out", type=Path, default=ROOT / "artifacts" / "task40" / "prefix.json")
            p.add_argument("--seeds", type=int, default=2)
            p.add_argument("--horizons", type=int, nargs="+", default=[500, 1000, 1500, 2000])
        else:
            p.add_argument("--out", type=Path, default=ROOT / "artifacts" / "task40" / "rollouts.jsonl")

    p = sub.add_parser("report")
    p.add_argument("--input", type=Path, default=ROOT / "artifacts" / "task40" / "rollouts.jsonl")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts" / "task40" / "summary.json")

    args = parent.parse_args()
    {"manifest": build_manifest, "prefix": prefix, "measure": measure, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
