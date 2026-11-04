"""Run a task234 drill bank headless: student, production teacher, wide teacher.

The game rules, replay, search and value nets are the existing ones (AGENTS.md §6.6):
this file is a thin short-horizon driver around ``human_demo.replay`` and
``search_play._search``.  It stops at a fixed tick horizon instead of the terminal so
one drill costs a few decisions, then judges the engine state mechanically.

Arms:
  student  -- r101 alone (the current student checkpoint)
  teacher  -- production qc parameters (verify 9, student candidates 2, horizon 2500)
  wide     -- wider/longer search (per-card 2, candidates 3, samples 4, verify 12, horizon 3500)

On fog/roof/boss the production policy is the GPT behaviour-clone prior
(``--rollout-policy-gpt``); the student arm still measures r101, which is the point of
the comparison.  What each arm can see is the archive's own fog_observation.

Usage:
  python3 scripts/drill_run.py --bank configs/task234_drills.json \
      --archive-root main=/path/to/PvZAgent --archive-root gen=artifacts/task234/gen_frames \
      --student .../student.pt --teacher-prior .../gptbc.pt \
      --value .../value.pt --value-roof .../value_roof.pt --static-value .../static.pt \
      --executable ~/pvz-mac-collect/build2/pvz-portable \
      --resource-dir ~/.cache/pvz-research-resources \
      --arms student,teacher,wide --seeds 3 --workers 4 --horizon 900 \
      --out artifacts/task234/run
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]

import search_play  # noqa: E402

PRIOR_TERRAINS = {"fog", "roof", "boss"}
TEACHER = dict(per_card=1, student_candidates=2, samples=3, verify_samples=9,
               horizon=2500, min_gap=300, save_candidates=0)
WIDE = dict(per_card=2, student_candidates=3, samples=4, verify_samples=12,
            horizon=3500, min_gap=150, save_candidates=0)
ARM_NAMES = ("student", "teacher", "teacher_r101", "wide")


def parse_roots(values):
    roots = {}
    for entry in values:
        name, sep, path = entry.partition("=")
        if not sep:
            raise SystemExit("--archive-root needs NAME=PATH")
        roots[name] = Path(path).expanduser().resolve()
    return roots


def make_args(arm, terrain, out_dir: Path, student: Path, value: Path):
    """search_play's own parser supplies the defaults; only the arm overrides are set."""
    args = search_play.parser().parse_args([
        "--rollout-policy", str(student), "--value", str(value),
        "--manifest", str(ROOT / "configs/human_demo_tasks.json"), "--task-id", "drill",
        "--out", str(out_dir),
    ])
    args.rollout_sim_skip = 61
    args.rollout_policy_gpt = None
    args.save_candidates = 0
    if arm in ("teacher", "teacher_r101", "wide"):
        for key, value_ in (TEACHER if arm != "wide" else WIDE).items():
            setattr(args, key, value_)
        if terrain == "fog":
            args.save_candidates = 1     # qc default for levels 31-40
    return args


def export_values(value: Path, roof_value: Path, static: Path, keep):
    from export_value_net import export_value_net
    from pvz_value_net import load_value_net
    out = {}
    for name, path in (("value", value), ("roof", roof_value), ("static", static)):
        target = Path(keep.name) / f"{name}.bin"
        export_value_net(load_value_net(path), str(target))
        out[name] = str(target)
    return out


_WORKER = {}


def init_worker(config):
    import torch
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    search_play._init_worker(config["value_bins"]["value"], config["student"],
                             config["sim_skip"], config["value_bins"]["static"])
    from pvz_rollout_policy import export_student, load_student
    gpt = load_student(config["teacher_prior"])
    gpt_bin = str(Path(config["tmp"]) / "gpt.bin")
    export_student(gpt, gpt_bin)
    search_play._W["student_nets"] = {"r101": search_play._W["fast"], "gpt": gpt}
    search_play._W["student_bins"] = {"r101": search_play._W["rollout_file"], "gpt": gpt_bin}
    search_play._W["lean_rollouts"] = True
    search_play._W["batch_waits"] = True
    search_play._W["rollout_sim_skip"] = config["sim_skip"]
    from pvz_event_env import EventWaitEnv
    env = EventWaitEnv(config["resource_dir"], executable=config["executable"], headless=True)
    env.load_rollout_net(search_play._W["rollout_file"])
    search_play._W["env"] = env
    _WORKER["env"] = env
    _WORKER["config"] = config


def _ready_mowers(obs):
    return sum(1 for d in obs.get("defenses") or [] if d.get("state") == 1)


def _on_board(obs):
    """Zombie ids still on board, from a privileged frame when the engine allows it."""
    frame = obs
    try:
        frame = _WORKER["env"].privileged_state()
    except Exception:
        pass
    out = {}
    for z in frame.get("zombies") or []:
        if isinstance(z, dict):
            if not z.get("on_board"):
                continue
            out[int(z["id"])] = dict(row=int(z["row"]), type=int(z["type"]),
                                     x=float(z["x"]), hp=int(z.get("health") or 0))
        elif z[5]:
            out[int(z[9])] = dict(row=int(z[0]), type=int(z[1]), x=float(z[2]), hp=int(z[4]))
    return out


def _plant_type(obs, row, col):
    for p in obs.get("plants") or []:
        if p["row"] == row and p["col"] == col and not p.get("squished"):
            return int(p["type"])
    return None


def _packet_type(obs, packet):
    for p in obs.get("packets") or []:
        if isinstance(p, dict):
            if int(p["index"]) == packet:
                value = p.get("imitater_type")
                return int(value if value is not None and int(value) >= 0 else p["type"])
        elif int(p[0]) == packet:
            return int(p[1])
    return None


def _sequence(actions):
    """Shovel -> plant -> replant on the same cell, within the drill horizon."""
    out = []
    for index, first in enumerate(actions):
        if first["type"] != "shovel":
            continue
        cell = (first["row"], first["col"])
        planted = None
        for second in actions[index + 1:]:
            if second["tick"] - first["tick"] > 1200:
                break
            if second["type"] != "plant" or (second["row"], second["col"]) != cell:
                continue
            if planted is None:
                planted = second
                continue
            if (first.get("plant_before") == second.get("plant_type")
                    and second.get("plant_type") != planted.get("plant_type")):
                out.append(dict(cell=list(cell), removed=first.get("plant_before"),
                                planted=planted.get("plant_type"),
                                replanted=second.get("plant_type"),
                                shovel_tick=first["tick"], replant_tick=second["tick"]))
            break
    return out


def restore_prefix(env, archive, decision, frames, meta):
    """Restore the archive's state at ``decision`` with the existing replay path.

    Human-demo archives carry per-operation ticks and are replayed by
    ``human_demo.replay`` (rejections and the exact RNG included).  Collection
    archives (search_play frames) have no operation column; for those the recorded
    actions are the prefix, and the reset context comes from the archive itself.
    """
    from pvz_env import TaskSpec, PlayerProfileContext
    from pvz_tasks import profile_for_deck
    if (archive / "ticks.jsonl").exists() and "operation" in frames[min(decision - 1, len(frames) - 1)]:
        from human_demo import replay
        obs, _meta, prefix = replay(env, archive, decision=decision)
        return obs, prefix
    if meta.get("environment_task"):
        config = dict(meta["environment_task"])
        config["profile"] = PlayerProfileContext(**config["profile"])
        config["preplanted"] = tuple(tuple(p) for p in config.get("preplanted") or ())
        deck = meta["task"]["deck"]
        obs, _ = env.reset(deck=deck, task=TaskSpec(**config))
    else:
        # Collection frames predate the full environment_task block.  The deck is in
        # frame 0's seed bank; level/seed/multiplier are in the archive meta.
        deck = [int(p[1]) for p in frames[0].get("packets") or []]
        if not deck or any(not 0 <= seed_type < 49 for seed_type in deck):
            raise ValueError("archive has neither environment_task nor a valid seed bank")
        level = int((meta.get("context") or {}).get("level") or frames[0]["scene"]["level"])
        seed = int((meta.get("task") or {}).get("seed"))
        multiplier = float((meta.get("context") or {}).get("multiplier") or 1.0)
        obs, _ = env.reset(deck=deck, task=TaskSpec(
            level=level, seed=seed, playthrough=2,
            profile=profile_for_deck(deck), zombie_count_multiplier=multiplier))
    for index in range(1, decision):
        action = frames[index].get("action") or frames[index].get("input_action")
        if action is None:
            delta = int(frames[index + 1]["tick"]) - int(frames[index]["tick"])
            action = dict(type="wait", ticks=max(1, delta))
        obs, _r, _d, _t, _info = env.step(action)
        if int(obs["tick"]) != int(frames[index]["tick"]):
            raise ValueError(f"prefix differs at frame {index}: {obs['tick']} != {frames[index]['tick']}")
    return obs, frames[:decision]


def _run_one(job):
    cfg, drill, arm, seed, horizon = (_WORKER["config"], job["drill"], job["arm"],
                                      job["seed"], job["horizon"])
    env = _WORKER["env"]
    from pvz_frame_store import frame_path, read_frames
    from pvz_student_observation import StudentHistory
    from pvz_value_net import observation_context
    from render_episode import _frame
    import torch

    started = time.time()
    archive = cfg["roots"][drill["source_root"]] / drill["archive"]
    frames = read_frames(frame_path(archive))
    meta = json.loads((archive / "meta.json").read_text())
    obs, prefix = restore_prefix(env, archive, int(drill["decision"]), frames, meta)
    if obs.get("terminal"):
        return dict(status="terminal_at_root", drill_id=drill.get("uid", drill["id"]), arm=arm, seed=seed)
    env.reseed(seed)
    torch.manual_seed(seed)
    terrain = drill["terrain"]
    prior = "gpt" if (arm in ("teacher", "wide") and terrain in PRIOR_TERRAINS) else "r101"
    search_play._W["fast"] = search_play._W["student_nets"][prior]
    env.load_rollout_net(search_play._W["student_bins"][prior])
    env.load_value_net(cfg["value_bins"]["roof" if terrain in ("roof", "boss") else "value"])
    env.load_value_net(cfg["value_bins"]["static"], static=True)

    fast_state = {}
    if prefix:
        fast_state["frame"] = prefix[-2] if len(prefix) > 1 else None
        fast_state["action"] = prefix[-1].get("action")
        if search_play._W["fast"].obs != "v1":
            history = StudentHistory(search_play._W["fast"].obs)
            for frame in prefix[:-1]:
                history.extra(frame)
            fast_state["history"] = history
        search_play._W["student_history"] = fast_state.get("history")
    else:
        search_play._W["student_history"] = None
    search_play._W["context"] = observation_context(obs)
    search_play._W["demo_actions"] = []
    args = make_args(arm, terrain, cfg["out"], cfg["student"], cfg["value"])
    stats = {"searched": 0, "overrides": 0, "shovels": 0, "override_kinds": Counter(),
             "last_search_tick": -10 ** 9, "last_combo_tick": -10 ** 9,
             "last_targets": None, "pending_upgrade": None}
    root_tick = int(obs["tick"])
    root_ready = _ready_mowers(obs)
    targets = [int(t["id"]) for t in drill.get("targets") or []]
    root_targets = {t["id"]: t for t in drill.get("targets") or []}
    actions, mower_used, house_breach = [], 0, False
    plants_lost = 0
    try:
        for _ in range(400):
            if obs.get("terminal"):
                break
            if int(obs["tick"]) - root_tick >= horizon:
                break
            policy = search_play._fast_action(obs, fast_state, env)
            search_play._W["student_history"] = fast_state.get("history")
            action = policy
            if arm in ("teacher", "teacher_r101", "wide"):
                action = search_play._search(
                    env, obs, policy, args,
                    (seed * 10000 + len(actions)) & 0x7fffffff, stats)
            fast_state["action"] = action
            before = dict(row=action.get("row"), col=action.get("col"),
                          plant_before=_plant_type(obs, action.get("row"), action.get("col"))
                          if action.get("type") in ("plant", "shovel") else None)
            obs, _reward, _done, _truncated, info = env.step(action)
            rejected = not info.get("ok")
            if rejected:
                obs, _reward, _done, _truncated, info = env.step(dict(type="wait", ticks=60))
            plants_lost += int((info.get("events") or {}).get("plants_eaten", 0))
            pipe = _packet_type(obs, int(action["packet"])) if action.get("type") == "plant" else None
            actions.append(dict(tick=int(obs["tick"]),
                                type=action.get("type"), packet=action.get("packet"),
                                row=action.get("row"), col=action.get("col"),
                                rejected=rejected, plant_type=pipe,
                                plant_before=before["plant_before"]))
            if action.get("type") == "shovel":
                stats["shovels"] += 1
            ready = _ready_mowers(obs)
            if ready < root_ready:
                mower_used = max(mower_used, root_ready - ready)
            if obs.get("result") == 2:
                house_breach = True
        final_tick = int(obs["tick"])
        alive = _on_board(obs)
        alive_targets = [tid for tid in targets if tid in alive]
        success = (not alive_targets and mower_used == 0 and not house_breach)
        return dict(
            status="ok", drill_id=drill.get("uid", drill["id"]), arm=arm, seed=seed, terrain=terrain,
            level=drill["level"], prior=prior, success=bool(success),
            target_count=len(targets), targets_alive=len(alive_targets),
            alive=alive_targets, target_types=sorted({root_targets[t]["type"] for t in targets}),
            mower_used=mower_used, house_breach=bool(house_breach),
            plants_lost=plants_lost, root_tick=root_tick, final_tick=final_tick,
            tick_delta=final_tick - root_tick, decisions=len(actions),
            searched=stats["searched"], overrides=stats["overrides"], shovels=stats["shovels"],
            override_kinds=dict(stats["override_kinds"]),
            sequences=_sequence(actions),
            actions=actions,
            seconds=round(time.time() - started, 3),
        )
    except Exception as exc:
        return dict(status="error", drill_id=drill.get("uid", drill["id"]), arm=arm, seed=seed,
                    error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-2000:])


def run_one(job):
    try:
        return _run_one(job)
    except Exception as exc:
        return dict(status="error", drill_id=job["drill"].get("uid", job["drill"]["id"]), arm=job["arm"],
                    seed=job["seed"], error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc()[-2000:])


def select_drills(bank, args):
    drills = bank["drills"]
    if args.drill_id:
        wanted = set(args.drill_id)
        drills = [d for d in drills if d["id"] in wanted or d.get("uid") in wanted]
    if args.terrain:
        wanted = set(args.terrain.split(","))
        drills = [d for d in drills if d["terrain"] in wanted]
    if args.per_terrain:
        chosen, counts = [], Counter()
        for drill in drills:
            if counts[drill["terrain"]] >= args.per_terrain:
                continue
            counts[drill["terrain"]] += 1
            chosen.append(drill)
        drills = chosen
    if args.limit:
        drills = drills[:args.limit]
    return drills


def summarize(rows, drills, arms):
    by_arm = defaultdict(list)
    by_arm_terrain = defaultdict(list)
    by_drill = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("status") != "ok":
            continue
        by_arm[row["arm"]].append(row)
        by_arm_terrain[(row["arm"], row["terrain"])].append(row)
        by_drill[row["drill_id"]][row["arm"]].append(row)

    def block(rows_):
        n = len(rows_)
        if not n:
            return dict(games=0)
        return dict(
            games=n,
            success=sum(r["success"] for r in rows_),
            rate=round(sum(r["success"] for r in rows_) / n, 4),
            mean_plants_lost=round(statistics.mean(r["plants_lost"] for r in rows_), 2),
            mower_games=sum(1 for r in rows_ if r["mower_used"]),
            house_games=sum(1 for r in rows_ if r["house_breach"]),
            mean_tick_delta=round(statistics.mean(r["tick_delta"] for r in rows_), 1),
            mean_decisions=round(statistics.mean(r["decisions"] for r in rows_), 2),
            shovels=sum(r["shovels"] for r in rows_),
            multi_step_games=sum(1 for r in rows_ if r["sequences"]),
            searched_per_game=round(statistics.mean(r["searched"] for r in rows_), 2),
            overrides_per_game=round(statistics.mean(r["overrides"] for r in rows_), 2),
        )

    summary = dict(games=len(rows), arms={}, by_arm_terrain={}, drills={},
                   statuses=dict(Counter(r.get("status") for r in rows)))
    for arm in arms:
        summary["arms"][arm] = block(by_arm.get(arm, []))
    for (arm, terrain), rows_ in sorted(by_arm_terrain.items()):
        summary["by_arm_terrain"][f"{arm}|{terrain}"] = block(rows_)
    for drill_id, per_arm in by_drill.items():
        drill = next(d for d in drills if d.get("uid", d["id"]) == drill_id)
        entry = dict(terrain=drill["terrain"], level=drill["level"],
                     types=[t["name"] for t in drill.get("targets") or []])
        for arm in arms:
            rows_ = per_arm.get(arm, [])
            wins = sum(r["success"] for r in rows_)
            entry[arm] = dict(games=len(rows_), wins=wins,
                              rate=round(wins / len(rows_), 4) if rows_ else None,
                              se=round(math.sqrt((wins / len(rows_)) * (1 - wins / len(rows_)) / len(rows_)), 4)
                              if rows_ else None)
        summary["drills"][drill_id] = entry
    # Between-drill variance of the per-drill success rate, by arm: the "exam" noise if
    # one treats each drill as one item and draws k futures.
    for arm in arms:
        rates = [e[arm]["rate"] for e in summary["drills"].values() if e[arm]["rate"] is not None]
        summary["arms"][arm]["between_drill_sd"] = round(statistics.pstdev(rates), 4) if len(rates) > 1 else 0.0
        # Seed noise inside one drill, averaged over drills: the exam's per-item noise.
        within = []
        for entry in summary["drills"].values():
            item = entry[arm]
            if item["rate"] is not None and item["games"]:
                within.append(math.sqrt(item["rate"] * (1 - item["rate"]) / item["games"]))
        summary["arms"][arm]["within_drill_se"] = round(statistics.mean(within), 4) if within else 0.0
        summary["arms"][arm]["drills"] = len(rates)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--archive-root", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--teacher-prior", type=Path)
    parser.add_argument("--value", type=Path, required=True)
    parser.add_argument("--value-roof", type=Path)
    parser.add_argument("--static-value", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--resource-dir", type=Path,
                        default=Path.home() / ".cache/pvz-research-resources")
    parser.add_argument("--arms", default="student,teacher,wide")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=88000000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=None,
                        help="override the bank's horizon (default: each drill's)")
    parser.add_argument("--terrain", help="comma-separated terrain filter")
    parser.add_argument("--per-terrain", type=int, help="take the first N drills of each terrain")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--drill-id", action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        parser.error("--workers must be 1..6 (the Mac collector is shared with the user)")
    if not args.teacher_prior:
        parser.error("--teacher-prior is required for the production teacher")
    if not args.value_roof:
        parser.error("--value-roof is required for roof drills")
    roots = parse_roots(args.archive_root)
    bank = json.loads(args.bank.read_text())
    drills = select_drills(bank, args)
    if not drills:
        parser.error("no drills selected")
    arms = [a for a in args.arms.split(",") if a]
    for arm in arms:
        if arm not in ARM_NAMES:
            parser.error(f"unknown arm {arm}")
    jobs = [dict(drill=drill, arm=arm, seed=args.seed_base + index,
                 horizon=args.horizon or int(drill.get("horizon") or 900))
            for drill in drills for arm in arms for index in range(args.seeds)]
    args.out.mkdir(parents=True, exist_ok=True)
    import tempfile
    keep = tempfile.TemporaryDirectory(prefix="task234-values-")
    value_bins = export_values(args.value, args.value_roof, args.static_value, keep)
    config = dict(roots=roots, student=str(args.student), teacher_prior=str(args.teacher_prior),
                  value_bins=value_bins, sim_skip=61, resource_dir=str(args.resource_dir),
                  executable=str(args.executable), out=str(args.out), tmp=keep.name,
                  value=str(args.value))
    print(f"{len(drills)} drills, {len(jobs)} games on {args.workers} workers", flush=True)
    rows = []
    started = time.time()
    with (args.out / "results.jsonl").open("w") as stream:
        ctx = mp.get_context("fork")
        with ctx.Pool(args.workers, initializer=init_worker, initargs=(config,)) as pool:
            for row in pool.imap_unordered(run_one, jobs):
                rows.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                if len(rows) % 10 == 0:
                    print(f"{len(rows)}/{len(jobs)} ({time.time() - started:.0f}s)", flush=True)
    (args.out / "jobs.json").write_text(json.dumps(jobs, indent=1) + "\n")
    summary = summarize(rows, drills, arms)
    summary.update(seconds=round(time.time() - started, 1), drill_count=len(drills),
                   games=len(jobs), workers=args.workers,
                   bank=str(args.bank), student=str(args.student),
                   teacher_prior=str(args.teacher_prior), value=str(args.value),
                   value_roof=str(args.value_roof), static_value=str(args.static_value))
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(summary["arms"], ensure_ascii=False, indent=1))
    keep.cleanup()


if __name__ == "__main__":
    main()
