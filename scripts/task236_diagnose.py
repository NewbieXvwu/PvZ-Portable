"""task236 stage 1: value-net estimate vs paired short-horizon simulation around a shovel.

For each drill root the same board is branched into four action plans:

  wait          the no-shovel control (one 60-tick decision)
  shovel_only   dig the chosen plant, do not replant
  replant_same  dig it and put the same SeedType back in the same cell
  human_replace dig it and plant the card the human actually used there (human drills only)

Every plan is scored twice: the leaf value net on the post-action board (the "immediate
estimate") and a real engine continuation to a short horizon with RESEED futures shared by
all plans (common random numbers).  This is measurement only: no production path changes.

Usage:
  python3 scripts/task236_diagnose.py --bank configs/task234_drills.json \
      --archive-root main=/Users/newbiexvwu/PvZAgent \
      --archive-root userpick=/Users/newbiexvwu/PvZAgent/.claude/worktrees/userpick \
      --archive-root imitater=/Users/newbiexvwu/PvZAgent/.claude/worktrees/imitater \
      --student ~/pvz-mac-collect/models/student_f5b85ed7fa00.pt \
      --teacher-prior ~/pvz-mac-collect/models/gptbc.pt \
      --value ~/pvz-mac-collect/models/value.pt \
      --value-roof ~/pvz-mac-collect/models/value_roof.pt \
      --static-value ~/pvz-mac-collect/models/static.pt \
      --executable ~/pvz-mac-collect/build-shovel3/pvz-portable \
      --workers 4 --seeds 4 --horizon 1200 --out artifacts/task236/diagnose
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
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]

import drill_run  # noqa: E402
from pvz_frame_store import frame_path, read_frames  # noqa: E402

PRIOR_TERRAINS = {"fog", "roof", "boss"}
PLANT_NAMES = {
    0: "peashooter", 1: "sunflower", 2: "cherrybomb", 3: "wallnut", 4: "potatomine",
    5: "snowpea", 6: "chomper", 7: "repeater", 8: "puffshroom", 9: "sunshroom",
    10: "fumeshroom", 11: "gravebuster", 12: "hypnoshroom", 13: "scaredyshroom",
    14: "iceshroom", 15: "doomshroom", 16: "lilypad", 17: "squash", 18: "threepeater",
    19: "tanglekelp", 20: "jalapeno", 21: "spikeweed", 22: "torchwood", 23: "TallNut",
    24: "sea shroom", 25: "plantern", 26: "cactus", 27: "blover", 28: "split pea",
    29: "starfruit", 30: "pumpkin", 31: "magnetshroom", 32: "cabbagepult",
    33: "flowerpot", 34: "kernelpult", 35: "coffeebean", 36: "garlic", 37: "umbrellaleaf",
    38: "marigold", 39: "melonpult", 40: "gatlingpea", 41: "twinsunflower",
    42: "gloomshroom", 43: "cattail", 44: "wintermelon", 45: "goldmagnet",
    46: "spikerock", 47: "cobcannon", 48: "imitater",
}

_WORKER = {}


def init_worker(config):
    import torch
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from pvz_event_env import EventWaitEnv
    env = EventWaitEnv(config["resource_dir"], executable=config["executable"], headless=True)
    env.load_value_net(config["bins"]["value"])
    env.load_value_net(config["bins"]["static"], static=True)
    _WORKER["env"] = env
    _WORKER["config"] = config


def export_models(args, keep):
    from export_value_net import export_value_net
    from pvz_value_net import load_value_net
    from pvz_rollout_policy import export_student, load_student
    bins = {}
    for name, path in (("value", args.value), ("roof", args.value_roof), ("static", args.static_value)):
        target = Path(keep.name) / f"{name}.bin"
        export_value_net(load_value_net(path), str(target))
        bins[name] = str(target)
    for name, path in (("reference", args.reference_value), ("reference_roof", args.reference_roof)):
        if path:
            target = Path(keep.name) / f"{name}.bin"
            export_value_net(load_value_net(path), str(target))
            bins[name] = str(target)
    for name, path in (("student", args.student), ("gpt", args.teacher_prior)):
        target = Path(keep.name) / f"{name}.bin"
        export_student(load_student(path), str(target))
        bins[name] = str(target)
    return bins


def plan_actions(drill, obs):
    """Return {plan: action list} for one root, using only the board and the drill record."""
    cell = (int(drill["row"]), int(drill["col"]))
    cell_plants = [p for p in obs.get("plants") or []
                   if (p["row"], p["col"]) == cell and not p.get("squished")]
    packets = {}
    for p in obs.get("packets") or []:
        packets.setdefault(int(p["type"]), []).append(p)
    if not cell_plants:
        return {}, None
    # The shovel's own top-plant order (Board::GetTopPlantAt TOPPLANT_ANY):
    # flying -> normal -> pumpkin -> under (flowerpot / lilypad).
    def rank(plant):
        kind = int(plant["type"])
        if kind == 30:
            return 2
        if kind in (16, 33):
            return 3
        return 1
    removed = int(min(cell_plants, key=rank)["type"])
    if rank(min(cell_plants, key=rank)) == 3:
        return {}, removed
    plans = {"wait": [dict(type="wait", ticks=60)], "shovel_only": []}
    # The engine digs the top plant on the cell; the plan asks for the recorded one.
    plans["shovel_only"] = [dict(type="shovel", row=cell[0], col=cell[1])]
    def first_active(seed_type):
        for p in packets.get(seed_type, []):
            if p.get("active") and int(p.get("cooldown") or 0) == 0:
                return p
        return None
    packet = first_active(removed)
    if packet:
        plans["replant_same"] = [dict(type="shovel", row=cell[0], col=cell[1]),
                                 dict(type="plant", packet=int(packet["index"]), row=cell[0], col=cell[1])]
    wanted = drill.get("planted_type")
    if wanted is not None and int(wanted) != removed and removed is not None:
        packet = first_active(int(wanted))
        if packet:
            plans["human_replace"] = [dict(type="shovel", row=cell[0], col=cell[1]),
                                      dict(type="plant", packet=int(packet["index"]), row=cell[0], col=cell[1])]
    return plans, removed


def run_drill(job):
    cfg = _WORKER["config"]
    drill = job["drill"]
    env = _WORKER["env"]
    started = time.time()
    try:
        archive = cfg["roots"][drill["source_root"]] / drill["archive"]
        frames = read_frames(frame_path(archive))
        meta = json.loads((archive / "meta.json").read_text())
        obs, _prefix = drill_run.restore_prefix(env, archive, int(drill["decision"]), frames, meta)
        terrain = drill["terrain"]
        env.load_value_net(cfg["bins"]["roof" if terrain in ("roof", "boss") else "value"])
        prior = "gpt" if terrain in PRIOR_TERRAINS else "student"
        env.load_rollout_net(cfg["bins"][prior])
        if "row" not in drill or "col" not in drill:
            targets = drill.get("targets") or []
            if not targets:
                raise ValueError("drill has no cell and no target")
            row = int(targets[0]["row"])
            lane = [p for p in obs.get("plants") or []
                    if int(p["row"]) == row and int(p["type"]) not in (16, 30, 33)
                    and not p.get("squished")]
            if not lane:
                raise ValueError(f"no diggable plant in target row {row}")
            drill = dict(drill, row=row, col=max(int(p["col"]) for p in lane))
        root_id = env.snapshot_fast()
        root_tick = int(obs["tick"])
        plans, removed = plan_actions(drill, obs)
        cell_plant = None
        for p in obs.get("plants") or []:
            if (p["row"], p["col"]) == (int(drill["row"]), int(drill["col"])) and not p.get("squished"):
                info = dict(type=int(p["type"]), health=int(p["health"]), max_health=int(p["max_health"]),
                            state=int(p["state"]), state_countdown=int(p["state_countdown"]),
                            launch_counter=int(p["launch_counter"]), launch_rate=int(p["launch_rate"]),
                            shooting_counter=int(p["shooting_counter"]))
                if removed is not None and int(p["type"]) == removed:
                    cell_plant = info
        targets = drill.get("targets") or []
        cell_x = 40 + 80 * int(drill["col"]) + 40
        urgent_types = [t for t in targets if int(t["row"]) == int(drill["row"])
                        and float(t["x"]) - cell_x <= 160.0]
        out = dict(id=drill.get("uid", drill["id"]), terrain=terrain, level=int(drill["level"]),
                   row=int(drill["row"]), col=int(drill["col"]), removed=removed,
                   removed_name=PLANT_NAMES.get(removed, str(removed)),
                   planted_type=drill.get("planted_type"), human=bool(drill.get("human")),
                   root_tick=root_tick, sun=int(obs.get("sun") or 0),
                   cell_plant=cell_plant,
                   urgent=bool(urgent_types), urgent_targets=urgent_types,
                   plans={}, horizons=job["horizons"], seeds=job["seeds"])
        for name, actions in plans.items():
            entry = dict(actions=actions, applicable=True)
            env.restore_fast(root_id)
            post_tick = root_tick
            ok = True
            for action in actions:
                _obs, _r, _d, _t, info = env.step(action)
                if not info.get("ok"):
                    ok = False
                    entry["error"] = f"rejected {action.get('type')}"
                    break
                post_tick = int(_obs["tick"])
            if not ok:
                entry["applicable"] = False
                out["plans"][name] = entry
                continue
            entry["post_tick"] = post_tick
            env.load_value_net(cfg["bins"]["roof" if terrain in ("roof", "boss") else "value"])
            entry["leaf"] = float(env.leaf_values()[0])
            if "reference" in cfg["bins"]:
                reference = "reference_roof" if terrain in ("roof", "boss") else "reference"
                env.load_value_net(cfg["bins"].get(reference, cfg["bins"]["reference"]))
            values = {}
            for horizon in job["horizons"]:
                vals = []
                for seed in job["seeds"]:
                    env.restore_fast(root_id)
                    for action in actions:
                        env.step(action)
                    result = env.rollout(None, until_tick=post_tick + horizon,
                                         seed=int(seed), score_value=True)
                    vals.append(float(result["value"]))
                values[str(horizon)] = vals
                entry[f"sim_{horizon}"] = round(statistics.mean(vals), 6)
                entry[f"sim_sd_{horizon}"] = round(statistics.pstdev(vals), 6) if len(vals) > 1 else 0.0
            entry["values"] = values
            out["plans"][name] = entry
        env.restore_fast(root_id)
        out["seconds"] = round(time.time() - started, 2)
        out["status"] = "ok"
        return out
    except Exception as exc:
        return dict(status="error", id=drill.get("uid", drill.get("id")),
                    error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc()[-2000:], seconds=round(time.time() - started, 2))


def human_drills(bank):
    out = []
    for index, sequence in enumerate(bank.get("human_sequences") or []):
        out.append(dict(
            id=f"human{index}:{Path(sequence['archive']).name}:r{sequence['root_frame']}",
            uid=f"human{index}:{Path(sequence['archive']).name}:r{sequence['root_frame']}",
            source_root=sequence["source_root"], archive=sequence["archive"],
            decision=int(sequence["decision"]), level=int(sequence["level"]),
            terrain=sequence["terrain"], row=int(sequence["row"]), col=int(sequence["col"]),
            planted_type=int(sequence["planted"]["type"]),
            replanted_type=int(sequence["replanted"]["type"]),
            targets=sequence.get("targets") or [], human=True))
    return out


def select_drills(bank, args):
    drills = human_drills(bank)
    wanted = set(args.eval_ids or bank.get("eval_drill_ids") or [])
    if args.per_terrain:
        chosen, counts = [], Counter()
        for drill in bank["drills"]:
            if drill["id"] not in wanted:
                continue
            if counts[drill["terrain"]] >= args.per_terrain:
                continue
            counts[drill["terrain"]] += 1
            chosen.append(dict(drill, uid=drill.get("uid", drill["id"]), human=False))
        drills.extend(chosen)
    if args.drill_id:
        keep = set(args.drill_id)
        drills = [d for d in drills if d["id"] in keep or d.get("uid") in keep]
    return drills


def mean(values):
    return statistics.mean(values) if values else None


def summarize(rows, horizons):
    ok = [r for r in rows if r.get("status") == "ok"]
    summary = dict(drills=len(ok), errors=[r for r in rows if r.get("status") != "ok"],
                   by_plan={}, by_urgency={}, by_terrain={})
    plan_names = ["wait", "shovel_only", "replant_same", "human_replace"]
    for name in plan_names:
        entries = [r["plans"][name] for r in ok if name in r["plans"] and r["plans"][name].get("applicable")]
        if not entries:
            continue
        block = dict(drills=len(entries))
        for horizon in horizons:
            key = f"sim_{horizon}"
            paired = []
            for r in ok:
                plans = r["plans"]
                if name not in plans or not plans[name].get("applicable") or "wait" not in plans:
                    continue
                if not plans["wait"].get("applicable"):
                    continue
                paired.append((plans[name][f"sim_{horizon}"], plans["wait"][f"sim_{horizon}"],
                               plans[name]["leaf"], plans["wait"]["leaf"]))
            if not paired:
                continue
            sim_deltas = [a - b for a, b, _c, _d in paired]
            leaf_deltas = [c - d for _a, _b, c, d in paired]
            errors = [ld - sd for ld, sd in zip(leaf_deltas, sim_deltas)]
            block[f"h{horizon}"] = dict(
                n=len(paired),
                mean_sim_delta=round(mean(sim_deltas), 5),
                mean_leaf_delta=round(mean(leaf_deltas), 5),
                mean_error=round(mean(errors), 5),
                sd_sim_delta=round(statistics.pstdev(sim_deltas), 5) if len(sim_deltas) > 1 else 0.0,
                sd_error=round(statistics.pstdev(errors), 5) if len(errors) > 1 else 0.0,
                # How often the value net gets the *sign* of the change wrong.
                sign_disagree=sum(1 for ld, sd in zip(leaf_deltas, sim_deltas) if (ld > 0) != (sd > 0)),
            )
        summary["by_plan"][name] = block
    for urgency in (True, False):
        block = {}
        subset = [r for r in ok if bool(r.get("urgent")) == urgency]
        for name in plan_names:
            paired = []
            for r in subset:
                plans = r["plans"]
                if name not in plans or not plans[name].get("applicable") or "wait" not in plans:
                    continue
                for horizon in horizons:
                    paired.append((plans[name][f"sim_{horizon}"], plans["wait"][f"sim_{horizon}"],
                                   plans[name]["leaf"], plans["wait"]["leaf"], horizon))
            if not paired:
                continue
            out = {}
            for horizon in horizons:
                rows_h = [p for p in paired if p[4] == horizon]
                if not rows_h:
                    continue
                sim = [a - b for a, b, _c, _d, _h in rows_h]
                leaf = [c - d for _a, _b, c, d, _h in rows_h]
                out[f"h{horizon}"] = dict(n=len(rows_h), mean_sim_delta=round(mean(sim), 5),
                                          mean_leaf_delta=round(mean(leaf), 5),
                                          mean_error=round(mean([x - y for x, y in zip(leaf, sim)]), 5),
                                          sign_disagree=sum(1 for x, y in zip(leaf, sim) if (x > 0) != (y > 0)))
            block[name] = out
        summary["by_urgency"]["urgent" if urgency else "economic"] = block
    for terrain in sorted({r["terrain"] for r in ok}):
        block = {}
        subset = [r for r in ok if r["terrain"] == terrain]
        for name in plan_names:
            paired = []
            for r in subset:
                plans = r["plans"]
                if name not in plans or not plans[name].get("applicable") or "wait" not in plans:
                    continue
                p = plans[name]
                horizon = horizons[-1]
                paired.append((p[f"sim_{horizon}"], plans["wait"][f"sim_{horizon}"], p["leaf"], plans["wait"]["leaf"]))
            if not paired:
                continue
            sim = [a - b for a, b, _c, _d in paired]
            leaf = [c - d for _a, _b, c, d in paired]
            block[name] = dict(n=len(paired), mean_sim_delta=round(mean(sim), 5),
                               mean_leaf_delta=round(mean(leaf), 5),
                               mean_error=round(mean([x - y for x, y in zip(leaf, sim)]), 5),
                               sign_disagree=sum(1 for x, y in zip(leaf, sim) if (x > 0) != (y > 0)))
        summary["by_terrain"][terrain] = block
    # The 10-18 symptom: digging a plant without replanting raises the leaf value.
    removal = []
    for r in ok:
        plans = r["plans"]
        if "shovel_only" not in plans or not plans["shovel_only"].get("applicable"):
            continue
        removal.append((plans["shovel_only"]["leaf"], r))
    if removal:
        raised = [leaf for leaf, _r in removal]
        summary["removal_leaf"] = dict(
            n=len(removal),
            positive=sum(1 for r in ok if "shovel_only" in r["plans"]
                         and r["plans"]["shovel_only"].get("applicable")
                         and r["plans"]["shovel_only"]["leaf"] > r.get("root_leaf", 0)),
            mean_leaf=round(mean(raised), 5))
    return summary


def attach_root_leaf(rows):
    for r in rows:
        if r.get("status") != "ok":
            continue
        wait = r["plans"].get("wait")
        if wait and wait.get("applicable"):
            r["root_leaf"] = wait["leaf"]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--archive-root", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--teacher-prior", type=Path, required=True)
    parser.add_argument("--value", type=Path, required=True)
    parser.add_argument("--value-roof", type=Path, required=True)
    parser.add_argument("--static-value", type=Path, required=True)
    parser.add_argument("--reference-value", type=Path,
                        help="fixed rollout-end judge, independent of the tested leaf network")
    parser.add_argument("--reference-roof", type=Path,
                        help="fixed roof judge; defaults to --reference-value")
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--resource-dir", type=Path,
                        default=Path.home() / ".cache/pvz-research-resources")
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--seed-base", type=int, default=23600000)
    parser.add_argument("--horizon", type=int, default=1200)
    parser.add_argument("--horizon2", type=int, default=600)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--per-terrain", type=int, default=12)
    parser.add_argument("--eval-ids", nargs="*")
    parser.add_argument("--drill-id", action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        parser.error("--workers must be 1..6")
    roots = drill_run.parse_roots(args.archive_root)
    bank = json.loads(args.bank.read_text())
    drills = select_drills(bank, args)
    if not drills:
        parser.error("no drills selected")
    args.out.mkdir(parents=True, exist_ok=True)
    keep = tempfile.TemporaryDirectory(prefix="task236-values-", dir=args.out)
    bins = export_models(args, keep)
    seeds = [args.seed_base + index for index in range(args.seeds)]
    horizons = sorted({args.horizon, args.horizon2})
    config = dict(roots=roots, bins=bins, resource_dir=str(args.resource_dir),
                  executable=str(args.executable))
    jobs = [dict(drill=drill, seeds=seeds, horizons=horizons) for drill in drills]
    print(f"{len(drills)} drills, {len(jobs) * (2 + 2) * len(seeds) * len(horizons)} rollouts max, "
          f"{args.workers} workers", flush=True)
    rows = []
    started = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers, initializer=init_worker, initargs=(config,)) as pool:
        for row in pool.imap_unordered(run_drill, jobs):
            rows.append(row)
            print(f"{len(rows)}/{len(jobs)} {row.get('id')} {row.get('status')} "
                  f"{row.get('seconds')}s ({time.time() - started:.0f}s)", flush=True)
    attach_root_leaf(rows)
    (args.out / "rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n")
    summary = summarize(rows, horizons)
    summary.update(seconds=round(time.time() - started, 1), seeds=seeds, horizons=horizons,
                   bank=str(args.bank), student=str(args.student),
                   teacher_prior=str(args.teacher_prior), value=str(args.value),
                   value_roof=str(args.value_roof), static_value=str(args.static_value),
                   reference_value=str(args.reference_value) if args.reference_value else None,
                   reference_roof=str(args.reference_roof) if args.reference_roof else None)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    keep.cleanup()


if __name__ == "__main__":
    main()
