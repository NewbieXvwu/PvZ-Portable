"""Measure the v5/v6 leaf values behind task72b's 94 fog states (wait vs static_best).

The original task72b probe recorded, for every state, one value per candidate and
horizon, but shipped no protocol.  Two things *are* recoverable from the recorded
identity: the decision seed (future_seed * 10000 + decision, search_play.py:671) and
the three sampling futures derived from it (decision_seed * 1000 + k,
search_play.py:407).  The reconstruction in task78_task72_rebuild.py already replays
the original trajectories and matches all 94 states, so the remaining unknown is only
which actions were scored.

The enumeration is recoverable too.  The frozen recorder (task78_task72_rebuild.py,
Recorder.before) rebuilds each state's cell list and static_best from the legal
packet-8 (cactus) plants only, which is exactly the reference's cell count at every
state; the probe therefore scores one plant per legal cactus cell plus wait, keyed
"row,col".  It leaves the frozen reconstruction policy untouched (the trajectory
stays byte-identical to the one bind2 matched) and only adds the scoring rollouts, so
the measured values can be diffed against the recorded ones at all three horizons.

Nothing here writes labels, changes a model, or starts a second game; one run is one
game.
"""
from __future__ import annotations

import fcntl
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]
from task78_data import MAIN, sha  # noqa: E402

REFERENCE = ROOT / "MEMORY/docs/task72b-probe-rows.jsonl"
TASK = "demo_fog_33_cactus_snow_m200"
SEED = 20261033
TAKEOVERS = (229, 306, 382)
FUTURES = tuple(range(101, 109))
HORIZONS = (2500, 4000, 6000)
SAMPLES = 3
DEMO = ROOT.parent / "cq/artifacts/human_demo/raw/seed_20261033" / TASK
STUDENT = MAIN / "artifacts/student/dw4i_s1/model.pt"
STATIC = Path("/home/newbiexvwu/pvz-logs/mac_branch1/model.pt")
MANIFEST = ROOT / "artifacts/task78/task72_rebuild_mac_branch1/tasks.json"
REBUILD = ROOT / "artifacts/task78/task72_rebuild_mac_branch1"
OUT = ROOT / "artifacts/task78/task72_probe94"


def reference_rows():
    rows = [json.loads(line) for line in REFERENCE.read_text().splitlines() if line.strip()]
    if len(rows) != 94:
        raise ValueError("exactly 94 original reference states required")
    return rows


PACKET = 8


def cell_key(action):
    return "wait" if action.get("type") == "wait" else "%d,%d" % (int(action["row"]), int(action["col"]))


def cells_of(obs):
    """The reference's cell list: legal cactus (packet 8) plants, one action per cell."""
    plants = (obs.get("legal_actions") or {}).get("plants") or []
    cells = sorted([a["row"], a["col"]] for a in plants if a["packet"] == PACKET)
    return [dict(type="plant", packet=PACKET, row=row, col=col) for row, col in cells]


def takeover_states(takeover):
    """The reference states of this takeover, keyed by (future, decision, tick).

    The three takeovers restart the decision counter, so the recorded (seed, decision)
    is ambiguous between them; the tick from the frozen reconstruction separates them.
    """
    if takeover not in TAKEOVERS:
        raise ValueError("only the original takeovers 229/306/382 are supported")
    keys = set()
    for future in FUTURES:
        path = REBUILD / f"n{takeover}/f{future}/signatures.jsonl"
        for line in path.read_text().splitlines():
            row = json.loads(line)
            keys.add((row["seed"], row["decision"], row["tick"]))
    return {(row["seed"], row["decision"], row["tick"]): row
            for row in reference_rows() if (row["seed"], row["decision"], row["tick"]) in keys}


def recorded_actions(takeover, future):
    """The actions the frozen reconstruction executed, one per decision.

    The frame archive stores one line per decision boundary (frames[0] is the state the
    takeover starts in), and each line carries the action that produced the next one.
    Replaying them keeps the trajectory identical whatever value net is loaded, so the
    two arms differ only in the leaf scores the preregistered gate compares.
    """
    from pvz_frame_store import frame_path, read_frames
    path = REBUILD / "frames" / f"seed_{SEED}_f{future}" / f"task72b_n{takeover}_fast_search"
    frames = read_frames(frame_path(path))
    return [frames[index + 1]["action"] for index in range(len(frames) - 1)]


def resource_check():
    available = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                         if line.startswith("MemAvailable:"))) * 1024
    if available < 8 * 1024 ** 3:
        raise RuntimeError("yield to qc: probe requires available memory >=8 GiB")


def leaf_value(teacher, env, obs, root, action, horizon, decision_seed):
    """The value search_play assigns a candidate: mean of --samples terminal-or-leaf scores."""
    scores = []
    for k in range(SAMPLES):
        leaf = teacher._rollout_policy(env, obs, action, obs["tick"] + horizon, root,
                                       (decision_seed * 1000 + k) % 2 ** 32)
        terminal = teacher._score_terminal_or_value(leaf)
        scores.append(terminal if terminal is not None else leaf["value"])
    if not all(math.isfinite(v) for v in scores):
        raise ValueError("non-finite leaf value")
    return sum(scores) / len(scores)


def probe(takeover, future, value, label, out=OUT, slot=0, sim_skip=61):
    """Play the frozen reconstruction trajectory once and score wait/static_best at its states."""
    if takeover not in TAKEOVERS or future not in FUTURES:
        raise ValueError("only the original takeovers and futures are supported")
    states = takeover_states(takeover)
    recorded = recorded_actions(takeover, future)
    import search_play as teacher

    rows = []
    original_search = teacher._search

    def search(env, obs, policy_action, args, decision_seed, stats):
        index = decision_seed % 10000
        if index >= len(recorded):
            raise ValueError("recorded actions ended at decision %d" % index)
        action = recorded[index]
        state = states.get((decision_seed // 10000, index, int(obs["tick"])))
        if state is None:
            return action
        chosen = state["chosen"] if isinstance(state["chosen"], dict) else json.loads(state["chosen"])
        if action != chosen:
            raise ValueError("replayed action differs from the reference at decision %d" % index)
        actions = cells_of(obs)
        if sorted([a["row"], a["col"]] for a in actions) != sorted([list(c) for c in state["cells"]]):
            raise ValueError("cell list differs from the reference at decision %d" % state["decision"])
        root = env.snapshot_fast()
        try:
            scored = teacher._static_scores(env, root, actions, obs.get("sun_income_rate"))
            best = max(scored, key=lambda item: item[0])[1]
            if [best["row"], best["col"]] != list(state["static_best"]):
                raise ValueError("static_best cell differs from the reference at decision %d" % state["decision"])
            candidates = [teacher.NOOP] + actions
            record = dict(task=args.task_id, seed=future, decision=state["decision"], tick=int(obs["tick"]),
                          takeover=takeover, static_best=list(state["static_best"]), values={})
            for horizon in HORIZONS:
                record["values"][str(horizon)] = {
                    cell_key(c): leaf_value(teacher, env, obs, root, c, horizon, decision_seed)
                    for c in candidates}
            rows.append(record)
        finally:
            env.restore_fast(root)
            env.release_snapshot(root)
        return action

    args = teacher.parser().parse_args([
        "--manifest", str(MANIFEST), "--task-id", TASK, "--out", str(out / label / f"n{takeover}_f{future}"),
        "--value", str(value), "--rollout-policy", str(STUDENT), "--static-value", str(STATIC),
        "--demo-archive", str(DEMO), "--demo-decision", str(takeover),
        "--future-seeds", str(future), "--workers", "1", "--verify-samples", "9",
        "--student-candidates", "2", "--executable", str(ROOT / "build/pvz-portable"),
        "--resource-dir", "/home/newbiexvwu/.cache/pvz-research-resources"])
    resource_check()
    os.nice(19)
    (out / label).mkdir(parents=True, exist_ok=True)
    with (ROOT / f"artifacts/task78/probe-{slot}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        teacher._search = search
        keep, bins = teacher._export_values(str(value), str(STATIC))
        teacher._init_worker(bins[0], str(STUDENT), sim_skip, bins[1])
        try:
            job = dict(args=args, seed=SEED, future_seed=future, arm="fast_search")
            result = teacher._play(job, None)
        finally:
            teacher._close_worker_env()
            keep.cleanup()
            teacher._search = original_search
    if result.get("terminal") is not True:
        raise ValueError("probe trajectory did not reach terminal")
    path = out / label / f"n{takeover}_f{future}.jsonl"
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + chr(10))
    return path, len(rows), states


def load(label, out=OUT):
    rows = []
    for path in sorted((out / label).glob("n*_f*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    if len({(r["takeover"], r["seed"], r["decision"], r["tick"]) for r in rows}) != len(rows):
        raise ValueError("duplicate probe states in " + label)
    return rows


def compare(label, horizon, out=OUT):
    """Diff the measured wait/static_best values against task72b's recorded ones."""
    measured = {(r["takeover"], r["seed"], r["decision"], r["tick"]): r for r in load(label, out)}
    diffs, missing, cells = [], [], []
    for row in reference_rows():
        key_state = (row["seed"], row["decision"], row["tick"])
        takeover = next((t for t in TAKEOVERS if (t,) + key_state in measured), None)
        if takeover is None:
            missing.append(list(key_state))
            continue
        got = measured[(takeover,) + key_state]["values"][str(horizon)]
        expected = row["values"][str(horizon)]
        if sorted(expected) != sorted(got):
            cells.append([row["seed"], row["decision"], sorted(expected), sorted(got)])
            continue
        for candidate in sorted(expected):
            diffs.append([row["seed"], row["decision"], candidate, expected[candidate],
                          got[candidate], abs(got[candidate] - expected[candidate])])
    finite = [d[5] for d in diffs]
    report = dict(label=label, horizon=horizon, states=len(reference_rows()), candidate_set_mismatch=cells,
                  matched=len(reference_rows()) - len(missing),
                  missing=missing, values=len(diffs), max_abs_diff=max(finite) if finite else None,
                  within_rounding=sum(1 for d in finite if d <= 5e-5),
                  within_milli=sum(1 for d in finite if d <= 1e-3),
                  worst=sorted(diffs, key=lambda d: -d[5])[:5])
    if missing:
        report["verdict"] = "state_mismatch"
    elif cells:
        report["verdict"] = "candidate_mismatch"
    elif finite and report["max_abs_diff"] <= 5e-5:
        report["verdict"] = "reproduced"
    else:
        report["verdict"] = "value_mismatch"
    return report


def signs(values, static_best):
    """The per-horizon sign of value(static_best) - value(wait); None where either is missing."""
    key = "%d,%d" % tuple(static_best)
    out = {}
    for horizon in HORIZONS:
        table = values.get(str(horizon)) or {}
        if "wait" not in table or key not in table:
            out[horizon] = None
            continue
        gap = table[key] - table["wait"]
        out[horizon] = 0 if abs(gap) < 5e-9 else (1 if gap > 0 else -1)
    return out


def score(labels=("v5r", "v6"), out=OUT):
    """The preregistered stability gate: direction flips across 2500/4000/6000 for v5 and v6."""
    measured = {label: {(r["takeover"], r["seed"], r["decision"], r["tick"]): r for r in load(label, out)}
                for label in labels}
    per_state = []
    for row in reference_rows():
        key_state = (row["seed"], row["decision"], row["tick"])
        entry = dict(seed=row["seed"], decision=row["decision"], tick=row["tick"],
                     takeover=next((t for t in TAKEOVERS if (t,) + key_state in measured[labels[0]]), None))
        if entry["takeover"] is None:
            continue
        for label in labels:
            entry[label] = signs(measured[label][(entry["takeover"],) + key_state]["values"],
                                 row["static_best"])
        per_state.append(entry)
    report = dict(states=len(reference_rows()), measured_states=len(per_state), arms={})
    for label in labels:
        evaluable = [e for e in per_state if all(e[label][h] is not None for h in HORIZONS)]
        zeros = [e for e in evaluable if any(e[label][h] == 0 for h in HORIZONS)]
        directional = [e for e in evaluable if all(e[label][h] != 0 for h in HORIZONS)]
        flips = [e for e in directional if len({e[label][h] for h in HORIZONS}) > 1]
        report["arms"][label] = dict(evaluable=len(evaluable), zero_direction=len(zeros),
                                     directional=len(directional), flips=len(flips),
                                     flip_rate=(len(flips) / len(directional)) if directional else None,
                                     flip_states=[[e["takeover"], e["seed"], e["decision"]] for e in flips])
    paired = [e for e in per_state
              if all(all(e[label][h] is not None and e[label][h] != 0 for h in HORIZONS) for label in labels)]
    report["paired_states"] = len(paired)
    if len(labels) == 2 and paired:
        import random
        clusters = {}
        for e in paired:
            flip = {label: len({e[label][h] for h in HORIZONS}) > 1 for label in labels}
            clusters.setdefault(e["seed"], []).append(flip)
        keys = sorted(clusters)
        rng = random.Random(780)
        diffs = []
        for _ in range(10000):
            drawn = [rng.choice(keys) for _ in keys]
            a = b = n = 0
            for key in drawn:
                for flip in clusters[key]:
                    n += 1
                    a += flip[labels[0]]
                    b += flip[labels[1]]
            diffs.append(b / n - a / n)
        diffs.sort()
        report["flip_rate_difference"] = dict(
            point=(report["arms"][labels[1]]["flips"] / report["arms"][labels[1]]["directional"]
                   - report["arms"][labels[0]]["flips"] / report["arms"][labels[0]]["directional"]),
            ci95=[diffs[250], diffs[9749]], clusters=len(keys), resamples=10000, seed=780)
        base = report["arms"][labels[0]]["flip_rate"]
        report["gate"] = dict(
            v6_at_most_half_v5=(report["arms"][labels[1]]["flip_rate"] is not None and base is not None
                                and report["arms"][labels[1]]["flip_rate"] <= base / 2),
            upper_bound_at_most_zero=report["flip_rate_difference"]["ci95"][1] <= 0)
        report["gate"]["passed"] = bool(report["gate"]["v6_at_most_half_v5"]
                                        and report["gate"]["upper_bound_at_most_zero"])
    return report


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("probe")
    run.add_argument("--takeover", type=int, required=True)
    run.add_argument("--future", type=int, required=True)
    run.add_argument("--value", required=True)
    run.add_argument("--label", required=True)
    run.add_argument("--slot", type=int, default=0)
    run.add_argument("--sim-skip", type=int, default=61)
    tally = sub.add_parser("score")
    tally.add_argument("--out", default=str(OUT))
    check = sub.add_parser("compare")
    check.add_argument("--label", required=True)
    check.add_argument("--horizon", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "probe":
        path, count, states = probe(args.takeover, args.future, args.value, args.label,
                                    slot=args.slot, sim_skip=args.sim_skip)
        print(json.dumps(dict(path=str(path), states=count, expected=len(states))))
    elif args.command == "compare":
        print(json.dumps(compare(args.label, args.horizon), indent=1))
    else:
        print(json.dumps(score(out=Path(args.out)), indent=1))


if __name__ == "__main__":
    main()
