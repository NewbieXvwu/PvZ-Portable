"""Can a value net rank the search's candidate moves? Ground truth instead of guessing from game results.

From real positions of student games, every candidate (wait, and the student's best cell for each card) is
played and then continued by the student, in the C++ simulator, ``--samples`` times with different random
futures (the same futures for every candidate). Each run gives
  * the leaf after ``--horizon`` ticks -- what the search hands to the value net, and
  * the final result of playing on to the end -- the ground truth for that leaf.
Per value net it reports how often the net orders two candidates the way their true win rates do
(pairs that differ by at least ``--min-gap``), and the true win rate lost by picking the net's favourite
instead of the truly best candidate (regret), next to the regret of the student's own choice.

With ``--dump`` the leaves themselves (features, task context, the result of that run) are also saved, as
training data where the only difference between neighbouring samples is the move (scripts/train_value_branches.py).

Usage:
  python scripts/value_ranking_check.py --manifest artifacts/stage4/day_l7_hard.json --task-id day_l7_z15_eval \
      --rollout-policy artifacts/student/v3all/model.pt --values artifacts/value_net/v2/model.pt,... --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]

_W: dict = {}


def _init(args) -> None:
    import torch
    torch.set_num_threads(1)
    from pvz_event_env import EventWaitEnv
    from pvz_rollout_policy import export_student, load_student
    from pvz_value_net import load_value_net
    _W["args"] = args
    _W["student"] = load_student(args.rollout_policy)
    _W["values"] = {name: load_value_net(path) for name, path in args.value_map.items()}
    _W["tmp"] = tempfile.TemporaryDirectory()
    weights = Path(_W["tmp"].name) / "student.bin"
    export_student(_W["student"], weights)
    _W["env"] = EventWaitEnv(Path(args.resource_dir))
    _W["weights"] = weights


def _candidates(net, frame, previous, legal) -> list[dict]:
    """Wait, plus the student's favourite legal cell for every card that can be planted."""
    import torch
    from pvz_rollout_policy import LAWN_COLS, WAIT, features
    board, scalars = features(frame, previous)
    with torch.inference_mode():
        h, g = net.encode(board[None], scalars[None])
        out = [dict(WAIT)]
        for packet in sorted({a["packet"] for a in legal}):
            cells = net.cell_logits(h, g, torch.tensor([packet]))[0].tolist()
            best = max((a for a in legal if a["packet"] == packet), key=lambda a: cells[a["row"] * LAWN_COLS + a["col"]])
            out.append({"type": "plant", "packet": packet, "row": best["row"], "col": best["col"]})
    return out


def _game(seed_task) -> tuple[list[dict], list[tuple]]:
    import torch
    from pvz_env import TaskSpec
    from pvz_rollout_policy import act
    from pvz_value_net import featurize, observation_context
    from render_episode import _frame
    from scripted_baseline import profile_for_deck
    args, env, net = _W["args"], _W["env"], _W["student"]
    seed, kw = seed_task
    task = TaskSpec(level=kw["level"], seed=seed, playthrough=2, profile=profile_for_deck(tuple(kw["deck"])),
                    **{k: v for k, v in kw["task_extra"].items() if v is not None})
    obs, _ = env.reset(deck=tuple(kw["deck"]), task=task)
    env.load_rollout_net(_W["weights"])
    context = torch.from_numpy(observation_context(obs))
    rng = random.Random(seed)
    previous, previous_action, positions, dumped = None, None, [], []
    while not obs["terminal"]:
        frame = _frame(obs)
        frame["action"] = previous_action
        legal = (obs.get("legal_actions") or {}).get("plants") or []
        action = act(net, frame, previous, legal)
        if legal and obs["wave"] >= args.min_wave and rng.random() < args.position_rate:
            position, leaves = _evaluate(env, obs, frame, previous, legal, action, seed, len(positions), context)
            positions.append(position)
            dumped += leaves
        previous, previous_action = frame, action
        obs, _r, done, _t, info = env.step(action)
        if not info.get("ok"):
            obs, *_ = env.step({"type": "wait", "ticks": 60})
            previous_action = {"type": "wait", "ticks": 60}
        if done:
            break
    return positions, dumped


def _evaluate(env, obs, frame, previous, legal, student_action, seed, index, context) -> tuple[dict, list[tuple]]:
    import torch
    from pvz_value_net import featurize
    from render_episode import _frame
    args = _W["args"]
    cands = _candidates(_W["student"], frame, previous, legal)
    student_key = json.dumps(student_action, sort_keys=True)
    root = env.snapshot()
    outcomes, leaves, waves = [], [], []
    try:
        for cand in cands:
            wins, frames, reached = [], [], []
            for k in range(args.samples):
                future = (seed * 1000 + index * 50 + k) % 2 ** 32
                leaf = env.rollout(cand, obs["tick"] + args.horizon, sim_skip=args.sim_skip, snapshot_id=root, seed=future)
                if leaf.get("terminal"):
                    end = leaf
                    frames.append(None)
                else:
                    frames.append(_frame(leaf))
                    end = env.rollout(None, 0, sim_skip=args.sim_skip)
                wins.append(float(end.get("result") == 1)); reached.append(end["wave"])
            outcomes.append(wins); leaves.append(frames); waves.append(sum(reached) / len(reached))
    finally:
        env.restore(root)
        env.release_snapshot(root)
    dumped = []
    if args.dump:   # (board, scalars + context, won, seed, position, candidate) for every live leaf; numpy, because
                    # thousands of tensors sent back through the pool run out of file descriptors
        for c, (wins, frames) in enumerate(zip(outcomes, leaves)):
            for f, w in zip(frames, wins):
                if f is not None:
                    b, s = featurize(f)
                    dumped.append((b.half().numpy(), torch.cat([s, context]).numpy(), w, seed, index, c))
    values = {}
    for name, vnet in _W["values"].items():
        per_cand = []
        for wins, frames in zip(outcomes, leaves):
            live = [f for f in frames if f is not None]
            scores = []
            if live:
                b, s = zip(*(featurize(f) for f in live))
                s = torch.stack(s)
                if vnet.context:
                    s = torch.cat([s, context.expand(len(live), -1)], dim=1)
                with torch.inference_mode():
                    scores = torch.sigmoid(vnet(torch.stack(b), s)).tolist()
            it = iter(scores)
            per_cand.append(sum(next(it) if f is not None else w for f, w in zip(frames, wins)) / len(wins))
        values[name] = per_cand
    return {"seed": seed, "tick": obs["tick"], "wave": obs["wave"],
            "candidates": [json.dumps(c, sort_keys=True) for c in cands], "student": student_key,
            "true": [sum(w) / len(w) for w in outcomes], "true_wave": waves, "values": values}, dumped


def report(positions: list[dict], min_gap: float, margin: float) -> dict:
    out = {"positions": len(positions)}
    names = list(positions[0]["values"]) if positions else []
    student_regret = []
    for p in positions:
        q = p["true"]
        s = p["candidates"].index(p["student"]) if p["student"] in p["candidates"] else 0
        student_regret.append(max(q) - q[s])
    out["student_regret"] = sum(student_regret) / max(1, len(student_regret))
    out["oracle_gain_positions"] = sum(r >= min_gap for r in student_regret)
    for name in names:
        agree = pairs = wave_agree = wave_pairs = 0
        regret, search_regret = [], []
        for p in positions:
            q, v, w = p["true"], p["values"][name], p["true_wave"]
            for i in range(len(w)):
                for j in range(i + 1, len(w)):
                    if abs(w[i] - w[j]) >= 1.0:       # secondary truth when nearly every future loses
                        wave_pairs += 1
                        wave_agree += (w[i] - w[j]) * (v[i] - v[j]) > 0
            for i in range(len(q)):
                for j in range(i + 1, len(q)):
                    if abs(q[i] - q[j]) >= min_gap:
                        pairs += 1
                        agree += (q[i] - q[j]) * (v[i] - v[j]) > 0
            best = max(range(len(v)), key=v.__getitem__)
            regret.append(max(q) - q[best])
            # what the anchored search would do: keep the student's move unless another beats it by `margin`
            s = p["candidates"].index(p["student"]) if p["student"] in p["candidates"] else 0
            pick = best if v[best] > v[s] + margin else s
            search_regret.append(max(q) - q[pick])
        out[name] = {"pair_accuracy": agree / max(1, pairs), "pairs": pairs,
                     "wave_pair_accuracy": wave_agree / max(1, wave_pairs), "wave_pairs": wave_pairs,
                     "argmax_regret": sum(regret) / max(1, len(regret)),
                     "anchored_search_regret": sum(search_regret) / max(1, len(search_regret))}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--task-id", required=True)
    p.add_argument("--seeds", type=int, default=32)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--rollout-policy", required=True)
    p.add_argument("--values", required=True, help="name=path,... or path,... (name = parent directory); none = only --dump")
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--horizon", type=int, default=3000)
    p.add_argument("--sim-skip", type=int, default=21)
    p.add_argument("--min-wave", type=int, default=3)
    p.add_argument("--position-rate", type=float, default=0.03, help="chance that a decision with legal plants is evaluated")
    p.add_argument("--min-gap", type=float, default=0.25)
    p.add_argument("--margin", type=float, default=0.02)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--resource-dir", default="/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dump", type=Path, help="also save every live leaf with its result here (torch file)")
    args = p.parse_args()
    args.value_map = {} if args.values == "none" else dict((kv.split("=", 1) if "=" in kv else (Path(kv).parent.name, kv)) for kv in args.values.split(","))
    from render_episode import load_training_task, training_task_kwargs
    task = load_training_task(Path(args.manifest), args.task_id)
    jobs = [(s, training_task_kwargs(task, s)) for s in task["seeds"][args.seed_offset:args.seed_offset + args.seeds]]
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args,)) as pool:
        positions, dumped = [], []
        for done, (game, leaves) in enumerate(pool.imap_unordered(_game, jobs), 1):
            positions += game; dumped += leaves
            if done % 16 == 0:
                print(f"{done}/{len(jobs)} games, {len(positions)} positions, {len(dumped)} leaves", flush=True)
    result = {"config": {k: str(v) for k, v in vars(args).items()}, "summary": report(positions, args.min_gap, args.margin),
              "positions": positions}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result))
    if args.dump:
        import numpy as np
        import torch
        b, s, won, seed, pos, cand = zip(*dumped)
        torch.save({"boards": torch.from_numpy(np.stack(b)), "scalars": torch.from_numpy(np.stack(s)), "won": torch.tensor(won),
                    "seed": torch.tensor(seed), "position": torch.tensor(pos), "candidate": torch.tensor(cand),
                    "config": result["config"]}, args.dump)
    print(json.dumps(result["summary"], indent=1))


if __name__ == "__main__":
    main()
