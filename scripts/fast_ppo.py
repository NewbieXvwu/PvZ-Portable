"""PPO on the small student network (python/pvz_rollout_policy.py), now that RL is cheap.

The 7.9M-parameter pi made PPO cost ~11 h per 1M decisions, so RL from scratch never got off the
ground. The student is ~0.5M parameters and plays a game in ~4 s, which restores the premise of
DESIGN.md §1 (the simulator's speed sets the pace). This trains it with PPO, starting from a
student checkpoint, on any task in a manifest -- e.g. night levels, where no teacher exists.

Policy: plant-or-wait (Bernoulli) -> packet (categorical over legal packets) -> cell (categorical
over that packet's legal cells), the same heads as the student, plus a value head. Steps with no
legal plant are forced waits: they carry rewards but no policy gradient.
Reward: +1 win / -1 loss at the end, +0.02 each time a new wave starts.

Usage:
  python scripts/fast_ppo.py --init artifacts/student/v2/model.pt --manifest <m.json> --task-id <id> \
      --eval-manifest <m.json> --eval-task-id <id> --out artifacts/fast_ppo/run1
"""
from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESOURCE_DIR = "/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN"

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]
from pvz_rollout_policy import COLS, LAWN_COLS, PACKETS, ROWS, WAIT, RolloutNet, features  # noqa: E402

CELLS = ROWS * LAWN_COLS
WAVE_BONUS = 0.02


class ActorCritic(RolloutNet):
    def __init__(self, width: int = 64):
        super().__init__(width)
        self.value = nn.Sequential(nn.Linear(2 * width, width), nn.ReLU(), nn.Linear(width, 1))

    def heads(self, board, scalars):
        h = self.trunk(board)
        s = self.scalar(scalars)
        g = torch.cat([h.mean(dim=(2, 3)), s], dim=1)
        return h, g

    def cell_logits(self, h, g, packet):
        e = self.packet_embedding(packet)[:, :, None, None].expand(-1, -1, ROWS, COLS)
        ctx = g[:, :, None, None].expand(-1, -1, ROWS, COLS)
        return self.cell(torch.cat([h, e, ctx], dim=1))[:, 0, :, :LAWN_COLS].flatten(1)


def _masks(legal: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    packet_mask = np.zeros(PACKETS, dtype=bool)
    cell_mask = np.zeros((PACKETS, CELLS), dtype=bool)
    for a in legal:
        if a["packet"] < PACKETS and a["row"] < ROWS and a["col"] < LAWN_COLS:
            packet_mask[a["packet"]] = True
            cell_mask[a["packet"], a["row"] * LAWN_COLS + a["col"]] = True
    return packet_mask, cell_mask


@torch.inference_mode()
def _choose(net: ActorCritic, board, scalars, packet_mask, cell_mask, greedy: bool, rng: np.random.Generator):
    h, g = net.heads(board[None], scalars[None])
    value = float(net.value(g)[0, 0])
    if not packet_mask.any():
        return dict(WAIT), None, value
    p_plant = float(torch.sigmoid(net.plant(g)[0, 0]))
    plant = p_plant > 0.5 if greedy else rng.random() < p_plant
    if not plant:
        return dict(WAIT), (0, -1, -1), value
    plog = net.packet(g)[0].masked_fill(~torch.from_numpy(packet_mask), -1e9)
    packet = int(plog.argmax()) if greedy else int(torch.multinomial(F.softmax(plog, 0), 1))
    clog = net.cell_logits(h, g, torch.tensor([packet]))[0].masked_fill(~torch.from_numpy(cell_mask[packet]), -1e9)
    cell = int(clog.argmax()) if greedy else int(torch.multinomial(F.softmax(clog, 0), 1))
    return ({"type": "plant", "packet": packet, "row": cell // LAWN_COLS, "col": cell % LAWN_COLS},
            (1, packet, cell), value)


_NET: ActorCritic | None = None
_ENV = None


def _worker_init(resource_dir: str) -> None:
    global _NET, _ENV
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)
    _NET = ActorCritic().eval()
    from pvz_event_env import EventWaitEnv
    _ENV = EventWaitEnv(resource_dir, headless=True)


def _play(job: dict) -> dict:
    from pvz_env import TaskSpec
    from render_episode import _frame, load_training_task, training_task_kwargs
    from scripted_baseline import profile_for_deck
    if job.get("weights") is not None:
        _NET.load_state_dict(torch.load(io.BytesIO(job["weights"]), weights_only=False))
    kw = training_task_kwargs(load_training_task(job["manifest"], job["task_id"]), job["seed"])
    task = TaskSpec(level=kw["level"], seed=kw["seed"], playthrough=2, profile=profile_for_deck(tuple(kw["deck"])),
                    **{k: v for k, v in kw["task_extra"].items() if v is not None})
    obs, _ = _ENV.reset(deck=tuple(kw["deck"]), task=task)
    rng = np.random.default_rng(job["seed"] * 7919 + job.get("iteration", 0))
    steps, previous, last_action = [], None, None
    planted: dict[int, int] = {}
    wave = obs["wave"]
    while not obs["terminal"] and len(steps) < job["max_actions"]:
        frame = _frame(obs)
        frame["action"] = last_action
        board, scalars = features(frame, previous)
        packet_mask, cell_mask = _masks((obs.get("legal_actions") or {}).get("plants") or [])
        action, choice, value = _choose(_NET, board, scalars, packet_mask, cell_mask, job["greedy"], rng)
        obs, _r, done, _t, info = _ENV.step(action)
        if not info.get("ok"):
            obs, _r, done, _t, info = _ENV.step(dict(WAIT))
            action = dict(WAIT)
        reward = WAVE_BONUS * max(0, obs["wave"] - wave)
        wave = obs["wave"]
        if obs["terminal"]:
            reward += 1.0 if obs.get("result") == 1 else -1.0
        if not job["greedy"]:
            steps.append({"board": board.half().numpy(), "scalars": scalars.numpy(), "packet_mask": packet_mask,
                          "cell_mask": cell_mask, "choice": choice, "value": value, "reward": reward})
        if action["type"] == "plant":
            planted[action["packet"]] = planted.get(action["packet"], 0) + 1
        previous, last_action = frame, action
        if done:
            break
    return {"seed": job["seed"], "won": obs.get("result") == 1, "wave": obs["wave"], "steps": steps, "planted": planted}


def _logp_entropy(net: ActorCritic, batch: dict, device: str):
    board, scalars = batch["board"].to(device).float(), batch["scalars"].to(device)
    h, g = net.heads(board, scalars)
    value = net.value(g)[:, 0]
    planted = batch["plant"].to(device)
    pl = net.plant(g)[:, 0]
    logp = torch.where(planted, F.logsigmoid(pl), F.logsigmoid(-pl))
    p = torch.sigmoid(pl)
    entropy = -(p * F.logsigmoid(pl) + (1 - p) * F.logsigmoid(-pl))
    if planted.any():
        idx = torch.nonzero(planted).squeeze(1)
        packet = batch["packet"].to(device)[idx]
        plog = net.packet(g[idx]).masked_fill(~batch["packet_mask"].to(device)[idx], -1e9)
        clog = net.cell_logits(h[idx], g[idx], packet)
        cmask = batch["cell_mask_chosen"].to(device)[idx]
        clog = clog.masked_fill(~cmask, -1e9)
        extra = F.log_softmax(plog, 1).gather(1, packet[:, None])[:, 0] + \
            F.log_softmax(clog, 1).gather(1, batch["cell"].to(device)[idx][:, None])[:, 0]
        logp = logp.index_add(0, idx, extra)
    return logp, entropy, value


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init", required=True, help="student checkpoint (RolloutNet state_dict)")
    p.add_argument("--manifest", required=True)
    p.add_argument("--task-id", required=True)
    p.add_argument("--eval-manifest", required=True)
    p.add_argument("--eval-task-id", required=True)
    p.add_argument("--eval-seeds", type=int, default=128)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--games", type=int, default=112, help="games per PPO iteration")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--entropy", type=float, default=0.003)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--max-actions", type=int, default=3000)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--resource-dir", default=DEFAULT_RESOURCE_DIR)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    from render_episode import load_training_task
    train_seeds = load_training_task(args.manifest, args.task_id)["seeds"]
    eval_seeds = load_training_task(args.eval_manifest, args.eval_task_id)["seeds"][: args.eval_seeds]

    net = ActorCritic()
    missing, unexpected = net.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=False)["state_dict"],
                                              strict=False)
    assert all(k.startswith("value.") for k in missing) and not unexpected, (missing, unexpected)
    net.to(args.device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0)
    log = (args.out / "log.jsonl").open("a")
    started = time.time()

    def weights() -> bytes:
        buffer = io.BytesIO()
        torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, buffer)
        return buffer.getvalue()

    with mp.get_context("spawn").Pool(args.workers, initializer=_worker_init, initargs=(args.resource_dir,)) as pool:
        def evaluate(iteration: int) -> dict:
            w = weights()
            res = list(pool.imap_unordered(_play, [{"seed": s, "manifest": args.eval_manifest, "task_id": args.eval_task_id,
                                                    "greedy": True, "weights": w, "max_actions": args.max_actions}
                                                   for s in eval_seeds]))
            out = {"iteration": iteration, "eval_won": sum(r["won"] for r in res), "eval_games": len(res),
                   "eval_mean_wave": round(sum(r["wave"] for r in res) / len(res), 2), "seconds": round(time.time() - started)}
            log.write(json.dumps(out) + "\n"); log.flush(); print(json.dumps(out), flush=True)
            return out

        evaluate(0)
        rng = np.random.default_rng(0)
        for iteration in range(1, args.iterations + 1):
            w = weights()
            seeds = rng.choice(train_seeds, size=args.games, replace=len(train_seeds) < args.games)
            games = list(pool.imap_unordered(_play, [{"seed": int(s), "manifest": args.manifest, "task_id": args.task_id,
                                                      "greedy": False, "weights": w, "iteration": iteration,
                                                      "max_actions": args.max_actions} for s in seeds]))
            rows, adv, ret = [], [], []
            for game in games:
                steps = game["steps"]
                values = [s["value"] for s in steps] + [0.0]
                gae = 0.0
                game_adv = [0.0] * len(steps)
                for t in reversed(range(len(steps))):
                    delta = steps[t]["reward"] + args.gamma * values[t + 1] - values[t]
                    gae = delta + args.gamma * args.lam * gae
                    game_adv[t] = gae
                for t, s in enumerate(steps):
                    if s["choice"] is None:
                        continue
                    rows.append(s); adv.append(game_adv[t]); ret.append(game_adv[t] + s["value"])
            if not rows:
                continue
            batch_all = {
                "board": torch.from_numpy(np.stack([r["board"] for r in rows])),
                "scalars": torch.from_numpy(np.stack([r["scalars"] for r in rows])),
                "plant": torch.tensor([r["choice"][0] == 1 for r in rows]),
                "packet": torch.tensor([max(0, r["choice"][1]) for r in rows]),
                "cell": torch.tensor([max(0, r["choice"][2]) for r in rows]),
                "packet_mask": torch.from_numpy(np.stack([r["packet_mask"] for r in rows])),
                "cell_mask_chosen": torch.from_numpy(np.stack([r["cell_mask"][max(0, r["choice"][1])] for r in rows])),
            }
            advantages = torch.tensor(adv, dtype=torch.float32)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            returns = torch.tensor(ret, dtype=torch.float32)
            with torch.no_grad():
                old_logp = torch.cat([_logp_entropy(net, {k: v[i:i + 4096] for k, v in batch_all.items()}, args.device)[0].cpu()
                                      for i in range(0, len(rows), 4096)])
            stats = []
            for _epoch in range(args.epochs):
                perm = torch.randperm(len(rows))
                for i in range(0, len(rows), 4096):
                    idx = perm[i:i + 4096]
                    mb = {k: v[idx] for k, v in batch_all.items()}
                    logp, entropy, value = _logp_entropy(net, mb, args.device)
                    ratio = torch.exp(logp - old_logp[idx].to(args.device))
                    a = advantages[idx].to(args.device)
                    policy_loss = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
                    value_loss = F.mse_loss(value, returns[idx].to(args.device))
                    loss = policy_loss + 0.5 * value_loss - args.entropy * entropy.mean()
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                    stats.append((policy_loss.item(), value_loss.item(), entropy.mean().item(),
                                  float(((ratio - 1).abs() > args.clip).float().mean())))
            s = np.array(stats).mean(0)
            out = {"iteration": iteration, "train_won": sum(g["won"] for g in games), "train_games": len(games),
                   "train_mean_wave": round(sum(g["wave"] for g in games) / len(games), 2), "decisions": len(rows),
                   "policy_loss": round(s[0], 4), "value_loss": round(s[1], 4), "entropy": round(s[2], 4),
                   "clip_frac": round(s[3], 4), "seconds": round(time.time() - started),
                   "planted_per_game": {k: round(sum(g["planted"].get(k, 0) for g in games) / len(games), 2)
                                        for k in sorted({k for g in games for k in g["planted"]})}}
            log.write(json.dumps(out) + "\n"); log.flush(); print(json.dumps(out), flush=True)
            if iteration % args.eval_every == 0 or iteration == args.iterations:
                evaluate(iteration)
                torch.save({"state_dict": {k: v.cpu() for k, v in net.state_dict().items()}, "iteration": iteration},
                           args.out / f"model_iter{iteration}.pt")


if __name__ == "__main__":
    main()
