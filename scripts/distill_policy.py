"""Distil "pi + search" back into pi (DESIGN.md §13, stage 3 -- expert iteration).

Input: trajectories recorded by ``scripts/search_play.py --record DIR`` (the actions actually
played: mostly pi's own, plus the few steps where the search re-placed a plant).

Starting from pi's checkpoint we fine-tune with behaviour cloning (-log p(recorded action), the
loss of scripts/bc_pipeline/train_bc_gru_sequence.py). pi already gives its own actions
probability ~1, so their gradient is ~0 and the signal sits on the ~6 search-changed steps per
game. Training therefore uses short windows:

  * one no-grad pass of the *original* pi over every game caches the GRU state at every
    ``--window`` steps;
  * each epoch trains on every window that contains a search change, plus an equal number of
    random windows of pi's own play (so it keeps what it already does well), each window
    starting from its cached GRU state.

The cached states come from the original weights and drift slightly as pi changes; with a small
learning rate that error is small, and the real test is playing games afterwards anyway.

Validation (held-out seeds) reports how often greedy pi now picks the search's choice on changed
steps, and how often it still reproduces its own plants elsewhere.

The output keeps the RL checkpoint layout (``config`` + ``state_dict``), so
``search_play.py --policy`` and ``render_episode.load_checkpoint_policy`` load it unchanged.

Usage:
  python scripts/distill_policy.py --policy <pi.pt> --data artifacts/distill/v0b_train/rows \
      --out artifacts/distill/pi_v0b_distilled.pt --device mps
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]

import torch  # noqa: E402

from pvz_agent_model import GameplayModelV1, replay_log_probs, select_action  # noqa: E402


def load_games(data: Path) -> list[dict]:
    games = []
    for path in sorted(data.glob("*.pkl.z")):
        game = pickle.loads(zlib.decompress(path.read_bytes()))
        game["path"] = str(path)
        games.append(game)
    return games


_CACHE_MODEL = None


def _cache_init(policy: str) -> None:
    global _CACHE_MODEL
    torch.set_num_threads(1)
    checkpoint = torch.load(policy, map_location="cpu", weights_only=False)
    _CACHE_MODEL = GameplayModelV1(checkpoint["config"])
    _CACHE_MODEL.load_state_dict(checkpoint["state_dict"], strict=False)


def _cache_job(job: tuple[list[str], int]) -> list[list]:
    paths, window = job
    games = [pickle.loads(zlib.decompress(Path(p).read_bytes())) for p in paths]
    cache_hidden(_CACHE_MODEL, games, window, batch=len(games))
    return [g["hidden"] for g in games]


def cache_hidden_parallel(policy: str, games: list[dict], window: int, workers: int) -> None:
    """Same as ``cache_hidden`` with the original weights, split over CPU processes.

    Building token batches row by row is Python-bound (~3 ms/row on CPU or MPS alike), so
    processes, not a faster device, are what shortens this pass.
    """
    import multiprocessing as mp
    jobs = [([g["path"] for g in games[i:i + 4]], window) for i in range(0, len(games), 4)]
    with mp.get_context("spawn").Pool(workers, initializer=_cache_init, initargs=(policy,)) as pool:
        for start, hiddens in zip(range(0, len(games), 4), pool.imap(_cache_job, jobs)):
            for g, hidden in zip(games[start:start + 4], hiddens):
                g["hidden"] = hidden


@torch.inference_mode()
def cache_hidden(model, games: list[dict], window: int, batch: int = 32) -> None:
    """Store the GRU state at the start of every window: ``game["hidden"][k]`` for rows k*window.."""
    model.eval()
    for first in range(0, len(games), batch):
        group = games[first:first + batch]
        for g in group:
            g["hidden"] = [None]
        longest = max(len(g["rows"]) for g in group)
        state = {id(g): None for g in group}
        for offset in range(0, longest, window):
            active = [g for g in group if offset < len(g["rows"])]
            _outputs, hidden_out = model.forward_sequences(
                [g["rows"][offset:offset + window] for g in active], [state[id(g)] for g in active])
            for index, g in enumerate(active):
                state[id(g)] = hidden_out[:, index].clone()
                g["hidden"].append(state[id(g)].cpu())


def _is_signal(row: dict, soft: bool) -> bool:
    return bool(row.get("search_targets")) if soft else bool(row.get("search_override"))


def windows(games: list[dict], window: int, rng: random.Random, regular_ratio: float,
            plain_needs_plant: bool = True, soft: bool = False) -> list[tuple[int, int]]:
    changed, plain = [], []
    for gi, g in enumerate(games):
        for k in range((len(g["rows"]) + window - 1) // window):
            rows = g["rows"][k * window:(k + 1) * window]
            if any(_is_signal(r, soft) for r in rows):
                changed.append((gi, k))
            elif not plain_needs_plant or any(r["action"]["type"] == "plant" for r in rows):
                plain.append((gi, k))
    extra = rng.sample(plain, min(len(plain), int(len(changed) * regular_ratio)))
    return changed + extra


def _soft_loss(model, outputs, flat, logp, args, prior_outputs=None):
    """-sum_c q_c log pi(c) on searched steps, q = softmax(search value / tau); -log pi(a) elsewhere.

    AlphaZero distils the search's preference distribution, not its single argmax: when two cells
    are nearly tied, pi is asked to spread probability between them instead of copying a noisy pick.
    Every candidate's log-probability reuses the same forward output, so this costs no extra forward.
    """
    alt_rows, alt_outputs, alt_prior, owners, q_values, groups = [], [], [], [], [], []
    for index, (row, output) in enumerate(zip(flat, outputs)):
        targets = row.get("search_targets")
        if not targets:
            continue
        groups.append((len(alt_rows), len(targets), [v for _c, v in targets]))
        for cand, _v in targets:
            alt_rows.append(dict(row, action={k: cand[k] for k in ("type", "packet", "row", "col") if k in cand}))
            alt_outputs.append(output)
            if prior_outputs is not None:
                alt_prior.append(prior_outputs[index])
            owners.append(index)
    if prior_outputs is not None and alt_rows:
        # q(c) ~ pi_old(c) * exp((value_c - max value) / tau): the search re-weights what pi already
        # considers plausible. A plain softmax over values gave ~6% to a sunflower in column 8
        # (log pi_old = -186), which the short look-ahead could not tell was absurd, and pi collapsed.
        with torch.inference_mode():
            prior_logp, _ = replay_log_probs(args.prior_model, alt_prior, alt_rows)
        prior_logp = prior_logp.float().cpu()
    for start, count, values in groups:
        scores = torch.tensor(values, dtype=torch.float32) / args.tau
        if prior_outputs is not None:
            scores = scores + prior_logp[start:start + count]
        q_values.extend(torch.softmax(scores, dim=0).tolist())
    own = torch.tensor([0.0 if row.get("search_targets") else 1.0 for row in flat], device=logp.device)
    total = -(logp.float() * own).sum()
    count = own.sum()
    if alt_rows:
        alt_logp, _ = replay_log_probs(model, alt_outputs, alt_rows)
        q = torch.tensor(q_values, device=alt_logp.device)
        total = total - args.soft_weight * (alt_logp.float() * q).sum()
        count = count + args.soft_weight * len(set(owners))
    return total / count


def same_action(pred: dict, target: dict) -> bool:
    if pred.get("type") != target.get("type"):
        return False
    keys = {"plant": ("packet", "row", "col"), "shovel": ("row", "col"), "wait": ("ticks", "until")}[target["type"]]
    return all(pred.get(k) == target.get(k) for k in keys)


def run(model, games, picks, args, device, optimizer=None, rng=None) -> dict:
    train = optimizer is not None
    model.train(train)
    order = list(picks)
    if rng is not None:
        rng.shuffle(order)
    totals = {"loss": 0.0, "rows": 0, "changed_rows": 0, "changed_match": 0, "own_plants": 0, "own_match": 0}
    for first in range(0, len(order), args.batch_windows):
        batch = order[first:first + args.batch_windows]
        sequences = [games[gi]["rows"][k * args.window:(k + 1) * args.window] for gi, k in batch]
        hiddens = [None if games[gi]["hidden"][k] is None else games[gi]["hidden"][k].to(device)
                   for gi, k in batch]
        with torch.set_grad_enabled(train):
            outputs, _ = model.forward_sequences(sequences, hiddens)
            flat = [row for sequence in sequences for row in sequence]
            logp, _ = replay_log_probs(model, outputs, flat)
            if args.target in ("soft", "prior"):
                prior = None
                if args.target == "prior":
                    with torch.inference_mode():
                        prior, _ = args.prior_model.forward_sequences(sequences, hiddens)
                loss = _soft_loss(model, outputs, flat, logp, args, prior)
            else:
                weights = torch.tensor([args.override_weight if r.get("search_override") else 1.0 for r in flat],
                                       device=logp.device)
                loss = -(logp.float() * weights).sum() / weights.sum()
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        else:
            with torch.inference_mode():
                for row, output in zip(flat, outputs):
                    if row["action"]["type"] != "plant":
                        continue
                    pred, _, _ = select_action(model, output, row["legal"], deterministic=True)
                    if args.target == "soft" and row.get("search_targets"):
                        best = max(row["search_targets"], key=lambda cv: cv[1])[0]
                        totals["changed_rows"] += 1
                        totals["changed_match"] += int(same_action(pred, best))
                        continue
                    hit = same_action(pred, row["action"])
                    key = "changed" if row.get("search_override") else "own"
                    totals[f"{key}_rows" if key == "changed" else "own_plants"] += 1
                    totals[f"{key}_match"] += int(hit)
        totals["loss"] += float(loss.detach()) * len(flat)
        totals["rows"] += len(flat)
    totals["loss"] /= max(1, totals["rows"])
    if not train:
        totals["changed_accuracy"] = totals["changed_match"] / max(1, totals["changed_rows"])
        totals["own_plant_accuracy"] = totals["own_match"] / max(1, totals["own_plants"])
    return totals


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--batch-windows", type=int, default=64)
    p.add_argument("--regular-ratio", type=float, default=3.0)
    p.add_argument("--override-weight", type=float, default=0.3)
    p.add_argument("--target", choices=("hard", "soft", "prior"), default="hard",
                   help="hard: imitate the executed action; soft: match softmax(search values / tau) on searched steps")
    p.add_argument("--tau", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=1, help="window sampling / shuffling seed (vary per round)")
    p.add_argument("--trainable", default="",
                   help="comma-separated parameter-name prefixes to train (others frozen), e.g. plant_cell_query,cell_key")
    p.add_argument("--soft-weight", type=float, default=1.0)
    p.add_argument("--max-windows", type=int, default=0, help="sample at most this many signal windows per run (0 = all)")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--cache-workers", type=int, default=14)
    p.add_argument("--device", default="cpu",
                   help="cpu by default: the pass is Python-bound, and MPS leaked memory here (50 GB by epoch 2)")
    p.add_argument("--threads", type=int, default=14)
    p.add_argument("--hidden-cache", type=Path,
                   help="reuse this GRU-state cache even if --policy differs (continue from an epoch checkpoint)")
    args = p.parse_args()
    started = time.time()

    games = load_games(args.data)
    seeds = sorted({g["seed"] for g in games})
    random.Random(0).shuffle(seeds)
    val_seeds = set(seeds[: max(1, int(len(seeds) * args.val_fraction))])
    changed = sum(r.get("search_override", False) for g in games for r in g["rows"])
    searched = sum(bool(r.get("search_targets")) for g in games for r in g["rows"])
    print(f"{len(games)} games, {sum(len(g['rows']) for g in games)} rows, {changed} search changes, "
          f"{searched} searched steps, "
          f"{len(val_seeds)} validation seeds", flush=True)

    checkpoint = torch.load(args.policy, map_location="cpu", weights_only=False)
    model = GameplayModelV1(checkpoint["config"])
    missing, _unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    if missing:
        raise RuntimeError(f"checkpoint is missing {len(missing)} parameters")
    torch.set_num_threads(args.threads)
    model.to(args.device)
    cache_path = args.hidden_cache or args.data.with_name(args.data.name + f".hidden_w{args.window}.pt")
    cache = torch.load(cache_path, weights_only=False) if cache_path.exists() else None
    if cache and (args.hidden_cache or cache.get("policy") == str(args.policy)) \
            and set(cache["hidden"]) >= {g["path"] for g in games}:
        for g in games:
            g["hidden"] = cache["hidden"][g["path"]]
    else:
        cache_hidden_parallel(args.policy, games, args.window, args.cache_workers)
        torch.save({"policy": str(args.policy), "hidden": {g["path"]: g["hidden"] for g in games}}, cache_path)
    print(f"cached GRU states ({time.time() - started:.0f}s)", flush=True)

    rng = random.Random(args.seed)
    train_ids = [i for i, g in enumerate(games) if g["seed"] not in val_seeds]
    val_ids = set(range(len(games))) - set(train_ids)
    all_windows = windows(games, args.window, rng, args.regular_ratio, soft=args.target == "soft")
    if args.max_windows and len(all_windows) > args.max_windows:
        all_windows = rng.sample(all_windows, args.max_windows)
    train_picks = [w for w in all_windows if w[0] in set(train_ids)]
    val_picks = [w for w in all_windows if w[0] in val_ids]
    if args.trainable:
        # Train only these parameter groups. Freezing the trunk and the GRU keeps the cached
        # hidden states exact: when the GRU itself changes, window training (which starts every
        # window from the *original* pi's state) teaches the model for states it never reaches in
        # play, and soft-target round 1 collapsed to 17/256 that way.
        prefixes = tuple(args.trainable.split(","))
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith(prefixes)
        print("trainable:", sum(p.numel() for p in model.parameters() if p.requires_grad), "parameters", flush=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    if args.target == "prior":
        args.prior_model = GameplayModelV1(checkpoint["config"])
        args.prior_model.load_state_dict(checkpoint["state_dict"], strict=False)
        args.prior_model.to(args.device).eval()
    cpu_state = lambda: {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    history = [{"epoch": 0, "val": run(model, games, val_picks, args, args.device)}]
    print(json.dumps(history[-1]), flush=True)
    best_state, best_loss = cpu_state(), history[0]["val"]["loss"]
    for epoch in range(1, args.epochs + 1):
        train = run(model, games, train_picks, args, args.device, optimizer, rng)
        val = run(model, games, val_picks, args, args.device)
        history.append({"epoch": epoch, "train_loss": train["loss"], "val": val,
                        "seconds": round(time.time() - started)})
        print(json.dumps(history[-1]), flush=True)
        if val["loss"] < best_loss:
            best_loss, best_state = val["loss"], cpu_state()
        epoch_out = dict(checkpoint)
        epoch_out["state_dict"] = cpu_state()
        epoch_out["distillation"] = {"epoch": epoch, "val": val, "from": str(args.policy)}
        torch.save(epoch_out, args.out.with_name(f"{args.out.stem}_epoch{epoch}.pt"))

    out = dict(checkpoint)
    out["state_dict"] = best_state
    out["distillation"] = {"from": str(args.policy), "data": str(args.data),
                           "args": {k: str(v) for k, v in vars(args).items() if k != "prior_model"},
                           "history": history, "val_seeds": sorted(val_seeds), "best_val_loss": best_loss,
                           "train_windows": len(train_picks), "val_windows": len(val_picks)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    args.out.with_suffix(".json").write_text(json.dumps(out["distillation"], indent=1))
    print(f"wrote {args.out} ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
