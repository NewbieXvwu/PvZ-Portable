"""Where the 184-second PPO update actually goes.

The recorded T5 update benchmark (``artifacts/t5/perf/ppo_update_2000_flex_saved.json``)
says one 2,000-episode update costs ~191 s and performs 1,120 optimizer steps, i.e.
~170 ms per step.  A step feeds 16 chunks x 16 steps = 256 transitions through a
4-layer, 192-wide encoder; the arithmetic for that is on the order of a few
milliseconds even at a fraction of the RTX 5080's fp32 rate.  So the step is not
compute-bound, and the question is what else it is doing.

This probe answers that without guessing:

* it rebuilds a rollout batch whose *shapes* match the recorded one (2000 episodes,
  127,877 transitions, ~80 tokens per transition), so the number of optimizer steps
  and the tensor sizes match the benchmark;
* it times the update end to end and reports **per-optimizer-step** cost, which is
  the quantity that transfers to the real run;
* it runs ``torch.profiler`` over the update and reports the sum of CPU time spent
  inside ATen operators against the wall clock.  The gap between the two is time
  spent in Python and numpy -- list comprehensions, dict construction, numpy
  staging -- which no ATen entry can account for.  That gap is the headline number.

Run it with the mise interpreter (the managed one has no numpy/torch):

    /Users/newbiexvwu/.local/share/mise/installs/python/3.14/bin/python3 \
        scripts/ppo_update_profile.py --episodes 128
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from pvz_agent_model import (  # noqa: E402
    FEATURE_COUNT,
    GameplayModelV1,
    configure_torch_threads,
    resolve_device,
)
from train_pvz_ppo import add_advantages, train_update  # noqa: E402

# Token mix measured over 197 real decisions (scripts/token_input_audit.py):
# mean 80.3 tokens, of which 54 are cells, 6 lanes, 6 seed packets, ~5 defenses,
# ~3 zombies, ~2 roster entries, and one each of global/profile/grid_item/plant.
# Ordered exactly as ``observation_tokens`` appends them, so ``cell_index`` and
# ``packet_index`` point at the same kinds of token the real encoder sees.
BASE_KINDS = (
    (0, 1),    # global
    (1, 1),    # profile
    (2, 54),   # cell
    (10, 6),   # lane
    (3, 1),    # plant
    (4, 3),    # zombie
    (5, 0),    # projectile
    (6, 5),    # defense
    (7, 1),    # grid_item
    (8, 6),    # seed_packet
    (9, 2),    # zombie_roster
)

# Recorded benchmark: 127,877 transitions over 2,000 episodes.
MEAN_TRANSITIONS_PER_EPISODE = 127_877 / 2000
# Recorded benchmark: 1,120 optimizer steps at sequence_length=16, chunks=16.
RECORDED_OPTIMIZER_STEPS = 1120
RECORDED_UPDATE_SECONDS = 191.0


def _layout(kinds: list[int]) -> tuple[int, int]:
    """First cell token index and first seed-packet token index."""
    cell_base = kinds.index(2)
    packet_base = kinds.index(8)
    return cell_base, packet_base


def build_episode(rng: random.Random, index: int, mean_transitions: float,
                  numpy_rng: np.random.Generator) -> dict[str, Any]:
    """One synthetic episode with real packed-token structure.

    Only the shapes matter for cost, but the token ids are kept in range so the
    embeddings behave like the real ones (a padded/out-of-range id would change
    nothing about the arithmetic, but it would make the run less representative).
    """
    length = max(2, int(rng.gauss(mean_transitions, mean_transitions * 0.35)))
    transitions = []
    for position in range(length):
        kinds: list[int] = []
        for kind, count in BASE_KINDS:
            # Zombies, projectiles, defenses and grid items are the only kinds whose
            # count moves decision to decision; the rest are structurally fixed.
            if kind in (4, 5, 6, 7):
                count = max(0, count + rng.randint(-1, 1) if rng.random() < 0.5 else count)
            kinds.extend([kind] * count)
        count = len(kinds)
        ids = np.zeros((count, 5), dtype=np.int8)
        ids[:, 0] = np.asarray(kinds, dtype=np.int8)
        ids[:, 1] = numpy_rng.integers(1, 30, size=count).astype(np.int8)
        ids[:, 2] = numpy_rng.integers(1, 4, size=count).astype(np.int8)
        ids[:, 3] = -1
        ids[:, 4] = -1
        for position_in_row, kind in enumerate(kinds):
            if kind in (2, 3, 4, 5, 6, 7, 10):
                ids[position_in_row, 3] = rng.randint(0, 5)
                ids[position_in_row, 4] = rng.randint(0, 8)
        features = numpy_rng.random((count, FEATURE_COUNT)).astype(np.float16)
        cell_base, packet_base = _layout(kinds)
        packets = rng.randint(1, 6)
        packed = {
            "ids": ids,
            "features": features,
            "cell_index": np.arange(cell_base, cell_base + 54, dtype=np.uint16),
            "packet_ids": np.arange(packets, dtype=np.uint8),
            "packet_index": np.arange(packet_base, packet_base + packets, dtype=np.uint16),
        }
        legal = {
            "packets": tuple(range(packets)),
            "plant_mask": tuple(0x3FF for _ in range(packets)),
            "shovel_mask": 0x3FF,
            "wait": True,
        }
        action_kind = rng.random()
        if action_kind < 0.6:
            action: dict[str, Any] = {"type": "plant", "packet": rng.randrange(packets),
                                      "row": rng.randint(0, 5), "col": rng.randint(0, 8)}
        elif action_kind < 0.7:
            action = {"type": "shovel", "row": rng.randint(0, 5), "col": rng.randint(0, 8)}
        else:
            action = {"type": "wait", "ticks": rng.choice((60, 150, 300))}
        transitions.append({
            "decision_index": position,
            "tokens": packed,
            "wave": rng.randint(1, 5),
            "legal": legal,
            "previous_action": transitions[-1]["action"] if transitions else None,
            "elapsed_since_previous_observation": rng.choice((60, 150, 300)),
            "events": {"zombies_killed": rng.randint(0, 4), "sun_produced": rng.randint(0, 25)},
            "critic_extra": [rng.random() for _ in range(16)],
            "action": action,
            "log_prob": -rng.random() * 2.0,
            "value": rng.random() * 2 - 1,
            "potential": rng.random(),
            # ``add_advantages`` discounts by wall-clock ticks, so every transition
            # needs one; the recorded rollouts only ever contain 60/150/300.
            "action_duration_ticks": rng.choice((60, 150, 300)),
        })
    for transition in transitions[:-1]:
        transition["reward"] = 0.0
    transitions[-1]["reward"] = -1.0
    return {"seed": index, "task_id": f"synthetic_{index}", "won": False, "result": -1,
            "transitions": transitions}


def build_batch(episodes: int, seed: int, mean_transitions: float) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    numpy_rng = np.random.default_rng(seed)
    batch = [build_episode(rng, index, mean_transitions, numpy_rng) for index in range(episodes)]
    add_advantages(batch, 0.95)
    return batch


def _object_bytes(episodes: list[dict[str, Any]]) -> int:
    """Rough Python-object size of the rollout batch, as the benchmark reports it."""
    total = 0
    for episode in episodes:
        total += sys.getsizeof(episode)
        for transition in episode["transitions"]:
            total += sys.getsizeof(transition)
            packed = transition["tokens"]
            total += packed["ids"].nbytes + packed["features"].nbytes
            total += packed["cell_index"].nbytes + packed["packet_index"].nbytes
    return total


class _PhaseTimer:
    """Accumulates wall time for named phases of the update."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def add(self, name: str, seconds: float) -> None:
        self.totals[name] = self.totals.get(name, 0.0) + seconds
        self.counts[name] = self.counts.get(name, 0) + 1


def _patch_phases(timer: _PhaseTimer) -> None:
    """Time ``forward_sequences`` and ``replay_log_probs`` at their real call sites.

    Wrapping the two functions rather than reimplementing the update body means the
    numbers describe the code that actually runs; nothing here can drift away from
    ``train_pvz_ppo.train_update``.
    """
    original_forward = GameplayModelV1.forward_sequences

    def timed_forward(self, sequences, hiddens):  # type: ignore[no-untyped-def]
        started = time.perf_counter()
        result = original_forward(self, sequences, hiddens)
        timer.add("forward_sequences", time.perf_counter() - started)
        return result

    GameplayModelV1.forward_sequences = timed_forward  # type: ignore[assignment]

    module = sys.modules["train_pvz_ppo"]
    original_replay = module.replay_log_probs

    def timed_replay(model, outputs, transitions):  # type: ignore[no-untyped-def]
        started = time.perf_counter()
        result = original_replay(model, outputs, transitions)
        timer.add("replay_log_probs", time.perf_counter() - started)
        return result

    module.replay_log_probs = timed_replay


def profile_update(episodes: list[dict[str, Any]], args: argparse.Namespace,
                   device: torch.device) -> dict[str, Any]:
    """Run the real ``train_update`` twice: once plainly, once under the profiler.

    The plain run gives the wall clock and the phase split; the profiled run gives
    ATen CPU time.  The difference between the profiled wall clock and its ATen
    total is time spent outside every ATen operator -- Python and numpy staging.
    """
    torch.manual_seed(args.seed)
    model = GameplayModelV1().to(device)
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    transitions = sum(len(episode["transitions"]) for episode in episodes)
    chunks = sum(-(-len(episode["transitions"]) // args.sequence_length) for episode in episodes)
    timer = _PhaseTimer()
    _patch_phases(timer)

    # Warm up once so lazy initialisation (fused relation bias compile, allocator
    # growth) does not land in the measured run.
    warm_episodes = episodes[: max(1, min(8, len(episodes)))]
    warm_model = copy.deepcopy(model)
    warm_optimizer = torch.optim.AdamW(warm_model.parameters(), lr=1e-4)
    random.seed(args.seed)
    train_update(warm_model, warm_episodes, warm_optimizer, device, 1,
                 args.sequence_length, 0.2, 0.5, 0.01, minibatch_chunks=args.minibatch_chunks,
                 attention_backend=args.attention_backend)
    del warm_model, warm_optimizer

    random.seed(args.seed)
    timer.totals.clear()
    timer.counts.clear()
    started = time.perf_counter()
    losses = train_update(model, episodes, optimizer, device, args.ppo_epochs,
                          args.sequence_length, 0.2, 0.5, 0.01,
                          minibatch_chunks=args.minibatch_chunks,
                          attention_backend=args.attention_backend)
    wall = time.perf_counter() - started

    optimizer_steps = timer.counts.get("forward_sequences", 0)
    result: dict[str, Any] = {
        "device": str(device),
        "episodes": len(episodes),
        "transitions": transitions,
        "chunks": chunks,
        "optimizer_steps": optimizer_steps,
        "wall_seconds": wall,
        "ms_per_optimizer_step": 1000.0 * wall / max(1, optimizer_steps),
        "transitions_per_second": transitions * args.ppo_epochs / wall,
        "losses": losses,
        "phases": {
            name: {
                "total_seconds": timer.totals[name],
                "calls": timer.counts[name],
                "ms_per_call": 1000.0 * timer.totals[name] / timer.counts[name],
                "share_of_wall": timer.totals[name] / wall,
            }
            for name in timer.totals
        },
    }
    result["phases"]["everything_else"] = {
        "total_seconds": wall - sum(timer.totals.values()),
        "calls": optimizer_steps,
        "ms_per_call": 1000.0 * (wall - sum(timer.totals.values())) / max(1, optimizer_steps),
        "share_of_wall": (wall - sum(timer.totals.values())) / wall,
    }
    return result


def aten_cpu_time(episodes: list[dict[str, Any]], args: argparse.Namespace,
                  device: torch.device) -> dict[str, Any]:
    """ATen CPU time for a bounded number of optimizer steps, plus the top operators.

    Profiling the whole update is not affordable: the profiler keeps one record per
    ATen call and a large update makes millions of them (measured: 8.4 GiB resident at
    128 episodes and still climbing).  Only the first ``--profile-steps`` optimizer
    steps are recorded.  That is enough because the per-step cost is flat: the
    recorded benchmark runs 1,120 steps at a constant ~170 ms each.
    """
    torch.manual_seed(args.seed)
    model = GameplayModelV1().to(device)
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    random.seed(args.seed)
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    remaining = {"steps": args.profile_steps}
    original_forward = GameplayModelV1.forward_sequences
    profiler_holder: dict[str, Any] = {}

    class _Enough(Exception):
        pass

    def bounded_forward(self, sequences, hiddens):  # type: ignore[no-untyped-def]
        if remaining["steps"] <= 0:
            raise _Enough
        remaining["steps"] -= 1
        if "profiler" not in profiler_holder:
            profiler_holder["profiler"] = profile(
                activities=activities, record_shapes=False, with_stack=False)
            profiler_holder["profiler"].__enter__()
        return original_forward(self, sequences, hiddens)

    GameplayModelV1.forward_sequences = bounded_forward  # type: ignore[assignment]
    started = time.perf_counter()
    try:
        train_update(model, episodes, optimizer, device, args.ppo_epochs,
                     args.sequence_length, 0.2, 0.5, 0.01,
                     minibatch_chunks=args.minibatch_chunks,
                     attention_backend=args.attention_backend)
    except _Enough:
        pass
    finally:
        GameplayModelV1.forward_sequences = original_forward  # type: ignore[assignment]
    profiled_steps = args.profile_steps - remaining["steps"]
    wall = time.perf_counter() - started
    profiler = profiler_holder.get("profiler")
    if profiler is not None:
        profiler.__exit__(None, None, None)

    rows = []
    aten_total = 0.0
    if profiler is not None:
        for event in profiler.key_averages():
            self_cpu = float(event.self_cpu_time_total) / 1000.0
            aten_total += self_cpu
            rows.append({
                "name": event.key,
                "self_cpu_seconds": self_cpu,
                "calls": event.count,
                "us_per_call": (float(event.self_cpu_time_total) / event.count) if event.count else 0.0,
            })
    rows.sort(key=lambda row: -row["self_cpu_seconds"])
    return {
        "profiled_steps": profiled_steps,
        "profiled_wall_seconds": wall,
        "ms_per_profiled_step": 1000.0 * wall / max(1, profiled_steps),
        "aten_self_cpu_seconds": aten_total,
        "aten_share_of_wall": aten_total / wall if wall else None,
        "outside_aten_seconds": wall - aten_total,
        "outside_aten_share_of_wall": (wall - aten_total) / wall if wall else None,
        "top_operators": rows[: args.top],
    }


def micro_benchmark(episodes: list[dict[str, Any]], args: argparse.Namespace,
                    device: torch.device) -> dict[str, Any]:
    """Time the individual expressions ``train_update`` runs once per minibatch.

    These are verbatim copies of the lines in ``train_pvz_ppo.train_update`` and
    ``pvz_agent_model.replay_log_probs``, evaluated on real transitions at the real
    minibatch size.  A copy can drift from its original, so the point of doing it here
    is that the phase timers above already measure the *whole* function at the real
    call site; this section only apportions that total, and every number is reported
    next to the total it has to fit inside.
    """
    torch.manual_seed(args.seed)
    model = GameplayModelV1().to(device)
    model.eval()
    flat = [transition for episode in episodes for transition in episode["transitions"]]
    size = args.minibatch_chunks * args.sequence_length
    flat = flat[:size]
    sequences = [flat]
    # Grad is left enabled: the backward measurement below needs a real graph, and
    # building one is what the update does anyway.
    outputs, _ = model.forward_sequences(sequences, [None])
    results: dict[str, Any] = {"minibatch_transitions": len(flat)}

    def time_it(name: str, repeats: int, function: Any) -> Any:
        best = float("inf")
        value = None
        for _ in range(repeats):
            started = time.perf_counter()
            value = function()
            elapsed = time.perf_counter() - started
            best = min(best, elapsed)
        results[name] = {"ms": 1000.0 * best, "repeats": repeats}
        return value

    time_it("belief_cat", 5, lambda: torch.cat([output["belief"] for output in outputs], dim=0))
    time_it("type_logits_stack", 5, lambda: torch.stack([o["type_logits"] for o in outputs]))
    time_it("wait_logits_stack", 5, lambda: torch.stack([o["wait_logits"] for o in outputs]))
    time_it("cell_keys_stack", 5, lambda: torch.stack([o["cell_keys"] for o in outputs]))
    time_it("critic_extra_tensor", 5, lambda: torch.tensor(
        [transition["critic_extra"] for transition in flat], dtype=torch.float32, device=device))
    time_it("old_log_prob_tensor", 5, lambda: torch.tensor(
        [transition["log_prob"] for transition in flat], dtype=torch.float32, device=device))
    time_it("returns_tensor", 5, lambda: torch.tensor(
        [transition["return"] for transition in flat], dtype=torch.float32, device=device))

    # ``normalized_advantage`` is stored as a 0-dim tensor, so stacking it launches one
    # kernel per element -- a different cost class from the float lists above.
    for transition, advantage in zip(flat, torch.zeros(len(flat))):
        transition["normalized_advantage"] = advantage
    time_it("advantage_stack", 5, lambda: torch.stack(
        [transition["normalized_advantage"] for transition in flat]))

    def rebuild_hidden() -> None:
        for index in range(len(flat)):
            outputs[index]["hidden"] = None

    time_it("output_dict_touch", 5, rebuild_hidden)

    # What the whole minibatch body costs outside forward_sequences, so the micro
    # numbers above can be judged against it.
    belief = torch.cat([output["belief"] for output in outputs], dim=0)
    extras = torch.tensor([transition["critic_extra"] for transition in flat],
                          dtype=torch.float32, device=device)
    values = model.privileged_value_batch(belief, extras).squeeze(-1)
    returns = torch.tensor([transition["return"] for transition in flat],
                           dtype=torch.float32, device=device)
    loss = torch.nn.functional.mse_loss(values, returns)
    time_it("backward_only", 3, lambda: torch.autograd.backward(loss, retain_graph=True))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=128,
                        help="synthetic episodes; the recorded benchmark used 2,000")
    parser.add_argument("--mean-transitions", type=float, default=MEAN_TRANSITIONS_PER_EPISODE)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--minibatch-chunks", type=int, default=16)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--attention-backend", choices=("auto", "dense", "flex"), default="auto")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--profile-steps", type=int, default=6,
                        help="optimizer steps to record with torch.profiler; the "
                             "profiler retains one record per ATen call, so profiling "
                             "a whole update exhausts memory")
    parser.add_argument("--skip-profiler", action="store_true")
    parser.add_argument("--micro", action="store_true",
                        help="also time the individual per-minibatch expressions")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    configure_torch_threads(args.threads)
    device = resolve_device(args.device)

    started = time.perf_counter()
    episodes = build_batch(args.episodes, args.seed, args.mean_transitions)
    build_seconds = time.perf_counter() - started
    transitions = sum(len(episode["transitions"]) for episode in episodes)

    print("=" * 78)
    print("PPO update profile")
    print("=" * 78)
    print(f"device            {device}   threads={args.threads}")
    print(f"episodes          {len(episodes)}   (recorded benchmark: 2000)")
    print(f"transitions       {transitions}   (recorded benchmark: 127877)")
    print(f"mean tokens/step  "
          f"{np.mean([len(t['tokens']['ids']) for e in episodes for t in e['transitions']]):.1f}")
    per_chunk = [len(t["tokens"]["ids"]) for e in episodes for t in e["transitions"]]
    print(f"token padding     real {np.mean(per_chunk):.1f}/transition, but a minibatch "
          f"pads every row to the batch max; observed max {max(per_chunk)}"
          f" -> {100 * (1 - np.mean(per_chunk) / max(per_chunk)):.0f}% of encoder slots are padding")
    print(f"sequence_length   {args.sequence_length}   minibatch_chunks {args.minibatch_chunks}"
          f"   ppo_epochs {args.ppo_epochs}")
    print(f"python objects    {_object_bytes(episodes) / 2**20:.1f} MiB for the rollout batch")
    print(f"build             {build_seconds:.1f}s")
    print()

    plain = profile_update(episodes, args, device)
    print("-" * 78)
    print("wall clock, through the real train_update")
    print("-" * 78)
    print(f"optimizer steps   {plain['optimizer_steps']}   "
          f"(recorded benchmark: {RECORDED_OPTIMIZER_STEPS})")
    print(f"wall              {plain['wall_seconds']:.2f}s")
    print(f"per step          {plain['ms_per_optimizer_step']:.1f} ms")
    print(f"transitions/s     {plain['transitions_per_second']:.1f}   "
          f"(recorded benchmark: 693.5 on an RTX 5080)")
    print()
    print(f"{'phase':<24}{'calls':>8}{'total s':>10}{'ms/call':>10}{'share':>9}")
    for name, phase in sorted(plain["phases"].items(), key=lambda item: -item[1]["total_seconds"]):
        print(f"{name:<24}{phase['calls']:>8}{phase['total_seconds']:>10.2f}"
              f"{phase['ms_per_call']:>10.2f}{100 * phase['share_of_wall']:>8.1f}%")
    print()

    if args.micro:
        micro = micro_benchmark(episodes, args, device)
        print("-" * 78)
        print("micro-benchmark: the per-minibatch expressions, on real transitions")
        print("-" * 78)
        print(f"minibatch         {micro.pop('minibatch_transitions')} transitions")
        print(f"{'expression':<24}{'ms':>10}{'repeats':>9}")
        for name, row in sorted(micro.items(), key=lambda item: -item[1]["ms"]):
            print(f"{name:<24}{row['ms']:>10.2f}{row['repeats']:>9}")
        print()
        print("compare against the per-step totals printed above: the sum of these is what")
        print("the update pays every minibatch of every epoch, and none of it is device work.")
        print()

    if not args.skip_profiler:
        profiled = aten_cpu_time(episodes, args, device)
        print("-" * 78)
        print("ATen CPU time vs wall clock (torch.profiler)")
        print("-" * 78)
        print(f"recorded steps    {profiled['profiled_steps']}"
              f"   wall {profiled['profiled_wall_seconds']:.2f}s"
              f"   {profiled['ms_per_profiled_step']:.1f} ms/step")
        print(f"inside ATen ops   {profiled['aten_self_cpu_seconds']:.2f}s"
              f"   {100 * profiled['aten_share_of_wall']:.1f}%")
        print(f"OUTSIDE ATen ops  {profiled['outside_aten_seconds']:.2f}s"
              f"   {100 * profiled['outside_aten_share_of_wall']:.1f}%"
              "   <- Python + numpy, no operator accounts for it")
        print()
        print(f"{'operator':<44}{'calls':>10}{'self s':>10}{'us/call':>10}")
        for row in profiled["top_operators"]:
            name = row["name"] if len(row["name"]) <= 43 else row["name"][:40] + "..."
            print(f"{name:<44}{row['calls']:>10}{row['self_cpu_seconds']:>10.2f}"
                  f"{row['us_per_call']:>10.2f}")
        print()
        print("projection onto the recorded 1,120-step update")
        print("-" * 78)
        scale = RECORDED_OPTIMIZER_STEPS / max(1, plain["optimizer_steps"])
        per_step = plain["ms_per_optimizer_step"]
        print(f"  this profile runs {plain['optimizer_steps']} steps at {per_step:.1f} ms")
        print(f"  x{scale:.2f} -> {per_step * RECORDED_OPTIMIZER_STEPS / 1000.0:.1f}s "
              f"vs the recorded {RECORDED_UPDATE_SECONDS:.0f}s")
        for name, phase in sorted(plain["phases"].items(), key=lambda item: -item[1]["total_seconds"]):
            projected = phase["ms_per_call"] * RECORDED_OPTIMIZER_STEPS / 1000.0
            print(f"    {name:<22}{projected:>8.1f}s")
        print()

    if args.json:
        args.json.write_text(json.dumps({
            "plain": plain,
            "episodes": len(episodes),
            "transitions": transitions,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
