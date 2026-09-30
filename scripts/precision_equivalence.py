"""Does a bf16 PPO update still take the same steps as an fp32 one?

``artifacts/t5/perf/ppo_update_lr_1e4.json`` records one 16-episode update at
14.371 s in fp32 against 3.73 s in bf16 on an RTX 5080 -- 3.85x, the largest single
number anywhere in the repository.  It was never acted on, because bf16 is not
bit-equivalent and nobody had argued what "equivalent enough" means here.  This
script makes that argument measurable instead of rhetorical.

Two things have to hold for the speedup to be usable:

1. **The update must go where the fp32 update goes.**  A PPO update is only
   meaningful relative to the policy that collected the rollout, so the quantity
   that has to survive is the ratio ``exp(new_logp - old_logp)``.  The script
   updates the same batch twice -- once in fp32, once under bf16 autocast -- from
   the same seed, then evaluates both resulting policies on **held-out** shards
   and reports the distribution of the log-probability difference per decision.
   ``clip_epsilon`` (0.2 by default) is the scale that matters: bf16 noise has to
   sit far below it, or the clip starts firing on rounding rather than on learning.

2. **The losses must not move.**  The script reports both loss vectors and the
   per-parameter relative drift of the updated state dicts, against the bf16 unit
   roundoff (2**-8 = 3.9e-3) as the natural yardstick.

It also measures the bf16 **forward** noise with the weights held fixed, which is
the noise floor the update comparison has to be read against: if a single forward
already moves the log-probs by X, no update can be expected to do better.

``--attention-backend`` has to be given explicitly and matched to the question.  The
recorded 3.85x came from ``ppo_update_benchmark.py``, whose default backend is
``dense``; production runs ``auto``, which is FlexAttention on CUDA.  The two
backends do not just differ in speed -- they disagree about the *sign* of bf16's
effect, so a number measured on one says nothing about the other.

Run it on a CUDA machine with real rollout shards::

    python scripts/precision_equivalence.py \
        --data-dir artifacts/t5/runs/run_2/.seed_jobs/update_0001/<digest> \
        --episodes 16 --attention-backend flex \
        --output artifacts/t5/perf/precision_equivalence_flex.json
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
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
    GameplayModelV1,
    configure_torch_threads,
    replay_log_probs,
    resolve_device,
)
from pvz_seed_jobs import read_episode  # noqa: E402
from train_pvz_ppo import add_advantages, train_update  # noqa: E402

AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}
BF16_UNIT_ROUNDOFF = 2.0 ** -8


def load_episodes(data_dir: Path, offset: int, count: int) -> list[dict[str, Any]]:
    paths = sorted(data_dir.glob("seed_*.npz"))[offset:offset + count]
    if len(paths) < count:
        raise ValueError(f"need {count} shards at offset {offset} in {data_dir}, "
                         f"found {len(paths)}")
    return [read_episode(path) for path in paths]


def precision_context(device: torch.device, precision: str) -> Any:
    if precision in AMP_DTYPES and device.type == "cuda":
        return torch.autocast("cuda", dtype=AMP_DTYPES[precision])
    return nullcontext()


def run_update(episodes: list[dict[str, Any]], warmup_episodes: list[dict[str, Any]],
               device: torch.device, precision: str, sequence_length: int,
               minibatch_chunks: int, ppo_epochs: int, learning_rate: float,
               attention_backend: str, seed: int
               ) -> tuple[dict[str, float], dict[str, torch.Tensor], float, int]:
    """One real ``train_update`` at the requested precision, after a warm-up.

    The warm-up is not optional.  ``torch.compile`` is entered lazily by both the
    fused relation-bias assembly and FlexAttention, and a new dtype compiles again
    from scratch.  Without a warm-up the first precision timed absorbs the first
    compile and the second absorbs the second, which is easily enough to invert
    the comparison -- measured: it turned a 2.0x win into a 2.6x loss.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    model = GameplayModelV1().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    warm_model = copy.deepcopy(model)
    warm_optimizer = torch.optim.AdamW(warm_model.parameters(), lr=learning_rate)
    random.seed(seed)
    with precision_context(device, precision):
        train_update(warm_model, warmup_episodes, warm_optimizer, device, 1,
                     sequence_length, 0.2, 0.5, 0.01,
                     minibatch_chunks=minibatch_chunks,
                     attention_backend=attention_backend)
    del warm_model, warm_optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    random.seed(seed)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with precision_context(device, precision):
        losses = train_update(model, episodes, optimizer, device, ppo_epochs,
                              sequence_length, 0.2, 0.5, 0.01,
                              minibatch_chunks=minibatch_chunks,
                              attention_backend=attention_backend)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    state = {key: value.detach().float().cpu() for key, value in model.state_dict().items()}
    return losses, state, elapsed, len(episodes)


def policy_log_probs(model: GameplayModelV1, transitions: list[dict[str, Any]],
                     device: torch.device, precision: str) -> tuple[np.ndarray, np.ndarray]:
    """Log-probability and entropy of the recorded actions, per transition."""
    model.eval()
    with torch.no_grad(), precision_context(device, precision):
        outputs, _ = model.forward_sequences([transitions], [None])
        log_prob, entropy = replay_log_probs(model, outputs, transitions)
    return (log_prob.detach().float().cpu().numpy(),
            entropy.detach().float().cpu().numpy())


def state_drift(reference: dict[str, torch.Tensor],
                subject: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Per-parameter drift of the updated state dicts, in units of bf16 roundoff."""
    worst_name, worst_relative, worst_absolute = "", 0.0, 0.0
    exact = 0
    relative_values: list[float] = []
    for key, reference_value in reference.items():
        subject_value = subject[key]
        difference = (subject_value - reference_value).abs()
        if torch.equal(subject_value, reference_value):
            exact += 1
        absolute = float(difference.max().item())
        relative = float((difference / reference_value.abs().clamp_min(1e-3)).max().item())
        relative_values.append(relative)
        if relative > worst_relative:
            worst_name, worst_relative, worst_absolute = key, relative, absolute
    return {
        "parameters": len(reference),
        "bitwise_identical": exact,
        "worst_parameter": worst_name,
        "worst_relative_difference": worst_relative,
        "worst_absolute_difference": worst_absolute,
        "median_relative_difference": float(np.median(relative_values)),
        "worst_relative_in_bf16_roundoffs": worst_relative / BF16_UNIT_ROUNDOFF,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--warmup-episodes", type=int, default=4,
                        help="episodes in the per-precision warm-up; the timed runs "
                             "must not pay for a torch.compile")
    parser.add_argument("--held-out-offset", type=int, default=500,
                        help="first shard index used for the held-out policy comparison")
    parser.add_argument("--held-out-episodes", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--minibatch-chunks", type=int, default=16)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--attention-backend", choices=("auto", "dense", "flex"), default="auto")
    parser.add_argument("--precision", default="bf16", choices=sorted(AMP_DTYPES))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    configure_torch_threads(1)
    device = resolve_device(args.device)
    if args.precision in AMP_DTYPES and device.type != "cuda":
        parser.error(f"{args.precision} autocast needs CUDA")

    episodes = load_episodes(args.data_dir, 0, args.episodes)
    held_out = load_episodes(args.data_dir, args.held_out_offset, args.held_out_episodes)
    held_out_transitions = [t for episode in held_out for t in episode["transitions"]]
    add_advantages(episodes, 0.95)
    add_advantages(held_out, 0.95)
    transitions = sum(len(episode["transitions"]) for episode in episodes)
    print(f"torch {torch.__version__}  device {device}  attention={args.attention_backend}")
    print(f"update batch   {args.episodes} episodes / {transitions} transitions")
    print(f"held-out       {len(held_out_transitions)} transitions from "
          f"shards {args.held_out_offset}..{args.held_out_offset + args.held_out_episodes}",
          flush=True)

    torch.manual_seed(args.seed)
    reference_model = GameplayModelV1().to(device)
    reference_state = {key: value.detach().float().cpu()
                       for key, value in reference_model.state_dict().items()}
    del reference_model

    runs: dict[str, dict[str, Any]] = {}
    warmup_episodes = episodes[:max(1, min(args.warmup_episodes, len(episodes)))]
    for precision in ("fp32", args.precision):
        if precision in runs:
            continue
        # ``run_update`` reseeds both generators, so both runs see the same model
        # initialisation and the same minibatch order.
        losses, state, elapsed, count = run_update(
            episodes, warmup_episodes, device, precision, args.sequence_length,
            args.minibatch_chunks, args.ppo_epochs, args.learning_rate,
            args.attention_backend, args.seed)
        runs[precision] = {"losses": losses, "state": state, "seconds": elapsed}
        print(f"{precision:>4}: {elapsed:8.3f} s   "
              f"policy_loss {losses['policy_loss']:.6f}   "
              f"value_loss {losses['value_loss']:.6f}   "
              f"entropy {losses['entropy']:.6f}   "
              f"grad_norm {losses['gradient_norm']:.6f}", flush=True)

    reference, subject = runs["fp32"], runs[args.precision]
    speedup = reference["seconds"] / subject["seconds"]

    # The held-out policy comparison runs both updated models in fp32, so the only
    # difference between them is the weights the update produced.
    torch.manual_seed(args.seed)
    evaluation_model = GameplayModelV1().to(device)
    policies = {}
    for precision, run in (("fp32", reference), (args.precision, subject)):
        evaluation_model.load_state_dict(run["state"])
        log_prob, entropy = policy_log_probs(
            evaluation_model, held_out_transitions, device, "fp32")
        policies[precision] = {"log_prob": log_prob, "entropy": entropy}

    delta = policies[args.precision]["log_prob"] - policies["fp32"]["log_prob"]
    entropy_delta = policies[args.precision]["entropy"] - policies["fp32"]["entropy"]

    # Noise floor: the same fp32 weights, forwarded under autocast.  Any update
    # comparison has to be read against this, not against zero.
    evaluation_model.load_state_dict(reference["state"])
    autocast_log_prob, _ = policy_log_probs(
        evaluation_model, held_out_transitions, device, args.precision)
    floor_delta = autocast_log_prob - policies["fp32"]["log_prob"]

    ratio = np.exp(delta)
    report = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch_version": torch.__version__,
        "data_dir": str(args.data_dir.resolve()),
        "attention_backend": args.attention_backend,
        "episodes": args.episodes,
        "transitions": transitions,
        "optimizer": {"sequence_length": args.sequence_length,
                      "minibatch_chunks": args.minibatch_chunks,
                      "ppo_epochs": args.ppo_epochs,
                      "learning_rate": args.learning_rate},
        "runs": {name: {"seconds": run["seconds"], "losses": run["losses"]}
                 for name, run in runs.items()},
        "speedup": speedup,
        "state_drift_fp32_vs_subject": state_drift(reference["state"], subject["state"]),
        "state_drift_initial_vs_fp32": state_drift(reference_state, reference["state"]),
        "held_out_policy": {
            "transitions": len(held_out_transitions),
            "max_abs_logprob_delta": float(np.abs(delta).max()),
            "mean_abs_logprob_delta": float(np.abs(delta).mean()),
            "p99_abs_logprob_delta": float(np.percentile(np.abs(delta), 99)),
            "max_abs_entropy_delta": float(np.abs(entropy_delta).max()),
            "ratio_min": float(ratio.min()),
            "ratio_max": float(ratio.max()),
            "clip_epsilon": 0.2,
            "decisions_with_ratio_outside_clip": int(
                np.count_nonzero((ratio < 0.8) | (ratio > 1.2))),
            "forward_noise_floor_max_abs_logprob_delta": float(np.abs(floor_delta).max()),
            "forward_noise_floor_mean_abs_logprob_delta": float(np.abs(floor_delta).mean()),
        },
        "bf16_unit_roundoff": BF16_UNIT_ROUNDOFF,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
