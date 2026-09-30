"""Watch a running T5 training job without touching it.

The trainer rewrites ``training_state.json`` every update and ``learning_curve.json``
every evaluation, but nothing ever reads them while the job runs -- so a three-minute
update looks like a hang.  This is a read-only observer: it polls the two files and
prints what changed.

It deliberately uses only the standard library.  ``requirements.txt`` is torch plus
numpy, and adding a dashboard dependency to watch a training run is a bad trade.

Usage::

    # one snapshot, then exit (safe to call from anywhere)
    python scripts/watch_training.py --state-dir artifacts/t5/run

    # follow until interrupted
    python scripts/watch_training.py --state-dir artifacts/t5/run --follow

``--state-dir`` is the trainer's ``--output-dir``: the directory holding
``training_state.json`` and ``learning_curve.json``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

STATE_NAME = "training_state.json"
CURVE_NAME = "learning_curve.json"


def _load(path: Path) -> Any | None:
    """Return the parsed JSON, or ``None`` if it is missing or unreadable.

    The trainer writes through ``atomic_json`` (write to a temp file, then
    ``os.replace``), so a partially written file should be impossible -- but a
    watcher that crashes on a transient read error is worse than one that waits.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _rolling_summary(state: dict[str, Any]) -> tuple[int, float, float, float, list[tuple[str, float]]]:
    """Per-task rolling win rates, worst first."""
    passes = state.get("recent_passes") or {}
    rates = []
    for task_id, values in passes.items():
        if values:
            rates.append((task_id, sum(values) / len(values)))
    if not rates:
        return 0, 0.0, 0.0, 0.0, []
    ordered = sorted(rate for _, rate in rates)
    worst = sorted(rates, key=lambda item: item[1])[:3]
    return len(rates), ordered[0], statistics.median(ordered), ordered[-1], worst


def _render(state: dict[str, Any], curve: Any | None, previous_curve_len: int) -> tuple[str, int]:
    lines: list[str] = []
    runs = state.get("runs") or []
    run = runs[-1] if runs else {}
    update = state.get("last_update") or {}
    debug = state.get("last_training_debug") or {}
    timing = debug.get("timing_seconds") or {}
    losses = update.get("losses") or {}

    count, lowest, median, highest, worst = _rolling_summary(state)
    lines.append(
        f"run={run.get('run_number', '?')} status={run.get('status', '?')} "
        f"update={update.get('update', 0)} "
        f"episodes={state.get('cumulative_episodes', 0):,} "
        f"({update.get('episodes', 0)}/update)"
    )
    if count:
        lines.append(
            f"  task win rate (rolling 64, {count} tasks): "
            f"min={lowest:.2f} median={median:.2f} max={highest:.2f}"
        )
        lines.append("  worst: " + ", ".join(f"{name}={rate:.2f}" for name, rate in worst))
    lines.append(
        f"  term={debug.get('terminal_outcome_mean_recent', 0.0):+.3f} "
        f"shaping={debug.get('mean_abs_shaping_reward_recent', 0.0):.4f} "
        f"advantage[{debug.get('gae_advantage_mean', 0.0):+.3f}"
        f"/{debug.get('gae_advantage_std', 0.0):.3f}]"
    )
    lines.append(
        f"  pi={losses.get('policy_loss', 0.0):.4f} v={losses.get('value_loss', 0.0):.4f} "
        f"H={losses.get('entropy', 0.0):.3f} |grad|={debug.get('gradient_norm', 0.0):.3f}"
    )
    if timing:
        lines.append(
            f"  rollout={timing.get('rollout')}s update={timing.get('ppo_update')}s "
            f"digest={timing.get('episode_digest')}s hash={timing.get('model_state_sha256')}s "
            f"ckpt={timing.get('checkpoint_save')}s "
            f"-> {timing.get('episodes_per_hour')} episodes/hour"
        )
    profile = debug.get("mean_episode_profile_seconds") or {}
    if profile and all(value is not None for value in profile.values()):
        total = sum(profile.values()) or 1.0
        lines.append("  per episode: " + ", ".join(
            f"{name}={value * 1e3:.1f}ms ({value / total * 100:.0f}%)"
            for name, value in sorted(profile.items(), key=lambda kv: -kv[1])))
    else:
        lines.append("  per episode: (not recorded -- an older state file, or cached shards)")

    if isinstance(curve, list) and len(curve) > previous_curve_len:
        lines.append(f"  learning curve: {len(curve)} evaluation points")
        for row in curve[previous_curve_len:]:
            recent = row.get("train_task_pass_rates_recent_64") or {}
            values = [v for v in recent.values() if v is not None]
            median = f"{statistics.median(values):.2f}" if values else "n/a"
            lines.append(
                f"    @{row.get('cumulative_training_episodes', 0):,} ep  "
                f"gate={_fmt(row.get('heldout_cap3_x1_pass_rate'))} "
                f"stage0={_fmt(row.get('stage0_pass_rate'))} "
                f"train_median={median}"
            )
    return "\n".join(lines), len(curve) if isinstance(curve, list) else previous_curve_len


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state-dir", type=Path, required=True,
                        help="the trainer's --output-dir")
    parser.add_argument("--follow", action="store_true",
                        help="keep polling instead of printing one snapshot")
    parser.add_argument("--interval", type=float, default=10.0,
                        help="seconds between polls in --follow mode (default 10)")
    args = parser.parse_args()

    state_path = args.state_dir / STATE_NAME
    curve_path = args.state_dir / CURVE_NAME
    if not state_path.exists():
        raise SystemExit(f"no {STATE_NAME} in {args.state_dir}; is --state-dir the trainer's --output-dir?")

    printed_curve = 0
    last_signature: tuple[Any, ...] | None = None
    while True:
        state = _load(state_path)
        curve = _load(curve_path)
        if state is None:
            if not args.follow:
                raise SystemExit(f"{state_path} is unreadable")
            time.sleep(args.interval)
            continue
        update = state.get("last_update") or {}
        signature = (state.get("cumulative_episodes"), update.get("update"),
                     len(curve) if isinstance(curve, list) else 0)
        if signature != last_signature:
            last_signature = signature
            body, printed_curve = _render(state, curve, printed_curve)
            stamp = time.strftime("%H:%M:%S")
            print(f"--- {stamp} " + "-" * 46, flush=True)
            print(body, flush=True)
        if not args.follow:
            return
        if state.get("stop_reason"):
            print(f"stopped: {state['stop_reason']}", flush=True)
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
