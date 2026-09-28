"""What is the search's error budget?  Measure the top1-vs-top2 decision margin.

If the winning candidate leads the runner-up by ~1e-2, then any numerical error
below ~1e-3 cannot change the decision, and precision is not a binding
constraint.  If margins are ~1e-7, precision matters a great deal.

Run:  python3 decision_margin.py [resource_dir]
"""
from __future__ import annotations

import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

import torch  # noqa: E402

from pvz_env import PvZEnv, training_task  # noqa: E402
from pvz_search import SearchTeacher  # noqa: E402
from pvz_search_value import SearchValueModel  # noqa: E402

def _resource_dir(argument: str | None) -> Path:
    """Resolve the game resource directory from the argument or ``$PVZ_RESOURCE_DIR``.

    The path is deliberately not baked in -- this repository is public, and a
    personal download directory is nobody else's business.
    """
    candidate = argument or os.environ.get("PVZ_RESOURCE_DIR")
    if not candidate:
        raise SystemExit("pass the resource directory as an argument or set PVZ_RESOURCE_DIR")
    return Path(candidate)
LEVEL = 7
SEEDS = (30000, 30001, 30002, 30003)
DECISION_CAP = 40
BUDGET = 256
CANDIDATE_LIMIT = 8
HORIZON = 900


def main() -> int:
    resource_dir = _resource_dir(sys.argv[1] if len(sys.argv) > 1 else None)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = SearchValueModel().eval()

    margins: list[float] = []
    spreads: list[float] = []
    ties = 0
    decisions = 0

    env = PvZEnv(resource_dir=resource_dir)
    try:
        for seed in SEEDS:
            observation, _ = env.reset(level=LEVEL, task=training_task(seed, LEVEL))
            teacher = SearchTeacher(env, value_model=model, simulation_budget=BUDGET,
                                    candidate_limit=CANDIDATE_LIMIT, horizon_ticks=HORIZON)
            for _ in range(DECISION_CAP):
                if observation.get("terminal"):
                    break
                advice = teacher.advice(observation)
                decisions += 1
                scores = [score for _, score in advice.candidates]
                if len(scores) > 1:
                    spreads.append(max(scores) - min(scores))
                    if advice.best_second_margin is None:
                        ties += 1
                    else:
                        margins.append(abs(advice.best_second_margin))
                observation, _, terminal, truncated, _ = env.step(advice.action)
                if terminal or truncated:
                    break
    finally:
        env.close()

    def describe(label: str, values: list[float]) -> None:
        if not values:
            print(f"  {label:<26} (no samples)")
            return
        ordered = sorted(values)
        def quantile(q: float) -> float:
            return ordered[min(len(ordered) - 1, int(q * len(ordered)))]
        print(f"  {label:<26} n={len(ordered):4d}  "
              f"min={ordered[0]:.3e}  p10={quantile(0.10):.3e}  "
              f"median={statistics.median(ordered):.3e}  p90={quantile(0.90):.3e}  "
              f"max={ordered[-1]:.3e}")

    print(f"real simulator, level {LEVEL}, seeds {SEEDS}, budget={BUDGET}\n")
    print(f"  decisions taken: {decisions}   (ties on outcome rank: {ties})")
    describe("top1-top2 margin", margins)
    describe("candidate score spread", spreads)

    if not margins:
        return 0

    print()
    print("  误差预算：margin 低于阈值 ε 的决策占比（这些决策在误差 > ε 时可能翻转）")
    print(f"  {'ε':>10} {'margin < ε 的占比':>20}   说明")
    notes = {
        0.0: "精确并列 —— 任何误差都可能翻转，但翻转无质量损失（两者同分）",
        1e-9: "double 舍入量级",
        1e-7: "float32 舍入量级",
        6.6e-5: "bf16 前向实测误差（但 bf16 在本机慢 8 倍，不是可选项）",
        2.5e-5: "fp16 前向实测误差（同样更慢）",
        1e-3: "宽松的近似值模型",
        1e-2: "激进的近似值模型",
    }
    for epsilon in sorted(notes):
        count = sum(1 for value in margins if value < epsilon)
        share = 100.0 * count / len(margins)
        bar = "#" * int(round(share / 2))
        print(f"  {epsilon:>10.1e} {share:>18.1f}%   {bar} {notes[epsilon]}")

    strict = [value for value in margins if value > 0.0]
    if strict:
        print(f"\n  排除精确并列后：最小 {min(strict):.3e}，"
              f"中位 {statistics.median(strict):.3e}")
        for epsilon in (1e-7, 1e-5, 6.6e-5):
            count = sum(1 for value in strict if value < epsilon)
            print(f"    其中 margin < {epsilon:.1e} 的：{count}/{len(strict)} "
                  f"({100.0 * count / len(strict):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
