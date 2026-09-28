"""Does raising ``torch.set_num_threads`` change results, and what does it buy?

The training entry points used to pin ``torch.set_num_threads(1)`` right next to
``random.seed`` / ``torch.manual_seed``, so the intent was clearly reproducibility
rather than speed.  On Apple Silicon that pin costs nothing (Accelerate already
saturates one thread), but on a desktop x86 CPU the same pin measured 2.0-2.2x on
the batch-of-1 search shape and 1.6-1.7x on the training step, depending on which
script did the measuring.  That is what moved the pin into ``--threads``; the
numbers below are the evidence for the default of 4.

This script answers the two questions that decide whether the pin should move:

1. **Correctness** -- does the thread count change the outputs *bitwise*?
   Checked on a forward pass, on the student network's outputs, and on the
   parameters after one optimizer step.
2. **Speed** -- how much does each thread count buy on the real shapes?

Run:  python3 thread_effect.py
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

import torch  # noqa: E402

from pvz_agent_model import GameplayModelV1  # noqa: E402
from pvz_search_value import SEARCH_VALUE_FEATURES, SearchValueModel  # noqa: E402
from test_search_value_features import _observation  # noqa: E402

SEED = 1234
THREAD_COUNTS = (1, 2, 4, 8, 16)
OBSERVATION = _observation(
    plants=[{"type": i % 12, "imitater_type": -1, "row": i % 6, "col": (i * 3) % 9,
             "health": 300, "max_health": 300, "squished": False, "state": 0,
             "state_countdown": 0, "launch_counter": 0, "launch_rate": 0,
             "shooting_counter": 0, "wake_up_counter": 0, "asleep": False,
             "bungee_state": 0, "target_zombie_id": -1} for i in range(30)],
    zombies=[{"type": i % 20, "row": i % 6, "x": 200.0 + i * 40, "y": 90.0,
              "body_health": 200, "body_max_health": 200, "helm_health": 0,
              "helm_max_health": 0, "shield_health": 0, "shield_max_health": 0,
              "phase": 0, "phase_counter": 0, "velocity_x": -0.5, "chilled": 0,
              "buttered": 0, "ice_trap": 0, "has_head": True, "has_arm": True,
              "has_object": False, "is_eating": i % 5 == 0,
              "target_col": -1, "target_row": -1} for i in range(15)])


def digest(tensors) -> str:
    sha = hashlib.sha256()
    for tensor in tensors:
        sha.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return sha.hexdigest()[:16]


def parameter_digest(model) -> str:
    return digest(list(model.parameters()))


def value_outputs() -> list[torch.Tensor]:
    torch.manual_seed(SEED)
    model = SearchValueModel().eval()
    features = torch.randn(8, SEARCH_VALUE_FEATURES)
    with torch.inference_mode():
        return [model(features)]


def gameplay_outputs() -> list[torch.Tensor]:
    torch.manual_seed(SEED)
    model = GameplayModelV1().eval()
    with torch.no_grad():
        out = model.step(OBSERVATION)
    return [out["value"], out["type_logits"], out["aux_outcome"], out["belief"]]


def value_forward_digest() -> str:
    return digest(value_outputs())


def gameplay_digest() -> str:
    return digest(gameplay_outputs())


def trained_parameters(steps: int = 64) -> list[torch.Tensor]:
    """One imitation-style gradient-accumulation window; return the weights."""
    torch.manual_seed(SEED)
    model = GameplayModelV1()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    model.train()
    hidden = None
    terms = []
    for _ in range(steps):
        out = model.step(OBSERVATION, hidden, None, 150, {})
        hidden = out["hidden"]
        terms.append(out["value"].sum() + out["type_logits"].sum())
    optimizer.zero_grad(set_to_none=True)
    torch.stack(terms).mean().backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return [p.detach().clone() for p in model.parameters()]


def training_step_digest() -> str:
    return digest(trained_parameters())


def max_abs_diff(left: list[torch.Tensor], right: list[torch.Tensor]) -> float:
    return max(float((a - b).abs().max()) for a, b in zip(left, right))


def max_rel_diff(left: list[torch.Tensor], right: list[torch.Tensor]) -> float:
    """Scale-free comparison: |a-b| / max(|b|) over the whole tensor set."""
    scale = max(float(b.abs().max()) for b in right) or 1.0
    return max_abs_diff(left, right) / scale


def timed(function, iterations: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        function()
    started = time.perf_counter()
    for _ in range(iterations):
        function()
    return (time.perf_counter() - started) / iterations * 1e6


def os_cpu_count() -> int:
    """``torch.get_os_cpu_count`` is missing on some torch builds."""
    getter = getattr(torch, "get_os_cpu_count", None)
    if getter is not None:
        return getter()
    import os
    return os.cpu_count() or torch.get_num_threads()


def main() -> None:
    print(f"torch {torch.__version__}   cores={os_cpu_count()}   "
          f"default threads={torch.get_num_threads()}")
    print()

    # ---------------------------------------------------------------- correctness
    print("=" * 100)
    print("1. 线程数会不会改变结果？（逐位比较）")
    print("=" * 100)
    signatures: dict[str, dict[int, str]] = {"value forward": {}, "gameplay step": {},
                                             "after one optimizer step": {}}
    original = torch.get_num_threads()
    for threads in THREAD_COUNTS:
        torch.set_num_threads(threads)
        signatures["value forward"][threads] = value_forward_digest()
        signatures["gameplay step"][threads] = gameplay_digest()
        signatures["after one optimizer step"][threads] = training_step_digest()
    torch.set_num_threads(original)

    all_same = True
    for label, by_thread in signatures.items():
        unique = sorted(set(by_thread.values()))
        stable = len(unique) == 1
        all_same &= stable
        verdict = "全部逐位一致" if stable else f"**{len(unique)} 种不同结果**"
        print(f"  {label:<28} {verdict}   {by_thread}")
    print()
    print("  结论：", "线程数不影响数值 -> 该 pin 只影响速度" if all_same
          else "线程数会改变数值 -> 提高线程数会破坏逐位可复现")
    print()

    # ------------------------------------------------------------- how much, exactly
    print("=" * 100)
    print("1b. 差异到底有多大？（不是「不同」，而是「差多少」）")
    print("=" * 100)
    print("   误差预算（真实模拟器 160 次决策实测）: 1e-5 ~ 1e-4，低于它翻不动搜索决策")
    print()
    reference: dict[str, list[torch.Tensor]] = {}
    measured: dict[int, dict[str, tuple[float, float]]] = {}
    for threads in THREAD_COUNTS:
        torch.set_num_threads(threads)
        current = {"value forward": value_outputs(),
                   "gameplay step": gameplay_outputs(),
                   "after 64-step window": trained_parameters()}
        if threads == 1:
            reference = current
            continue
        measured[threads] = {
            label: (max_abs_diff(tensors, reference[label]),
                    max_rel_diff(tensors, reference[label]))
            for label, tensors in current.items()
        }
    torch.set_num_threads(original)

    for label in ("value forward", "gameplay step", "after 64-step window"):
        print(f"  {label}")
        for threads in THREAD_COUNTS:
            if threads == 1:
                continue
            absolute, relative = measured[threads][label]
            verdict = ("远低于预算" if relative < 1e-6 else
                       "低于预算" if relative < 1e-5 else
                       "**接近/超过预算**")
            print(f"    threads={threads:<3} max|d|={absolute:.3e}   "
                  f"相对={relative:.3e}   {verdict}")
    print()

    # ---------------------------------------------------------------------- speed
    print("=" * 100)
    print("2. 每种线程数买到多少速度")
    print("=" * 100)
    torch.manual_seed(SEED)
    value_model = SearchValueModel().eval()
    features = torch.randn(1, SEARCH_VALUE_FEATURES)
    batch = torch.randn(256, SEARCH_VALUE_FEATURES)
    torch.manual_seed(SEED)
    game_model = GameplayModelV1().eval()

    def leaf():
        with torch.inference_mode():
            return value_model(features)

    def batched():
        with torch.inference_mode():
            return value_model(batch)

    def step_forward():
        with torch.no_grad():
            return game_model.step(OBSERVATION)

    def bc_window():
        game_model.train()
        hidden = None
        terms = []
        for _ in range(64):
            out = game_model.step(OBSERVATION, hidden, None, 150, {})
            hidden = out["hidden"]
            terms.append(out["value"].sum() + out["type_logits"].sum())
        torch.stack(terms).mean().backward()
        game_model.zero_grad(set_to_none=True)

    print(f"  {'threads':>7} {'leaf (1,4116)':>14} {'value batch=256':>16} "
          f"{'step forward':>14} {'64-step fwd+bwd':>17}")
    rows = []
    for threads in THREAD_COUNTS:
        torch.set_num_threads(threads)
        leaf_us = timed(leaf, 3000)
        batched_us = timed(batched, 200)
        step_us = timed(step_forward, 20)
        bc_us = timed(bc_window, 3, warmup=1) / 64
        rows.append((threads, leaf_us, batched_us, step_us, bc_us))
        print(f"  {threads:>7} {leaf_us:>13.2f}u {batched_us:>15.2f}u "
              f"{step_us:>13.1f}u {bc_us:>16.1f}u")
    torch.set_num_threads(original)

    best_leaf = min(rows, key=lambda r: r[1])
    best_bc = min(rows, key=lambda r: r[4])
    base = next(r for r in rows if r[0] == 1)
    print()
    print(f"  相对 threads=1：leaf 最快 {best_leaf[1] / base[1]:.2f}x（threads={best_leaf[0]}），"
          f"64 步训练最快 {best_bc[4] / base[4]:.2f}x（threads={best_bc[0]}）")

    # ---------------------------------------------------------------- net ledger
    print()
    print("=" * 100)
    print("3. 决策矩阵 —— 线程数 × 设备，对一次完整训练运行的总账")
    print("=" * 100)
    print("   rollout = 192 集 × 104 次决策 × 252 次叶子估值 = 5,031,936 次 predict()")
    print("   train   = 32,768 次 model.step()（含反向）")
    print()

    accel = None
    if torch.cuda.is_available():
        accel = torch.device("cuda")
        accel_name = "cuda"
    elif torch.backends.mps.is_available():
        accel = torch.device("mps")
        accel_name = "mps"
    else:
        accel_name = None

    accel_value = accel_game = None
    if accel is not None:
        torch.manual_seed(SEED)
        accel_value = SearchValueModel().eval().to(accel)
        torch.manual_seed(SEED)
        accel_game = GameplayModelV1().eval().to(accel)

    def sync():
        if accel is None:
            return
        if accel.type == "cuda":
            torch.cuda.synchronize()
        else:
            torch.mps.synchronize()

    def predict_full(model, obs):
        return model.predict(obs)

    def train_window(model):
        model.train()
        hidden = None
        terms = []
        for _ in range(64):
            out = model.step(OBSERVATION, hidden, None, 150, {})
            hidden = out["hidden"]
            terms.append(out["value"].sum() + out["type_logits"].sum())
        torch.stack(terms).mean().backward()
        model.zero_grad(set_to_none=True)

    def timed_sync(function, iterations, warmup=3):
        for _ in range(warmup):
            function()
        sync()
        started = time.perf_counter()
        for _ in range(iterations):
            function()
        sync()
        return (time.perf_counter() - started) / iterations * 1e6

    LEAVES = 192 * 104 * 252
    TRAIN_STEPS = 32768
    cpu_value = SearchValueModel().eval()

    print(f"  {'配置':<26} {'leaf predict':>13} {'train step':>12} "
          f"{'rollout':>10} {'train':>10} {'总计':>10}")
    matrix = []
    for threads in THREAD_COUNTS:
        torch.set_num_threads(threads)
        leaf_us = timed_sync(lambda: predict_full(cpu_value, OBSERVATION), 500)
        train_us = timed_sync(lambda: train_window(game_model), 3, warmup=1) / 64
        rollout = LEAVES * leaf_us / 1e6 / 60
        train = TRAIN_STEPS * train_us / 1e6 / 60
        matrix.append((f"cpu, threads={threads}", leaf_us, train_us, rollout, train,
                       rollout + train))
    if accel is not None:
        torch.set_num_threads(1)
        leaf_us = timed_sync(lambda: predict_full(accel_value, OBSERVATION), 200)
        train_us = timed_sync(lambda: train_window(accel_game), 3, warmup=1) / 64
        rollout = LEAVES * leaf_us / 1e6 / 60
        train = TRAIN_STEPS * train_us / 1e6 / 60
        matrix.append((f"{accel_name}, cpu threads=1", leaf_us, train_us, rollout, train,
                       rollout + train))
        # Split: search on the CPU (batch-of-1 hates accelerators), training on the GPU.
        torch.set_num_threads(best_leaf[0])
        leaf_us = timed_sync(lambda: predict_full(cpu_value, OBSERVATION), 500)
        rollout = LEAVES * leaf_us / 1e6 / 60
        matrix.append((f"{accel_name} train + cpu threads={best_leaf[0]}", leaf_us, train_us,
                       rollout, train, rollout + train))
    torch.set_num_threads(original)

    for label, leaf_us, train_us, rollout, train, total in matrix:
        print(f"  {label:<26} {leaf_us:>12.1f}u {train_us:>11.0f}u "
              f"{rollout:>9.1f}m {train:>9.1f}m {total:>9.1f}m")
    best = min(matrix, key=lambda r: r[5])
    worst = max(matrix, key=lambda r: r[5])
    print()
    print(f"  最优：{best[0]}  {best[5]:.1f} min")
    print(f"  最差：{worst[0]}  {worst[5]:.1f} min")
    print(f"  差距：{worst[5] / best[5]:.2f}x")


if __name__ == "__main__":
    main()
