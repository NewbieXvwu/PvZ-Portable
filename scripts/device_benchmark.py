"""Which device actually wins on this machine's workload shapes?

Every model invocation in this project is a **batch-of-1** forward pass: the
search value model is evaluated once per search leaf (~250 times per decision)
and ``GameplayModelV1.step`` carries a GRU hidden state, so it is called one
observation at a time.  That is the opposite of what accelerators are good at,
so the device choice has to be measured rather than assumed.

This script runs the same six sections on whatever accelerator the host offers
(CUDA on NVIDIA, MPS on Apple Silicon, nothing on a plain CPU box) and prints a
comparable table plus a computed net ledger for a full training run.

Run:  python3 device_benchmark.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

import torch  # noqa: E402

from pvz_agent_model import GameplayModelV1, observation_tokens  # noqa: E402
from pvz_search_value import SearchValueModel, search_value_features  # noqa: E402
from test_search_value_features import _observation  # noqa: E402

CPU = torch.device("cpu")


def _accelerator() -> torch.device | None:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return None


ACCEL = _accelerator()
ACCEL_NAME = {"cuda": "CUDA", "mps": "MPS"}.get(ACCEL.type if ACCEL else "", "n/a")

# Measured on the real simulator at budget=256 / level 7 / seed 30000.  Used only
# by the net ledger in section 8 so it is derived from data, not asserted.
LEAVES_PER_DECISION = 252
DECISIONS_PER_EPISODE = 104
EPISODES_PER_RUN = 192
TRAINING_STEPS = 32768

MEASURED: dict[str, tuple[float, float]] = {}


def sync() -> None:
    if ACCEL is None:
        return
    if ACCEL.type == "cuda":
        torch.cuda.synchronize()
    elif ACCEL.type == "mps":
        torch.mps.synchronize()


def timed(function, iterations: int, warmup: int = 5) -> float:
    for _ in range(warmup):
        function()
    sync()
    started = time.perf_counter()
    for _ in range(iterations):
        function()
    sync()
    return (time.perf_counter() - started) / iterations * 1e6


def compare(label: str, cpu_fn, accel_fn, iterations: int, key: str | None = None) -> None:
    cpu_us = timed(cpu_fn, iterations)
    if accel_fn is None:
        print(f"  {label:<40} CPU {cpu_us:9.2f} us   {ACCEL_NAME} n/a")
        return
    accel_us = timed(accel_fn, max(5, iterations // 4))
    ratio = cpu_us / accel_us
    verdict = f"{ACCEL_NAME} faster" if ratio > 1.1 else ("CPU faster" if ratio < 0.9 else "tie")
    print(f"  {label:<40} CPU {cpu_us:9.2f} us   {ACCEL_NAME} {accel_us:9.2f} us   "
          f"{ratio:5.2f}x  -> {verdict}")
    if key:
        MEASURED[key] = (cpu_us, accel_us)


def main() -> None:
    print(f"torch {torch.__version__}   accelerator: {ACCEL_NAME} ({ACCEL})")
    print(f"cuda build {torch.version.cuda}   cudnn {torch.backends.cudnn.version()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"gpu        {props.name}  cc={props.major}.{props.minor}  "
              f"sm={props.multi_processor_count}  mem={props.total_memory / 2**30:.1f} GiB")
        print(f"arch list  {torch.cuda.get_arch_list()}")
    print(f"num_threads={torch.get_num_threads()}  interop={torch.get_num_interop_threads()}")
    print()

    observation = _observation()
    dense_observation = _observation(
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

    # ---------------------------------------------------------------- value model
    print("=" * 104)
    print("1. SearchValueModel —— 搜索叶子，形状 (1, 4116)，每次决策约 250 次")
    print("=" * 104)
    cpu_model = SearchValueModel().eval()
    accel_model = (SearchValueModel().eval().to(ACCEL)
                   if ACCEL is not None else None)
    if accel_model is not None:
        accel_model.load_state_dict({k: v.clone() for k, v in cpu_model.state_dict().items()})
    features = search_value_features(observation).unsqueeze(0)
    # Keep a device-resident copy so "pure compute" and "compute + host-to-device
    # copy" are measured separately -- conflating them hides which one dominates.
    accel_features = None if ACCEL is None else features.to(ACCEL)
    with torch.inference_mode():
        compare("forward only (张量已驻留设备)",
                lambda: cpu_model(features),
                None if accel_model is None else lambda: accel_model(accel_features),
                3000, key="leaf_forward")
        compare("predict (含特征构造 + H2D 拷贝)",
                lambda: cpu_model.predict(observation),
                None if accel_model is None else lambda: accel_model.predict(observation),
                1000, key="leaf_predict")

    # --------------------------------------------------------------- gameplay model
    print()
    print("=" * 104)
    print("2. GameplayModelV1.step —— 学生网络，每次决策 1 次")
    print("=" * 104)
    cpu_game = GameplayModelV1().eval()
    accel_game = GameplayModelV1().eval().to(ACCEL) if ACCEL is not None else None
    if accel_game is not None:
        accel_game.load_state_dict({k: v.clone() for k, v in cpu_game.state_dict().items()})
    with torch.no_grad():
        compare("step (小棋盘)", lambda: cpu_game.step(observation),
                None if accel_game is None else lambda: accel_game.step(observation), 30,
                key="step_small")
        compare("step (30 plants / 15 zombies)", lambda: cpu_game.step(dense_observation),
                None if accel_game is None else lambda: accel_game.step(dense_observation), 30,
                key="step_dense")

    # ------------------------------------------------------------------- batching
    print()
    print("=" * 104)
    print("3. 批量形状 —— GEMM 才真正 FLOP-bound（训练里用得到吗？见第 7 段）")
    print("=" * 104)
    batch = torch.randn(256, 4116)
    accel_batch = None if ACCEL is None else batch.to(ACCEL)
    with torch.inference_mode():
        compare("value forward batch=256 (张量已驻留)",
                lambda: cpu_model(batch),
                None if accel_model is None else lambda: accel_model(accel_batch), 500,
                key="value_batch256")
        compare("value forward batch=256 (含 H2D 拷贝)",
                lambda: cpu_model(batch),
                None if accel_model is None else lambda: accel_model(batch.to(ACCEL)), 500)

    print()
    print("=" * 104)
    print("4. 学生网络 encoder batch=64")
    print("   （注意 RelationAttention 只接受 1-D 的 kinds/rows/cols，relation bias 在 batch 上广播）")
    print("=" * 104)
    tokens, _ = observation_tokens(dense_observation)
    kinds = tokens["kinds"]
    rows = tokens["rows"]
    cols = tokens["cols"]
    batch_size = 64

    def encoder_forward(model, device):
        with torch.no_grad():
            x = (model.kind_embedding(kinds.to(device)) + model.category_embedding(tokens["categories"].to(device)))
            x = x + model.variant_embedding(tokens["variants"].to(device))
            x = x + model.feature_projection(tokens["features"].to(device))
            x = x + model.row_embedding((rows.to(device) + 1).clamp(0, 7))
            x = x + model.col_embedding((cols.to(device) + 1).clamp(0, 10))
            x = x.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
            for layer in model.encoder:
                x = layer(x, kinds.to(device), rows.to(device), cols.to(device))
            return model.encoder_norm(x)

    compare("encoder batch=64 (含 H2D 拷贝)", lambda: encoder_forward(cpu_game, CPU),
            None if accel_game is None else lambda: encoder_forward(accel_game, ACCEL), 30,
            key="encoder64")

    # ------------------------------------------------------------------ low precision
    print()
    print("=" * 104)
    print("5. 低精度：CPU 侧在两种架构上分别实测")
    print("=" * 104)
    weight = cpu_model.network[0].weight
    bias = cpu_model.network[0].bias
    hidden_weight = cpu_model.network[2].weight
    hidden_bias = cpu_model.network[2].bias
    out_weight = cpu_model.network[4].weight
    out_bias = cpu_model.network[4].bias
    with torch.inference_mode():
        reference = cpu_model(features).numpy()

        def manual(x, w, b, hw, hb, ow, ob):
            h = torch.nn.functional.silu(torch.nn.functional.linear(x, w, b))
            h = torch.nn.functional.silu(torch.nn.functional.linear(h, hw, hb))
            return torch.tanh(torch.nn.functional.linear(h, ow, ob)).squeeze(-1)

        baseline = timed(lambda: manual(features, weight, bias, hidden_weight,
                                        hidden_bias, out_weight, out_bias), 3000)
        print(f"  {'fp32 基线（手工函数式）':<40} {baseline:9.2f} us")

        for label, dtype in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
            cast = (weight.to(dtype), bias.to(dtype), hidden_weight.to(dtype),
                    hidden_bias.to(dtype), out_weight.to(dtype), out_bias.to(dtype))

            def cpu_low(_cast=cast, _dtype=dtype):
                return manual(features.to(_dtype), *_cast).float()

            micros = timed(cpu_low, 3000)
            delta = abs(reference - cpu_low().numpy()).max()
            print(f"  {'CPU ' + label + ' 权重+输入':<40} {micros:9.2f} us   "
                  f"{baseline / micros:5.2f}x   max|d|={delta:.3e}")

        if ACCEL is not None and accel_model is not None:
            for label, dtype in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
                a_w = weight.to(ACCEL, dtype)
                a_b = bias.to(ACCEL, dtype)
                a_hw = hidden_weight.to(ACCEL, dtype)
                a_hb = hidden_bias.to(ACCEL, dtype)
                a_ow = out_weight.to(ACCEL, dtype)
                a_ob = out_bias.to(ACCEL, dtype)
                a_x = features.to(ACCEL, dtype)

                def accel_low():
                    return manual(a_x, a_w, a_b, a_hw, a_hb, a_ow, a_ob).float()

                micros = timed(accel_low, 3000)
                delta = abs(reference - accel_low().cpu().numpy()).max()
                print(f"  {ACCEL_NAME + ' ' + label + ' 权重+输入':<40} {micros:9.2f} us   "
                      f"{baseline / micros:5.2f}x   max|d|={delta:.3e}")

    # -------------------------------------------------------------------- threading
    print()
    print("=" * 104)
    print("6. CPU 线程数对搜索形状的影响（batch=1 的 GEMV 本来就并行不起来）")
    print("=" * 104)
    original_threads = torch.get_num_threads()
    core_count = torch.get_os_cpu_count() if hasattr(torch, "get_os_cpu_count") else (torch.get_num_threads())
    thread_grid = sorted({1, 2, 4, 8, original_threads, core_count})
    for threads in thread_grid:
        torch.set_num_threads(threads)
        with torch.inference_mode():
            single = timed(lambda: cpu_model(features), 3000)
            batched = timed(lambda: cpu_model(batch), 200)
        print(f"  threads={threads:<3} batch=1: {single:8.2f} us      batch=256: {batched:9.2f} us")
    torch.set_num_threads(original_threads)

    # ------------------------------------------- the *real* training step shape
    print()
    print("=" * 104)
    print("7. 真实训练内循环 —— 学生网络逐条 step() + 反向（batch 只是梯度累积窗口，不是张量维度）")
    print("=" * 104)
    print("   pvz_imitation.train: for start in range(0, len(steps), 64) 里仍是逐条 model.step()，")
    print("   GRU 隐状态串行传递 —— 所以 batch=64 的 encoder 收益在真实路径里拿不到。")

    def forward_window(model, steps: int):
        """`steps` sequential forwards with the GRU hidden carried along."""
        hidden = None
        terms = []
        for _ in range(steps):
            out = model.step(dense_observation, hidden, None, 150, {})
            hidden = out["hidden"]
            terms.append(out["value"].sum() + out["type_logits"].sum() + out["aux_outcome"].sum())
        return torch.stack(terms).mean()

    def window(model, device, steps: int, backward: bool):
        model.train()
        term = forward_window(model, steps)
        if backward:
            term.backward()
            model.zero_grad(set_to_none=True)

    def time_window(model, device, steps: int, backward: bool, rounds: int = 5) -> float:
        window(model, device, 4, backward)
        sync()
        started = time.perf_counter()
        for _ in range(rounds):
            window(model, device, steps, backward)
        sync()
        return (time.perf_counter() - started) / (rounds * steps) * 1e6

    for threads in (1, original_threads):
        torch.set_num_threads(threads)
        print(f"\n   threads={threads}")
        for label, backward in (("forward only", False), ("forward + backward", True)):
            cpu_us = time_window(cpu_game, CPU, 64, backward)
            if accel_game is None:
                print(f"     {label:<20} CPU {cpu_us:10.2f} us/step")
                continue
            accel_us = time_window(accel_game, ACCEL, 64, backward)
            verdict = f"{ACCEL_NAME} faster" if accel_us < cpu_us else "CPU faster"
            print(f"     {label:<20} CPU {cpu_us:10.2f} us/step   "
                  f"{ACCEL_NAME} {accel_us:10.2f} us/step   {cpu_us / accel_us:5.2f}x  -> {verdict}")
            if backward and threads == 1:
                MEASURED["train_step"] = (cpu_us, accel_us)
    torch.set_num_threads(original_threads)

    # ----------------------------------------------------------------- net ledger
    print()
    print("=" * 104)
    print("8. 净账 —— 用本机实测的每单位成本推算一次完整训练运行")
    print("=" * 104)
    print(f"   rollout: {EPISODES_PER_RUN} 集 × {DECISIONS_PER_EPISODE} 次决策 × "
          f"{LEAVES_PER_DECISION} 次叶子估值 = {EPISODES_PER_RUN * DECISIONS_PER_EPISODE * LEAVES_PER_DECISION:,} 次前向")
    print(f"   train  : {TRAINING_STEPS:,} 次 model.step()（8 epochs × 64 集 × ≤64 步）")
    print()

    if "leaf_predict" not in MEASURED:
        print("   本机没有可用加速器，跳过对比。")
        return

    leaf_cpu, leaf_accel = MEASURED["leaf_predict"]
    leaves = EPISODES_PER_RUN * DECISIONS_PER_EPISODE * LEAVES_PER_DECISION
    rollout_cpu = leaves * leaf_cpu / 1e6 / 60
    rollout_accel = leaves * leaf_accel / 1e6 / 60
    print(f"   叶子估值 {leaf_cpu:.1f} us -> {leaf_accel:.1f} us ({leaf_accel / leaf_cpu:.2f}x)")
    print(f"   rollout 搜索侧: CPU {rollout_cpu:6.1f} min   {ACCEL_NAME} {rollout_accel:6.1f} min"
          f"   ->  {rollout_accel - rollout_cpu:+.1f} min")

    if "train_step" in MEASURED:
        step_cpu, step_accel = MEASURED["train_step"]
        train_cpu = TRAINING_STEPS * step_cpu / 1e6 / 60
        train_accel = TRAINING_STEPS * step_accel / 1e6 / 60
        print(f"   训练 step  {step_cpu:.0f} us -> {step_accel:.0f} us ({step_accel / step_cpu:.2f}x)")
        print(f"   train 梯度侧  : CPU {train_cpu:6.1f} min   {ACCEL_NAME} {train_accel:6.1f} min"
              f"   ->  {train_accel - train_cpu:+.1f} min")
        net = (rollout_accel - rollout_cpu) + (train_accel - train_cpu)
        total_cpu = rollout_cpu + train_cpu
        print(f"   {'-' * 66}")
        print(f"   净差          : {net:+.1f} min（CPU 总计 {total_cpu:.1f} min，"
              f"{ACCEL_NAME} 总计 {total_cpu + net:.1f} min）")
        verdict = f"{ACCEL_NAME} 净亏" if net > 0 else f"{ACCEL_NAME} 净赚"
        print(f"   结论          : 值模型/搜索放 {ACCEL_NAME} 是{verdict} {abs(net):.1f} min")
    print()
    print("   注意：真实模拟器里 C++ 侧（快照保存/恢复）还占 advice() 的 ~66%，")
    print("   那部分与设备选择无关，所以实际差距会比上表小。")


if __name__ == "__main__":
    main()
