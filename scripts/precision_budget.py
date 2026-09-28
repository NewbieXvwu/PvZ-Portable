"""How much precision does this project actually need, and what would relaxing it buy?

Two separate questions:
  A. What is the error budget?  -> the distribution of search decision margins.
  B. What does spending that budget buy?  -> measured speed/error of each candidate.

Run:  python3 precision_budget.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from pvz_search_value import SEARCH_VALUE_FEATURES, SearchValueModel, search_value_features  # noqa: E402
from test_search_value_features import _observation  # noqa: E402

torch.set_num_threads(1)


def timed(function, iterations: int) -> float:
    function()
    started = time.perf_counter()
    for _ in range(iterations):
        function()
    return (time.perf_counter() - started) / iterations * 1e6


# --------------------------------------------------------------------------- #
# B. Error-for-speed candidates
# --------------------------------------------------------------------------- #
def report(label: str, reference: np.ndarray, candidate: np.ndarray, micros: float,
           baseline_micros: float) -> None:
    delta = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    print(f"  {label:<40} {micros:7.2f} us ({baseline_micros / micros:5.2f}x)  "
          f"max|d|={delta.max():.3e}  rows_changed={int((delta > 0).sum())}/{len(delta)}")


def main() -> None:
    torch.manual_seed(0)
    model = SearchValueModel().eval()

    observations = [
        _observation(),
        _observation(plants=[], zombies=[]),
        _observation(zombies=[{"type": i % 20, "row": i % 6, "x": 200.0 + i * 30,
                               "body_health": 200, "helm_health": 0, "shield_health": 0}
                              for i in range(15)]),
    ]
    rows = torch.stack([search_value_features(obs) for obs in observations])

    print("=" * 100)
    print("A. 特征向量的稀疏度（决定稀疏优化是否值得做）")
    print("=" * 100)
    for index, observation in enumerate(observations):
        vector = search_value_features(observation)
        nonzero = int((vector != 0).sum())
        print(f"  observation {index}: {nonzero:5d}/{SEARCH_VALUE_FEATURES} 非零 "
              f"({100.0 * nonzero / SEARCH_VALUE_FEATURES:5.2f}%)")

    print()
    print("=" * 100)
    print("B. 逐层成本（batch=1，即搜索叶子的实际形状）")
    print("=" * 100)
    with torch.inference_mode():
        x = rows[0:1]
        baseline = timed(lambda: model(x), 3000)
        print(f"  完整 3 层前向                     {baseline:7.2f} us")
        hidden = torch.randn(1, 128)
        for name, module, argument in (("Linear(4116,128)", model.network[0], x),
                                       ("Linear(128,128)", model.network[2], hidden),
                                       ("Linear(128,1)", model.network[4], hidden)):
            print(f"    {name:<30} {timed(lambda m=module, a=argument: m(a), 3000):7.2f} us")

    print()
    print("=" * 100)
    print("C. 放宽精度能换到什么（batch=1，输出经 tanh，范围 [-1,1]）")
    print("=" * 100)
    with torch.inference_mode():
        reference = model(rows).numpy()

        # --- C1: bf16 autocast ------------------------------------------------
        def bf16() -> torch.Tensor:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                return model(rows)

        micros = timed(bf16, 3000)
        report("bf16 autocast（CPU）", reference, bf16().float().numpy(), micros, baseline)

        # --- C2: half-precision weights ---------------------------------------
        half = SearchValueModel().eval()
        half.load_state_dict(model.state_dict())
        half = half.half()

        def fp16() -> torch.Tensor:
            return half(rows.half()).float()

        micros = timed(fp16, 3000)
        report("fp16 权重 + fp16 输入", reference, fp16().numpy(), micros, baseline)

        # --- C3: sparse first layer (exact gather, different summation order) --
        weight = model.network[0].weight
        bias = model.network[0].bias
        rest = model.network[1:]

        def sparse_first_layer() -> torch.Tensor:
            out = []
            for row in rows:
                index = torch.nonzero(row, as_tuple=False).squeeze(-1)
                out.append(bias + (weight[:, index] @ row[index]))
            hidden = torch.stack(out)
            return rest(hidden).squeeze(-1)

        micros = timed(sparse_first_layer, 3000)
        report("稀疏首层（nonzero gather）", reference, sparse_first_layer().numpy(), micros, baseline)

        # --- C4: bf16 matmul with fp32 accumulate (torch.backends) ------------
        def bf16_mm() -> torch.Tensor:
            h = F.silu(F.linear(rows.bfloat16(), weight.bfloat16(), bias.bfloat16()).float())
            h = F.silu(F.linear(h, model.network[2].weight, model.network[2].bias))
            return torch.tanh(F.linear(h, model.network[4].weight, model.network[4].bias)).squeeze(-1)

        micros = timed(bf16_mm, 3000)
        report("仅首层 bf16（fp32 累加）", reference, bf16_mm().numpy(), micros, baseline)

    print()
    print("=" * 100)
    print("D. 特征累加器降到 float32 能省多少")
    print("=" * 100)
    observation = observations[0]
    print(f"  search_value_features (float64, 当前)  {timed(lambda: search_value_features(observation), 3000):7.2f} us")

    def float32_features() -> np.ndarray:
        import pvz_search_value as module
        original = module.np.zeros

        def zeros32(shape, dtype=None):
            return original(shape, dtype=np.float32)

        module.np.zeros = zeros32
        try:
            return search_value_features(observation).numpy()
        finally:
            module.np.zeros = original

    micros = timed(float32_features, 3000)
    reference_features = search_value_features(observation).numpy()
    report("特征累加器 float32", reference_features, float32_features(), micros,
           timed(lambda: search_value_features(observation), 3000))

    print()
    print("  注：即使这一步免费，也只会把 predict 从 "
          f"{timed(lambda: model.predict(observation), 3000):.1f} us 降到约 "
          f"{timed(lambda: model.predict(observation), 3000) - timed(lambda: search_value_features(observation), 3000):.1f} us")


if __name__ == "__main__":
    main()
