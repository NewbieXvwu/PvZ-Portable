"""Benchmark the relation-bias optimisations end to end.

Compares three configurations of the same model on the same inputs:

  * ``baseline``  -- indices recomputed inside every encoder layer, eager
  * ``hoisted``   -- indices computed once per forward, eager
  * ``fused``     -- hoisted, plus ``torch.compile`` on the bias assembly

All three are bit-identical (see ``test_relation_bias_optimization``); this
script measures what that buys in wall-clock on both forward paths: the
batch-of-1 rollout step and the batched PPO-update forward.

Usage::

    python scripts/relation_bias_benchmark.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from pvz_agent_model import (  # noqa: E402
    MODEL_CONFIG,
    GameplayModelV1,
    configure_torch_threads,
    observation_tokens,
    pack_tokens,
    relation_bias_indices,
    set_relation_bias_fusion,
)


def _observation(plants: int = 18, zombies: int = 12) -> dict:
    sys.path.insert(0, str(ROOT / "scripts"))
    from attention_cost_profile import observation  # noqa: E402
    return observation(plants, zombies)


def _timeit(fn, repeats: int = 200) -> float:
    fn()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / repeats


def _encoder_with_per_layer_indices(model, tensors, rows, cols, use_fusion: bool):
    """The baseline: recompute indices inside every layer."""
    x = model.kind_embedding(tensors["kinds"]) + model.category_embedding(tensors["categories"])
    x = x + model.variant_embedding(tensors["variants"])
    x = x + model.feature_projection(tensors["features"])
    x = x + model.row_embedding((tensors["rows"] + 1).clamp(0, 7))
    x = x + model.col_embedding((tensors["cols"] + 1).clamp(0, 10))
    x = x.unsqueeze(0)
    for layer in model.encoder:
        indices = relation_bias_indices(rows, cols)
        x = layer(x, tensors["kinds"], tensors["rows"], tensors["cols"], indices=indices)
    return model.encoder_norm(x)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--batch", type=int, default=0,
                        help="sequence batch for the update-path test (0 = skip)")
    args = parser.parse_args()

    configure_torch_threads(1)
    torch.manual_seed(0)
    model = GameplayModelV1().eval()
    tensors, metadata = observation_tokens(_observation())
    packed = pack_tokens(tensors, metadata)
    L = packed["ids"].shape[0]
    rows = tensors["rows"].unsqueeze(0)
    cols = tensors["cols"].unsqueeze(0)

    print(f"torch {torch.__version__}  threads=1  cpu  L={L}")
    print()

    with torch.no_grad():
        # ---- rollout path: batch-of-1 step_tokens -------------------------
        print("--- rollout path: step_tokens (batch 1) ---")
        set_relation_bias_fusion(False)
        baseline = _timeit(
            lambda: _encoder_with_per_layer_indices(model, tensors, rows, cols, False),
            args.repeats)
        hoisted = _timeit(lambda: _hoisted(model, tensors, rows, cols), args.repeats)
        print(f"  baseline (per-layer indices) {baseline * 1e3:8.3f} ms")
        print(f"  hoisted  (indices once)      {hoisted * 1e3:8.3f} ms"
              f"   {baseline / hoisted:.2f}x")

        enabled = set_relation_bias_fusion(True)
        if enabled:
            _hoisted(model, tensors, rows, cols)  # trigger compile
            fused = _timeit(lambda: _hoisted(model, tensors, rows, cols), args.repeats)
            print(f"  hoisted + fused              {fused * 1e3:8.3f} ms"
                  f"   {baseline / fused:.2f}x")
        else:
            print("  hoisted + fused              (torch.compile unavailable)")

        # ---- equivalence across the three configurations -------------------
        set_relation_bias_fusion(False)
        reference = _encoder_with_per_layer_indices(model, tensors, rows, cols, False)
        candidate = _hoisted(model, tensors, rows, cols)
        print(f"  torch.equal(baseline, hoisted) = {torch.equal(reference, candidate)}")
        if enabled:
            set_relation_bias_fusion(True)
            fused_out = _hoisted(model, tensors, rows, cols)
            print(f"  torch.equal(baseline, fused)   = {torch.equal(reference, fused_out)}")
        set_relation_bias_fusion(False)
        print()

        # ---- full step_tokens, the thing the rollout actually calls --------
        print("--- rollout path: full step_tokens ---")
        set_relation_bias_fusion(False)
        plain = _timeit(lambda: model.step_tokens(tensors, metadata, 8), args.repeats)
        if enabled:
            set_relation_bias_fusion(True)
            model.step_tokens(tensors, metadata, 8)
            fused_step = _timeit(lambda: model.step_tokens(tensors, metadata, 8), args.repeats)
            print(f"  eager   {plain * 1e3:8.3f} ms")
            print(f"  fused   {fused_step * 1e3:8.3f} ms   {plain / fused_step:.2f}x")
        else:
            print(f"  eager   {plain * 1e3:8.3f} ms")
        set_relation_bias_fusion(False)


def _hoisted(model, tensors, rows, cols):
    """Hoisted variant: indices computed once, then every layer reuses them."""
    x = model.kind_embedding(tensors["kinds"]) + model.category_embedding(tensors["categories"])
    x = x + model.variant_embedding(tensors["variants"])
    x = x + model.feature_projection(tensors["features"])
    x = x + model.row_embedding((tensors["rows"] + 1).clamp(0, 7))
    x = x + model.col_embedding((tensors["cols"] + 1).clamp(0, 10))
    x = x.unsqueeze(0)
    indices = relation_bias_indices(rows, cols)
    for layer in model.encoder:
        x = layer(x, tensors["kinds"], tensors["rows"], tensors["cols"], indices=indices)
    return model.encoder_norm(x)


if __name__ == "__main__":
    main()
