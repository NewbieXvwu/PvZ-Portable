"""Compare current relation attention with fused, local, and linear alternatives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from pvz_agent_model import FEATURE_COUNT, GameplayModelV1, TOKEN_ID_FIELDS, configure_torch_threads  # noqa: E402
from pvz_seed_jobs import read_numpy  # noqa: E402


def _prepare(data_dir: Path, episodes: int, frames_per_episode: int,
             model: GameplayModelV1, device: torch.device) -> tuple[torch.Tensor, ...]:
    paths = sorted(data_dir.glob("seed_*.npz"))[:episodes]
    if len(paths) < episodes:
        raise ValueError(f"need {episodes} episode shards, found {len(paths)}")
    transitions = [transition for path in paths
                   for transition in read_numpy(path)["transitions"][:frames_per_episode]]
    lengths = np.array([item["tokens"]["ids"].shape[0] for item in transitions], dtype=np.int64)
    count = len(transitions)
    max_tokens = int(lengths.max())
    ids = np.zeros((count, max_tokens, len(TOKEN_ID_FIELDS)), dtype=np.int64)
    features = np.zeros((count, max_tokens, FEATURE_COUNT), dtype=np.float16)
    key_mask = np.zeros((count, max_tokens), dtype=bool)
    for index, transition in enumerate(transitions):
        tokens = transition["tokens"]
        length = tokens["ids"].shape[0]
        ids[index, :length] = tokens["ids"]
        features[index, :length] = tokens["features"]
        key_mask[index, :length] = True
    ids_t = torch.from_numpy(ids).to(device)
    kinds, categories, variants, rows, cols = (ids_t[:, :, i] for i in range(5))
    x = (model.kind_embedding(kinds) + model.category_embedding(categories)
         + model.variant_embedding(variants)
         + model.feature_projection(torch.from_numpy(features).to(device=device, dtype=torch.float32))
         + model.row_embedding((rows + 1).clamp(0, 7))
         + model.col_embedding((cols + 1).clamp(0, 10)))
    return (x.detach(), kinds, rows, cols, torch.from_numpy(key_mask).to(device))


def _relation_bias(module: torch.nn.Module, kinds: torch.Tensor,
                   rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    kind_pair = module.kind_pair_bias[:, kinds[:, :, None], kinds[:, None, :]]
    row_known = (rows[:, :, None] >= 0) & (rows[:, None, :] >= 0)
    col_known = (cols[:, :, None] >= 0) & (cols[:, None, :] >= 0)
    row_delta = (rows[:, :, None] - rows[:, None, :]).clamp(-5, 5) + 5
    col_delta = (cols[:, :, None] - cols[:, None, :]).clamp(-8, 8) + 8
    row_bucket = torch.where(row_known, row_delta, 11)
    col_bucket = torch.where(col_known, col_delta, 17)
    same_cell = (row_known & col_known & (rows[:, :, None] == rows[:, None, :])
                 & (cols[:, :, None] == cols[:, None, :])).long()
    relation = (kind_pair + module.row_bias(row_bucket).permute(3, 0, 1, 2)
                + module.col_bias(col_bucket).permute(3, 0, 1, 2)
                + module.same_cell_bias(same_cell).permute(3, 0, 1, 2))
    return relation.permute(1, 0, 2, 3)


def _project(module: torch.nn.Module, attended: torch.Tensor) -> torch.Tensor:
    batch, heads, count, width = attended.shape
    return module.projection(attended.transpose(1, 2).contiguous().view(
        batch, count, heads * width))


def _qkv(module: torch.nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    batch, count, width = x.shape
    qkv = module.qkv(x).view(batch, count, 3, module.heads, module.head_width)
    return tuple(qkv[:, :, i].transpose(1, 2) for i in range(3))


def _dense(module: torch.nn.Module, x: torch.Tensor, kinds: torch.Tensor,
           rows: torch.Tensor, cols: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return module(x, kinds, rows, cols, mask)


def _sdpa_relation(module: torch.nn.Module, x: torch.Tensor, kinds: torch.Tensor,
                   rows: torch.Tensor, cols: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    query, key, value = _qkv(module, x)
    bias = _relation_bias(module, kinds, rows, cols)
    bias = bias.masked_fill(~mask[:, None, None, :], torch.finfo(bias.dtype).min)
    attended = F.scaled_dot_product_attention(query, key, value, attn_mask=bias)
    return _project(module, attended)


def _score_modifier(module: torch.nn.Module, kinds: torch.Tensor, rows: torch.Tensor,
                    cols: torch.Tensor, mask: torch.Tensor) -> Callable[..., torch.Tensor]:
    kind_bias = module.kind_pair_bias
    row_bias = module.row_bias.weight
    col_bias = module.col_bias.weight
    same_bias = module.same_cell_bias.weight

    def score_mod(score: torch.Tensor, batch: torch.Tensor, head: torch.Tensor,
                  query_index: torch.Tensor, key_index: torch.Tensor) -> torch.Tensor:
        query_kind = kinds[batch, query_index]
        key_kind = kinds[batch, key_index]
        query_row = rows[batch, query_index]
        key_row = rows[batch, key_index]
        query_col = cols[batch, query_index]
        key_col = cols[batch, key_index]
        row_known = (query_row >= 0) & (key_row >= 0)
        col_known = (query_col >= 0) & (key_col >= 0)
        row_bucket = torch.where(row_known, (query_row - key_row).clamp(-5, 5) + 5, 11)
        col_bucket = torch.where(col_known, (query_col - key_col).clamp(-8, 8) + 8, 17)
        same_cell = (row_known & col_known & (query_row == key_row) & (query_col == key_col)).long()
        relation = (kind_bias[head, query_kind, key_kind]
                    + row_bias[row_bucket, head] + col_bias[col_bucket, head]
                    + same_bias[same_cell, head])
        return (score + relation).masked_fill(~mask[batch, key_index], torch.finfo(score.dtype).min)

    return score_mod


def _local_block_mask(mask: torch.Tensor, window: int, block_size: int) -> Any:
    def local_mask(batch: torch.Tensor, head: torch.Tensor,
                   query_index: torch.Tensor, key_index: torch.Tensor) -> torch.Tensor:
        return mask[batch, key_index] & ((query_index - key_index).abs() <= window)

    q_len = mask.shape[1]
    return create_block_mask(
        local_mask, B=None, H=None, Q_LEN=q_len, KV_LEN=q_len,
        device=mask.device, BLOCK_SIZE=block_size,
    )


def _flex(module: torch.nn.Module, x: torch.Tensor, kinds: torch.Tensor,
          rows: torch.Tensor, cols: torch.Tensor, mask: torch.Tensor,
          function: Callable[..., torch.Tensor], score_mod: Callable[..., torch.Tensor],
          block_mask: Any = None) -> torch.Tensor:
    query, key, value = _qkv(module, x)
    attended = function(query, key, value, score_mod=score_mod, block_mask=block_mask)
    return _project(module, attended)


def _linear(module: torch.nn.Module, x: torch.Tensor, kinds: torch.Tensor,
            rows: torch.Tensor, cols: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    query, key, value = _qkv(module, x)
    query_features = F.elu(query) + 1
    key_features = (F.elu(key) + 1) * mask[:, None, :, None]
    key_value = torch.einsum("bhld,bhle->bhde", key_features, value)
    key_sum = key_features.sum(dim=-2)
    numerator = torch.einsum("bhld,bhde->bhle", query_features, key_value)
    denominator = torch.einsum("bhld,bhd->bhl", query_features, key_sum).clamp_min(1e-6)
    return _project(module, numerator / denominator.unsqueeze(-1))


def _sdpa_without_relation(module: torch.nn.Module, x: torch.Tensor,
                           mask: torch.Tensor) -> torch.Tensor:
    query, key, value = _qkv(module, x)
    attended = F.scaled_dot_product_attention(query, key, value,
                                              attn_mask=mask[:, None, None, :])
    return _project(module, attended)


def _benchmark(name: str, function: Callable[[torch.Tensor], torch.Tensor],
               x: torch.Tensor, module: torch.nn.Module, expected: torch.Tensor,
               repeats: int, device: torch.device) -> dict[str, Any]:
    times = []
    module.zero_grad(set_to_none=True)
    for iteration in range(repeats + 1):
        sample = x.detach().requires_grad_(True)
        module.zero_grad(set_to_none=True)
        if device.type == "cuda" and iteration == 0:
            torch.cuda.empty_cache()
        started = time.perf_counter()
        output = function(sample)
        loss = output.float().square().mean()
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if iteration:
            times.append(elapsed)
    return {
        "name": name,
        "median_forward_backward_seconds": statistics.median(times),
        "repeats": repeats,
        "max_abs_output_error": float((output.detach().float() - expected.detach().float()).abs().max()),
        "rms_output_error": float((output.detach().float() - expected.detach().float()).square().mean().sqrt()),
        "peak_cuda_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1024**2, 1)
        if device.type == "cuda" else None,
        "peak_cuda_reserved_mb": round(torch.cuda.max_memory_reserved(device) / 1024**2, 1)
        if device.type == "cuda" else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--frames-per-episode", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    configure_torch_threads(1)
    torch.manual_seed(0)
    model = GameplayModelV1().eval().to(device)
    x, kinds, rows, cols, mask = _prepare(
        args.data_dir, args.episodes, args.frames_per_episode, model, device)
    module = model.encoder[0].attention
    with torch.no_grad():
        expected = _dense(module, x, kinds, rows, cols, mask)
    batch, count, width = x.shape
    print(f"benchmark shape: frames={batch}, tokens={count}, width={width}", flush=True)
    results = []

    def run(name: str, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        results.append(_benchmark(name, fn, x, module, expected, args.repeats, device))
        print(json.dumps(results[-1], sort_keys=True), flush=True)

    run("dense_relation", lambda sample: _dense(module, sample, kinds, rows, cols, mask))
    run("sdpa_relation", lambda sample: _sdpa_relation(module, sample, kinds, rows, cols, mask))
    run("sdpa_without_relation", lambda sample: _sdpa_without_relation(module, sample, mask))
    flex = torch.compile(flex_attention, dynamic=True)
    exact_score_mod = _score_modifier(module, kinds, rows, cols, mask)
    run("flex_relation", lambda sample: _flex(
        module, sample, kinds, rows, cols, mask, flex, exact_score_mod))

    # Sliding local attention uses relation bias within a token window. The mask is
    # block sparse; global relation bias itself is retained for every in-window pair.
    local_block_mask = _local_block_mask(mask, args.window, args.block_size)
    run("flex_local_relation", lambda sample: _flex(
        module, sample, kinds, rows, cols, mask, flex, exact_score_mod, local_block_mask))
    run("linear_without_relation", lambda sample: _linear(module, sample, kinds, rows, cols, mask))

    report = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "shape": {"frames": batch, "tokens": count, "width": width},
        "relation_bias": "learned kind, row delta, column delta and same-cell terms",
        "local_window": args.window,
        "results": results,
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
