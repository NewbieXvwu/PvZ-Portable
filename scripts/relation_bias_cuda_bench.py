"""What the relation bias costs on CUDA, and whether precomputing it helps.

The recorded sweep (``artifacts/t5/perf/attention_sweep.json``, RTX 5080, 256 frames
x 89 tokens) says one relation-attention layer costs 18.92 ms forward+backward when
the bias is computed inside FlexAttention's ``score_mod``, against 3.31 ms with no
bias at all.  Four layers put 62.4 ms of the 170.5 ms optimizer step on the bias --
36.6% -- which is the largest single identified cost in the T5 update.  Every number
behind that came from a CPU-side profile or from the sweep itself, and the two
directions ``PPO_UPDATE_ANATOMY.md`` §9 proposed for it had never been run on CUDA.

This script runs them, and it splits each variant into forward-only and
forward+backward so the 36.6% can be attributed rather than quoted:

  ``dense_relation``          reference output, eager path with the fused assembly
  ``dense_relation_eager``    same, with ``PVZ_RELATION_BIAS_FUSION`` forced off
  ``sdpa_relation``           precomputed bias handed to SDPA as an attention mask
  ``sdpa_without_relation``   no bias at all
  ``flex_mask_only``          score_mod applies the key mask and nothing else
  ``flex_relation``           production score_mod: buckets + 4 lookups per element
  ``flex_hoisted_buckets``    buckets precomputed outside, score_mod does the 4 lookups
  ``flex_precomputed_bias``   bias materialised as (batch, heads, L, L); one load
  ``flex_precomputed_bias_fused``  the same, but the materialisation is compiled

The last four differ only in where the work happens, and their spread answers the
question the anatomy file left open: is the cost the arithmetic in ``score_mod``, the
gathers it performs, or the layout of the operands it gathers from.  Every variant is
timed forward-only and forward+backward separately, because the two turn out not to
resemble each other at all.

A ``--term-repeats`` section then breaks the assembly's backward into its four
lookups, so a cost that lands in one term can be named rather than averaged.

It also A/Bs the §6.2 layout change on CUDA.  That change was justified by strided
reads on the CPU; ``permute`` is free on CUDA and the reasoning need not transfer.

Run it on a CUDA machine with real rollout shards::

    python scripts/relation_bias_cuda_bench.py \
        --data-dir artifacts/t5/runs/run_2/.seed_jobs/update_0001/<digest> \
        --episodes 16 --frames-per-episode 16 \
        --output artifacts/t5/perf/relation_bias_cuda.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "python"))

try:
    import attention_benchmark as AB  # noqa: E402
except ImportError as exc:  # pragma: no cover - CUDA/torch-version dependent
    raise SystemExit(f"this probe needs torch's flex_attention: {exc}") from exc

import pvz_agent_model as model_module  # noqa: E402
from pvz_agent_model import (  # noqa: E402
    GameplayModelV1,
    configure_torch_threads,
    fused_relation_bias,
    relation_bias_from_indices,
    relation_bias_indices,
    set_relation_bias_fusion,
)

ScoreMod = Callable[..., torch.Tensor]


# --------------------------------------------------------------------------- #
# bias assembly
# --------------------------------------------------------------------------- #

def _buckets(rows: torch.Tensor, cols: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """The pre-§6.2 bucket construction, inlined so the layout A/B can compare."""
    row_known = (rows[:, :, None] >= 0) & (rows[:, None, :] >= 0)
    col_known = (cols[:, :, None] >= 0) & (cols[:, None, :] >= 0)
    row_bucket = torch.where(row_known, (rows[:, :, None] - rows[:, None, :]).clamp(-5, 5) + 5, 11)
    col_bucket = torch.where(col_known, (cols[:, :, None] - cols[:, None, :]).clamp(-8, 8) + 8, 17)
    same_cell = (row_known & col_known & (rows[:, :, None] == rows[:, None, :])
                 & (cols[:, :, None] == cols[:, None, :])).long()
    return row_bucket, col_bucket, same_cell


def bias_production(module: torch.nn.Module, kinds: torch.Tensor,
                    rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    """The shipped assembly: buckets from :func:`relation_bias_indices`, one permute."""
    indices = relation_bias_indices(rows, cols)
    return relation_bias_from_indices(
        kinds, indices.row_bucket, indices.col_bucket, indices.same_cell,
        module.kind_pair_bias, module.row_bias.weight,
        module.col_bias.weight, module.same_cell_bias.weight)


def bias_pre_layout(module: torch.nn.Module, kinds: torch.Tensor,
                    rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    """The assembly as it stood before §6.2: three ``permute``d operands per add."""
    row_bucket, col_bucket, same_cell = _buckets(rows, cols)
    relation = module.kind_pair_bias[:, kinds[:, :, None], kinds[:, None, :]]
    relation = relation + module.row_bias.weight[row_bucket].permute(3, 0, 1, 2)
    relation = relation + module.col_bias.weight[col_bucket].permute(3, 0, 1, 2)
    relation = relation + module.same_cell_bias.weight[same_cell].permute(3, 0, 1, 2)
    return relation.permute(1, 0, 2, 3)


def bias_fused(module: torch.nn.Module, kinds: torch.Tensor,
               rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    """The shipped assembly as the eager attention path actually runs it.

    ``RelationAttention`` calls the assembly through ``torch.compile`` whenever
    ``PVZ_RELATION_BIAS_FUSION`` is on, which it is by default.  Measuring the plain
    Python function instead would describe code the dense path does not execute.
    """
    compiled = fused_relation_bias()
    if compiled is None:
        raise RuntimeError("torch.compile is unavailable for the relation-bias assembly")
    indices = relation_bias_indices(rows, cols)
    return compiled(kinds, indices.row_bucket, indices.col_bucket, indices.same_cell,
                    module.kind_pair_bias, module.row_bias.weight,
                    module.col_bias.weight, module.same_cell_bias.weight)


# --------------------------------------------------------------------------- #
# variants
# --------------------------------------------------------------------------- #

def _masked(score: torch.Tensor, mask: torch.Tensor, batch: torch.Tensor,
            key_index: torch.Tensor) -> torch.Tensor:
    return score.masked_fill(~mask[batch, key_index], torch.finfo(score.dtype).min)


def make_variants(module: torch.nn.Module, kinds: torch.Tensor, rows: torch.Tensor,
                  cols: torch.Tensor, mask: torch.Tensor,
                  flex: Any) -> list[tuple[str, Callable[[torch.Tensor], torch.Tensor]]]:
    """Every variant, as ``(name, callable)`` pairs taking only the encoder input.

    Each closure is built once and reused for every repeat: a fresh ``score_mod``
    function object makes ``torch.compile`` recompile, which would show up as a
    per-call cost that has nothing to do with the variant.
    """
    kind_bias = module.kind_pair_bias
    row_bias = module.row_bias.weight
    col_bias = module.col_bias.weight
    same_bias = module.same_cell_bias.weight
    indices = relation_bias_indices(rows, cols)
    hoisted_row = indices.row_bucket
    hoisted_col = indices.col_bucket
    hoisted_same = indices.same_cell
    bias_cell: dict[str, torch.Tensor] = {}
    initial_fusion = model_module.RELATION_BIAS_FUSION

    def flex_mask_only_score(score: torch.Tensor, batch: torch.Tensor, head: torch.Tensor,
                             query_index: torch.Tensor, key_index: torch.Tensor) -> torch.Tensor:
        return _masked(score, mask, batch, key_index)

    def flex_relation_score(score: torch.Tensor, batch: torch.Tensor, head: torch.Tensor,
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
        same_cell = (row_known & col_known & (query_row == key_row)
                     & (query_col == key_col)).long()
        relation = (kind_bias[head, query_kind, key_kind]
                    + row_bias[row_bucket, head] + col_bias[col_bucket, head]
                    + same_bias[same_cell, head])
        return _masked(score + relation, mask, batch, key_index)

    def flex_hoisted_score(score: torch.Tensor, batch: torch.Tensor, head: torch.Tensor,
                           query_index: torch.Tensor, key_index: torch.Tensor) -> torch.Tensor:
        relation = (kind_bias[head, kinds[batch, query_index], kinds[batch, key_index]]
                    + row_bias[hoisted_row[batch, query_index, key_index], head]
                    + col_bias[hoisted_col[batch, query_index, key_index], head]
                    + same_bias[hoisted_same[batch, query_index, key_index], head])
        return _masked(score + relation, mask, batch, key_index)

    def flex_precomputed_score(score: torch.Tensor, batch: torch.Tensor, head: torch.Tensor,
                               query_index: torch.Tensor, key_index: torch.Tensor) -> torch.Tensor:
        return _masked(score + bias_cell["bias"][batch, head, query_index, key_index],
                       mask, batch, key_index)

    def run_flex(score_mod: ScoreMod | None, prepare: Callable[[], None] | None = None
                 ) -> Callable[[torch.Tensor], torch.Tensor]:
        def run(x: torch.Tensor) -> torch.Tensor:
            if prepare is not None:
                prepare()
            query, key, value = AB._qkv(module, x)
            attended = flex(query, key, value, score_mod=score_mod)
            return AB._project(module, attended)
        return run

    def prepare_bias() -> None:
        bias_cell["bias"] = bias_production(module, kinds, rows, cols)

    def prepare_bias_fused() -> None:
        bias_cell["bias"] = bias_fused(module, kinds, rows, cols)

    def dense_eager(x: torch.Tensor) -> torch.Tensor:
        # ``module`` consults the module-level fusion flag on every call, so the
        # variant that has it off has to hold it off for the whole timing loop.
        # ``set_relation_bias_fusion`` returns the value in effect *after* the call,
        # so the restore has to name the original value rather than its return.
        set_relation_bias_fusion(False)
        try:
            return AB._dense(module, x, kinds, rows, cols, mask)
        finally:
            set_relation_bias_fusion(initial_fusion)

    return [
        ("dense_relation", lambda x: AB._dense(module, x, kinds, rows, cols, mask)),
        ("dense_relation_eager", dense_eager),
        ("sdpa_relation", lambda x: AB._sdpa_relation(module, x, kinds, rows, cols, mask)),
        ("sdpa_without_relation", lambda x: AB._sdpa_without_relation(module, x, mask)),
        ("flex_mask_only", run_flex(flex_mask_only_score)),
        ("flex_relation", run_flex(flex_relation_score)),
        ("flex_hoisted_buckets", run_flex(flex_hoisted_score)),
        ("flex_precomputed_bias", run_flex(flex_precomputed_score, prepare_bias)),
        ("flex_precomputed_bias_fused", run_flex(flex_precomputed_score, prepare_bias_fused)),
    ]


# --------------------------------------------------------------------------- #
# timing
# --------------------------------------------------------------------------- #

def time_variant(function: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor,
                 repeats: int, device: torch.device) -> tuple[float, float, torch.Tensor]:
    """Median forward-only and forward+backward seconds, plus the last output."""
    forward_times: list[float] = []
    total_times: list[float] = []
    output = None
    for iteration in range(repeats + 2):
        sample = x.detach().requires_grad_(True)
        started = time.perf_counter()
        output = function(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_seconds = time.perf_counter() - started

        loss = output.float().square().mean()
        started = time.perf_counter()
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        total_seconds = forward_seconds + (time.perf_counter() - started)
        del loss

        if iteration >= 2:  # two warm-up calls: torch.compile and the allocator
            forward_times.append(forward_seconds)
            total_times.append(total_seconds)
    assert output is not None
    return statistics.median(forward_times), statistics.median(total_times), output.detach()


def bench_assembly(name: str, function: Callable[[], torch.Tensor], repeats: int,
                   device: torch.device) -> dict[str, Any]:
    """Forward-only and forward+backward cost of the bias assembly on its own."""
    forward_times: list[float] = []
    total_times: list[float] = []
    value = None
    for iteration in range(repeats + 2):
        started = time.perf_counter()
        value = function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_seconds = time.perf_counter() - started

        loss = value.float().square().mean()
        started = time.perf_counter()
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        total_seconds = forward_seconds + (time.perf_counter() - started)
        del loss
        if iteration >= 2:
            forward_times.append(forward_seconds)
            total_times.append(total_seconds)
    assert value is not None
    return {
        "name": name,
        "median_forward_seconds": statistics.median(forward_times),
        "median_forward_backward_seconds": statistics.median(total_times),
        "median_backward_seconds": statistics.median(total_times) - statistics.median(forward_times),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="directory of seed_*.npz rollout shards")
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--frames-per-episode", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--term-repeats", type=int, default=3,
                        help="repeats for the per-lookup backward breakdown, which is "
                             "far slower than the assembled variants")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    configure_torch_threads(1)
    torch.manual_seed(0)
    model = GameplayModelV1().eval().to(device)
    x, kinds, rows, cols, mask = AB._prepare(
        args.data_dir, args.episodes, args.frames_per_episode, model, device)
    module = model.encoder[0].attention
    with torch.no_grad():
        expected = AB._dense(module, x, kinds, rows, cols, mask)
    batch, count, width = x.shape
    heads = module.heads
    print(f"torch {torch.__version__}  device {device}  "
          f"shape frames={batch} tokens={count} width={width} heads={heads}", flush=True)

    # The layout A/B must come first: both layouts have to see the same parameters.
    # ``bias_fused`` is deliberately called with grad enabled -- it is the call that
    # compiles, and a graph built under ``no_grad`` would not produce a backward for
    # the timing loop below to measure.
    with torch.no_grad():
        shipped = bias_production(module, kinds, rows, cols)
        previous = bias_pre_layout(module, kinds, rows, cols)
    fused = bias_fused(module, kinds, rows, cols)
    with torch.no_grad():
        layout_identical = bool(torch.equal(shipped, previous))
        fused_identical = bool(torch.equal(shipped, fused))
        layout_max = float((shipped - previous).abs().max().item())
        fused_max = float((shipped - fused).abs().max().item())
    del fused
    print(f"layout A/B: torch.equal(shipped, pre_6.2) = {layout_identical}  "
          f"max|diff| = {layout_max:.3e}", flush=True)
    print(f"fusion A/B: torch.equal(shipped, fused) = {fused_identical}  "
          f"max|diff| = {fused_max:.3e}", flush=True)

    flex = torch.compile(torch.nn.attention.flex_attention.flex_attention, dynamic=True)
    variants = make_variants(module, kinds, rows, cols, mask, flex)

    results = []
    for name, function in variants:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        module.zero_grad(set_to_none=True)
        forward_seconds, total_seconds, output = time_variant(
            function, x, args.repeats, device)
        row = {
            "name": name,
            "median_forward_seconds": forward_seconds,
            "median_forward_backward_seconds": total_seconds,
            "median_backward_seconds": total_seconds - forward_seconds,
            "max_abs_output_error": float(
                (output.float() - expected.float()).abs().max().item()),
            "peak_cuda_allocated_mb": (
                round(torch.cuda.max_memory_allocated(device) / 1024 ** 2, 1)
                if device.type == "cuda" else None),
        }
        results.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    assembly = []
    for name, function in (
        ("assembly_shipped_eager", lambda: bias_production(module, kinds, rows, cols)),
        ("assembly_shipped_fused", lambda: bias_fused(module, kinds, rows, cols)),
        ("assembly_pre_6_2_eager", lambda: bias_pre_layout(module, kinds, rows, cols)),
    ):
        assembly.append(bench_assembly(name, function, args.repeats, device))
        print(json.dumps(assembly[-1], sort_keys=True), flush=True)

    # Which of the four lookups the assembly's backward actually pays for.  The
    # ``distinct_outputs`` column is the number of floats the gradient has to be
    # reduced into: a lookup whose index space is tiny turns its backward into a
    # many-to-few accumulation, which is a different cost class from a gather.
    row_bucket, col_bucket, same_cell = _buckets(rows, cols)
    terms = [
        ("term_kind_pair_bias", len(torch.unique(kinds)) ** 2 * heads,
         lambda: module.kind_pair_bias.permute(1, 2, 0)[kinds[:, :, None], kinds[:, None, :]]),
        ("term_row_bias", 12 * heads, lambda: module.row_bias.weight[row_bucket]),
        ("term_col_bias", 18 * heads, lambda: module.col_bias.weight[col_bucket]),
        ("term_same_cell_bias", 2 * heads, lambda: module.same_cell_bias.weight[same_cell]),
    ]
    term_rows = []
    for name, distinct, function in terms:
        row = bench_assembly(name, function, args.term_repeats, device)
        row["distinct_outputs"] = int(distinct)
        term_rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    # The two deltas the anatomy file is built on, restated against this run.
    by_name = {row["name"]: row for row in results}
    with_bias = by_name["flex_relation"]["median_forward_backward_seconds"]
    without_bias = by_name["flex_mask_only"]["median_forward_backward_seconds"]
    layer_delta = with_bias - without_bias
    report = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch_version": torch.__version__,
        "shape": {"frames": batch, "tokens": count, "width": width, "heads": heads},
        "repeats": args.repeats,
        "layout_ab": {
            "torch_equal": layout_identical,
            "max_abs_difference": layout_max,
        },
        "fusion_ab": {
            "torch_equal": fused_identical,
            "max_abs_difference": fused_max,
        },
        "variants": results,
        "bias_assembly": assembly,
        "bias_assembly_terms": term_rows,
        "bias_cost_per_layer": {
            "flex_relation_minus_flex_mask_only_seconds": layer_delta,
            "flex_relation_over_flex_mask_only": with_bias / without_bias,
            "four_layers_seconds": 4 * layer_delta,
            "forward_only_seconds": (
                by_name["flex_relation"]["median_forward_seconds"]
                - by_name["flex_mask_only"]["median_forward_seconds"]),
            "backward_only_seconds": (
                by_name["flex_relation"]["median_backward_seconds"]
                - by_name["flex_mask_only"]["median_backward_seconds"]),
        },
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
