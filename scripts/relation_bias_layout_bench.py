"""Measure the layout cost of assembling the relation bias.

``relation_bias_from_indices`` gathers four ``(B, L, L, H)`` tables and permutes
each one to ``(H, B, L, L)`` before summing, then permutes the result back.  That
is four non-contiguous copies of ``B*L*L*H`` floats per encoder layer, and in the
PPO update ``B`` is the whole minibatch (256 transitions), so the copies move
hundreds of megabytes per forward.

Every term is a function of a pair index and a learned table, so the four terms
can equally be gathered and summed in ``(B, L, L, H)`` order and permuted once.
Each element's addition order is unchanged, so the result is bit-identical.

Run:  python scripts/relation_bias_layout_bench.py [--device cpu] [--threads 1]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from pvz_agent_model import TOKEN_KINDS, relation_bias_indices  # noqa: E402

HEADS = 6
# Real shapes: one rollout decision (batch 1, 108 tokens) and one PPO update
# minibatch (256 transitions, ~80 tokens each).
SHAPES = ((1, 108), (256, 80), (16, 80))


def current(kinds, indices, kind_pair_bias, row_w, col_w, same_w):
    relation = kind_pair_bias[:, kinds[:, :, None], kinds[:, None, :]]
    relation = relation + row_w[indices.row_bucket].permute(3, 0, 1, 2)
    relation = relation + col_w[indices.col_bucket].permute(3, 0, 1, 2)
    relation = relation + same_w[indices.same_cell].permute(3, 0, 1, 2)
    return relation.permute(1, 0, 2, 3)


def single_permute(kinds, indices, kind_pair_bias, row_w, col_w, same_w):
    """Same four gathers and the same addition order, in ``(B, L, L, H)`` order.

    ``kind_pair_bias`` is ``(H, K, K)``, so moving the head dim last first makes
    it ``(K, K, H)`` and the same pair indexing yields ``(B, L, L, H)`` -- the
    layout the three ``nn.Embedding`` gathers already produce.  That permute
    touches ``H*K*K`` elements (726), not ``B*L*L*H``.
    """
    relation = kind_pair_bias.permute(1, 2, 0)[kinds[:, :, None], kinds[:, None, :]]
    relation = relation + row_w[indices.row_bucket]
    relation = relation + col_w[indices.col_bucket]
    relation = relation + same_w[indices.same_cell]
    return relation.permute(0, 3, 1, 2)


def narrow_indices(indices, dtype):
    """The pair indices only ever take a handful of values, so int64 is waste.

    ``torch`` accepts int32 and uint8 as index dtypes but not int16.
    """
    return type(indices)(
        row_bucket=indices.row_bucket.to(dtype),
        col_bucket=indices.col_bucket.to(dtype),
        same_cell=indices.same_cell.to(dtype),
    )


def timeit(fn, repeats: int) -> tuple[float, float]:
    fn()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1e3)
    return statistics.median(samples), min(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--layers", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    torch.manual_seed(0)

    kinds_vocab = len(TOKEN_KINDS)
    print(f"torch {torch.__version__}  threads={args.threads}  device={device}")
    print(f"heads={HEADS}  kind vocabulary={kinds_vocab}  layers={args.layers}\n")

    for batch, length in SHAPES:
        rows = torch.randint(0, 6, (batch, length), device=device)
        cols = torch.randint(0, 9, (batch, length), device=device)
        kinds = torch.randint(0, kinds_vocab, (batch, length), device=device)
        indices = relation_bias_indices(rows, cols)
        narrow = narrow_indices(indices, torch.int32)
        kind_pair_bias = torch.randn(HEADS, kinds_vocab, kinds_vocab, device=device)
        row_w = torch.randn(12, HEADS, device=device)
        col_w = torch.randn(18, HEADS, device=device)
        same_w = torch.randn(2, HEADS, device=device)
        args_tuple = (kinds, indices, kind_pair_bias, row_w, col_w, same_w)
        narrow_args = (kinds, narrow, kind_pair_bias, row_w, col_w, same_w)

        reference = current(*args_tuple)
        variants = {
            "current (4 permutes)": lambda: current(*args_tuple),
            "single permute": lambda: single_permute(*args_tuple),
            "single permute + int32 indices": lambda: single_permute(*narrow_args),
        }
        exact = all(torch.equal(reference, fn()) for fn in variants.values())

        index_mb = batch * length * length * 3 * 8 / 1e6
        print(f"batch={batch:4d}  tokens={length:4d}  "
              f"bias tensor {batch * HEADS * length * length * 4 / 1e6:.1f} MB  "
              f"int64 pair indices {index_mb:.1f} MB  bit-identical={exact}")
        for name, fn in variants.items():
            median, best = timeit(fn, args.repeats)
            print(f"    {name:32s} {median:7.3f} ms   (best {best:7.3f})  "
                  f"x{args.layers} layers = {median * args.layers:7.3f} ms")
        print()


if __name__ == "__main__":
    main()
