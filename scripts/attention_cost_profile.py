"""Profile the cost of GameplayModelV1's relation attention, and test fusion.

Motivation
----------
LLM-side sparse/compressed attention (CSA2, QSA, MSA, SGA, IndexShare, KDA)
targets contexts of 10^5-10^6 tokens where a KV cache grows with the context.
This model's context is ~100 tokens and it has no KV cache at all, so the
question is what actually costs time.  This script answers it with measurements:

  * token count for representative board densities,
  * per-decision breakdown (model forward / tokenization / sampling),
  * inside ``step_tokens``: encoder vs GRU vs heads,
  * inside one attention layer: the six stages,
  * operator count of the relation-bias construction,
  * eager vs ``torch.compile`` on that construction, with a bit-exactness check.

Findings (Apple M5 Pro, CPU, 1 thread, torch 2.13.0, L=102):
  relation-bias build = 67.1% of attention, ~38% of ``step_tokens``,
  117 dispatched operators, and ``torch.compile`` removes 1.86x of it bit-exactly.

Run::

    python scripts/attention_cost_profile.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from pvz_agent_model import (  # noqa: E402
    MODEL_CONFIG,
    GameplayModelV1,
    configure_torch_threads,
    legal_summary,
    observation_tokens,
    pack_tokens,
    relation_bias_from_indices,
    relation_bias_indices,
    select_action,
)

ROW_COUNT, COL_COUNT = 6, 9


def _cell(row: int, col: int) -> dict:
    return {"row": row, "col": col, "terrain": 1, "row_type": 1,
            "plant_types": [], "grid_item_types": []}


def _plant(index: int) -> dict:
    return {"type": index % 8, "imitater_type": -1, "row": index % 6, "col": index % 9,
            "health": 300, "max_health": 300, "state": 0, "state_countdown": 0,
            "launch_counter": 0, "launch_rate": 0, "shooting_counter": 0,
            "wake_up_counter": 0, "asleep": False, "squished": False,
            "bungee_state": 0, "target_zombie_id": -1}


def _zombie(index: int) -> dict:
    return {"type": index % 6, "row": index % 6, "x": 700.0 - index * 12, "y": 90.0,
            "body_health": 200, "body_max_health": 200,
            "helm_health": 0, "helm_max_health": 0,
            "shield_health": 0, "shield_max_health": 0,
            "phase": 0, "phase_counter": 0, "velocity_x": -0.5,
            "chilled": 0, "buttered": 0, "ice_trap": 0,
            "has_head": True, "has_arm": True, "has_object": False,
            "is_eating": False, "target_col": -1, "target_row": -1}


def observation(plants: int = 18, zombies: int = 12, projectiles: int = 6) -> dict:
    """A schema-accurate observation with realistic mid-game entity counts.

    Includes the derived-feature inputs the environment supplies
    (``wave_timer`` and ``sun_income_rate``), which ``derive_observation_features``
    reads directly.
    """
    return {
        "terminal": False, "result": 0, "tick": 3000, "wave": 8, "wave_count": 20,
        "sun": 425, "night": False, "pool": False, "fog": False, "roof": False,
        "zombie_count_multiplier": 1.0,
        "wave_timer": 1800,
        "sun_income_rate": 41.7,
        "player_profile": {
            "playthrough": 2, "seed_slot_count": 6, "owned_upgrade_plants": [],
            "imitater_owned": False, "first_aid_owned": False,
            "pool_cleaner_owned": False, "roof_cleaner_owned": False,
            "rake_charges_remaining": 0,
        },
        "cells": [_cell(r, c) for r in range(ROW_COUNT) for c in range(COL_COUNT)],
        "plants": [_plant(i) for i in range(plants)],
        "zombies": [_zombie(i) for i in range(zombies)],
        "projectiles": [{"type": 0, "row": i % 6, "x": 420.0, "y": 90.0, "z": 0.0,
                         "vx": 6.0, "vy": 0.0, "vz": 0.0, "motion": 0,
                         "damage": 20, "age": 10, "target_zombie_id": 1}
                        for i in range(projectiles)],
        "defenses": [{"type": 0, "row": 0, "state": 1, "x": 40.0, "y": 30.0}],
        "grid_items": [],
        "packets": [{"index": i, "type": i, "imitater_type": -1, "cost": 100,
                     "cooldown": 0, "refresh_time": 3000, "active": True}
                    for i in range(6)],
        "loadout_context": {"zombie_roster": [0, 1, 2]},
        "legal_actions": {
            "plants": [{"packet": p, "row": 1, "col": 2 + p} for p in range(3)],
            "shovels": [[3, 2]], "wait": True,
        },
    }


def build_indices(rows, cols):
    """The layer-independent pair indices, as production computes them."""
    return relation_bias_indices(rows, cols)


def build_relation(att, kinds, rows, cols, indices=None):
    """Production's relation-bias assembly, called through the real function.

    This used to be a hand-written copy of ``RelationAttention``'s dense path.
    A copy drifts: it kept reporting the pre-2026 four-``permute`` cost long
    after production had collapsed them into one, so the profile silently
    measured code that no longer ran.  It now calls production.
    """
    if indices is None:
        indices = relation_bias_indices(rows, cols)
    return relation_bias_from_indices(
        kinds, indices.row_bucket, indices.col_bucket, indices.same_cell,
        att.kind_pair_bias, att.row_bias.weight, att.col_bias.weight,
        att.same_cell_bias.weight)


def timeit(fn, repeats: int = 300) -> float:
    fn()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / repeats


def main() -> None:
    configure_torch_threads(1)
    torch.manual_seed(0)
    model = GameplayModelV1().eval()
    width, heads, layers = MODEL_CONFIG["width"], MODEL_CONFIG["heads"], MODEL_CONFIG["layers"]
    ff = MODEL_CONFIG["ff_width"]
    head_width = width // heads

    print(f"torch {torch.__version__}  threads=1  cpu")
    print(f"params {sum(p.numel() for p in model.parameters()):,}   config {MODEL_CONFIG}")
    print()

    print("--- context size vs board density ---")
    for plants, zombies in ((1, 1), (18, 12), (24, 20)):
        tensors, metadata = observation_tokens(observation(plants, zombies))
        print(f"  {plants:2d} plants / {zombies:2d} zombies -> L = {pack_tokens(tensors, metadata)['ids'].shape[0]}")
    print()

    obs = observation()
    tensors, metadata = observation_tokens(obs)
    legal = legal_summary(obs["legal_actions"])
    L = pack_tokens(tensors, metadata)["ids"].shape[0]
    wave = obs["wave"]

    with torch.no_grad():
        output = model.step_tokens(tensors, metadata, wave)

        print(f"--- per-decision breakdown (L={L}) ---")
        parts = {
            "step_tokens (model forward)": timeit(lambda: model.step_tokens(tensors, metadata, wave)),
            "observation_tokens": timeit(lambda: observation_tokens(obs), 2000),
            "select_action": timeit(lambda: select_action(model, output, legal)),
            "legal_summary": timeit(lambda: legal_summary(obs["legal_actions"]), 2000),
        }
        total = sum(parts.values())
        for name, value in sorted(parts.items(), key=lambda kv: -kv[1]):
            print(f"  {name:28s} {value * 1e3:8.3f} ms  {value / total * 100:5.1f}%")
        print(f"  {'TOTAL':28s} {total * 1e3:8.3f} ms")
        print()

        def encoder_only():
            x = model.kind_embedding(tensors["kinds"]) + model.category_embedding(tensors["categories"])
            x = x + model.variant_embedding(tensors["variants"])
            x = x + model.feature_projection(tensors["features"])
            x = x + model.row_embedding((tensors["rows"] + 1).clamp(0, 7))
            x = x + model.col_embedding((tensors["cols"] + 1).clamp(0, 10))
            x = x.unsqueeze(0)
            for layer in model.encoder:
                x = layer(x, tensors["kinds"], tensors["rows"], tensors["cols"])
            return model.encoder_norm(x)

        encoder = timeit(encoder_only)
        hidden = torch.zeros(MODEL_CONFIG["gru_layers"], 1, MODEL_CONFIG["gru_width"])
        recurrent_input = torch.randn(1, 1, width + 128)
        gru = timeit(lambda: model.belief(recurrent_input, hidden))
        step = parts["step_tokens (model forward)"]
        print("--- inside step_tokens ---")
        print(f"  encoder ({layers} relation layers)  {encoder * 1e3:8.3f} ms  {encoder / step * 100:5.1f}%")
        print(f"  GRU                          {gru * 1e3:8.3f} ms  {gru / step * 100:5.1f}%")
        print(f"  embeddings + action heads    {(step - encoder - gru) * 1e3:8.3f} ms"
              f"  {(step - encoder - gru) / step * 100:5.1f}%")
        print()

        layer = model.encoder[0]
        att = layer.attention
        normed = layer.attention_norm(
            (model.kind_embedding(tensors["kinds"])
             + model.category_embedding(tensors["categories"])
             + model.variant_embedding(tensors["variants"])
             + model.feature_projection(tensors["features"])
             + model.row_embedding((tensors["rows"] + 1).clamp(0, 7))
             + model.col_embedding((tensors["cols"] + 1).clamp(0, 10))).unsqueeze(0))

        kinds = tensors["kinds"].unsqueeze(0)
        rows = tensors["rows"].unsqueeze(0)
        cols = tensors["cols"].unsqueeze(0)

        qkv = att.qkv(normed).view(1, L, 3, heads, head_width).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        scores = torch.matmul(query, key.transpose(-2, -1)) * (head_width ** -0.5)
        # Production computes the pair indices once per forward and shares them
        # across all four layers, so the assembly is measured with them given.
        indices = relation_bias_indices(rows, cols)
        relation = build_relation(att, kinds, rows, cols, indices)
        attended = torch.softmax(scores + relation, dim=-1)
        av = torch.matmul(attended, value).transpose(1, 2).contiguous().view(1, L, width)

        stages = {
            "pair indices (once/forward)": timeit(lambda: build_indices(rows, cols)),
            "bias assembly (per layer)": timeit(
                lambda: build_relation(att, kinds, rows, cols, indices)),
            "softmax": timeit(lambda: torch.softmax(scores + relation, dim=-1)),
            "QKV projection": timeit(lambda: att.qkv(normed)),
            "score matmul QK^T": timeit(
                lambda: torch.matmul(query, key.transpose(-2, -1)) * (head_width ** -0.5)),
            "AV matmul": timeit(lambda: torch.matmul(attended, value)),
            "output projection": timeit(lambda: att.projection(av)),
        }
        stage_total = sum(stages.values())
        print(f"--- inside one attention layer (L={L}, heads={heads}, head_width={head_width}) ---")
        for name, value in sorted(stages.items(), key=lambda kv: -kv[1]):
            print(f"  {name:22s} {value * 1e3:7.3f} ms  {value / stage_total * 100:5.1f}%")
        print(f"  {'SUM':22s} {stage_total * 1e3:7.3f} ms")
        print()

        macs_attention = heads * L * L * head_width * 2
        macs_ffn = L * width * ff * 2 * 3
        bias_share = stages["bias assembly (per layer)"] / stage_total * 100
        print("--- attention is time-bound, not compute-bound ---")
        print(f"  attention MACs/layer {macs_attention:>12,}  ({macs_attention / (macs_attention + macs_ffn) * 100:.1f}% of layer MACs)")
        print(f"  FFN MACs/layer       {macs_ffn:>12,}")
        print(f"  bias assembly is {bias_share:.1f}% of the layer's attention time while "
              f"attention matmuls are {macs_attention / (macs_attention + macs_ffn) * 100:.1f}% of its MACs")
        print()

        print("--- operator count of one bias assembly (indices given) ---")
        from torch.profiler import ProfilerActivity, profile

        build_relation(att, kinds, rows, cols, indices)
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            build_relation(att, kinds, rows, cols, indices)
        counts = {e.key: e.count for e in prof.key_averages() if e.count}
        print(f"  dispatched operators: {sum(counts.values())}")
        for key, count in sorted(counts.items(), key=lambda kv: -kv[1])[:10]:
            print(f"    {count:>4} x {key[:66]}")
        print()

        print("--- operator count of one pair-index build (per forward) ---")
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            build_indices(rows, cols)
        counts = {e.key: e.count for e in prof.key_averages() if e.count}
        print(f"  dispatched operators: {sum(counts.values())}")
        for key, count in sorted(counts.items(), key=lambda kv: -kv[1])[:10]:
            print(f"    {count:>4} x {key[:66]}")
        print()

        print("--- eager vs torch.compile on the bias assembly ---")
        eager = build_relation(att, kinds, rows, cols, indices)
        t_eager = timeit(lambda: build_relation(att, kinds, rows, cols, indices))
        try:
            compiled = torch.compile(build_relation, dynamic=True)
            compiled(att, kinds, rows, cols)
            fused = compiled(att, kinds, rows, cols)
            t_fused = timeit(lambda: compiled(att, kinds, rows, cols), 200)
            print(f"  eager     {t_eager * 1e3:7.3f} ms")
            print(f"  compiled  {t_fused * 1e3:7.3f} ms   speedup {t_eager / t_fused:.2f}x")
            print(f"  torch.equal(eager, compiled) = {torch.equal(eager, fused)}"
                  f"   max|diff| = {(eager - fused).abs().max().item():.3e}")
            saved = layers * (t_eager - t_fused)
            print(f"  projected saving: {saved * 1e3:.3f} ms per decision"
                  f"  ({saved / total * 100:.1f}% of per-decision cost)")
        except Exception as exc:  # noqa: BLE001
            print(f"  torch.compile unavailable: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
