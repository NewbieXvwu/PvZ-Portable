"""Equivalence tests for the hoisted/fused relation-bias path.

The encoder now computes the relation-bias indices once and reuses them across
layers, and can optionally fuse the bias assembly with ``torch.compile``.  Both
changes must be bit-identical to the original per-layer inline construction: the
bias tables are per-layer parameters, so only the *indices* are shareable, and
sharing them is a pure code-motion change.

These tests pin that with ``torch.equal`` (not a tolerance) because a tolerance
would hide exactly the kind of silent drift this refactor could introduce.
"""

from __future__ import annotations

import os
from pathlib import Path
import unittest
from unittest import mock

import torch

from pvz_agent_model import (
    RELATION_BIAS_FUSION,
    GameplayModelV1,
    RelationBiasIndices,
    configure_torch_threads,
    env_flag,
    fused_relation_bias,
    relation_bias_from_indices,
    relation_bias_indices,
    set_relation_bias_fusion,
    use_fused_relation_bias,
    observation_tokens,
    pack_tokens,
    unpack_tokens,
)
from test_agent_model import observation


def _reference_relation(attention, kinds, rows, cols):
    """The original per-layer inline construction, verbatim."""
    kind_pair = attention.kind_pair_bias[:, kinds[:, :, None], kinds[:, None, :]]
    row_known = (rows[:, :, None] >= 0) & (rows[:, None, :] >= 0)
    col_known = (cols[:, :, None] >= 0) & (cols[:, None, :] >= 0)
    row_delta = (rows[:, :, None] - rows[:, None, :]).clamp(-5, 5) + 5
    col_delta = (cols[:, :, None] - cols[:, None, :]).clamp(-8, 8) + 8
    row_bucket = torch.where(row_known, row_delta, 11)
    col_bucket = torch.where(col_known, col_delta, 17)
    same_cell = (row_known & col_known & (rows[:, :, None] == rows[:, None, :])
                 & (cols[:, :, None] == cols[:, None, :])).long()
    relation = kind_pair
    relation = relation + attention.row_bias(row_bucket).permute(3, 0, 1, 2)
    relation = relation + attention.col_bias(col_bucket).permute(3, 0, 1, 2)
    relation = relation + attention.same_cell_bias(same_cell).permute(3, 0, 1, 2)
    return relation.permute(1, 0, 2, 3), row_bucket, col_bucket, same_cell


class RelationBiasIndicesTests(unittest.TestCase):
    def setUp(self) -> None:
        configure_torch_threads(1)
        torch.manual_seed(0)
        self.model = GameplayModelV1().eval()
        self.attention = self.model.encoder[0].attention
        tensors, metadata = observation_tokens(observation())
        self.tensors = tensors
        self.metadata = metadata
        self.kinds = tensors["kinds"].unsqueeze(0)
        self.rows = tensors["rows"].unsqueeze(0)
        self.cols = tensors["cols"].unsqueeze(0)

    def test_indices_match_the_inline_construction(self) -> None:
        _, row_bucket, col_bucket, same_cell = _reference_relation(
            self.attention, self.kinds, self.rows, self.cols)
        indices = relation_bias_indices(self.rows, self.cols)
        self.assertTrue(torch.equal(indices.row_bucket, row_bucket))
        self.assertTrue(torch.equal(indices.col_bucket, col_bucket))
        self.assertTrue(torch.equal(indices.same_cell, same_cell))

    def test_bias_tensor_is_bit_identical(self) -> None:
        expected, row_bucket, col_bucket, same_cell = _reference_relation(
            self.attention, self.kinds, self.rows, self.cols)
        actual = relation_bias_from_indices(
            self.kinds, row_bucket, col_bucket, same_cell,
            self.attention.kind_pair_bias, self.attention.row_bias.weight,
            self.attention.col_bias.weight, self.attention.same_cell_bias.weight)
        self.assertTrue(torch.equal(expected, actual))

    def test_indices_are_independent_of_layer_parameters(self) -> None:
        """Two layers with different bias tables must receive the same indices."""
        second = self.model.encoder[1].attention
        with torch.no_grad():
            second.row_bias.weight.add_(1.5)
            second.col_bias.weight.add_(-0.75)
            second.same_cell_bias.weight.add_(2.0)
        indices = relation_bias_indices(self.rows, self.cols)
        expected_second, row_bucket, col_bucket, same_cell = _reference_relation(
            second, self.kinds, self.rows, self.cols)
        actual_second = relation_bias_from_indices(
            self.kinds, indices.row_bucket, indices.col_bucket, indices.same_cell,
            second.kind_pair_bias, second.row_bias.weight,
            second.col_bias.weight, second.same_cell_bias.weight)
        self.assertTrue(torch.equal(expected_second, actual_second))


class RelationBiasFusionTests(unittest.TestCase):
    def setUp(self) -> None:
        configure_torch_threads(1)
        torch.manual_seed(0)
        self.model = GameplayModelV1().eval()
        self.attention = self.model.encoder[0].attention
        tensors, metadata = observation_tokens(observation())
        self.kinds = tensors["kinds"].unsqueeze(0)
        self.rows = tensors["rows"].unsqueeze(0)
        self.cols = tensors["cols"].unsqueeze(0)
        self.addCleanup(set_relation_bias_fusion, False)

    def test_fused_path_is_bit_identical_when_available(self) -> None:
        if fused_relation_bias() is None:
            self.skipTest("torch.compile unavailable in this build")
        indices = relation_bias_indices(self.rows, self.cols)
        eager = relation_bias_from_indices(
            self.kinds, indices.row_bucket, indices.col_bucket, indices.same_cell,
            self.attention.kind_pair_bias, self.attention.row_bias.weight,
            self.attention.col_bias.weight, self.attention.same_cell_bias.weight)
        fused = fused_relation_bias()(
            self.kinds, indices.row_bucket, indices.col_bucket, indices.same_cell,
            self.attention.kind_pair_bias, self.attention.row_bias.weight,
            self.attention.col_bias.weight, self.attention.same_cell_bias.weight)
        self.assertTrue(torch.equal(eager, fused))
        self.assertEqual((eager - fused).abs().max().item(), 0.0)

    def test_enabling_fusion_does_not_change_model_output(self) -> None:
        tensors, metadata = observation_tokens(observation())
        packed = pack_tokens(tensors, metadata)
        with torch.no_grad():
            set_relation_bias_fusion(False)
            plain = self.model.step_tokens(tensors, metadata, 3)
            set_relation_bias_fusion(True)
            restored, _ = unpack_tokens(packed, torch.device("cpu"))
            fused = self.model.step_tokens(restored, metadata, 3)
        self.assertTrue(torch.equal(plain["belief"], fused["belief"]))
        self.assertTrue(torch.equal(plain["packet_logits"], fused["packet_logits"]))
        self.assertTrue(torch.equal(plain["cell_tokens"], fused["cell_tokens"]))
        self.assertTrue(torch.equal(plain["cell_keys"], fused["cell_keys"]))
        self.assertTrue(torch.equal(plain["value"], fused["value"]))


class RelationBiasFusionDefaultTests(unittest.TestCase):
    """The fusion is on by default; the env var is the documented escape hatch."""

    def test_env_flag_accepts_the_usual_off_spellings(self) -> None:
        for raw in ("", "0", "false", "FALSE", " no ", "off"):
            with self.subTest(raw=raw):
                with mock.patch.dict(os.environ, {"PVZ_TEST_FLAG": raw}):
                    self.assertFalse(env_flag("PVZ_TEST_FLAG", default=True))
        for raw in ("1", "true", "yes", "on"):
            with self.subTest(raw=raw):
                with mock.patch.dict(os.environ, {"PVZ_TEST_FLAG": raw}):
                    self.assertTrue(env_flag("PVZ_TEST_FLAG", default=True))

    def test_env_flag_falls_back_to_the_default_when_unset(self) -> None:
        with mock.patch.dict(os.environ):
            os.environ.pop("PVZ_TEST_FLAG", None)
            self.assertTrue(env_flag("PVZ_TEST_FLAG", default=True))
            self.assertFalse(env_flag("PVZ_TEST_FLAG", default=False))

    def test_fusion_default_is_wired_to_on(self) -> None:
        """A default-off fusion would silently lose the measured 1.3-1.4x."""
        source = (Path(__file__).resolve().parent / "pvz_agent_model.py").read_text()
        self.assertIn('env_flag("PVZ_RELATION_BIAS_FUSION", default=True)', source)
        # The module-level value must be exactly what that call would produce, so an
        # ambient PVZ_RELATION_BIAS_FUSION=0 in the test environment cannot mask a
        # regression in the wiring.
        self.assertEqual(
            RELATION_BIAS_FUSION,
            env_flag("PVZ_RELATION_BIAS_FUSION", default=True))

    def test_disabling_the_fusion_restores_the_eager_chain(self) -> None:
        try:
            self.assertFalse(set_relation_bias_fusion(False))
            self.assertFalse(use_fused_relation_bias())
        finally:
            set_relation_bias_fusion(True)


class EncoderHoistingEquivalenceTests(unittest.TestCase):
    """The hoisted path must match a model that computes indices per layer."""

    def test_step_tokens_matches_per_layer_index_computation(self) -> None:
        configure_torch_threads(1)
        torch.manual_seed(0)
        model = GameplayModelV1().eval()
        tensors, metadata = observation_tokens(observation())
        kinds = tensors["kinds"].unsqueeze(0)
        rows = tensors["rows"].unsqueeze(0)
        cols = tensors["cols"].unsqueeze(0)

        with torch.no_grad():
            hoisted = model.step_tokens(tensors, metadata, 3)

            # Re-run the encoder the old way: indices recomputed inside each layer.
            x = model.kind_embedding(tensors["kinds"]) + model.category_embedding(tensors["categories"])
            x = x + model.variant_embedding(tensors["variants"])
            x = x + model.feature_projection(tensors["features"])
            x = x + model.row_embedding((tensors["rows"] + 1).clamp(0, 7))
            x = x + model.col_embedding((tensors["cols"] + 1).clamp(0, 10))
            x = x.unsqueeze(0)
            for layer in model.encoder:
                indices = relation_bias_indices(rows, cols)
                x = layer(x, tensors["kinds"], tensors["rows"], tensors["cols"],
                          indices=indices)
            manual = model.encoder_norm(x)

        # cell_tokens / packet_tokens are the encoder output gathered at the 54
        # cell and the active packet positions, so comparing them compares the
        # encoder result for those tokens directly.
        cell_ids = [metadata["cell_tokens"][cell] for cell in range(54)]
        packet_ids = sorted(metadata["packet_tokens"])
        packet_token_ids = [metadata["packet_tokens"][packet] for packet in packet_ids]
        self.assertTrue(torch.equal(manual[0, cell_ids], hoisted["cell_tokens"]))
        self.assertTrue(torch.equal(manual[0, packet_token_ids], hoisted["packet_tokens"]))
        self.assertEqual(manual.shape[1], pack_tokens(tensors, metadata)["ids"].shape[0])


if __name__ == "__main__":
    unittest.main()
