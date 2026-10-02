"""Explicit new-stage weight transfer and self-contained stage continuation."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

import torch

from pvz_agent_model import GameplayModelV1, configure_torch_threads, model_architecture_version
from pvz_common import sha256_file
from pvz_initialization import transfer_weights, validate_transfer
from pvz_research import ROOT, capture_rng, load_config, restore_rng

SMALL = dict(layers=1, width=32, heads=4, ff_width=64, gru_layers=1, gru_width=16,
             critic_width=24, critic_layers=1, input_flags=0)


class InitializationTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_torch_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.source = Path(self.temporary.name) / "parent.pt"
        torch.manual_seed(3)
        self.parent = GameplayModelV1(SMALL).eval()
        optimizer = torch.optim.AdamW(self.parent.parameters(), lr=.01)
        sum(p.square().sum() for p in self.parent.parameters()).backward(); optimizer.step()
        self.checkpoint = dict(state_dict=self.parent.state_dict(), optimizer_state_dict=optimizer.state_dict(),
            config=self.parent.config, model_architecture_version=model_architecture_version(self.parent.config),
            research_version=1, experiment_identity="parent-identity",
            experiment_config=dict(initialization_seed=3), provenance=dict(commit="parent-source"),
            training_state=dict(phase="ready", experiment_id="parent", updates=23,
                counters=dict(episodes=391, decisions=47856, ticks=4964050), wall_seconds=2361.79))
        self.save_parent()
        self.settings = dict(method="weights_transfer_v1", source_checkpoint=str(self.source),
                             source_sha256=sha256_file(self.source), note="New complete-level curriculum")

    def save_parent(self):
        torch.save(self.checkpoint, self.source)

    def test_all_parameters_transfer_and_new_optimizer_and_rng_stay_fresh(self):
        torch.manual_seed(71)
        model = GameplayModelV1(SMALL).eval()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        before_rng = torch.get_rng_state().clone()
        record = transfer_weights(model, self.settings, self.source)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, self.parent.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(optimizer.state, {})
        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-4)
        torch.testing.assert_close(before_rng, torch.get_rng_state(), rtol=0, atol=0)
        self.assertEqual(record["source_counters"], self.checkpoint["training_state"]["counters"])
        record["source_counters"]["decisions"] = 0
        self.assertEqual(self.checkpoint["training_state"]["counters"]["decisions"], 47856)

    def test_config_keeps_transfer_explicit_but_parent_is_not_needed_to_parse_resume(self):
        config = json.loads((ROOT / "experiments/t5/frontier_v1/seed0.json").read_text())
        config["initialization"] = self.settings
        path = Path(self.temporary.name) / "config.json"; path.write_text(json.dumps(config))
        self.source.unlink()  # A temporary fixture, never a real research checkpoint.
        loaded, _, _ = load_config(path)
        self.assertEqual(loaded["initialization"], self.settings)
        del config["initialization"]["source_sha256"]
        path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "weight transfer"):
            load_config(path)

    def test_source_identity_and_model_semantics_must_match(self):
        for field, value in (("config", {**self.parent.config, "input_flags": 7}),
                             ("model_architecture_version", 1), ("research_version", 0)):
            original = self.checkpoint[field]; self.checkpoint[field] = value; self.save_parent()
            settings = {**self.settings, "source_sha256": sha256_file(self.source)}
            with self.assertRaisesRegex(ValueError, "same-configuration"):
                transfer_weights(GameplayModelV1(SMALL), settings, self.source)
            self.checkpoint[field] = original

    def test_changed_source_or_unfinished_boundary_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "differs from frozen config"):
            transfer_weights(GameplayModelV1(SMALL), {**self.settings, "source_sha256": "0" * 64}, self.source)
        self.checkpoint["training_state"]["phase"] = "partial_optimizer_update"; self.save_parent()
        with self.assertRaisesRegex(ValueError, "complete research boundary"):
            transfer_weights(GameplayModelV1(SMALL), {**self.settings, "source_sha256": sha256_file(self.source)}, self.source)

    def test_complete_stage_checkpoint_continues_without_parent_file(self):
        model = GameplayModelV1(SMALL).eval()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        record = transfer_weights(model, self.settings, self.source)
        stage_rng = random.Random(91)
        state = dict(updates=0, counters=dict(episodes=0, decisions=0, ticks=0), initialization_provenance=record)
        self.assertEqual(state["counters"]["decisions"], 0)
        sum(p.square().sum() for p in model.parameters()).backward(); optimizer.step(); optimizer.zero_grad()
        stage = Path(self.temporary.name) / "stage.pt"
        torch.save(dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(), rng=capture_rng(stage_rng), state=state), stage)
        sum(p.square().sum() for p in model.parameters()).backward(); optimizer.step()
        expected = copy.deepcopy(model.state_dict()); expected_draw = stage_rng.random()
        self.source.unlink()
        saved = torch.load(stage, weights_only=False)
        restored = GameplayModelV1(SMALL).eval(); restored.load_state_dict(saved["state_dict"])
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=.5); restored_optimizer.load_state_dict(saved["optimizer"])
        restored_rng = random.Random(); restore_rng(saved["rng"], restored_rng)
        sum(p.square().sum() for p in restored.parameters()).backward(); restored_optimizer.step()
        for key, value in restored.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
        self.assertEqual(restored_rng.random(), expected_draw)
        self.assertEqual(saved["state"]["initialization_provenance"]["source_counters"]["decisions"], 47856)

    def test_missing_or_implicit_source_and_note_are_rejected(self):
        for values in ({"note": ""}, {"source_checkpoint": ""}, {"source_sha256": "not-a-digest"},
                       {"method": "reuse_last_run"}):
            with self.assertRaises(ValueError):
                validate_transfer({**self.settings, **values})


if __name__ == "__main__":
    unittest.main()
