#!/usr/bin/env python3
"""Convert a supervised BC checkpoint to the explicit research weight-transfer format."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
from pvz_agent_model import GameplayModelV1, model_architecture_version  # noqa: E402
from pvz_common import ENV_PROTOCOL_VERSION, OBSERVATION_VERSION, TASK_VERSION  # noqa: E402
from pvz_value import VALUE_SEMANTICS  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--training-results", type=Path)
    parser.add_argument("--collection-manifest", type=Path)
    args = parser.parse_args()

    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    training_path = (args.training_results or source.with_name("training_results.json")).expanduser().resolve()
    collection_path = (args.collection_manifest or source.with_name("collection_manifest.json")).expanduser().resolve()
    source_checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    required_keys = {"config", "model_state_dict", "epoch", "validation_loss"}
    optional_gru_keys = {"state_dict", "sequence_mode", "sequence_length"}
    if (not required_keys <= set(source_checkpoint)
            or set(source_checkpoint) - required_keys - optional_gru_keys):
        raise ValueError("input is not the declared behavior-cloning checkpoint format")
    results = json.loads(training_path.read_text(encoding="utf-8"))
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    metrics = results["validation_metrics"]["exact_action_by_teacher_type"]
    plant_accuracy = metrics["plant"]["accuracy"]
    sequence_mode = (results["training"].get("sequence_mode")
                     or source_checkpoint.get("sequence_mode"))
    if not sequence_mode:
        raise ValueError("BC checkpoint provenance requires an explicit sequence_mode")
    config = source_checkpoint["config"]
    model = GameplayModelV1(config)
    model.load_state_dict(source_checkpoint["model_state_dict"], strict=True)
    converted = {
        "state_dict": source_checkpoint["model_state_dict"],
        "model_architecture_version": model_architecture_version(config),
        "value_semantics": VALUE_SEMANTICS,
        "config": config,
        "provenance": {
            "protocol_version": ENV_PROTOCOL_VERSION,
            "observation_version": OBSERVATION_VERSION,
            "task_version": TASK_VERSION,
            "search_label_version": None,
            "source_kind": "behavior_cloning",
            "source_path": str(source),
            "source_sha256": sha256(source),
            "training_results_path": str(training_path),
            "training_results_sha256": sha256(training_path),
            "collection_manifest_path": str(collection_path),
            "collection_manifest_sha256": sha256(collection_path),
            "validation_plant_accuracy": plant_accuracy,
            "validation_accuracy": {
                action: value["accuracy"] for action, value in metrics.items()
            },
            "validation_loss": source_checkpoint["validation_loss"],
            "best_epoch": source_checkpoint["epoch"],
            "sequence_mode": sequence_mode,
            "source_training_device": results.get("device", results["training"].get("device")),
            "collection_simulator_sha256": collection["simulator_sha256"],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, output)
    print(json.dumps({"output": str(output), "sha256": sha256(output),
                      "model_architecture_version": converted["model_architecture_version"],
                      "validation_plant_accuracy": plant_accuracy}, indent=2))


if __name__ == "__main__":
    main()
