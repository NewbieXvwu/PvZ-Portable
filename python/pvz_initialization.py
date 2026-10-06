"""Explicit weight transfer into a new research stage; complete resume is separate."""
from __future__ import annotations

import copy
from pathlib import Path
import re
from typing import Any

import torch

from pvz_agent_model import GameplayModelV1, model_architecture_version
from pvz_common import sha256_file


def validate_transfer(settings: dict[str, Any]) -> None:
    if (set(settings) != {"method", "source_checkpoint", "source_sha256", "note"}
            or settings["method"] != "weights_transfer_v1"
            or not isinstance(settings["source_checkpoint"], str) or not settings["source_checkpoint"]
            or not isinstance(settings["note"], str) or not settings["note"].strip()
            or not isinstance(settings["source_sha256"], str)
            or re.fullmatch("[0-9a-f]{64}", settings["source_sha256"]) is None):
        raise ValueError("weight transfer needs explicit source checkpoint, SHA256 and stage note")


def transfer_weights(model: GameplayModelV1, settings: dict[str, Any], source: Path) -> dict:
    """Load every same-configuration parameter; leave new optimizer/RNG untouched.

    A stage resume uses its own complete checkpoint and does not call this
    function. It therefore does not depend on the parent file being available.
    """
    validate_transfer(settings)
    if sha256_file(source) != settings["source_sha256"]:
        raise ValueError("weight-transfer source checkpoint differs from frozen config")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    provenance = checkpoint.get("provenance", {})
    source_kind = ("research" if checkpoint.get("research_version") == 1
                   else provenance.get("source_kind"))
    supported_source = (source_kind == "research" or source_kind == "behavior_cloning")
    if (not supported_source or checkpoint.get("config") != model.config
            or checkpoint.get("model_architecture_version") != model_architecture_version(model.config)):
        raise ValueError("weight transfer requires a same-configuration research or declared BC checkpoint")
    if source_kind == "behavior_cloning" and not all(
            provenance.get(key) is not None for key in
            ("source_path", "source_sha256", "validation_plant_accuracy", "sequence_mode")):
        raise ValueError("BC checkpoint provenance is incomplete")
    state = checkpoint.get("training_state")
    if source_kind == "research":
        if state is None or state["phase"] not in {"ready", "pending_evaluation", "initial_evaluation"}:
            raise ValueError("weight transfer source is not a complete research boundary")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    result = {"kind": "weights_transfer_v1", "source_kind": source_kind,
            "source_checkpoint": str(source),
            "source_sha256": settings["source_sha256"], "note": settings["note"],
            "source_provenance": copy.deepcopy(provenance),
            "new_stage_state": "new AdamW, declared RNG seeds, zero stage counters and new curriculum; source budget recorded separately"}
    if source_kind == "research":
        result.update({"source_experiment_id": state["experiment_id"],
                       "source_experiment_identity": checkpoint["experiment_identity"],
                       "source_initialization_seed": checkpoint["experiment_config"]["initialization_seed"],
                       "source_updates": state["updates"], "source_counters": copy.deepcopy(state["counters"]),
                       "source_wall_seconds": state["wall_seconds"],
                       "source_commit": provenance["commit"],
                       "source_initialization_provenance": copy.deepcopy(state.get("initialization_provenance"))})
    else:
        result.update({"source_validation_plant_accuracy": provenance.get("validation_plant_accuracy"),
                       "source_validation_accuracy": provenance.get("validation_accuracy"),
                       "source_sequence_mode": provenance.get("sequence_mode")})
    return result
