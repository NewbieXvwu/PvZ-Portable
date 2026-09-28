"""Checkpoint and provenance helpers for search-imitation training."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import sys
from pathlib import Path
from typing import Any

import torch

from pvz_agent_model import GameplayModelV1, MODEL_ARCHITECTURE_VERSION, MODEL_CONFIG
from pvz_common import (
    ENV_PROTOCOL_VERSION,
    OBSERVATION_VERSION,
    TASK_VERSION,
    TRAINING_SEED,
    VALUE_RANGE,
    git_metadata,
    sha256_file,
)
from pvz_env import PlayerProfileContext
from pvz_value import SEARCH_LABEL_VERSION, VALUE_GAMMA, VALUE_SEMANTICS

__all__ = ["checkpoint_metadata", "provenance", "save_checkpoint", "sha256_file", "task_signature"]


def checkpoint_metadata(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Return everything a checkpoint carries except its tensors.

    Both the search-value summary and the benchmark checkpoint record used to
    re-implement this one-liner; they now share it so the two summaries can never
    drift apart.
    """
    return {key: value for key, value in checkpoint.items() if key != "state_dict"}


def task_signature(level: int, deck: list[int] | tuple[int, ...], zombie_count_multiplier: float,
                   resource_dir: str | Path) -> dict[str, Any]:
    resources = Path(resource_dir).expanduser().resolve()
    return {
        "level": level,
        "deck": list(deck),
        "playthrough": 2,
        "profile": asdict(PlayerProfileContext()),
        "zombie_count_multiplier": zombie_count_multiplier,
        "resource_sha256": {
            "main.pak": sha256_file(resources / "main.pak"),
            "properties/partner.xml": sha256_file(resources / "properties" / "partner.xml"),
        },
    }


def _jsonable_cli_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def provenance(args: argparse.Namespace, data_paths: dict[str, Path], train_seeds: list[int],
               dagger_seeds: list[int], device: torch.device) -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    git_sha, git_dirty = git_metadata(root)
    resource_dir = Path(args.resource_dir).expanduser().resolve()
    cli = {key: _jsonable_cli_value(value) for key, value in vars(args).items()}
    cli["resolved_device"] = str(device)
    return {
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "protocol_version": ENV_PROTOCOL_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "task_version": TASK_VERSION,
        "search_label_version": SEARCH_LABEL_VERSION,
        "value_range": list(VALUE_RANGE),
        "value_gamma": VALUE_GAMMA,
        "value_semantics": VALUE_SEMANTICS,
        "command": {"argv": list(sys.argv), "arguments": cli},
        "trajectory_sha256": {name: sha256_file(path) for name, path in data_paths.items()},
        "resource_sha256": {
            "main.pak": sha256_file(resource_dir / "main.pak"),
            "properties/partner.xml": sha256_file(resource_dir / "properties" / "partner.xml"),
        },
        "random_seeds": {
            "python": TRAINING_SEED,
            "torch": TRAINING_SEED,
            "search_episodes": train_seeds,
            "dagger_episodes": dagger_seeds,
        },
        "model_config": MODEL_CONFIG,
    }


def save_checkpoint(path: Path, model: GameplayModelV1, metadata: dict[str, Any], **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model_architecture_version": MODEL_ARCHITECTURE_VERSION,
        "value_semantics": VALUE_SEMANTICS,
        "config": MODEL_CONFIG,
        "level": metadata["command"]["arguments"]["level"],
        "deck": metadata["command"]["arguments"]["deck"],
        "profile": "Adventure-II, six slots, no store items",
        "provenance": metadata,
        **fields,
    }, path)
