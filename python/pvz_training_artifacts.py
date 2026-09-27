"""Checkpoint and provenance helpers for search-imitation training."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from pvz_agent_model import GameplayModelV1, MODEL_ARCHITECTURE_VERSION, MODEL_CONFIG
from pvz_value import SEARCH_LABEL_VERSION, VALUE_GAMMA, VALUE_SEMANTICS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance(args: argparse.Namespace, data_paths: dict[str, Path], train_seeds: list[int],
               dagger_seeds: list[int], device: torch.device) -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    try:
        git_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                                 capture_output=True, text=True).stdout.strip()
        git_dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True,
                                        capture_output=True, text=True).stdout)
    except (OSError, subprocess.CalledProcessError):
        git_sha, git_dirty = None, None
    resource_dir = Path(args.resource_dir).expanduser().resolve()
    cli = vars(args).copy()
    cli["output_dir"] = str(args.output_dir)
    cli["resolved_device"] = str(device)
    return {
        "git_sha": git_sha, "git_dirty": git_dirty,
        "observation_version": 2, "task_version": 2, "search_label_version": SEARCH_LABEL_VERSION,
        "value_range": [-1, 1], "value_gamma": VALUE_GAMMA, "value_semantics": VALUE_SEMANTICS,
        "command": {"argv": list(sys.argv), "arguments": cli},
        "trajectory_sha256": {name: sha256_file(path) for name, path in data_paths.items()},
        "resource_sha256": {
            "main.pak": sha256_file(resource_dir / "main.pak"),
            "properties/partner.xml": sha256_file(resource_dir / "properties" / "partner.xml"),
        },
        "random_seeds": {"python": 17, "torch": 17, "search_episodes": train_seeds,
                         "dagger_episodes": dagger_seeds},
        "model_config": MODEL_CONFIG,
    }


def save_checkpoint(path: Path, model: GameplayModelV1, metadata: dict[str, Any], **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "model_architecture_version": MODEL_ARCHITECTURE_VERSION,
                "value_semantics": VALUE_SEMANTICS,
                "config": MODEL_CONFIG, "level": metadata["command"]["arguments"]["level"],
                "deck": metadata["command"]["arguments"]["deck"],
                "profile": "Adventure-II, six slots, no store items", "provenance": metadata, **fields}, path)
