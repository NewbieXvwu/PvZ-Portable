"""Dependency-free constants and helpers shared by every layer.

This module deliberately imports nothing from the rest of the project: it is the
bottom of the dependency graph, so ``pvz_env`` (which must not import the training
modules) and every training entry point can share one definition of the protocol
version and the OOM badness knobs.
"""

from __future__ import annotations

from pathlib import Path

ENV_PROTOCOL_VERSION = 4

# OOM badness chosen by role.  The kernel kills the highest oom_score_adj first when memory runs
# out, so collection (the cheapest work to lose) goes before training, and both go before the
# controller and the WSL entry point (init, sshd, tmux keep the default 0).
OOM_SCORE_WORKER = 1000
OOM_SCORE_CONTROLLER = 500
OOM_SCORE_TRAINING = 300


def set_oom_score_adj(score: int) -> bool:
    """Raise this process's OOM badness; returns False where the kernel has no such knob.

    Raising one's own score needs no privilege.  macOS has no /proc and Windows has no
    oom_score_adj, so the write is simply skipped there (task34).
    """
    try:
        Path("/proc/self/oom_score_adj").write_text(f"{score}\n")
    except OSError:
        return False
    return True
