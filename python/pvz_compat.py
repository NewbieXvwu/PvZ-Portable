"""Minimal registry of the backward-compatibility paths still in the tree.

task33 (branch `audit`, commit 10df7165) builds the full registry — every LEGACY(<id>) marker plus
every EXPERIMENT(<id>) switch, checked both ways by python/test_compat_registry.py. That file is not
on this branch yet, so this is the smallest version that keeps the one legacy path task32's archive
conversion creates: **reading** frames.jsonl. When task33 merges, its file replaces this one; the
only entry here is already worded the way task33 words it.

Nothing here has side effects: it is data plus two helpers.
"""
from __future__ import annotations

from datetime import date

# id -> {what, kept_for, since, remove_when, where}
LEGACY: dict[str, dict] = {
    "jsonl-archives": {
        "what": "frames.jsonl 旧档案的读取路径（写入分支只服务它）",
        "kept_for": "2026-10-22 以前写成 JSONL 的档案：台式机上尚未转换的存量、HF 上传的旧包、"
                    "以及 whatif 重放要读的旧存档（AGENTS §8 第 4 条）。Mac 上的 6 659 个档案已转 npz，"
                    "但读取路径不随本地文件消失而删除",
        "since": "2026-10-01",
        "remove_when": "所有机器与 HF 上都不再有 frames.jsonl，且没有 whatif 存档需要重放",
        "where": ["python/pvz_frame_store.py:frame_paths", "python/pvz_frame_store.py:frame_bytes"],
    },
}

EXPERIMENTAL_FLAGS: dict[str, dict] = {}


def legacy_ids() -> set[str]:
    return set(LEGACY)


def expired(today: date | None = None) -> list[str]:
    """Experiment switches whose decide-by date has passed: decide or delete."""
    today = today or date.today()
    return [key for key, entry in EXPERIMENTAL_FLAGS.items()
            if date.fromisoformat(entry["decide_by"]) < today]
