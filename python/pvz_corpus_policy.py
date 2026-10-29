"""Explicit corpus retirements approved by the main controller, never a missing-file fallback."""

# task165, 2026-10-30: all 256 task20 archives were deleted and have no backup.
# Keep this exact allowlist shared by saved recipes, packer and trainer.
ABANDONED_LABELS = frozenset(('d10orig1_fast_search', 'd10orig2_fast_search'))


def retained_labels(labels):
    return [label for label in labels if label not in ABANDONED_LABELS]
