#!/usr/bin/env python3
"""task84: build one ablation variant's training pack.

The pack CLI cannot pass read_cache_dir, so a cold run recomputes every episode
feature (hours). This runner calls pack() directly with the recipe's resolved row
options plus the production read cache, exactly like the pipeline's pack step.

Usage: python3 scripts/task84_pack_variant.py {a|b}
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]

from pack_training_rows import pack  # noqa: E402
from pvz_recipe import training_row_options  # noqa: E402
from pvz_training_pack import archive_fingerprint, pack_options  # noqa: E402

variant = sys.argv[1]
recipe = json.loads((ROOT / ("artifacts/task84/variant_%s/recipe.json" % variant)).read_text())
opts = training_row_options(recipe)
data = ROOT / "artifacts/task84/frames"
options = pack_options(set(opts["labels"]), opts["wait_keep"], opts["obs"], opts["cells"],
                       opts["aux_horizon"], opts["packet_head"], opts["with_context"],
                       opts["label_weight"],
                       sources=archive_fingerprint(data, set(opts["labels"])))
started = time.time()
target = pack(data, options,
              cache_dir=Path("/home/newbiexvwu/pvz-logs/curriculum_state.pipeline/cache"),
              precompute_workers=int(recipe.get("precompute_workers", 8)),
              read_cache_dir=Path("/home/newbiexvwu/PvZ-search/artifacts/student/v2_cache"))
print("pack: %s (%.0f s)" % (target, time.time() - started), flush=True)
