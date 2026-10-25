#!/usr/bin/env python3
"""task84: build a disjoint tail shard of a variant's missing label segments.

The main pack() walks labels in sorted order with one memory-capped builder. This
shard takes the tail of the still-missing labels and builds them in reverse, so the
two processes meet in the middle without duplicating work. Segment contracts are
per-label (segment_options), so shards interleave safely with the main pack.

Usage: python3 scripts/task84_pack_shard.py {a|b} [count]
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]

from pack_training_rows import pack  # noqa: E402
from pvz_frame_store import frame_paths  # noqa: E402
from pvz_training_pack import (archive_fingerprint, fingerprint_paths, pack_options,  # noqa: E402
                               segment_options, segment_path)
from pvz_recipe import training_row_options  # noqa: E402

def main():
    variant = sys.argv[1]
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    recipe = json.loads((ROOT / ("artifacts/task84/variant_%s/recipe.json" % variant)).read_text())
    opts = training_row_options(recipe)
    data = ROOT / "artifacts/task84/frames"
    labels = sorted(opts["labels"])
    catalog = {}
    for path in frame_paths(data):
        if path.parent.name in set(labels):
            catalog.setdefault(path.parent.name, []).append(path)
    options = pack_options(set(labels), opts["wait_keep"], opts["obs"], opts["cells"], opts["aux_horizon"],
                           opts["packet_head"], opts["with_context"], opts["label_weight"],
                           sources=archive_fingerprint(data, set(labels)))
    missing = []
    for label in labels:
        contract = segment_options(options, label, fingerprint_paths(data, catalog[label]))
        if not segment_path(data, contract).is_file():
            missing.append(label)
    shard = missing[-count:] if count else missing
    print("missing %d, shard %d: %s..%s" % (len(missing), len(shard), shard[:1], shard[-1:]), flush=True)
    started = time.time()
    target = pack(data, pack_options(set(shard), opts["wait_keep"], opts["obs"], opts["cells"],
                                     opts["aux_horizon"], opts["packet_head"], opts["with_context"],
                                     opts["label_weight"],
                                     sources=archive_fingerprint(data, set(shard))),
                  cache_dir=Path("/home/newbiexvwu/pvz-logs/curriculum_state.pipeline/cache"),
                  precompute_workers=int(recipe.get("precompute_workers", 8)),
                  read_cache_dir=Path("/home/newbiexvwu/PvZ-search/artifacts/student/v2_cache"),
                  quiet=False)
    print("shard done in %.0f s -> %s" % (time.time() - started, target), flush=True)



if __name__ == '__main__':
    main()
