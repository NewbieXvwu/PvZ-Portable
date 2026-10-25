#!/usr/bin/env python3
"""task88 step 4: simulate gate rounds on a copy of the live seed ledger.

Never touches the real ledger: it copies the file into a temp directory, then
allocates `--rounds` blocks of the v3 pair count through the production
`promotion_gate.allocate_seed_block`, and reports every block plus the checks
that matter (pairwise disjoint, inside a registered pool, ordered by window).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]
import pvz_promotion
import promotion_gate


def simulate(ledger_path, rounds, total):
    with tempfile.TemporaryDirectory() as folder:
        copy = Path(folder) / "gate_seed_ledger.json"
        if ledger_path.exists():
            shutil.copyfile(ledger_path, copy)
        before = json.loads(copy.read_text()) if copy.exists() else dict(allocated=[])
        blocks = [promotion_gate.allocate_seed_block(copy, total) for _ in range(rounds)]
        after = json.loads(copy.read_text())
    spans = sorted(blocks)
    disjoint = all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
    in_pool = all(any(low <= a and b <= high for low, high in pvz_promotion.SEED_POOLS_V2)
                  for a, b in blocks)
    windows = [next(index for index, (low, high) in enumerate(pvz_promotion.SEED_POOLS_V2)
                    if low <= a and b <= high) for a, b in blocks]
    return dict(
        ledger=str(ledger_path), rounds=rounds, seeds_per_round=total,
        allocated_before=len(before.get("allocated", [])),
        allocated_after=len(after.get("allocated", [])),
        pools=[list(pool) for pool in pvz_promotion.SEED_POOLS_V2],
        windows_used=windows,
        first_block=list(blocks[0]), last_block=list(blocks[-1]),
        total_seeds=rounds * total,
        pairwise_disjoint=disjoint, every_block_inside_a_pool=in_pool,
        monotone_windows=windows == sorted(windows),
        blocks=[list(block) for block in blocks],
        ledger_written_pools=after.get("pools"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ledger", type=Path,
                        default=Path.home() / "pvz-logs/gate_seed_ledger.json")
    parser.add_argument("--rounds", type=int, default=250)
    parser.add_argument("--seeds", type=int, default=None,
                        help="seeds per round (default: the v3 pair count, 1920)")
    parser.add_argument("--out", type=Path, default=ROOT / "MEMORY/docs/gate-seed-dryrun.json")
    args = parser.parse_args()
    total = args.seeds or pvz_promotion.policy(3)["pair_count"]
    result = simulate(args.ledger, args.rounds, total)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    summary = {key: value for key, value in result.items() if key != "blocks"}
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
