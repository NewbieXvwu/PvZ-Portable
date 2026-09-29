"""Run the fixed T1 victory smoke cases against a local PvZ resource directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from pvz_env import PvZEnv  # noqa: E402
from scripted_baseline import run  # noqa: E402

LEVELS = (1, 2, 7)
SEEDS = (30000, 30001)
MAX_TICK = 120_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-dir", required=True)
    args = parser.parse_args()

    with PvZEnv(args.resource_dir) as env:
        for level in LEVELS:
            for seed in SEEDS:
                try:
                    result = run(env, seed, level)
                except AssertionError as error:
                    raise SystemExit(f"FAIL level {level} seed {seed}: {error}") from error
                print(
                    f"level {level} seed {seed}: won={result['won']} "
                    f"terminal={result['terminal']} tick={result['tick']}"
                )
                if not result["terminal"] or not result["won"]:
                    raise AssertionError(f"level {level} seed {seed} did not win: {result}")
                if result["tick"] > MAX_TICK:
                    raise AssertionError(f"level {level} seed {seed} exceeded {MAX_TICK} ticks")

    print(f"PASS: {len(LEVELS)} levels x {len(SEEDS)} seeds")


if __name__ == "__main__":
    main()
