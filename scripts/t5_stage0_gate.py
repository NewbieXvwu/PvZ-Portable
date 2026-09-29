"""Executable pass/fail gate for the T5 stage-0 learning signal."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from pvz_common import git_metadata  # noqa: E402
from pvz_seed_jobs import atomic_json  # noqa: E402


STATE_PATH = ROOT / "artifacts/t5/training_state.json"
GATE_PATH = ROOT / "artifacts/t5/stage0_gate.json"
MINIMUM_SAMPLES = 5 * 64
MINIMUM_PASS_RATE = 0.50


def evaluate(state_path: Path = STATE_PATH, gate_path: Path = GATE_PATH) -> dict:
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    evaluations = state.get("evaluations", [])
    last = evaluations[-1] if evaluations else {}
    stage0 = last.get("stage0_set", {})
    pass_rate = stage0.get("pass_rate")
    sample_count = stage0.get("sample_count", 0)
    passed = (
        isinstance(pass_rate, (int, float))
        and pass_rate >= MINIMUM_PASS_RATE
        and isinstance(sample_count, int)
        and sample_count >= MINIMUM_SAMPLES
    )
    revision, _ = git_metadata(ROOT)
    result = {
        "stage": "T5-stage0",
        "result": "pass" if passed else "fail",
        "pass_rate": pass_rate if isinstance(pass_rate, (int, float)) else 0.0,
        "sample_count": sample_count if isinstance(sample_count, int) else 0,
        "cumulative_episodes": last.get("cumulative_episodes", state.get("cumulative_episodes", 0)),
        "thresholds": {
            "pass_rate_min": MINIMUM_PASS_RATE,
            "sample_count_min": MINIMUM_SAMPLES,
        },
        "source": str(state_path.relative_to(ROOT)) if state_path.is_relative_to(ROOT) else str(state_path),
        "commit": revision,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(gate_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, default=STATE_PATH)
    parser.add_argument("--output", type=Path, default=GATE_PATH)
    args = parser.parse_args()
    result = evaluate(args.state.expanduser().resolve(), args.output.expanduser().resolve())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
