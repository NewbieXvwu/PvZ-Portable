#!/usr/bin/env python3
"""Count CUDA verification requests from archived frames, including rejected overrides."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from pvz_frame_store import frame_path, read_frames


def ledger(games, frames, label):
    rows = []
    for game in games:
        requests = game.get("cuda_batches", 0)
        if not requests:
            continue
        archive = frame_path(frames / f"seed_{game['seed']}" / f"{label}_{game['arm']}")
        records = read_frames(archive)
        verifications = [frame["verification"] for frame in records if frame.get("verification")]
        attempts = len(verifications)
        accepted = sum(bool(v["after_override"]) for v in verifications)
        search = game["searched"]
        rows.append({"seed": game["seed"], "archive": str(archive), "cuda_requests": requests,
                     "searches": search, "verification_requests": attempts,
                     "accepted_verifications": accepted, "rejected_verifications": attempts - accepted,
                     "other_requests": requests - search - attempts,
                     "verification_request_fraction": attempts / requests,
                     "cuda_seconds": game["cuda_seconds"],
                     "seconds_per_request": game["cuda_seconds"] / requests})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--games", type=Path, required=True, help="games.jsonl")
    p.add_argument("--frames", type=Path, required=True, help="root containing seed_X directories")
    p.add_argument("--label", required=True, help="frame label without the _fast_search suffix")
    args = p.parse_args()
    games = [json.loads(line) for line in args.games.read_text().splitlines() if line.strip()]
    print(json.dumps(ledger(games, args.frames, args.label), indent=2))


if __name__ == "__main__":
    main()
