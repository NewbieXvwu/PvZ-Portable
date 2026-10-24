"""Bound three remaining Python optimizations using real frames, trace counts and worker CPU."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]


def timed(function, repeats=10):
    samples = []
    for _ in range(5):
        start = time.process_time()
        for _ in range(repeats):
            function()
        samples.append((time.process_time() - start) / repeats)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--game", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from pvz_frame_store import read_frames
    from pvz_rollout_policy import load_student, features, packet_features, identity_scalars
    from pvz_student_observation import StudentHistory
    from pvz_value_net import context_vector
    torch.set_num_threads(1)
    net = load_student(args.student)
    frames = read_frames(args.frames)
    meta = json.loads((args.frames.parent / "meta.json").read_text())["context"]
    context = context_vector(meta["multiplier"], meta["roster"]) if net.context else None
    game = json.loads((args.game / "result.json").read_text())
    trace = json.loads((args.game / "trace.json").read_text())
    searches = [row for row in trace if row["searched"]]
    total_cpu = sum(game["host_cpu_excluding_noise_s"].values())
    actions = [[action for action, _value in row["targets"]] for row in searches]
    key = lambda a: (a["type"], a.get("packet"), a.get("row"), a.get("col"), a.get("ticks"))
    for candidates in actions:
        old = [json.dumps(a, sort_keys=True) for a in candidates]
        new = [key(a) for a in candidates]
        assert [[a == b for b in old] for a in old] == [[a == b for b in new] for a in new]
    old_keys = timed(lambda: [[json.dumps(a, sort_keys=True) for a in candidates] for candidates in actions])
    new_keys = timed(lambda: [[key(a) for a in candidates] for candidates in actions])

    selected = {i * (len(frames) - 1) // 31 for i in range(32)}
    history = StudentHistory(net.obs)
    samples = []
    for index, frame in enumerate(frames):
        previous = frames[index - 1] if index else None
        history.extra(frame)
        if index in selected:
            samples.append((frame, previous, deepcopy(history)))

    def encode_features(sample):
        frame, previous, history = sample
        board, scalars = (features(frame, previous, context) if net.obs == "v1" else
                          history.features(frame, previous, context))
        info = packet_features(frame)[None] if net.packet_head == "identity" else None
        if info is not None:
            scalars = identity_scalars(frame, scalars)
        return board[None], scalars[None], info

    inputs = [encode_features(sample) for sample in samples]
    for sample, expected in zip(samples, inputs):
        for actual, value in zip(encode_features(sample), expected):
            if value is not None:
                torch.testing.assert_close(actual, value, rtol=0, atol=0)
    feature_cpu = timed(lambda: [encode_features(sample) for sample in samples]) / len(samples)
    with torch.inference_mode():
        def encode():
            return [net.encode(b, s, sparse_input=True, packet_info=info) for b, s, info in inputs]
        outputs = encode()
        for actual, expected in zip(encode(), outputs):
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        network_cpu = timed(encode) / len(samples)
    # Cache hits cannot save more than the entire duplicated operation, even with zero lookup cost.
    # The fixed recipe serializes the same candidate in several deduplication/selection passes.
    # Eight passes conservatively cover those call sites; time a full pass over the real trace.
    rows = [dict(attempt="candidate keys: sorted JSON to tuples", cpu_s_saved_upper_bound=8*max(0, old_keys-new_keys),
                 passes_upper_bound=8, old_s=old_keys, prototype_s=new_keys, equivalent=True),
            dict(attempt="reuse student features for ranked_plants", cpu_s_saved_upper_bound=feature_cpu*len(searches),
                 samples=len(samples), seconds_per_duplicate=feature_cpu, equivalent=True),
            dict(attempt="reuse student encoder output for ranked_plants", cpu_s_saved_upper_bound=network_cpu*len(searches),
                 samples=len(samples), seconds_per_duplicate=network_cpu, equivalent=True)]
    for row in rows:
        row["game_fraction_upper_bound"] = row["cpu_s_saved_upper_bound"] / total_cpu
        row["below_two_percent"] = row["game_fraction_upper_bound"] < 0.02
    result = dict(game_cpu_s=total_cpu, searches=len(searches), frames=str(args.frames), attempts=rows,
                  scope="offline CPU microbenchmarks scaled by actual search count; not whole-game A/B results")
    args.out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
