"""Replay the pinned task37 archive corpus and compare direct Board encoding and logits."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]


def archived_task(meta, first_frame):
    """Use only archived task/hand data, never the current task-family deck rules."""
    from pvz_env import TaskSpec, profile_for_deck
    task, context = meta["task"], meta["context"]
    original = task.get("conveyor_start") == "original"
    deck = () if original else tuple(task.get("deck", [p[1] for p in first_frame["packets"]]))
    return deck, TaskSpec(level=context["level"], seed=task["seed"],
                         playthrough=task.get("playthrough", 2),
                         profile=profile_for_deck(deck, task.get("first_aid_owned", False)),
                         zombie_count_multiplier=context["multiplier"], wave_cap=task.get("wave_cap"),
                         preplanted=tuple(tuple(p) for p in task.get("preplanted") or ()),
                         conveyor_start="original" if original else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-index", type=Path, required=True)
    parser.add_argument("--value", type=Path, nargs="+", required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--resource-dir", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    from check_engine_value import corpus
    from export_value_net import export_value_net
    from pvz_event_env import EventWaitEnv
    from pvz_frame_store import read_frames
    from pvz_value_net import context_vector, encode_frame, load_value_net, observation_context
    from render_episode import _frame
    torch.set_num_threads(1)
    _groups, sources = corpus(Path("/"), args.corpus_index)
    result = dict(executable_sha256=hashlib.sha256(args.executable.read_bytes()).hexdigest(),
                  corpus_index_sha256=hashlib.sha256(args.corpus_index.read_bytes()).hexdigest(),
                  sources=sources, models=[])
    with tempfile.TemporaryDirectory() as tmp, EventWaitEnv(
            args.resource_dir, executable=args.executable) as env:
        for path in args.value:
            net = load_value_net(path)
            export_value_net(net, Path(tmp) / "value.bin")
            env.load_value_net(Path(tmp) / "value.bin")
            row = dict(checkpoint=str(path), checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       encoder=net.encoder_version, groups={})
            for source in sources:
                archive = Path(source["path"])
                meta = json.loads((archive.parent / "meta.json").read_text())
                frames = read_frames(archive)
                deck, task = archived_task(meta, frames[0])
                obs, _ = env.reset(deck=deck, task=task)
                selected = source["indices"]
                if selected != sorted(set(selected)):
                    raise ValueError(f"corpus indices must be unique and ordered: {archive}")
                # One reference batch per source (the pinned corpus has 32 rows per archive).
                expected_features = []
                for index in selected:
                    board, scalars = encode_frame(frames[index], net.encoder_version)
                    if net.context:
                        scalars = torch.cat([scalars, torch.from_numpy(context_vector(
                            meta["context"]["multiplier"], meta["context"]["roster"]))])
                    expected_features.append((board, scalars))
                with torch.inference_mode():
                    reference = []
                    for start in range(0, len(selected), 32):
                        batch = expected_features[start:start + 32]
                        reference.extend(net(torch.stack([b for b, _ in batch]),
                                             torch.stack([s for _, s in batch])).tolist())
                selected_slots = {index: slot for slot, index in enumerate(selected)}
                group = row["groups"].setdefault(source["group"], dict(frames=0, max_logit_error=0.0,
                                                                         max_feature_error=0.0, passed=True))
                for index in range(selected[-1] + 1):
                    if index:
                        obs, *_, info = env.step(frames[index]["action"])
                        rejected = not info.get("ok")
                        if rejected != bool(frames[index].get("action_rejected")):
                            raise AssertionError(f"archive action acceptance differs: {archive}, frame {index}")
                        if rejected:
                            obs, *_ = env.step(dict(type="wait", ticks=60))
                    if index not in selected_slots:
                        continue
                    slot = selected_slots[index]
                    b, s = expected_features[slot]
                    live_b, live_s = encode_frame(_frame(obs), net.encoder_version)
                    if net.context:
                        live_s = torch.cat([live_s, torch.from_numpy(observation_context(obs))])
                    label = f"{archive}, frame {index}, encoder {net.encoder_version}"
                    # First prove we reached the archived input, rather than quietly validating a different state.
                    np.testing.assert_array_equal(live_b.numpy(), b.numpy(), err_msg=f"replay board: {label}")
                    np.testing.assert_array_equal(live_s.numpy(), s.numpy(), err_msg=f"replay scalars: {label}")
                    env.leaf_values(trace_path=Path(tmp) / "board-features.bin")
                    actual = np.fromfile(Path(tmp) / "board-features.bin", dtype="<f4")
                    expected = np.concatenate([b.flatten().numpy(), s.numpy()])
                    error = float(np.max(np.abs(actual - expected)))
                    np.testing.assert_array_equal(actual, expected, err_msg=f"C++ Board features: {label}")
                    logit_error = abs(env._last_response["logits"][0] - reference[slot])
                    if logit_error >= 1e-5:
                        raise AssertionError(f"C++ Board logit error {logit_error}: {label}")
                    group["frames"] += 1
                    group["max_feature_error"] = max(group["max_feature_error"], error)
                    group["max_logit_error"] = max(group["max_logit_error"], logit_error)
                print(json.dumps(dict(checkpoint=str(path), source=str(archive), scene=source["group"],
                                      checked=len(selected), cumulative=group)), flush=True)
            if any(group["frames"] != 256 for group in row["groups"].values()) or len(row["groups"]) != 4:
                raise AssertionError("direct Board gate requires exactly 256 frames in each of four scenes")
            result["models"].append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pending = args.out.with_suffix(".json.tmp")
    pending.write_text(json.dumps(result, indent=2) + "\n")
    pending.replace(args.out)
    print(json.dumps(result["models"], indent=2))


if __name__ == "__main__":
    main()
