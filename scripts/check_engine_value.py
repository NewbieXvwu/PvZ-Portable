"""Compare engine logits against Python on 256 archived frames per terrain and checkpoint."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]


def corpus(root: Path):
    from pvz_frame_store import read_frames
    groups = {k: [] for k in ("day", "night", "pool", "conveyor")}
    sources = []
    for seed in sorted(root.iterdir()):
        if not seed.is_dir():
            continue
        for archive in sorted(seed.iterdir()):
            meta_path = archive / "meta.json"
            if not meta_path.is_file():
                continue
            meta = json.loads(meta_path.read_text())
            context = meta.get("context") or {}
            level = context.get("level") or meta.get("task", {}).get("level", 0)
            original = meta.get("task", {}).get("conveyor_start") == "original"
            group = "conveyor" if original else "day" if 1 <= level <= 9 else "night" if 11 <= level <= 20 else "pool" if 21 <= level <= 30 else None
            if group is None or len(groups[group]) >= 256 or not context:
                continue
            path = next((archive / f"frames.{extension}" for extension in ("npz", "jsonl")
                         if (archive / f"frames.{extension}").is_file()), None)
            if path is None:
                continue
            frames = read_frames(path)
            count = min(32, 256 - len(groups[group]), len(frames))
            indices = [i * (len(frames) - 1) // max(1, count - 1) for i in range(count)]
            if group == "conveyor" and not all(frames[i].get("conveyor_start") == "original" for i in indices):
                raise AssertionError(f"original conveyor metadata/frame mismatch: {path}")
            groups[group].extend((frames[i], context) for i in indices)
            sources.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                group=group, indices=indices))
        if all(len(rows) == 256 for rows in groups.values()):
            return groups, sources
    raise RuntimeError(f"insufficient corpus: { {key:len(rows) for key,rows in groups.items()} }")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--value", type=Path, nargs="+", required=True)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--resource-dir", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    from export_value_net import export_value_net
    from pvz_env import PvZEnv
    from pvz_value_net import context_vector, encode_frame, load_value_net
    torch.set_num_threads(1)
    groups, sources = corpus(args.frames)
    result = {"sources": sources, "models": []}
    with tempfile.TemporaryDirectory() as tmp, PvZEnv(args.resource_dir, executable=args.executable) as env:
        for path in args.value:
            net = load_value_net(path)
            export_value_net(net, Path(tmp) / "value.bin")
            env.load_value_net(Path(tmp) / "value.bin")
            row = dict(checkpoint=str(path), checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       encoder=net.encoder_version, groups={})
            for name, frames in groups.items():
                boards, scalars = [], []
                for frame, context in frames:
                    b, s = encode_frame(frame, net.encoder_version)
                    if net.context:
                        s = torch.cat([s, torch.from_numpy(context_vector(context["multiplier"], context["roster"]))])
                    boards.append(b); scalars.append(s)
                expected = []
                # Preserve the old teacher's exact forward batch size, including final partial batch.
                with torch.inference_mode():
                    for start in range(0, len(boards), 32):
                        expected.extend(net(torch.stack(boards[start:start+32]), torch.stack(scalars[start:start+32])).tolist())
                with (Path(tmp) / "features.bin").open("wb") as f:
                    for b, s in zip(boards, scalars):
                        f.write(b.numpy().astype("<f4").tobytes()); f.write(s.numpy().astype("<f4").tobytes())
                env.leaf_feature_values(Path(tmp) / "features.bin", len(frames))
                actual = env._last_response["logits"]
                error = float(np.max(np.abs(np.asarray(expected) - actual)))
                row["groups"][name] = dict(frames=len(frames), max_logit_error=error, passed=error < 1e-5)
                print(json.dumps({"model": str(path), "scene": name, **row["groups"][name]}), flush=True)
            result["models"].append(row)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    if not all(group["passed"] for model in result["models"] for group in model["groups"].values()):
        raise SystemExit("logit equivalence failed")


if __name__ == "__main__":
    main()
