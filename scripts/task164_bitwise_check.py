#!/usr/bin/env python3
"""task164: 新旧训练路径的逐位对比 + 提速倍数（同一子集、同种子、各 1 个 epoch）。

做法：把生产 segment 软链成一个"迷你语料"，按生产同样的 pack 契约写一个只含
segments 清单的 pack 文件，然后同一份 recipe 跑两遍训练器——一遍是 origin/prodint16
的原版，一遍是本分支的预取版——比较 model.pt 的每个张量是否 torch.equal。

只读生产 segment，所有写入都在 --work 指定的临时目录里。

用法：
  python3 scripts/task164_bitwise_check.py --segments 8 --work /tmp/task164_bit
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from pvz_recipe import load_recipe, training_row_options, validate  # noqa: E402
from pvz_training_pack import archive_fingerprint, pack_options, pack_path  # noqa: E402

SEGMENTS = Path.home() / "pvz-logs" / "curriculum_state.pipeline" / "frames" / "packs" / "segments"
BASE_RECIPE = Path.home() / "pvz-logs" / "curriculum_state.pipeline" / "round_000046" / "recipe.json"
OBS_BY_CHANNELS = {101: "v2", 117: "v2timer", 105: "v2sleep"}


def signature(path: Path):
    with np.load(path, allow_pickle=False) as stored:
        return (tuple(int(n) for n in stored["board_shape"]),
                tuple(int(n) for n in stored["scalar_shape"]), tuple(sorted(stored.files)))


def pick_segments(count: int) -> list[Path]:
    """Newest segment format only: mixing obs versions makes read_segments refuse the merge."""
    paths = sorted(SEGMENTS.glob("*.npz"))
    if not paths:
        raise SystemExit(f"no segments under {SEGMENTS}")
    want = signature(paths[-1])
    chosen = [path for path in reversed(paths) if signature(path) == want][:count]
    return sorted(chosen)


def build_corpus(work: Path, count: int, epochs: int = 1, device: str = "cuda") -> tuple[Path, dict, dict]:
    data = work / "data"
    segments = data / "packs" / "segments"
    segments.mkdir(parents=True, exist_ok=True)
    names = []
    for path in pick_segments(count):
        target = segments / path.name
        if not target.exists():
            target.symlink_to(path.resolve())
        names.append(f"segments/{path.name}")

    recipe = load_recipe(BASE_RECIPE)
    recipe["data"] = str(data)
    recipe["epochs"] = epochs
    recipe["save_epochs"] = 0
    recipe["seed"] = 1
    recipe["threads"] = 1
    recipe["device"] = device
    recipe["batch_size"] = 2048
    recipe["validation_batch_size"] = 4096
    recipe = validate(recipe, "training")

    options = pack_options(sources=archive_fingerprint(data, set(recipe["labels"])),
                           **{key: value for key, value in training_row_options(recipe).items()
                              if key != "source_weight"})
    path = pack_path(data, options)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, segments=np.array(names))
    return data, recipe, {"pack": str(path), "segments": len(names)}


def stage_old(work: Path, ref: str) -> Path:
    """A directory that looks like the repo to the pre-task164 trainer (ROOT/python + scripts/)."""
    old = work / "old"
    (old / "scripts").mkdir(parents=True, exist_ok=True)
    link = old / "python"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(ROOT / "python")
    source = subprocess.run(["git", "-C", str(ROOT), "show", f"{ref}:scripts/train_rollout_policy.py"],
                            capture_output=True, text=True, check=True).stdout
    (old / "scripts" / "train_rollout_policy.py").write_text(source)
    return old / "scripts" / "train_rollout_policy.py"


def run(trainer: Path, recipe: Path, out: Path) -> dict:
    if out.exists():
        shutil.rmtree(out)
    started = time.perf_counter()
    result = subprocess.run([sys.executable, str(trainer), "--recipe", str(recipe), "--out", str(out)],
                            capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    if result.returncode:
        print(result.stdout[-4000:], result.stderr[-4000:])
        raise SystemExit(f"{trainer} failed with {result.returncode}")
    metrics = json.loads((out / "metrics.json").read_text())
    return {"seconds": elapsed, "train_seconds": metrics["seconds"], "load_seconds": metrics["load_seconds"],
            "history": metrics["history"], "out": out}


def same(left, right) -> bool:
    """torch.equal, except that candidate padding is NaN and NaN != NaN (the values are still identical)."""
    import torch
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.is_floating_point():
        return bool(((left == right) | (torch.isnan(left) & torch.isnan(right))).all())
    return torch.equal(left, right)


def check_device_tensors(work: Path, trials: int = 3, rows: int = 2048) -> dict:
    """The GPU half: old inline assembly vs the pinned/side-stream one, tensor by tensor.

    CUDA training is not reproducible run to run (cuDNN backward), so weight equality can only be
    shown on the CPU; this is what proves the device path itself is unchanged.
    """
    import torch
    import train_rollout_policy as T
    recipe = json.loads((work / "recipe.json").read_text())
    options = T.training_row_options(recipe)
    options.pop("source_weight", None)
    loaded = T.load(Path(recipe["data"]), **options)
    boards, scalars, plant, packet, cell, weight, seed_of, values, cands = loaded[:9]
    auxiliary = loaded[9] if recipe["aux_weight"] else None
    packet_info = loaded[9 + bool(auxiliary is not None)] if recipe["packet_head"] == "identity" else None
    source = T.BatchSource(boards, scalars, plant, packet, cell, weight, packet_info, cands, auxiliary)
    device = torch.device("cuda")
    stream, event = torch.cuda.Stream(), torch.cuda.Event()
    names = ("board", "scalar", "plant", "packet", "cell", "weight", "packet_info", "cands", "aux")
    report = {}
    for trial in range(trials):
        idx = torch.randperm(len(plant))[:rows]
        old = (boards.batch(idx, device), scalars.batch(idx, device),
               plant[idx].to(device).float(), packet[idx].to(device).long(),
               cell[idx].to(device).long(), weight[idx].to(device),
               packet_info[idx].to(device) if packet_info is not None else None,
               cands[idx].to(device).float(),
               auxiliary[idx].to(device) if auxiliary is not None else None)
        new = T.load_batch(source, source.prepare(idx, pin=True), device, stream, event)
        torch.cuda.synchronize()
        bad = [name for name, left, right in zip(names, old, new)
               if (left is None) != (right is None) or (left is not None and not same(left, right))]
        report[f"trial{trial}"] = bad or "identical"
    return report


def compare(left: Path, right: Path) -> tuple[bool, list[str]]:
    import torch
    a = torch.load(left, map_location="cpu", weights_only=False)["state_dict"]
    b = torch.load(right, map_location="cpu", weights_only=False)["state_dict"]
    if set(a) != set(b):
        return False, [f"key mismatch: {set(a) ^ set(b)}"]
    bad = [key for key in a if not torch.equal(a[key], b[key])]
    return not bad, bad


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=int, default=8)
    parser.add_argument("--work", default="/tmp/task164_bit")
    parser.add_argument("--old-ref", default="origin/prodint16")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-run", action="store_true", help="only rebuild the corpus and recipe")
    parser.add_argument("--tensors", action="store_true", help="compare the device tensors old vs new")
    args = parser.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    data, recipe, info = build_corpus(work, args.segments, args.epochs, args.device)
    recipe_path = work / "recipe.json"
    recipe_path.write_text(json.dumps(recipe, indent=1))
    print(json.dumps(info), flush=True)
    if args.skip_run:
        print(recipe_path)
        return
    if args.tensors:
        print(json.dumps(check_device_tensors(work), ensure_ascii=False), flush=True)
        return

    old_trainer = stage_old(work, args.old_ref)
    new_trainer = ROOT / "scripts" / "train_rollout_policy.py"
    old = run(old_trainer, recipe_path, work / "out_old")
    new = run(new_trainer, recipe_path, work / "out_new")

    same, bad = compare(Path(old["out"]) / "model.pt", Path(new["out"]) / "model.pt")
    print(json.dumps({
        "identical": same,
        "differing_tensors": bad[:5],
        "old_seconds": round(old["train_seconds"], 1),
        "new_seconds": round(new["train_seconds"], 1),
        "speedup": round(old["train_seconds"] / new["train_seconds"], 3),
        "old_epochs": [round(h["seconds"], 1) for h in old["history"]],
        "new_epochs": [round(h["seconds"], 1) for h in new["history"]],
        "epoch_speedup": round(sum(h["seconds"] for h in old["history"])
                               / sum(h["seconds"] for h in new["history"]), 3),
        "old_val_loss": old["history"][0]["val_loss"],
        "new_val_loss": new["history"][0]["val_loss"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
