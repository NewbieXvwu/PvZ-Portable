"""task36 acceptance: the same rows through the archive loader and the pack loader train bit-identically.

Compares two finished runs (model.pt + metrics.json) written by the pre-merge trainer and the
packed trainer for the same seed, options and step budget: every epoch loss/metric must be equal
and every state-dict tensor must be bitwise equal.

  python3 scripts/task36_loader_parity.py --old /tmp/task36-accept-old --new /tmp/task36-accept-new
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def compare_numbers(old, new, prefix="") -> list[dict]:
    differences = []
    if isinstance(old, dict) and isinstance(new, dict):
        if set(old) != set(new):
            differences.append(dict(field=prefix, missing=sorted(set(old) ^ set(new))))
            return differences
        for key in old:
            differences += compare_numbers(old[key], new[key], f"{prefix}.{key}" if prefix else key)
    elif isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            differences.append(dict(field=prefix, old_rows=len(old), new_rows=len(new)))
            return differences
        for index, (a, b) in enumerate(zip(old, new)):
            differences += compare_numbers(a, b, f"{prefix}[{index}]")
    elif isinstance(old, float) and isinstance(new, float):
        if old != new:
            differences.append(dict(field=prefix, old=old, new=new))
    elif old != new:
        differences.append(dict(field=prefix, old=old, new=new))
    return differences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    old_metrics = json.loads((args.old / "metrics.json").read_text())
    new_metrics = json.loads((args.new / "metrics.json").read_text())
    history = compare_numbers(old_metrics["history"], new_metrics["history"], "history")

    old_state = load(args.old / "model.pt")["state_dict"]
    new_state = load(args.new / "model.pt")["state_dict"]
    tensors = {}
    if set(old_state) != set(new_state):
        tensors["__missing__"] = sorted(set(old_state) ^ set(new_state))
    else:
        for key in old_state:
            a, b = old_state[key], new_state[key]
            tensors[key] = dict(same=bool(torch.equal(a, b)), shape=list(a.shape), dtype=str(a.dtype),
                                max_abs_diff=(float((a.float() - b.float()).abs().max()) if a.shape == b.shape else None))
    mismatch = {key: value for key, value in tensors.items() if not (value.get("same") if isinstance(value, dict) else False)}
    result = dict(old=str(args.old), new=str(args.new), epochs=len(old_metrics["history"]),
                  history_differences=history, tensors=len(old_state), tensor_mismatches=mismatch,
                  bitwise=not history and not mismatch)
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    if not result["bitwise"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
