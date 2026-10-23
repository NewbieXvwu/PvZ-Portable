"""Which command-line switches of the hot-path scripts are ever actually set (task33).

Reads the argparse definitions out of the four hot-path scripts plus pvz_env.py, then cross-checks
each flag against (a) every run config.json under artifacts/ (a run records vars(args) verbatim) and
(b) every mention outside the defining file (scripts, shell wrappers, docs, queue scripts).

  python3 scripts/hotpath_flags.py [--artifacts DIR] [--root DIR]

Verdicts: never-set (only ever the default), constant (always one non-default value),
used (more than one value, or set in live code).
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGETS = ["scripts/search_play.py", "scripts/train_rollout_policy.py", "scripts/episode_query.py",
           "scripts/unified_panel.py", "scripts/curriculum_queue.py", "python/pvz_env.py"]


def definitions(path: Path):
    """Every add_argument in the file: dest, the flags that reach it, and its default."""
    tree = ast.parse(path.read_text())
    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        flags = [argument.value for argument in node.args
                 if isinstance(argument, ast.Constant) and isinstance(argument.value, str)]
        if not flags:
            continue
        keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
        if "dest" in keywords and isinstance(keywords["dest"], ast.Constant):
            dest = keywords["dest"].value
        else:
            long_flags = [flag for flag in flags if flag.startswith("--")]
            dest = (long_flags or flags)[0].lstrip("-").replace("-", "_")
        default = keywords.get("default")
        try:
            default = ast.literal_eval(default) if default is not None else None
        except ValueError:
            default = "<" + ast.unparse(default) + ">"
        action = keywords.get("action")
        if isinstance(action, ast.Constant) and action.value in ("store_true", "store_false"):
            default = action.value == "store_false"
        found.append({"dest": dest, "flags": flags, "default": default,
                      "line": node.lineno, "required": bool(keywords.get("required"))})
    return found


def normalise(value):
    if isinstance(value, bool):
        return str(value)
    if value is None:
        return "None"
    return str(value)


def observed(artifacts: Path, dest: str) -> Counter:
    counts = Counter()
    for path in artifacts.glob("**/config.json"):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if dest in data:
            counts[str(data[dest])] += 1
    return counts


def mentions(root: Path, flags, defining: Path):
    hits = []
    for flag in flags:
        if not flag.startswith("--"):
            continue
        out = subprocess.run(["git", "grep", "-l", "-F", "--", flag], cwd=root,
                             capture_output=True, text=True, check=False).stdout
        hits.extend(line for line in out.splitlines() if line and Path(root, line) != defining)
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("/Users/newbiexvwu/PvZAgent/artifacts"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    for target in TARGETS:
        path = args.root / target
        if not path.exists():
            print("== " + target + ": missing")
            continue
        print("")
        print("== " + target)
        for entry in definitions(path):
            values = observed(args.artifacts, entry["dest"])
            used = mentions(args.root, entry["flags"], path.resolve())
            distinct = {value for value in values} - {normalise(entry["default"])}
            if entry["required"]:
                verdict = "required"
            elif not used and not distinct:
                verdict = "never-set"
            elif used and not distinct and len(values) == 1:
                verdict = "constant"
            elif not used:
                verdict = "run-only"
            else:
                verdict = "used"
            print("%-10s %-32s default=%-18s runs=%s files=%d" % (
                verdict, entry["dest"], normalise(entry["default"])[:18],
                dict(values.most_common(4)), len(set(used))))


if __name__ == "__main__":
    main()
