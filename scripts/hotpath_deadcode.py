"""Module-level definitions nothing in the repository mentions any more (task33 dead code).

Counts word occurrences of every top-level def/class of the given modules across all source
extensions in the tree, ignoring the definition itself and its own module's docstring. A name with
no other occurrence is dead; one or two is a lead worth reading. Lists only, deletes nothing.

  python3 scripts/hotpath_deadcode.py [--modules ...]
"""
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUFFIXES = {".py", ".sh", ".cpp", ".h", ".md", ".json", ".txt", ".cu", ".cuh"}
SKIP = {"artifacts", "build", "vcpkg", "node_modules", ".git", "ios", "android", "wasm", "learn"}
DEFAULT_MODULES = ["python/pvz_frame_store.py", "python/pvz_training_rows.py", "python/pvz_rollout_policy.py",
                   "python/pvz_student_observation.py", "python/pvz_env.py", "python/pvz_event_env.py",
                   "python/pvz_value_net.py", "scripts/search_play.py", "scripts/train_rollout_policy.py",
                   "scripts/unified_panel.py", "scripts/curriculum_queue.py"]


def definitions(path: Path):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node.name, node.lineno


def corpus(root: Path) -> dict[Path, str]:
    texts = {}
    for path in root.rglob("*"):
        if path.suffix not in SUFFIXES or not path.is_file():
            continue
        if SKIP & set(path.relative_to(root).parts):
            continue
        try:
            texts[path] = path.read_text(errors="ignore")
        except OSError:
            continue
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modules", nargs="*", default=DEFAULT_MODULES)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--limit", type=int, default=2, help="report names with at most this many other mentions")
    args = parser.parse_args()
    texts = corpus(args.root)
    total_lines = 0
    for module in args.modules:
        path = args.root / module
        if not path.exists():
            continue
        entries = list(definitions(path))
        total_lines += len(path.read_text().splitlines())
        rows = []
        for name, line in entries:
            if name.startswith("__"):
                continue
            pattern = re.compile(r"\b" + re.escape(name) + r"\b")
            hits = []
            for other, text in texts.items():
                count = len(pattern.findall(text))
                if other == path:
                    count -= 1                      # the definition itself
                if count > 0:
                    hits.append((other.relative_to(args.root).as_posix(), count))
            if len(hits) <= args.limit:
                rows.append((name, line, hits))
        print("")
        print("== " + module + " (" + str(len(path.read_text().splitlines())) + " lines, "
              + str(len(entries)) + " definitions, " + str(len(rows)) + " with <= "
              + str(args.limit) + " other mentions)")
        for name, line, hits in sorted(rows, key=lambda row: row[1]):
            where = ", ".join(f"{file}:{count}" for file, count in hits) or "-"
            print("  line %-5d %-38s %s" % (line, name, where))
    print("")
    print("modules total lines: " + str(total_lines))


if __name__ == "__main__":
    main()
