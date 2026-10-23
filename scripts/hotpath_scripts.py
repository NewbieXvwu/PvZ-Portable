"""scripts/ entries nothing else names any more (task33: list only, delete nothing)."""
from __future__ import annotations
import re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
SUFFIXES = {".py", ".sh", ".cpp", ".h", ".md", ".json", ".txt"}
SKIP = {"artifacts", "build", "vcpkg", "node_modules", ".git", "ios", "android", "wasm"}
texts = {}
for path in ROOT.rglob("*"):
    if path.suffix not in SUFFIXES or not path.is_file():
        continue
    if SKIP & set(path.relative_to(ROOT).parts):
        continue
    try:
        texts[path] = path.read_text(errors="ignore")
    except OSError:
        continue
rows = []
for path in sorted((ROOT / "scripts").iterdir()):
    if not path.is_file() or path.suffix not in {".py", ".sh"}:
        continue
    hits = [other for other, text in texts.items() if other != path and path.name in text]
    rows.append((len(hits), path.name, len(path.read_text(errors="ignore").splitlines()), hits))
rows.sort()
for count, name, lines, hits in rows:
    if count <= 1:
        where = ", ".join(p.relative_to(ROOT).as_posix() for p in hits[:3]) or "-"
        print(f"{count}  {name:44s} {lines:5d} lines  {where}")
print("total scripts:", len(rows))
