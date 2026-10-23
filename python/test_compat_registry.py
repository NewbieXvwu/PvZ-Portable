"""The compatibility registry and the code agree (task33 2026-10-23).

Every LEGACY(<id>) / EXPERIMENT(<id>) marker in the tree must have an entry in python/pvz_compat.py,
and every entry must be marked somewhere -- otherwise deleting a branch leaves a stale entry, or
adding one leaves an unregistered switch. Experiment decide-by dates are enforced: once the date
has passed the test fails until the switch is turned into the only behaviour or deleted.
"""
from __future__ import annotations

import re
import sys
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import pvz_compat  # noqa: E402

SKIP = {"artifacts", "build", "vcpkg", "node_modules", ".git", "ios", "android", "wasm", ".claude"}
MARKER = re.compile(r"\b(LEGACY|EXPERIMENT)\(([a-z0-9-]+)\)")


def markers() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {"LEGACY": set(), "EXPERIMENT": set()}
    for path in ROOT.rglob("*"):
        if path.suffix not in {".py", ".cpp", ".h"} or not path.is_file():
            continue
        if SKIP & set(path.relative_to(ROOT).parts):
            continue
        for kind, name in MARKER.findall(path.read_text(errors="ignore")):
            found[kind].add(name)
    return found


class CompatRegistry(unittest.TestCase):
    def test_every_marker_is_registered(self) -> None:
        found = markers()
        self.assertEqual(found["LEGACY"] - set(pvz_compat.LEGACY), set(),
                         "LEGACY marker without an entry in pvz_compat.LEGACY")
        self.assertEqual(found["EXPERIMENT"] - set(pvz_compat.EXPERIMENTAL_FLAGS), set(),
                         "EXPERIMENT marker without an entry in pvz_compat.EXPERIMENTAL_FLAGS")

    def test_every_entry_is_marked(self) -> None:
        found = markers()
        self.assertEqual(set(pvz_compat.LEGACY) - found["LEGACY"], set(),
                         "pvz_compat.LEGACY entry nothing in the code marks")
        self.assertEqual(set(pvz_compat.EXPERIMENTAL_FLAGS) - found["EXPERIMENT"], set(),
                         "pvz_compat.EXPERIMENTAL_FLAGS entry nothing in the code marks")

    def test_legacy_entries_say_who_and_when(self) -> None:
        for name, entry in pvz_compat.LEGACY.items():
            for key in ("what", "kept_for", "since", "remove_when", "where"):
                self.assertTrue(str(entry.get(key, "")).strip(), f"{name} has no {key}")
            date.fromisoformat(entry["since"])

    def test_experiment_dates_are_open(self) -> None:
        self.assertEqual(pvz_compat.expired(date.today()), [],
                         "experiment switch past its decide-by date: turn it into the only "
                         "behaviour or delete it (pvz_compat.EXPERIMENTAL_FLAGS)")

    def test_experiment_flags_exist(self) -> None:
        for name, entry in pvz_compat.EXPERIMENTAL_FLAGS.items():
            script = ROOT / entry["script"]
            self.assertTrue(script.exists(), f"{name}: {entry['script']} missing")
            self.assertIn(entry["flag"], script.read_text(), f"{name}: {entry['flag']} not in {script}")
            date.fromisoformat(entry["decide_by"])


if __name__ == "__main__":
    unittest.main()
