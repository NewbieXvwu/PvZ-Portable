"""What the T5 training entry point can actually reach, and what nothing reaches.

The repository carries two generations of code.  T5/T6/T7 train a PPO policy on the
frozen 20-task family; the older T4 work trained a search teacher and then cloned it
with DAgger.  ``DESIGN.md`` §12 retired that teacher ("0 胜基座、冻结"), and the
current task list never mentions search, so a large part of ``python/`` is now only
reachable from itself.

Eyeballing import statements is not enough to say which part: ``train_pvz_ppo.py`` is
imported by the T5 trainer for four functions, so everything *it* imports comes along
even though the T5 trainer calls none of it.  This tool walks the import graph from a
set of entry points and then, separately, counts name-level references, so it can
distinguish three different things:

* **unreachable module** -- no import path from any entry point;
* **reachable module, unreferenced name** -- imported for a reason, but this
  particular function/constant is never called or read from outside its own file;
* **reachable and used** -- leave alone.

Usage:

    /Users/newbiexvwu/.local/share/mise/installs/python/3.14/bin/python3 \
        scripts/reachability_audit.py
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIRS = ("python", "scripts", "tests")

# ``python/`` holds libraries and exactly one executable entry point per generation;
# ``scripts/`` and ``tests/`` are leaves -- every file in them is something a human or
# a test runner invokes directly, so they are entry points by construction.  Treating
# them as leaves is what makes "unreachable" mean *dead library code* rather than
# "this benchmark is not imported by the trainer", which would be noise.
LIBRARY_DIR = "python"
DEFAULT_ENTRIES = (
    "python/train_pvz_ppo_task_family.py",   # T5 trainer
    "python/train_pvz_ppo.py",               # the PPO update it calls
)


def module_name(path: Path) -> str:
    return path.stem


def project_modules() -> dict[str, Path]:
    modules: dict[str, Path] = {}
    for directory in SOURCE_DIRS:
        for path in sorted((ROOT / directory).glob("*.py")):
            modules[module_name(path)] = path
    return modules


class ModuleFacts:
    """Definitions and references for one file, at the granularity of names.

    Two blind spots are deliberate and worth stating, because both make the tool
    under-report rather than over-report:

    * class methods are recorded under their bare name, so two classes that both
      define ``forward`` share one entry and a use of either marks both as live;
    * references are matched by identifier, so ``pvz_value.VALUE_GAMMA`` counts as a
      reference to ``VALUE_GAMMA`` from ``pvz_value``'s importer.

    Both only ever hide a dead name, never invent one, which is the right direction
    for a tool whose output is a list of things to go and read.
    """

    def __init__(self, path: Path, known: set[str]) -> None:
        self.path = path
        self.tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        self.imports: set[str] = set()
        self.defines: dict[str, int] = {}
        self.references: dict[str, int] = defaultdict(int)
        self._collect(known)

    def _record_definition(self, name: str, lineno: int) -> None:
        self.defines.setdefault(name, lineno)

    def _collect(self, known: set[str]) -> None:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in known:
                        self.imports.add(root)
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] in known and node.level == 0:
                    self.imports.add(node.module.split(".")[0])
            elif isinstance(node, ast.Name):
                self.references[node.id] += 1
            elif isinstance(node, ast.Attribute):
                # ``module.name`` or ``obj.name`` -- record the attribute so an import
                # used only as ``pvz_value.VALUE_GAMMA`` still counts as a reference.
                self.references[node.attr] += 1

        for node in self.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self._record_definition(node.name, node.lineno)
                if isinstance(node, ast.ClassDef):
                    # Methods are defined inside a class, so walking only ``body``
                    # would miss every one of them -- and a model file is almost
                    # entirely methods.
                    for child in node.body:
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            self._record_definition(child.name, child.lineno)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self._record_definition(target.id, node.lineno)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                self._record_definition(node.target.id, node.lineno)
            elif isinstance(node, (ast.If, ast.Try)):
                # Constants behind a feature flag are still definitions.
                for child in ast.walk(node):
                    if isinstance(child, ast.Assign):
                        for target in child.targets:
                            if isinstance(target, ast.Name):
                                self._record_definition(target.id, child.lineno)


def leaf_modules(modules: dict[str, Path]) -> list[str]:
    """Modules that are directly invokable rather than imported by the trainer.

    ``scripts/`` and ``tests/`` are leaves by location; ``python/test_*.py`` is a
    unittest module that lives beside the library it exercises but is run by the test
    runner, so nothing imports it and that is not evidence of anything.
    """
    return sorted(
        name for name, path in modules.items()
        if path.parent.name in ("scripts", "tests") or name.startswith("test_")
    )


def reachable(modules: dict[str, Path], entries: list[str]) -> tuple[set[str], dict[str, set[str]]]:
    facts = {name: ModuleFacts(path, set(modules)) for name, path in modules.items()}
    edges = {name: set(module.imports) for name, module in facts.items()}
    seen: set[str] = set()
    stack = []
    for entry in entries:
        name = Path(entry).stem
        if name not in modules:
            raise SystemExit(f"entry point {entry} is not a module under {SOURCE_DIRS}")
        stack.append(name)
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(edges.get(current, ()))
    return seen, edges


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entries", nargs="*", default=None,
                        help="library entry points; scripts/ and tests/ are added "
                             "automatically as leaves")
    parser.add_argument("--show-used", action="store_true",
                        help="also list names that are defined and used, for spot checks")
    parser.add_argument("--no-leaves", action="store_true",
                        help="do not treat scripts/ and tests/ as entry points; use this "
                             "to see what one library entry point alone pulls in")
    parser.add_argument("--closure-unused", action="store_true",
                        help="also report names defined in reachable modules that nothing "
                             "inside the reachable set ever calls")
    args = parser.parse_args()

    modules = project_modules()
    entries = [entry for entry in (args.entries or DEFAULT_ENTRIES)
               if Path(entry).stem in modules]
    if not args.no_leaves:
        entries += leaf_modules(modules)
    facts = {name: ModuleFacts(path, set(modules)) for name, path in modules.items()}
    seen, edges = reachable(modules, entries)

    def lines(name: str) -> int:
        return len(facts[name].path.read_text(encoding="utf-8").splitlines())

    library = {name: path for name, path in modules.items() if path.parent.name == LIBRARY_DIR}
    reachable_library = {name for name in library if name in seen}
    dead_library = set(library) - reachable_library

    print("=" * 78)
    print("reachability")
    print("=" * 78)
    print("  library entry points (python/):")
    for entry in DEFAULT_ENTRIES:
        if Path(entry).stem in modules:
            print(f"    {entry}")
    print(f"  leaves: {len(leaf_modules(modules))} files under scripts/ and tests/")
    print()
    print(f"python/ modules        {len(library):>3}   {sum(lines(n) for n in library):>6} lines")
    print(f"  reachable            {len(reachable_library):>3}   "
          f"{sum(lines(n) for n in reachable_library):>6} lines")
    print(f"  UNREACHABLE          {len(dead_library):>3}   "
          f"{sum(lines(n) for n in dead_library):>6} lines")
    print()

    print("-" * 78)
    print("UNREACHABLE library modules (no import path from any entry point or leaf)")
    print("-" * 78)
    print(f"{'module':<34}{'lines':>7}  imports (project)")
    for name in sorted(dead_library, key=lambda n: -lines(n)):
        imported = ", ".join(sorted(edges.get(name, ()))) or "-"
        print(f"{name:<34}{lines(name):>7}  {imported}")
    if not dead_library:
        print("  (none)")
    print()

    # Name-level: a name defined in a reachable library module that no other file
    # references and that its own file never uses is dead weight on the training path.
    external: dict[str, set[str]] = defaultdict(set)
    for name, module in facts.items():
        for reference in module.references:
            external[reference].add(name)

    print("-" * 78)
    print("UNREFERENCED names in REACHABLE library modules")
    print("-" * 78)
    print("(defined in python/, never named anywhere in the repository -- no other")
    print(" module references it and its own file never uses it)")
    print("test modules are skipped: their TestCase classes are found by the test")
    print("runner through reflection, so nothing is supposed to name them)")
    print()
    dead_total = 0
    for name in sorted(reachable_library, key=lambda n: -lines(n)):
        if name.startswith("test_"):
            continue
        module = facts[name]
        rows = []
        for defined, lineno in sorted(module.defines.items(), key=lambda item: item[1]):
            if defined.startswith("__"):
                continue
            if external[defined] - {name}:
                continue
            if module.references.get(defined, 0) == 0:
                rows.append((lineno, defined))
        if not rows:
            continue
        dead_total += len(rows)
        print(f"  {name}  ({lines(name)} lines)")
        for lineno, defined in rows:
            print(f"    line {lineno:>5}  {defined}")
    print()
    print(f"total unreferenced names in reachable non-test library modules: {dead_total}")

    # Names that exist inside the closure but that nothing inside the closure calls.
    # This is where T4 residue hides: ``train_pvz_ppo`` is imported by the T5 trainer
    # for four functions, and every other definition in it rides along unexecuted.
    if args.closure_unused:
        closure = set(entries) | seen
        inside: dict[str, set[str]] = defaultdict(set)
        for name, module in facts.items():
            if name not in closure:
                continue
            for reference in module.references:
                inside[reference].add(name)
        print()
        print("-" * 78)
        print("NAMES DEFINED BUT NEVER USED inside the closure")
        print("-" * 78)
        print("(the module is reachable, but nothing that runs calls this name)")
        print()
        total = 0
        for name in sorted(reachable_library, key=lambda n: -lines(n)):
            if name.startswith("test_"):
                continue
            module = facts[name]
            rows = []
            for defined, lineno in sorted(module.defines.items(), key=lambda item: item[1]):
                if defined.startswith("__"):
                    continue
                if inside[defined] - {name}:
                    continue
                if module.references.get(defined, 0) == 0:
                    rows.append((lineno, defined))
            if not rows:
                continue
            total += len(rows)
            print(f"  {name}  ({lines(name)} lines)")
            for lineno, defined in rows:
                print(f"    line {lineno:>5}  {defined}")
        print()
        print(f"total: {total}")

    if args.show_used:
        print()
        print("-" * 78)
        print("referenced names, by module (spot check)")
        print("-" * 78)
        for name in sorted(reachable_library):
            module = facts[name]
            used = sorted(defined for defined in module.defines
                          if external[defined] - {name})
            print(f"  {name}: {', '.join(used) if used else '-'}")


if __name__ == "__main__":
    sys.exit(main())
