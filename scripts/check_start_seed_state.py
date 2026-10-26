#!/usr/bin/env python3
"""task99: preflight for the demo-start seed renewal at a qc group boundary.

Read-only. Checks the live state file, the renewal ledger, the task manifest and
the committed seed inventory against each other, so the deployer does not have to
trust prose:

  inventory   the renewal pool must not overlap any registered segment
  config      every demo_start future block must stay outside the renewal pool
  ledger      pools == the registered pool; blocks inside it, ordered, never
              overlapping, each START_SEED_BLOCK seeds
  state       start cells (identified by `archive`) must have a seed table equal
              to their `seed_blocks` concatenation; offsets must not exceed it;
              every renewed segment must be present in the ledger
  cross-cell  two start cells must not share an *unused* seed (offset..end)

Exit code 1 if any check fails; the lines before it say which one.

Usage:
  python3 scripts/check_start_seed_state.py
  python3 scripts/check_start_seed_state.py --state /tmp/state.json --ledger /tmp/ledger.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]

from pvz_promotion import START_SEED_BLOCK, START_SEED_LEDGER_DEFAULT, START_SEED_POOL  # noqa: E402

DEFAULT_STATE = Path.home() / 'pvz-logs/curriculum_state.json'
DEFAULT_CONFIG = ROOT / 'configs/human_demo_tasks.json'
DEFAULT_INVENTORY = ROOT / 'MEMORY/docs/gate-seed-inventory.json'


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except OSError:
        return default


def check_inventory(inventory: dict | None) -> list[str]:
    if not inventory:
        return ['inventory: missing or unreadable']
    overlaps = [[max(START_SEED_POOL[0], low), min(START_SEED_POOL[1], high)]
                for low, high in inventory.get('occupied') or []
                if (low, high) != START_SEED_POOL and low < START_SEED_POOL[1] and START_SEED_POOL[0] < high]
    return [f'inventory: renewal pool {list(START_SEED_POOL)} overlaps {overlaps}'] if overlaps else []


def check_config(config: dict | None) -> tuple[list[str], list[str]]:
    problems, notes, owners = [], [], {}
    for task in (config or {}).get('tasks') or []:
        start = task.get('demo_start')
        if not start:
            continue
        seeds = [int(seed) for seed in start.get('future_seeds') or []]
        if not seeds:
            first = int(start.get('future_seed_start'))
            seeds = list(range(first, first + int(start.get('future_seed_count', 256))))
        if any(START_SEED_POOL[0] <= seed < START_SEED_POOL[1] for seed in seeds):
            problems.append(f'config: {task["task_id"]} block {seeds[0]}..{seeds[-1]} is inside '
                            f'the renewal pool {list(START_SEED_POOL)}')
        owners.setdefault((seeds[0], seeds[-1] + 1), []).append(task['task_id'])
    for block, names in sorted(owners.items()):
        if len(names) > 1:
            # 同一块被多个 task 登记：每个 task 有自己的 offset，都会从块头开始用同一批种子。
            notes.append(f'config: block {list(block)} is shared by {len(names)} tasks '
                         f'({", ".join(sorted(names))}); each starts at offset 0, so they would '
                         f'collect the same seeds')
    return problems, notes


def check_ledger(ledger: dict | None, exists: bool) -> list[str]:
    if not exists:
        return []
    if ledger is None:
        return ['ledger: unreadable']
    problems = []
    pools = [tuple(entry) for entry in ledger.get('pools') or [tuple(ledger.get('pool') or ())]]
    if pools != [START_SEED_POOL]:
        problems.append(f'ledger: pools {pools} != {[START_SEED_POOL]}')
    blocks = [tuple(block) for block in ledger.get('allocated') or []]
    for low, high in blocks:
        if not (START_SEED_POOL[0] <= low < high <= START_SEED_POOL[1]):
            problems.append(f'ledger: block [{low},{high}) outside the renewal pool')
        if high - low != START_SEED_BLOCK:
            problems.append(f'ledger: block [{low},{high}) is {high - low} seeds, expected {START_SEED_BLOCK}')
    ordered = sorted(blocks)
    for left, right in zip(ordered, ordered[1:]):
        if left[1] > right[0]:
            problems.append(f'ledger: blocks {list(left)} and {list(right)} overlap')
    return problems


def start_cells(state: dict) -> dict[str, dict]:
    return {name: cell for name, cell in (state.get('cells') or {}).items() if cell.get('archive')}


def check_state(state: dict | None, ledger_blocks: list[tuple]) -> tuple[list[str], list[str]]:
    """Return (problems, notes). Notes are informational (wasted blocks etc.)."""
    if state is None:
        return ['state: missing or unreadable'], []
    problems, notes = [], []
    offsets = state.get('offsets') or {}
    cells = start_cells(state)
    if not cells:
        notes.append('state: no start cells (nothing to check yet)')
    for name, cell in sorted(cells.items()):
        seeds = list(cell.get('seeds') or [])
        blocks = [tuple(block) for block in cell.get('seed_blocks') or []]
        if not seeds:
            problems.append(f'state: {name} has an empty seed table')
            continue
        if blocks:
            flat = [seed for low, high in blocks for seed in range(low, high)]
            if flat != seeds:
                problems.append(f'state: {name} seeds != seed_blocks concatenation '
                                f'({len(seeds)} seeds, {sum(high - low for low, high in blocks)} in blocks)')
            # Blocks are concatenated in table order, not numeric order: the original
            # manifest and the renewal pool are far apart; other cells can allocate
            # between two renewals of this cell.
            for block in blocks:
                if block[1] <= START_SEED_POOL[0] or START_SEED_POOL[1] <= block[0]:
                    continue
                if block not in ledger_blocks:
                    problems.append(f'state: {name} segment {list(block)} is not in the ledger')
        elif any(START_SEED_POOL[0] <= seed < START_SEED_POOL[1] for seed in seeds):
            problems.append(f'state: {name} has renewed seeds but no seed_blocks')
        if len(set(seeds)) != len(seeds):
            problems.append(f'state: {name} seed table contains duplicate seeds')
        offset = int(offsets.get(cell.get('task_id') or name, 0))
        if offset > len(seeds):
            problems.append(f'state: {name} offset {offset} exceeds its {len(seeds)} seeds')
        elif offset == len(seeds):
            notes.append(f'state: {name} is exactly at the end ({offset}); next plan must renew')
    referenced = {tuple(block) for cell in cells.values() for block in (cell.get('seed_blocks') or [])}
    for block in ledger_blocks:
        if block not in referenced:
            notes.append(f'ledger: block {list(block)} is not referenced by any start cell (wasted, never reused)')
    # Cross-cell: two cells must not share a seed they have not consumed yet.
    remaining = {}
    for name, cell in cells.items():
        offset = int(offsets.get(cell.get('task_id') or name, 0))
        remaining[name] = set((cell.get('seeds') or [])[offset:])
    names = sorted(remaining)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            shared = remaining[left] & remaining[right]
            if shared:
                problems.append(f'cross-cell: {left} and {right} share {len(shared)} unused seeds '
                                f'(e.g. {min(shared)})')
    return problems, notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--state', type=Path, default=DEFAULT_STATE)
    parser.add_argument('--ledger', type=Path, default=Path(START_SEED_LEDGER_DEFAULT))
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--inventory', type=Path, default=DEFAULT_INVENTORY)
    args = parser.parse_args()

    ledger = read_json(args.ledger)
    ledger_blocks = [tuple(block) for block in (ledger or {}).get('allocated') or []]
    config_problems, config_notes = check_config(read_json(args.config))
    problems = (check_inventory(read_json(args.inventory)) + config_problems
                + check_ledger(ledger, args.ledger.exists()))
    state_problems, notes = check_state(read_json(args.state), ledger_blocks)
    problems += state_problems
    notes = config_notes + notes
    for line in notes:
        print(f'note  {line}')
    for line in problems:
        print(f'FAIL  {line}')
    if problems:
        print(f'{len(problems)} problem(s)')
        raise SystemExit(1)
    print(f'ok    pool {list(START_SEED_POOL)}, {len(ledger_blocks)} ledger block(s), '
          f'{len(start_cells(read_json(args.state) or {}))} start cell(s)')


if __name__ == '__main__':
    main()
