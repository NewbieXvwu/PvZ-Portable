#!/usr/bin/env python3
"""Fail if the hand-mirrored Python layout of ``.cudaroot`` drifts from the C++ structs.

``python/pvz_cuda_root.py`` mirrors ``cuda/CudaRollout.h`` with ``struct.Struct`` format strings so a
machine with no compiler can still read an exported position.  That mirroring is the one place a
silently reordered field could survive: the bytes still parse, the counts still look plausible, and
the phase-1 comparison would then be comparing a zombie's health against a plant's.  This script is
the gate -- it compiles ``cuda/LayoutDump.cpp`` (g++, no link, no CUDA), and checks three things
against each other: the compiler's ``sizeof``/``offsetof``, the Python module's struct sizes, and the
positional field lists the module unzips with.

Usage: ``python3 scripts/cuda_layout_check.py`` (exit 1 and print every drift found).
"""
from __future__ import annotations

import pathlib
import re
import struct
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import pvz_cuda_root as layout  # noqa: E402


def camel(snake: str) -> str:
    head, *rest = snake.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in rest)


def dump_layout() -> dict[str, int]:
    source = ROOT / "cuda" / "LayoutDump.cpp"
    with tempfile.TemporaryDirectory() as work:
        binary = pathlib.Path(work) / "layout_dump"
        subprocess.run(["g++", "-std=c++17", "-I", str(source.parent), "-o", str(binary),
                        str(source)], check=True)
        output = subprocess.run([str(binary)], check=True, capture_output=True, text=True).stdout
    values: dict[str, int] = {}
    for line in output.splitlines():
        match = re.fullmatch(r"(\S+) (\d+)", line.strip())
        if match:
            values[match.group(1)] = int(match.group(2))
    return values


# RootRecord's scalar head, in the order the C++ struct declares it, with the repeats the Python
# format string expands.  Walking this list must reproduce both the compiler's offsets and the byte
# count of layout.ROOT_HEAD -- that is what ties the three sides together.
ROOT_HEAD_FIELDS: list[tuple[str, int]] = [
    ("magic", 1), ("version", 1),
    *[(name, 1) for name in layout.SCALAR_FIELDS],
    ("row_got_mowered", layout.ROWS),
    ("row_last_picked", layout.ROWS),
    ("row_second_last_picked", layout.ROWS),
    ("income_window_start", 1), ("income_count", 1), ("income_head", 1),
    ("income_tick", layout.INCOME_EVENTS), ("income_value", layout.INCOME_EVENTS),
    ("plant_count", 1), ("zombie_count", 1), ("projectile_count", 1),
    ("coin_count", 1), ("mower_count", 1), ("packet_count", 1),
]

ENTITY_RECORDS = (
    ("RootPlant", "PLANT_FIELDS", "PLANT_RECORD", layout.PLANT_SLOTS),
    ("RootZombie", "ZOMBIE_FIELDS", "ZOMBIE_RECORD", layout.ZOMBIE_SLOTS),
    ("RootProjectile", "PROJECTILE_FIELDS", "PROJECTILE_RECORD", layout.PROJECTILE_SLOTS),
    ("RootCoin", "COIN_FIELDS", "COIN_RECORD", layout.COIN_SLOTS),
    ("RootMower", "MOWER_FIELDS", "MOWER_RECORD", layout.MOWER_SLOTS),
    ("RootPacket", "PACKET_FIELDS", "PACKET_RECORD", layout.PACKET_SLOTS),
)


def check(dump: dict[str, int]) -> list[str]:
    problems: list[str] = []

    def expect(what: str, want: int, note: str = "") -> None:
        got = dump.get(what)
        if got is None:
            problems.append(f"{what}: missing from the C++ dump")
        elif got != want:
            problems.append(f"{what}: C++ says {got}, Python says {want} {note}".rstrip())

    if dump["kMagic"] != layout.MAGIC or dump["kVersion"] != layout.VERSION:
        problems.append(f"magic/version: C++ {dump['kMagic']:08x}/{dump['kVersion']} vs "
                        f"Python {layout.MAGIC:08x}/{layout.VERSION}")

    # The scalar head: every field offset, then the total against ROOT_HEAD.
    offset = 0
    for name, count in ROOT_HEAD_FIELDS:
        expect(f"RootRecord.{camel(name)}", offset)
        offset += 4 * count
    if offset != layout.ROOT_HEAD.size:
        problems.append(f"RootRecord head: {offset} bytes by field list, "
                        f"Python ROOT_HEAD is {layout.ROOT_HEAD.size}, C++ says "
                        f"{dump['RootRecord.plants']} before the entity arrays")

    # Entity records: size, field order, then their placement inside RootRecord.
    cursor = dump["RootRecord.plants"]
    for class_name, fields_attr, struct_attr, slots in ENTITY_RECORDS:
        fields = getattr(layout, fields_attr)
        record = getattr(layout, struct_attr)
        expect(f"{class_name}.size", record.size)
        for index, field in enumerate(fields):
            expect(f"{class_name}.{camel(field)}", index * 4)
        cursor += slots * record.size
    expect("RootRecord.waveCount", cursor)
    expect("RootRecord.waveZombieCount", cursor + 4)
    expect("RootRecord.waveTable", cursor + 4 + 4 * layout.MAX_WAVES)
    expect("RootRecord.supported",
           cursor + 4 + 4 * layout.MAX_WAVES + 4 * layout.MAX_WAVES * layout.WAVE_ZOMBIE_SLOTS)
    expect("RootRecord.size", layout.ROOT_SIZE)

    # LevelInfo: the head, the two def arrays and the trailing multiplier.
    expect("LevelInfo.level", 0)
    expect("LevelInfo.numWaves", 4)
    expect("LevelInfo.playableRows", 8)
    expect("LevelInfo.allowed", 12)
    expect("LevelInfo.zombies", 12 + 4 * dump["kZombieTypes"])
    expect("LevelInfo.plantCount", dump["LevelInfo.zombies"] + dump["kZombieTypes"] * dump["ZombieDef.size"])
    expect("LevelInfo.plants", dump["LevelInfo.plantCount"] + 4)
    expect("LevelInfo.zombieCountMultiplier",
           dump["LevelInfo.plants"] + 16 * dump["PlantDef.size"])
    expect("LevelInfo.size", layout.LEVEL_SIZE)
    for index, field in enumerate(layout.ZOMBIE_DEF_FIELDS):
        expect(f"ZombieDef.{camel(field)}", index * 4)
    for index, field in enumerate(layout.PLANT_DEF_FIELDS):
        expect(f"PlantDef.{camel(field)}", index * 4)
    return problems


def main() -> int:
    dump = dump_layout()
    problems = check(dump)
    if problems:
        print("cudaroot layout drift:")
        for problem in problems:
            print("  " + problem)
        return 1
    print(f"layout OK: root {layout.ROOT_SIZE} bytes, level {layout.LEVEL_SIZE} bytes, "
          f"{len(ENTITY_RECORDS)} entity record kinds, magic {layout.MAGIC:08x} v{layout.VERSION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
