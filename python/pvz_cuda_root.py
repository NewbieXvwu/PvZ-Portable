"""Read the ``.cudaroot`` record stream that ``CUDA_DUMP`` writes (layout: cuda/CudaRollout.h).

Pure Python and hand-mirrored from the C++ struct, so it doubles as a check on the layout: the
offsets asserted below are the ones the compiled exporter produced (``scripts/cuda_layout_check.py``
prints both sides and fails if they drift).  Used by the phase-1 comparison to confirm that what the
GPU uploads is what the original simulator held.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

MAGIC = 0x50564333            # "PVC3"
VERSION = 3
RECORD_LEVEL, RECORD_ROOT = 1, 2

INCOME_EVENTS = 160
PLANT_SLOTS, ZOMBIE_SLOTS, PROJECTILE_SLOTS = 60, 64, 96
COIN_SLOTS, MOWER_SLOTS, PACKET_SLOTS = 24, 6, 10
ROWS = 6

PLANT_FIELDS = ("cell", "type", "health", "max_health", "state", "state_countdown",
                "launch_counter", "launch_rate", "special")
ZOMBIE_FIELDS = ("type", "row", "body", "body_max", "helm", "helm_max", "shield", "shield_max",
                 "flying", "phase", "phase_counter", "state_bits", "chilled", "age", "from_wave",
                 "target_cell", "pos_x", "pos_y", "vel_x")
PROJECTILE_FIELDS = ("type", "row", "damage", "age", "pos_x", "pos_y", "vel_x", "vel_y")
COIN_FIELDS = ("value", "ticks_to_collect")
MOWER_FIELDS = ("row", "state", "timer", "flags", "chomp_counter", "pos_x")
PACKET_FIELDS = ("type", "active", "refresh_counter", "refresh_time", "cost")

SCALAR_FIELDS = ("tick", "sun", "wave", "num_waves", "zombie_count_down", "zombie_count_down_start",
                 "zombie_health_wave_start", "zombie_health_to_next_wave", "huge_wave_count_down",
                 "sun_count_down", "num_suns_fallen", "sun_money_produced", "zombies_killed",
                 "plants_eaten", "plants_planted", "triggered_lawn_mowers", "result")

RECORD_TAG = struct.Struct("<iii")
# RootRecord repeats magic/version at the head of its payload (LawnApp-side structs are self
# describing), then: 17 scalars, the 6 mowed-row flags, 6 + 6 row-picking floats, the income window
# head, the income events (tick/value interleaved so one slice reads one event), the entity counts.
ROOT_HEAD = struct.Struct("<ii" + "i" * 17 + "i" * ROWS + "f" * ROWS + "f" * ROWS
                          + "i" * 3 + "ii" * INCOME_EVENTS + "i" * 6)
ROOT_TAIL = struct.Struct("<ii")          # supported, unsupported reason
ZOMBIE_RECORD = struct.Struct("16i3f")
PLANT_RECORD = struct.Struct("9i")
PROJECTILE_RECORD = struct.Struct("4i4f")
COIN_RECORD = struct.Struct("2i")
MOWER_RECORD = struct.Struct("5if")
PACKET_RECORD = struct.Struct("5i")
ROOT_SIZE = ROOT_HEAD.size \
    + PLANT_SLOTS * PLANT_RECORD.size + ZOMBIE_SLOTS * ZOMBIE_RECORD.size \
    + PROJECTILE_SLOTS * PROJECTILE_RECORD.size + COIN_SLOTS * COIN_RECORD.size \
    + MOWER_SLOTS * MOWER_RECORD.size + PACKET_SLOTS * PACKET_RECORD.size + ROOT_TAIL.size
LEVEL_HEAD = struct.Struct("3i" + "i" * 34)
ZOMBIE_DEF = struct.Struct("9i2f")
LEVEL_PLANT_DEF = struct.Struct("6i")
LEVEL_SIZE = LEVEL_HEAD.size + 34 * ZOMBIE_DEF.size + 4 + 16 * LEVEL_PLANT_DEF.size + 4


def _records(blob: bytes):
    offset = 0
    while offset < len(blob):
        magic, version, kind = RECORD_TAG.unpack_from(blob, offset)
        if magic != MAGIC or version != VERSION:
            raise ValueError(f"bad cuda root tag at {offset}: {magic:08x}/{version}/{kind}")
        offset += RECORD_TAG.size
        if kind == RECORD_LEVEL:
            yield kind, blob[offset:offset + LEVEL_SIZE]
            offset += LEVEL_SIZE
        elif kind == RECORD_ROOT:
            yield kind, blob[offset:offset + ROOT_SIZE]
            offset += ROOT_SIZE
        else:
            raise ValueError(f"unknown cuda root record kind {kind} at {offset}")


def _entities(blob: bytes, offset: int, count: int, fields: tuple[str, ...],
              record: struct.Struct) -> list[dict]:
    return [dict(zip(fields, record.unpack_from(blob, offset + index * record.size)))
            for index in range(count)]


def parse_root(payload: bytes) -> dict[str, Any]:
    head = ROOT_HEAD.unpack_from(payload)
    if head[0] != MAGIC or head[1] != VERSION:
        raise ValueError(f"root record self-description mismatch: {head[:2]}")
    record: dict[str, Any] = dict(zip(SCALAR_FIELDS, head[2:2 + len(SCALAR_FIELDS)]))
    index = 2 + len(SCALAR_FIELDS)
    record["row_got_mowered"] = list(head[index:index + ROWS]); index += ROWS
    record["row_last_picked"] = list(head[index:index + ROWS]); index += ROWS
    record["row_second_last_picked"] = list(head[index:index + ROWS]); index += ROWS
    record["income_window_start"], record["income_count"], record["income_head"] = head[index:index + 3]
    index += 3
    record["income_events"] = [(tick, value) for tick, value in
                               zip(head[index:index + INCOME_EVENTS],
                                   head[index + INCOME_EVENTS:index + 2 * INCOME_EVENTS])
                               if value != 0]
    index += 2 * INCOME_EVENTS
    counts = head[index:index + 6]
    record["plant_count"], record["zombie_count"], record["projectile_count"] = counts[:3]
    record["coin_count"], record["mower_count"], record["packet_count"] = counts[3:]

    offset = ROOT_HEAD.size
    record["plants"] = _entities(payload, offset, record["plant_count"], PLANT_FIELDS, PLANT_RECORD)
    offset += PLANT_SLOTS * PLANT_RECORD.size
    record["zombies"] = _entities(payload, offset, record["zombie_count"], ZOMBIE_FIELDS, ZOMBIE_RECORD)
    offset += ZOMBIE_SLOTS * ZOMBIE_RECORD.size
    record["projectiles"] = _entities(payload, offset, record["projectile_count"],
                                      PROJECTILE_FIELDS, PROJECTILE_RECORD)
    offset += PROJECTILE_SLOTS * PROJECTILE_RECORD.size
    record["coins"] = _entities(payload, offset, record["coin_count"], COIN_FIELDS, COIN_RECORD)
    offset += COIN_SLOTS * COIN_RECORD.size
    record["mowers"] = _entities(payload, offset, record["mower_count"], MOWER_FIELDS, MOWER_RECORD)
    offset += MOWER_SLOTS * MOWER_RECORD.size
    record["packets"] = _entities(payload, offset, record["packet_count"], PACKET_FIELDS, PACKET_RECORD)
    offset += PACKET_SLOTS * PACKET_RECORD.size
    record["supported"], record["unsupported_reason"] = ROOT_TAIL.unpack_from(payload, offset)
    return record


def parse_level(payload: bytes) -> dict[str, Any]:
    head = LEVEL_HEAD.unpack_from(payload)
    info: dict[str, Any] = {"level": head[0], "num_waves": head[1], "playable_rows": head[2],
                            "allowed": list(head[3:])}
    offset = LEVEL_HEAD.size
    info["zombies"] = [dict(zip(("type", "value", "first_allowed_wave", "pick_weight", "body_health",
                                 "helm_health", "shield_health", "variant_count", "vel_x",
                                 "vel_min", "vel_max"), ZOMBIE_DEF.unpack_from(payload, offset + index * ZOMBIE_DEF.size)))
                       for index in range(34)]
    offset += 34 * ZOMBIE_DEF.size
    info["plant_count"] = struct.unpack_from("i", payload, offset)[0]
    offset += 4
    info["plants"] = [dict(zip(("seed_type", "cost", "refresh_time", "health", "launch_rate", "damage"),
                               LEVEL_PLANT_DEF.unpack_from(payload, offset + index * LEVEL_PLANT_DEF.size)))
                      for index in range(info["plant_count"])]
    offset += 16 * LEVEL_PLANT_DEF.size
    info["zombie_count_multiplier"] = struct.unpack_from("f", payload, offset)[0]
    return info


def read_cudaroot(path: str | Path) -> dict[str, Any]:
    blob = Path(path).read_bytes()
    level: dict[str, Any] | None = None
    roots: list[dict[str, Any]] = []
    for kind, payload in _records(blob):
        if kind == RECORD_LEVEL:
            level = parse_level(payload)
        else:
            roots.append(parse_root(payload))
    return {"level": level, "roots": roots}
