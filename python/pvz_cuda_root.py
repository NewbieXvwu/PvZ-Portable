"""Read the ``.cudaroot`` record stream that ``CUDA_DUMP`` writes (layout: cuda/CudaRollout.h).

Pure Python and hand-mirrored from the C++ struct, so it doubles as a check on the layout: the
offsets asserted below are the ones the compiled exporter produced (``scripts/cuda_layout_check.py``
prints both sides and fails if they drift).  Used by the phase-1 comparison to confirm that what the
GPU uploads is what the original simulator held.
"""
from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Any

MAGIC = 0x50564334            # "PVC4"
VERSION = 9                # explicit mHasObject bit for v2 (v8 lacked it; layout unchanged)
RECORD_LEVEL, RECORD_ROOT = 1, 2

INCOME_EVENTS = 160
PLANT_SLOTS, ZOMBIE_SLOTS, PROJECTILE_SLOTS = 60, 64, 96
COIN_SLOTS, MOWER_SLOTS, PACKET_SLOTS = 24, 6, 10
ROWS = 6
MAX_WAVES = 30
WAVE_ZOMBIE_SLOTS = 32

PLANT_FIELDS = ("cell", "type", "health", "max_health", "state", "state_countdown",
                "launch_counter", "launch_rate", "special", "extra_a", "extra_b", "on_board",
                "pos_x", "pos_y", "anim_time", "anim_rate", "anim_frames", "anim_loops",
                "born")
ZOMBIE_FIELDS = ("type", "row", "body", "body_max", "helm", "helm_max", "shield", "shield_max",
                 "flying", "phase", "phase_counter", "state_bits", "chilled", "age", "from_wave",
                 "target_cell", "pos_x", "pos_y", "vel_x", "anim_time", "anim_rate",
                 "anim_start", "anim_frames", "anim_loop_type", "anim_loops", "fade")
PROJECTILE_FIELDS = ("type", "row", "damage", "chill", "age", "pos_x", "pos_y", "vel_x", "vel_y")
COIN_FIELDS = ("value", "ticks_to_collect")
MOWER_FIELDS = ("row", "state", "timer", "flags", "chomp_counter", "pos_x")
PACKET_FIELDS = ("type", "active", "refresh_counter", "refresh_time", "cost")
PLANT_DEF_FIELDS = ("seed_type", "cost", "refresh_time", "health", "launch_rate", "damage",
                    "behavior", "shots", "prod_min", "prod_max", "prod_value", "chills",
                    "shoot_counter")
ZOMBIE_DEF_FIELDS = ("type", "value", "first_allowed_wave", "pick_weight", "body_health",
                     "helm_health", "shield_health", "variant_count", "vel_x", "vel_min", "vel_max")

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
ROOT_TAIL = struct.Struct("<iiii")        # history wave, history wave tick, supported, reason
# Board::mZombiesInWave is kept for corpus inspection. RESEED redraws unsent waves; CUDA continuation
# must use DrawFutureWave rather than this privileged table. It trails the entity arrays.
WAVE_BLOCK = struct.Struct("<i" + "i" * MAX_WAVES + "i" * (MAX_WAVES * WAVE_ZOMBIE_SLOTS))
ZOMBIE_RECORD = struct.Struct("16i5f5i")
PLANT_RECORD = struct.Struct("12i4f3i")   # ... anim_loops, born
PROJECTILE_RECORD = struct.Struct("5i4f")
COIN_RECORD = struct.Struct("2i")
MOWER_RECORD = struct.Struct("5if")
PACKET_RECORD = struct.Struct("5i")
ROOT_SIZE = ROOT_HEAD.size \
    + PLANT_SLOTS * PLANT_RECORD.size + ZOMBIE_SLOTS * ZOMBIE_RECORD.size \
    + PROJECTILE_SLOTS * PROJECTILE_RECORD.size + COIN_SLOTS * COIN_RECORD.size \
    + MOWER_SLOTS * MOWER_RECORD.size + PACKET_SLOTS * PACKET_RECORD.size \
    + WAVE_BLOCK.size + ROOT_TAIL.size
LEVEL_HEAD = struct.Struct("3i" + "i" * 34)
ZOMBIE_DEF = struct.Struct("9i2f")
LEVEL_PLANT_DEF = struct.Struct("13i")
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
    wave = WAVE_BLOCK.unpack_from(payload, offset)
    offset += WAVE_BLOCK.size
    wave_count = min(wave[0], MAX_WAVES)
    record["wave_count"] = wave[0]
    record["wave_zombie_count"] = list(wave[1:1 + MAX_WAVES])
    table = wave[1 + MAX_WAVES:]
    record["wave_table"] = [
        list(table[row * WAVE_ZOMBIE_SLOTS:row * WAVE_ZOMBIE_SLOTS
                    + min(record["wave_zombie_count"][row], WAVE_ZOMBIE_SLOTS)])
        for row in range(wave_count)]
    (record["history_wave"], record["history_wave_tick"],
     record["supported"], record["unsupported_reason"]) = ROOT_TAIL.unpack_from(payload, offset)
    return record


def parse_level(payload: bytes) -> dict[str, Any]:
    head = LEVEL_HEAD.unpack_from(payload)
    info: dict[str, Any] = {"level": head[0], "num_waves": head[1], "playable_rows": head[2],
                            "allowed": list(head[3:])}
    offset = LEVEL_HEAD.size
    info["zombies"] = [dict(zip(ZOMBIE_DEF_FIELDS,
                                ZOMBIE_DEF.unpack_from(payload, offset + index * ZOMBIE_DEF.size)))
                       for index in range(34)]
    offset += 34 * ZOMBIE_DEF.size
    info["plant_count"] = struct.unpack_from("i", payload, offset)[0]
    offset += 4
    info["plants"] = [dict(zip(PLANT_DEF_FIELDS,
                               LEVEL_PLANT_DEF.unpack_from(payload, offset + index * LEVEL_PLANT_DEF.size)))
                      for index in range(info["plant_count"])]
    offset += 16 * LEVEL_PLANT_DEF.size
    info["zombie_count_multiplier"] = struct.unpack_from("f", payload, offset)[0]
    return info


def read_cudaroot(path: str | Path) -> dict[str, Any]:
    blob = Path(path).read_bytes()
    level: dict[str, Any] = None
    roots: list[dict[str, Any]] = []
    for kind, payload in _records(blob):
        if kind == RECORD_LEVEL:
            level = parse_level(payload)
        else:
            roots.append(parse_root(payload))
    return {"level": level, "roots": roots}


# ---------------------------------------------------------------------------
# Rebuilding the *observation* a frame encoder would have seen
# ---------------------------------------------------------------------------
#
# The exporter copies the engine's fields; the value net never sees those, it sees the JSON the
# environment printed.  Feeding a record straight to ``pvz_value_net.featurize_numpy`` therefore only
# proves something when this translation is done in the open: every field below is what
# ``LawnApp::EnvironmentLeanObservation`` put in the array at the same position, and
# ``scripts/cuda_leaf_compare.py`` uses it to hold the CUDA core's own encoding against the Python one
# on the identical position.

ZB_EATING = 1 << 1            # cuda/CudaRollout.h:139; src/CudaExport.cpp:262 sets both of these
ZB_ON_BOARD = 1 << 8

INCOME_WINDOW_TICKS = 6000    # python/pvz_env.py's ``_sun_income_window_ticks``
CELL_PX = 80                  # Board::GridToPixelX(col) == col * 80 + 40
LAWN_XMIN = 40


def rendered_public_coordinate(value: float) -> float:
    """Round to six significant digits, half away from zero.

    ``std::ostringstream`` prints a double at its default precision of six significant digits, and
    that text is what the agent's JSON parses back -- so the value net was trained on the rounded
    number, not the engine's float.  ``cuda/SimCore.h``'s ``RenderedPublicCoordinate`` is the same
    function written for the device.  A tie between the two is unreachable in practice: the exact
    binary value of a float32 position is never precisely halfway at the seventh digit, so the
    rounding-mode difference (half away from zero here, half to even in ``printf``) never shows.
    """
    magnitude = abs(value)
    if magnitude <= 0.0:
        return value
    # Scientific exponent: 10**e <= |value| < 10**(e+1), negative for values below one.
    exponent = 0
    if magnitude >= 1.0:
        while exponent < 8 and magnitude >= 10.0 ** (exponent + 1):
            exponent += 1
    else:
        while exponent > -8 and magnitude < 10.0 ** exponent:
            exponent -= 1
    keep = exponent - 5
    scale = 10.0 ** keep if keep >= 0 else 1.0 / 10.0 ** (-keep)
    rounded = math.floor(magnitude / scale + 0.5) * scale
    return -rounded if value < 0.0 else rounded


def income_rate(record: dict[str, Any]) -> float:
    """``sun_income_rate`` as ``pvz_env`` annotated the observation with it: sun produced inside the
    last 6000 ticks, scaled to a per-1000-tick rate over the elapsed part of the episode."""
    tick = record["tick"]
    produced = sum(value for event_tick, value in record["income_events"]
                   if event_tick >= tick - INCOME_WINDOW_TICKS)
    elapsed = max(0, min(INCOME_WINDOW_TICKS, tick - record["income_window_start"]))
    return produced * 1000.0 / elapsed if elapsed else 0.0


def value_frame(record: dict[str, Any], level: dict[str, Any]) -> dict[str, Any]:
    """The ``render_episode._frame`` dict for one root, in the value net's encoding."""
    plants = [(entry["cell"] // 10, entry["cell"] % 10, entry["type"], entry["health"],
               entry["max_health"]) for entry in record["plants"]]   # cell = row * kCols(10) + col
    zombies = []
    for entry in record["zombies"]:
        bits = entry["state_bits"]
        zombies.append((entry["row"], entry["type"], rendered_public_coordinate(entry["pos_x"]),
                        entry["vel_x"], entry["body"] + entry["helm"] + entry["shield"],
                        bool(bits & ZB_ON_BOARD), bool(bits & ZB_EATING), entry["chilled"],
                        0, 0))
    # The exporter already dropped dead plants and out-of-range rows (src/CudaExport.cpp:213, :243) and
    # marks a squished plant's root unsupported, so what is here is what the observation listed.
    return {
        "tick": record["tick"],
        "wave": record["wave"],
        "sun": record["sun"],
        "sun_income_rate": income_rate(record),
        "plants": plants,
        "zombies": zombies,
        "mowers": [[entry["row"], entry["state"]] for entry in record["mowers"]],
        "packets": [[index, entry["type"], bool(entry["active"]), entry["refresh_counter"],
                     entry["refresh_time"], entry["cost"]]
                    for index, entry in enumerate(record["packets"])],
        # The Python derives lanes from the terrain grid; the record has no grid, and every terrain this
        # corpus uses has its playable rows as rows 0..n-1 (5 day-grass, 6 pool).
        "lanes": list(range(level["playable_rows"])),
        "scene": {"level": level["level"], "wave_count": record["num_waves"]},
        "result": record["result"],
        "terminal": record["result"] != 0,
    }


def value_context(level: dict[str, Any]) -> tuple[float, list[int]]:
    """``(zombie_count_multiplier, zombie roster)`` for ``pvz_value_net.context_vector``."""
    return level["zombie_count_multiplier"], [kind for kind, allowed in enumerate(level["allowed"])
                                             if allowed]
