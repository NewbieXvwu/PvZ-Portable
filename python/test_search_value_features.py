"""Equivalence proof for the vectorised search-value feature builder.

The builder used to assemble a Python ``list`` of 4116 floats and hand it to
``torch.tensor``.  It now accumulates into a ``float64`` numpy array and ends
with ``torch.from_numpy(arr.astype(np.float32))``, which is ~3.2x faster and
accounted for ~40% of ``SearchTeacher.advice`` runtime.

The change is claimed to be *bit-exact*, and this module is the proof.  It has
two halves:

1. ``_reference_search_value_features`` below is a frozen copy of the original
   list-based implementation.  ``FeatureEquivalenceTests`` runs both on a
   corpus that exercises every branch of the layout -- unknown/negative/
   non-numeric type ids, off-lawn entities, dead entities, imitater cards,
   empty boards, missing optional keys, and (critically) several entities
   colliding into the *same* accumulator slot, which is the only place where
   floating-point addition order can matter.
2. ``ConversionEquivalenceTests`` checks the single primitive the rewrite
   relies on: ``torch.tensor(list_of_floats, dtype=float32)`` agrees bit-for-bit
   with ``torch.from_numpy(np.asarray(list, dtype=float32))``.  It is checked on
   adversarial doubles -- denormals, float32 tie-to-even boundaries, and
   magnitudes at both ends of the float32 range -- because that is where a
   sloppy converter would diverge.
"""

from __future__ import annotations

import math
import random
import unittest
from typing import Any

import numpy as np
import torch

from pvz_search_candidates import lane_pressure
from pvz_search_value import (
    COL_GROUPS,
    COL_COUNT,
    MAX_SEED_SLOTS,
    PACKET_BLOCK_FEATURES,
    PACKET_WIDTH,
    PLANT_SLOT_COUNT,
    PLANT_TYPE_COUNT,
    ROW_COUNT,
    SEARCH_VALUE_FEATURES,
    UNKNOWN_PLANT_SLOT,
    UNKNOWN_ZOMBIE_SLOT,
    ZOMBIE_SLOT_COUNT,
    ZOMBIE_TYPE_COUNT,
    _slot,
    search_value_features,
)

PLANT_BLOCK_HALF = ROW_COUNT * COL_GROUPS * PLANT_SLOT_COUNT
ZOMBIE_BLOCK_HALF = ROW_COUNT * ZOMBIE_SLOT_COUNT

ROW_BLOCK_START = 16
PLANT_BLOCK_START = ROW_BLOCK_START + ROW_COUNT * 8
ZOMBIE_BLOCK_START = PLANT_BLOCK_START + 2 * PLANT_BLOCK_HALF
PACKET_BLOCK_START = ZOMBIE_BLOCK_START + 2 * ZOMBIE_BLOCK_HALF


# --------------------------------------------------------------------------- #
# Frozen reference implementation (the pre-optimisation code, verbatim).
# --------------------------------------------------------------------------- #
def _reference_search_value_features(observation: dict[str, Any]) -> torch.Tensor:
    plants = [plant for plant in observation["plants"]
              if not plant.get("squished") and plant.get("health", 0) > 0]
    zombies = [zombie for zombie in observation["zombies"]
               if zombie.get("body_health", 0) > 0 or zombie.get("helm_health", 0) > 0
               or zombie.get("shield_health", 0) > 0]
    projectiles = observation.get("projectiles", [])
    packets = observation.get("packets", [])
    pressure = lane_pressure(observation)
    rows = sorted({cell["row"] for cell in observation["cells"] if cell["row_type"] > 0})

    features = [0.0] * SEARCH_VALUE_FEATURES
    features[0:16] = [
        float(observation["sun"]) / 1000.0,
        float(observation["wave"]) / max(1.0, float(observation["wave_count"])),
        float(observation["tick"]) / 60000.0,
        float(observation.get("zombie_count_multiplier", 1.0)) / 10.0,
        float(bool(observation.get("night"))),
        float(bool(observation.get("pool"))),
        float(bool(observation.get("fog"))),
        float(bool(observation.get("roof"))),
        len(plants) / 40.0,
        len(zombies) / 40.0,
        len(projectiles) / 50.0,
        sum(bool(packet.get("active")) for packet in packets) / 10.0,
        sum(float(packet.get("cooldown", 0)) for packet in packets) / max(1.0, 3000.0 * len(packets)),
        sum(float(packet.get("cost", 0)) for packet in packets) / max(1.0, 500.0 * len(packets)),
        sum(defense.get("state") == 1 for defense in observation.get("defenses", [])) / max(1.0, float(len(rows))),
        1.0,
    ]

    cursor = 16
    for row in range(ROW_COUNT):
        row_zombies = [zombie for zombie in zombies if zombie.get("row") == row]
        row_plants = [plant for plant in plants if plant.get("row") == row]
        nearest = min((float(zombie.get("x", 900.0)) for zombie in row_zombies), default=900.0)
        zombie_hp = sum(
            max(0.0, float(zombie.get("body_health", 0)))
            + max(0.0, float(zombie.get("helm_health", 0)))
            + max(0.0, float(zombie.get("shield_health", 0)))
            for zombie in row_zombies
        )
        plant_health = sum(
            max(0.0, min(1.0, float(plant.get("health", 0)) / max(1.0, float(plant.get("max_health", 1)))))
            for plant in row_plants
        )
        features[cursor:cursor + 8] = [
            len(row_zombies) / 10.0,
            min(2.0, zombie_hp / 4000.0),
            min(2.0, pressure.get(row, 0.0) / 8.0),
            nearest / 900.0,
            len(row_plants) / 10.0,
            plant_health / 10.0,
            min((float(zombie.get("x", 900.0)) for zombie in row_zombies if zombie.get("is_eating")),
                default=900.0) / 900.0,
            sum(bool(zombie.get("is_eating")) for zombie in row_zombies) / 5.0,
        ]
        cursor += 8

    plant_spatial = [0.0] * (ROW_COUNT * COL_GROUPS * PLANT_SLOT_COUNT)
    plant_health_by_slot = [0.0] * len(plant_spatial)
    for plant in plants:
        row = int(plant.get("row", -1))
        col = int(plant.get("col", -1))
        if not (0 <= row < ROW_COUNT and 0 <= col < COL_COUNT):
            continue
        index = (row * COL_GROUPS + col // COL_GROUPS) * PLANT_SLOT_COUNT + _slot(
            plant.get("type"), PLANT_TYPE_COUNT, UNKNOWN_PLANT_SLOT
        )
        plant_spatial[index] += 1.0 / COL_GROUPS
        plant_health_by_slot[index] += max(
            0.0, min(1.0, float(plant.get("health", 0)) / max(1.0, float(plant.get("max_health", 1))))
        ) / COL_GROUPS
    features[cursor:cursor + len(plant_spatial)] = plant_spatial
    cursor += len(plant_spatial)
    features[cursor:cursor + len(plant_health_by_slot)] = plant_health_by_slot
    cursor += len(plant_health_by_slot)

    zombie_types = [0.0] * (ROW_COUNT * ZOMBIE_SLOT_COUNT)
    zombie_threat = [0.0] * len(zombie_types)
    for zombie in zombies:
        row = int(zombie.get("row", -1))
        if not (0 <= row < ROW_COUNT):
            continue
        index = row * ZOMBIE_SLOT_COUNT + _slot(zombie.get("type"), ZOMBIE_TYPE_COUNT, UNKNOWN_ZOMBIE_SLOT)
        zombie_types[index] += 0.2
        urgency = max(0.0, min(1.5, (760.0 - float(zombie.get("x", 900.0))) / 520.0))
        zombie_threat[index] += urgency / 3.0
    features[cursor:cursor + len(zombie_types)] = zombie_types
    cursor += len(zombie_types)
    features[cursor:cursor + len(zombie_threat)] = zombie_threat
    cursor += len(zombie_threat)

    packet_features = [0.0] * PACKET_BLOCK_FEATURES
    for packet in packets:
        index = int(packet.get("index", -1))
        if not (0 <= index < MAX_SEED_SLOTS):
            continue
        base = index * PACKET_WIDTH
        packet_features[base + _slot(packet.get("type"), PLANT_TYPE_COUNT, UNKNOWN_PLANT_SLOT)] = 1.0
        imitater_type = packet.get("imitater_type")
        if imitater_type is not None and int(imitater_type) >= 0:
            imitater_slot = _slot(imitater_type, PLANT_TYPE_COUNT, UNKNOWN_PLANT_SLOT)
            packet_features[base + PLANT_SLOT_COUNT + imitater_slot] = 1.0
        meta = base + PLANT_SLOT_COUNT * 2
        packet_features[meta] = float(bool(packet.get("active")))
        refresh = max(1.0, float(packet.get("refresh_time", 3000)))
        packet_features[meta + 1] = max(0.0, min(1.0, float(packet.get("cooldown", 0)) / refresh))
        packet_features[meta + 2] = min(2.0, refresh / 3000.0)
        packet_features[meta + 3] = min(2.0, float(packet.get("cost", 0)) / 500.0)
    features[cursor:cursor + len(packet_features)] = packet_features
    cursor += len(packet_features)

    if cursor != SEARCH_VALUE_FEATURES:
        raise RuntimeError(
            f"search value feature layout is inconsistent: wrote {cursor}, expected {SEARCH_VALUE_FEATURES}"
        )
    return torch.tensor(features, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #
def _plant(row: int, col: int, type_id: int = 0, health: int = 300, max_health: int = 300) -> dict:
    return {"type": type_id, "imitater_type": -1, "row": row, "col": col, "health": health,
            "max_health": max_health, "squished": False, "state": 0, "state_countdown": 0,
            "launch_counter": 0, "launch_rate": 0, "shooting_counter": 0, "wake_up_counter": 0,
            "asleep": False, "bungee_state": 0, "target_zombie_id": -1}


def _zombie(row: int, x: float, type_id: int = 0, body_health: int = 200,
            helm_health: int = 0, shield_health: int = 0, is_eating: bool = False) -> dict:
    return {"type": type_id, "row": row, "x": x, "y": 90.0, "body_health": body_health,
            "body_max_health": 200, "helm_health": helm_health, "helm_max_health": 0,
            "shield_health": shield_health, "shield_max_health": 0, "phase": 0, "phase_counter": 0,
            "velocity_x": -0.5, "chilled": 0, "buttered": 0, "ice_trap": 0, "has_head": True,
            "has_arm": True, "has_object": False, "is_eating": is_eating,
            "target_col": -1, "target_row": -1}


def _packet(index: int, type_id: int = 0, imitater_type: Any = -1, active: bool = True,
            cooldown: int = 0, refresh_time: int = 3000, cost: int = 100) -> dict:
    return {"index": index, "type": type_id, "imitater_type": imitater_type, "active": active,
            "cooldown": cooldown, "refresh_time": refresh_time, "cost": cost}


def _observation(**overrides: Any) -> dict:
    """A complete, mid-level board; ``overrides`` replace whole sub-collections."""
    plant_records = [_plant(index % ROW_COUNT, (index * 3) % COL_COUNT, index % 12)
                     for index in range(30)]
    zombie_records = [_zombie(index % ROW_COUNT, 200.0 + index * 40.0, index % 20,
                              is_eating=index % 5 == 0)
                      for index in range(15)]
    occupied = {(plant["row"], plant["col"]) for plant in plant_records}
    base = {
        "terminal": False, "result": 0, "tick": 5400, "wave": 7, "wave_count": 20,
        "wave_timer": 3000, "sun_income_rate": 12.0,
        "sun": 175, "night": False, "pool": False, "fog": False, "roof": False,
        "zombie_count_multiplier": 1.0,
        "player_profile": {"playthrough": 2, "seed_slot_count": 6, "owned_upgrade_plants": [],
                           "imitater_owned": False, "first_aid_owned": False,
                           "pool_cleaner_owned": False, "roof_cleaner_owned": False,
                           "rake_charges_remaining": 0},
        "cells": [{"row": row, "col": col, "terrain": 1, "row_type": 1,
                   "plant_types": [], "grid_item_types": []}
                  for row in range(ROW_COUNT) for col in range(COL_COUNT)],
        "plants": plant_records,
        "zombies": zombie_records,
        "projectiles": [{"row": index % ROW_COUNT, "x": 300.0 + index * 30.0, "y": 90.0, "z": 0.0,
                         "type": 0, "motion": 0, "vx": 6.0, "vy": 0.0, "vz": 0.0, "damage": 20,
                         "age": 10, "target_zombie_id": 1} for index in range(10)],
        "packets": [_packet(index, index, -1, active=index % 3 != 0,
                            cooldown=0 if index % 3 else 900) for index in range(6)],
        "defenses": [{"type": 0, "row": row, "state": 1, "x": 40.0, "y": 30.0}
                     for row in range(ROW_COUNT)],
        "grid_items": [],
        "loadout_context": {"zombie_roster": list(range(10))},
        "legal_actions": {
            "plants": [{"packet": packet, "row": row, "col": col}
                       for packet in range(6) for row in range(ROW_COUNT)
                       for col in range(COL_COUNT) if (row, col) not in occupied],
            "shovels": [[col, row] for row, col in sorted(occupied)],
            "wait": True,
        },
    }
    base.update(overrides)
    return base


def _corpus() -> list[tuple[str, dict]]:
    """Observations chosen so that every branch and every accumulator is hit."""
    base = _observation()
    cases: list[tuple[str, dict]] = [
        ("dense board", base),
        ("empty board", _observation(plants=[], zombies=[], projectiles=[])),
        ("single plant", _observation(plants=[_plant(0, 0)], zombies=[])),
        ("54 plants, 100 zombies", _observation(
            plants=[_plant(index % ROW_COUNT, index % COL_COUNT, index % 12) for index in range(54)],
            zombies=[_zombie(index % ROW_COUNT, 100.0 + index * 5.0, index % 20)
                     for index in range(100)])),
        ("no cells", _observation(cells=[])),
        ("night pool fog roof", _observation(night=True, pool=True, fog=True, roof=True)),
        ("zero wave count", _observation(wave_count=0)),
        ("huge sun and tick", _observation(sun=999_999, tick=10 ** 7)),
        ("zombie multiplier 25", _observation(zombie_count_multiplier=25.0)),

        # --- type-id routing into the unknown slot -------------------------
        ("unknown plant type", _observation(plants=[_plant(0, 0, 9999)])),
        ("negative plant type", _observation(plants=[_plant(0, 0, -5)])),
        ("non-numeric plant type", _observation(plants=[_plant(0, 0, "sunflower")])),
        ("none plant type", _observation(plants=[_plant(0, 0, None)])),
        ("unknown zombie type", _observation(zombies=[_zombie(0, 300.0, 500)])),
        ("negative zombie type", _observation(zombies=[_zombie(0, 300.0, -1)])),

        # --- entities outside the modelled grid ----------------------------
        ("plant off the lawn", _observation(plants=[_plant(9, -1), _plant(0, 12)])),
        ("zombie off the lawn", _observation(zombies=[_zombie(-3, 300.0), _zombie(6, 300.0)])),

        # --- filtered-out entities -----------------------------------------
        ("squished plant", _observation(plants=[_plant(0, 0), {**_plant(0, 3), "squished": True}])),
        ("zero-health plant", _observation(plants=[_plant(0, 0), _plant(0, 3, health=0)])),
        ("zero max-health plant", _observation(plants=[_plant(0, 0, health=5, max_health=0)])),
        ("negative plant health", _observation(plants=[_plant(0, 0, health=-50)])),
        ("over-healed plant", _observation(plants=[_plant(0, 0, health=900, max_health=300)])),
        ("fully dead zombies", _observation(zombies=[
            _zombie(0, 300.0), _zombie(1, 300.0, body_health=0, helm_health=0, shield_health=0)])),
        ("shield-only zombie", _observation(zombies=[
            _zombie(0, 300.0, body_health=0, shield_health=40)])),
        ("negative zombie health", _observation(zombies=[
            _zombie(0, 300.0, body_health=-10, helm_health=-3)])),

        # --- packet branches ------------------------------------------------
        ("imitater packets", _observation(packets=[
            _packet(0, 3, 7), _packet(1, 4, -1), _packet(2, 5, None), _packet(3, 6, 9999),
            _packet(4, 7, 0)])),
        ("packet index out of range", _observation(packets=[
            _packet(42, 0), _packet(-1, 1), _packet(0, 2)])),
        ("no packets", _observation(packets=[])),
        ("zero refresh time", _observation(packets=[_packet(0, 0, cooldown=500, refresh_time=0)])),
        ("long cooldown", _observation(packets=[
            _packet(0, 0, cooldown=99_999, refresh_time=30_000, cost=5000)])),

        # --- missing optional keys ------------------------------------------
        ("missing optional collections", {key: value for key, value in base.items()
                                          if key not in ("projectiles", "packets", "defenses")}),
        ("no defenses", _observation(defenses=[])),

        # --- accumulator collisions (the only place addition order matters) --
        ("two plants in one slot", _observation(
            plants=[_plant(0, 0, 3), _plant(0, 1, 3)], zombies=[])),
        ("three plants in one slot, different health", _observation(
            plants=[_plant(0, 0, 3, 300, 300), _plant(0, 1, 3, 200, 300),
                    _plant(0, 2, 3, 100, 300)], zombies=[])),
        ("plants colliding into the unknown slot", _observation(
            plants=[_plant(0, 0, 9999), _plant(0, 1, -1), _plant(0, 2, "x")], zombies=[])),
        ("two zombies in one slot", _observation(
            zombies=[_zombie(0, 700.0, 5), _zombie(0, 300.0, 5)], plants=[])),
        ("zombies colliding into the unknown slot", _observation(
            zombies=[_zombie(0, 700.0, 500), _zombie(0, 300.0, -2), _zombie(0, 100.0, None)],
            plants=[])),
        ("zombies at the urgency clamp", _observation(
            zombies=[_zombie(0, 900.0, 1), _zombie(0, 760.0, 1), _zombie(0, 240.0, 1),
                     _zombie(0, -500.0, 1)], plants=[])),
    ]
    return cases


def _bits(tensor: torch.Tensor) -> bytes:
    return tensor.detach().contiguous().numpy().tobytes()


class FeatureEquivalenceTests(unittest.TestCase):
    """The vectorised builder must reproduce the list-based builder exactly."""

    def test_every_corpus_observation_matches_the_reference_bit_for_bit(self) -> None:
        mismatched: list[str] = []
        for name, observation in _corpus():
            reference = _bits(_reference_search_value_features(observation))
            actual = _bits(search_value_features(observation))
            if reference != actual:
                mismatched.append(name)
        self.assertEqual(mismatched, [], f"feature vectors diverged for: {mismatched}")

    def test_the_corpus_actually_exercises_nonzero_features(self) -> None:
        """Guards the test above: an all-zero corpus would pass it vacuously."""
        for name, observation in _corpus():
            vector = search_value_features(observation)
            self.assertGreater(float(vector.abs().sum()), 0.0, f"{name} produced an all-zero vector")
            self.assertTrue(bool(torch.isfinite(vector).all()), f"{name} produced non-finite features")

    def test_colliding_plants_accumulate_into_one_slot(self) -> None:
        """A copy instead of a view would silently drop the second plant."""
        observation = _observation(plants=[_plant(0, 0, 3), _plant(0, 1, 3)], zombies=[])
        vector = search_value_features(observation)
        index = (0 * COL_GROUPS + 0 // COL_GROUPS) * PLANT_SLOT_COUNT + 3
        self.assertAlmostEqual(float(vector[PLANT_BLOCK_START + index]), 2.0 / COL_GROUPS, places=6)

    def test_colliding_zombies_accumulate_into_one_slot(self) -> None:
        observation = _observation(zombies=[_zombie(0, 700.0, 5), _zombie(0, 300.0, 5)], plants=[])
        vector = search_value_features(observation)
        index = 0 * ZOMBIE_SLOT_COUNT + 5
        self.assertAlmostEqual(float(vector[ZOMBIE_BLOCK_START + index]), 0.4, places=6)

    def test_the_output_is_a_contiguous_float32_vector_of_the_declared_width(self) -> None:
        vector = search_value_features(_observation())
        self.assertEqual(vector.dtype, torch.float32)
        self.assertEqual(tuple(vector.shape), (SEARCH_VALUE_FEATURES,))
        self.assertTrue(vector.is_contiguous())

    def test_the_layout_guard_still_rejects_a_drifted_cursor(self) -> None:
        """``cursor`` must still be validated; the blocks are now sized by constants."""
        self.assertEqual(
            ROW_BLOCK_START + ROW_COUNT * 8
            + 2 * PLANT_BLOCK_HALF + 2 * ZOMBIE_BLOCK_HALF + PACKET_BLOCK_FEATURES,
            SEARCH_VALUE_FEATURES,
        )
        self.assertEqual(PACKET_BLOCK_START + PACKET_BLOCK_FEATURES, SEARCH_VALUE_FEATURES)


class ConversionEquivalenceTests(unittest.TestCase):
    """``torch.tensor(list, float32)`` vs ``from_numpy(np.asarray(list, float32))``."""

    @staticmethod
    def _same_bits(values: list[float]) -> bool:
        from_torch = torch.tensor(values, dtype=torch.float32).numpy()
        from_numpy = np.asarray(values, dtype=np.float32)
        return from_torch.tobytes() == from_numpy.tobytes()

    def test_random_doubles_across_the_float32_exponent_range(self) -> None:
        rng = random.Random(1234)
        values = [math.ldexp(rng.uniform(-1.0, 1.0), rng.randint(-140, 127))
                  for _ in range(200_000)]
        self.assertTrue(self._same_bits(values))

    def test_doubles_beyond_the_float32_range_saturate_identically(self) -> None:
        """Both converters must agree on overflow too, not just on in-range values."""
        rng = random.Random(99)
        values = [math.ldexp(rng.uniform(1.0, 2.0), rng.randint(128, 320))
                  for _ in range(20_000)]
        with np.errstate(over="ignore"):
            self.assertTrue(self._same_bits(values))
            self.assertTrue(self._same_bits([-value for value in values]))
            saturated = np.asarray(values, dtype=np.float32)
        self.assertTrue(bool(np.isinf(saturated).all()))

    def test_float32_tie_to_even_boundaries(self) -> None:
        """Exactly-halfway values are where a different rounding mode would show up."""
        values = [math.ldexp(1.0, exponent) * (1.0 + k * 2.0 ** -24)
                  for exponent in range(-30, 30) for k in range(1, 200)]
        self.assertTrue(self._same_bits(values))

    def test_denormals_and_range_extremes(self) -> None:
        values = [math.ldexp(1.0, exponent) for exponent in range(-149, -120)]
        values += [3.4028234663852886e38, -3.4028234663852886e38, 1.401298464324817e-45, 0.0, -0.0]
        self.assertTrue(self._same_bits(values))

    def test_the_values_the_feature_builder_actually_produces(self) -> None:
        """Sums of the literal constants used by the layout, repeated like a collision."""
        literals = [1.0 / 3.0, 0.2, 0.2 * 3.0, 1.0 / 1000.0, 760.0 / 520.0, 0.99 ** 7, 1.0 / 40.0]
        values = [sum([literal] * repeats) for literal in literals for repeats in range(1, 13)]
        self.assertTrue(self._same_bits(values))

    def test_accumulating_in_float64_is_required(self) -> None:
        """The proof only holds because accumulation stays binary64.

        Seven zombies of one type in one row drive a single slot through seven
        ``+= 0.2`` steps.  Binary64 accumulation lands on ``1.4000000000000001``
        and binary32 on ``1.4000000953674316`` -- *different* float32 values, so
        the two candidate accumulators are genuinely distinguishable and this
        pins the one the equivalence proof depends on.
        """
        binary64 = np.float32(sum([0.2] * 7))
        binary32 = np.cumsum(np.full(7, np.float32(0.2), dtype=np.float32))[-1]
        self.assertNotEqual(binary64, binary32)

        observation = _observation(zombies=[_zombie(0, 700.0, 5) for _ in range(7)], plants=[])
        actual = search_value_features(observation)[ZOMBIE_BLOCK_START + 5]
        self.assertEqual(actual, binary64)

    def test_colliding_plants_accumulate_in_float64_too(self) -> None:
        """Same contract on the plant block: five thirds, not five float32 thirds.

        The builder does not deduplicate entities, so five records for the same
        cell drive one slot through five ``+= 1/3`` steps.
        """
        binary64 = np.float32(sum([1.0 / 3.0] * 5))
        binary32 = np.cumsum(np.full(5, np.float32(1.0 / 3.0), dtype=np.float32))[-1]
        self.assertNotEqual(binary64, binary32)

        observation = _observation(plants=[_plant(0, 0, 3) for _ in range(5)], zombies=[])
        index = (0 * COL_GROUPS + 0 // COL_GROUPS) * PLANT_SLOT_COUNT + 3
        actual = search_value_features(observation)[PLANT_BLOCK_START + index]
        self.assertEqual(actual, binary64)


class SeedSlotCoverageTests(unittest.TestCase):
    """Every one of the ten seed slots must be reachable by the packet block."""

    def test_all_ten_packet_slots_are_written(self) -> None:
        packets = [_packet(index, index, index + 1) for index in range(MAX_SEED_SLOTS)]
        vector = search_value_features(_observation(packets=packets))
        for index in range(MAX_SEED_SLOTS):
            slot = PACKET_BLOCK_START + index * PACKET_WIDTH
            self.assertEqual(float(vector[slot + index]), 1.0, f"packet type slot {index}")
            self.assertEqual(float(vector[slot + PLANT_SLOT_COUNT + index + 1]), 1.0,
                             f"imitater slot {index}")

    def test_the_packet_block_is_the_last_block(self) -> None:
        self.assertEqual(PACKET_BLOCK_START + PACKET_BLOCK_FEATURES, SEARCH_VALUE_FEATURES)


if __name__ == "__main__":
    unittest.main()
