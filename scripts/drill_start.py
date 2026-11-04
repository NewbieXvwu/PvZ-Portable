"""Shared drill-start mechanics (task235).

A drill cell reuses the demonstration-start convention (archive + decision + RESEED
future seed), but it is scored on a **local, short-horizon** success instead of the
whole level.  This module is the one implementation of the three pieces that would
otherwise fork between the task234 prototype (``scripts/drill_run.py``) and the
production collector (``scripts/search_play.py``):

* restore an archive at ``decision`` -- the existing ``human_demo.replay`` path when
  the archive has ``ticks.jsonl``, the recorded-action prefix path otherwise;
* read the on-board zombie ids from the engine (privileged when available);
* judge ``criterion``: every target gone/off board, no mower fired, no house breach.

No tactic is encoded here: the caller supplies the target ids and the horizon from
the mined drill.  The miner's roster/position rule lives in ``scripts/drill_mine.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_HORIZON = 1200
DEFAULT_CRITERION = dict(threats_neutralized=True, mower_unchanged=True,
                         house_unchanged=True, secondary="plants_lost")


def drill_spec(task_definition: dict | None) -> dict | None:
    """The ``drill_start`` block of a task definition, or None for every other task."""
    block = (task_definition or {}).get("drill_start")
    return block if isinstance(block, dict) else None


def drill_horizon(task_definition: dict | None) -> int | None:
    block = drill_spec(task_definition)
    return None if block is None else int(block.get("horizon", DEFAULT_HORIZON))


def drill_targets(task_definition: dict | None) -> list[dict]:
    block = drill_spec(task_definition)
    return list((block or {}).get("targets") or [])


def criterion_of(task_definition: dict | None) -> dict:
    block = drill_spec(task_definition)
    return dict(DEFAULT_CRITERION, **((block or {}).get("criterion") or {}))


def replayable_by_operations(archive: Path, frames, decision: int) -> bool:
    """A human/LLM import carries ticks.jsonl and per-frame ``operation`` numbers."""
    index = min(max(int(decision) - 1, 0), len(frames) - 1)
    return (Path(archive) / "ticks.jsonl").exists() and "operation" in frames[index]


def restore_prefix(env, archive: Path, decision: int, frames, meta: dict, *, fog_observation=None,
                   cache=None):
    """Replay once per exam item, optionally restoring the shared engine snapshot afterwards."""
    key = (str(archive), int(decision), fog_observation)
    if cache is not None:
        if cache.get('env') is env:
            if cache['key'] == key:
                return env.restore(cache['snapshot']), cache['prefix']
            env.release_snapshot(cache['snapshot'])
        cache.clear()
    obs, prefix = _restore_prefix(env, archive, decision, frames, meta,
                                  fog_observation=fog_observation)
    if cache is not None:
        cache.update(env=env, key=key, snapshot=env.snapshot(), prefix=prefix)
    return obs, prefix


def _restore_prefix(env, archive: Path, decision: int, frames, meta: dict, *, fog_observation=None):
    """Restore ``archive`` at ``decision`` and replay the action prefix.

    Human/LLM recordings replay through ``human_demo.replay`` (exact operation timeline,
    rejections and RNG included, and it owns ``DEMO_BEGIN``).  Collection archives have
    no operation column, so the recorded actions are the prefix and the reset context
    comes from the archive's own ``environment_task`` (older archives fall back to frame
    0's seed bank plus meta level/seed/multiplier).
    """
    archive = Path(archive)
    if replayable_by_operations(archive, frames, decision):
        from human_demo import replay
        obs, _meta, prefix = replay(env, archive, decision=decision,
                                    fog_observation=fog_observation)
        return obs, prefix
    from pvz_env import PlayerProfileContext, TaskSpec
    from pvz_tasks import profile_for_deck
    if meta.get("environment_task"):
        config = dict(meta["environment_task"])
        if fog_observation is not None:
            config["fog_observation"] = fog_observation
        config["profile"] = PlayerProfileContext(**config["profile"])
        config["preplanted"] = tuple(tuple(p) for p in config.get("preplanted") or ())
        obs, _ = env.reset(deck=meta["task"]["deck"], task=TaskSpec(**config))
    else:
        # Collection frames predate the full environment_task block.  The deck is in
        # frame 0's seed bank; level/seed/multiplier are in the archive meta.
        deck = [int(p[1]) for p in frames[0].get("packets") or []]
        if not deck or any(not 0 <= seed_type < 49 for seed_type in deck):
            raise ValueError("archive has neither environment_task nor a valid seed bank")
        level = int((meta.get("context") or {}).get("level") or frames[0]["scene"]["level"])
        seed = int((meta.get("task") or {}).get("seed"))
        multiplier = float((meta.get("context") or {}).get("multiplier") or 1.0)
        obs, _ = env.reset(deck=deck, task=TaskSpec(
            level=level, seed=seed, playthrough=2,
            profile=profile_for_deck(deck), zombie_count_multiplier=multiplier))
    for index in range(1, int(decision)):
        action = frames[index].get("action") or frames[index].get("input_action")
        if action is None:
            delta = int(frames[index + 1]["tick"]) - int(frames[index]["tick"])
            action = dict(type="wait", ticks=max(1, delta))
        obs, _r, _d, _t, _info = env.step(action)
        if int(obs["tick"]) != int(frames[index]["tick"]):
            raise ValueError(f"prefix differs at frame {index}: {obs['tick']} != {frames[index]['tick']}")
    return obs, frames[:int(decision)]


def horizon_reached(obs, root_tick: int, horizon: int | None) -> bool:
    """The short horizon is measured from the restored root, not from tick zero."""
    return horizon is not None and int(obs.get("tick") or 0) - int(root_tick) >= int(horizon)


def on_board(env, obs) -> dict[int, dict]:
    """On-board zombies by id, from a privileged frame when the engine exposes one."""
    frame = obs
    try:
        frame = env.privileged_state()
    except Exception:
        pass
    out = {}
    for z in frame.get("zombies") or []:
        if isinstance(z, dict):
            if not z.get("on_board"):
                continue
            out[int(z["id"])] = dict(row=int(z["row"]), type=int(z["type"]),
                                     x=float(z["x"]), hp=int(z.get("health") or 0))
        elif z[5]:
            out[int(z[9])] = dict(row=int(z[0]), type=int(z[1]), x=float(z[2]), hp=int(z[4]))
    return out


def judge(env, obs, targets, mowers_start: int, mowers_left: int, *, horizon: int,
          root_tick: int) -> dict:
    """The mined drill's mechanical local success, never a tactic or a heuristic."""
    alive = on_board(env, obs)
    target_ids = [int(target["id"]) for target in targets]
    alive_targets = [target_id for target_id in target_ids if target_id in alive]
    mower_used = max(0, int(mowers_start) - int(mowers_left))
    house_breach = obs.get("result") == 2
    success = (not alive_targets and mower_used == 0 and not house_breach)
    return dict(success=bool(success), target_count=len(target_ids),
                targets_alive=len(alive_targets), alive=alive_targets,
                mower_used=mower_used, house_breach=bool(house_breach),
                root_tick=int(root_tick), final_tick=int(obs.get("tick") or 0),
                tick_delta=int(obs.get("tick") or 0) - int(root_tick),
                horizon=int(horizon))


def read_drill_meta(archive: Path) -> dict:
    return json.loads((Path(archive) / "meta.json").read_text())
