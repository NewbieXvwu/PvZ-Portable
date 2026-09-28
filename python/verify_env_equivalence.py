"""Automated visible/headless and snapshot/restore equivalence regression.

Four paths have to agree tick for tick:

* ``-env`` (headless) and ``-env-visible`` (rendering) must share all game logic;
* ``SNAPSHOT``/``RESTORE`` must reproduce the saved state *exactly*, because every
  search branch is a rewind-and-replay of it;
* ``BRANCH_SNAPSHOT_FAST`` must agree with doing "restore, act, snapshot" by hand, since
  it exists only to save the search two command round trips;
* the privileged observation must be a faithful digest of that state, or none of the
  above can be checked.

Per level the harness asserts, in order:

1. **reset equivalence** -- observation and privileged-state digest.
2. **feature coverage** -- terrain, night, fog, gravestone grid items and the level's
   zombie roster are compared against a declared expectation.  A level that silently
   loses its fog fails here instead of passing as "two equally broken runs".
3. **step equivalence** -- every tick compared on observation, events and digest.
4. **card cooldown** -- a planted card enters cooldown, stays inactive, and its counter
   advances by exactly one per tick.

Then two whole-run phases follow:

5. **rewind stress** (``check_rewind_stress``) -- the same task is played three times in
   one process: twice plain and once with tick-neutral rewinds interleaved (snapshot,
   step, restore, replay the same step).  The two plain rollouts must agree (episode
   reproducibility) and the rewound one must agree with them (restore fidelity).
   Comparing headless against visible cannot catch a *lossy* restore, because both sides
   lose the same state and agree; comparing a rollout against itself can.
6. **branch equivalence** (``check_branch_equivalence``) -- for a batch of actions from
   one parent, ``BRANCH_SNAPSHOT_FAST`` must return exactly what restoring the parent and
   stepping the action returns, on both the observation and the child state digest.

Why the rewind stress is written this way
-----------------------------------------
A single restore reproduces the saved state trivially whenever nothing has happened
since, so the older "snapshot then restore in a loop" check passed on a build whose
restore was already broken.  What actually exposed the last restore bug was a *long*
rollout: the effect arrays have to churn for ~1700 ticks before the outgoing generation
aliases the incoming one and the teardown cascade kills a freshly restored reanimation.
The corruption is real but narrow -- at the project's dev seed (30000) it fires on levels
8 and 50 and not on 12/18/26/31/41, and at other seeds it may not fire at all within
budget.  So the stress runs over the whole level list rather than one hand-picked level,
and the seed is a parameter: this check is a genuine property, and the specific
(level, seed) pairs that provok it are just the cases that have been observed to.

The separate plain-vs-plain comparison exists because a second bug hid behind the first:
``DataArray`` object ids carry a generation key, the counter for it survived
``DataArrayFreeAll``, and so every episode in a process handed out different ids for the
same objects.  That left the *gameplay* identical -- plain observations matched -- but
made the privileged state, and therefore the save-game bytes and ``state_hash``, differ
between episodes.  Digests caught it; only the extra rollout says whether the rewind is
to blame.

Exit status is non-zero on the first divergence.  A simulator that dies mid-run counts as
one: the last regression in this area could surface either as a digest difference or as a
segfault, and both have to fail the run.

    python verify_env_equivalence.py --resource-dir <PvZ 1.2.0.1073 dir>
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pvz_common import canonical_digest
from pvz_env import PvZEnv, SimulatorExited, training_task

DECK = (0, 1, 2, 3, 4, 5)
SEED_PEASHOOTER, SEED_SNOW_PEA = 0, 5
SHOOTER_SEEDS = (SEED_PEASHOOTER, SEED_SNOW_PEA)
GRAVESTONE = 1
ZOMBIE_DANCER = 8

BACKGROUND_DAY, BACKGROUND_NIGHT, BACKGROUND_POOL = 0, 1, 2
BACKGROUND_FOG, BACKGROUND_ROOF, BACKGROUND_BOSS = 3, 4, 5

# How many cards the stress rollout plants before it switches to waiting.  A handful is
# enough to keep the effect system churning; more would just starve the level.
STRESS_PLANT_LIMIT = 12

EnvPair = tuple[PvZEnv, PvZEnv]


class Divergence(RuntimeError):
    """Raised on the first tick where two paths disagree."""


@dataclass(frozen=True)
class Coverage:
    """What a level must show on reset for it to count as covering a feature."""

    level: int
    label: str
    terrain: int
    night: bool
    fog: bool
    gravestones: bool
    roster_contains: tuple[int, ...] = ()


# Areas are ten levels each: 1-10 day, 11-20 night, 21-30 pool, 31-40 fog, 41-50 roof.
# Levels 5/10/20/25/30/40/45/50 are the conveyor-belt levels (Wall-nut Bowling, Little
# Trouble, Stormy Night, Bungee Blitz and the area bosses); level 50 is included for
# the boss terrain but its cards are consumed rather than cooled down, so it skips the
# cooldown contract.  Levels 9/19/29/39/49 are flag levels and 35 has no normal seed
# bank, so neither is used.
COVERAGE = (
    Coverage(8, "day", BACKGROUND_DAY, False, False, False, (2, 4)),
    Coverage(12, "night + gravestones", BACKGROUND_NIGHT, True, False, True),
    Coverage(18, "night + gravestones + dancing zombie", BACKGROUND_NIGHT, True, False, True, (ZOMBIE_DANCER,)),
    Coverage(26, "pool", BACKGROUND_POOL, False, False, False),
    Coverage(31, "fog", BACKGROUND_FOG, True, True, False),
    Coverage(41, "roof", BACKGROUND_ROOF, False, False, False),
    Coverage(50, "roof boss", BACKGROUND_BOSS, True, False, False),
)


def digest(env: PvZEnv) -> str:
    return canonical_digest(env.privileged_state())


def recorded_digest(env: PvZEnv) -> str:
    """The digest piggybacked on the last recorded operation (requires debug_replay)."""
    return env.episode["operations"][-1]["debug_state_sha256"]


def compare_pair(envs: EnvPair, action: dict[str, Any], where: str) -> dict[str, Any]:
    """Step both rendering paths once and require the results to be identical."""
    headless, visible = envs
    left = headless.step(action)
    right = visible.step(action)
    if not left[4]["ok"] or not right[4]["ok"]:
        raise Divergence(f"{where}: action rejected: {action} {left[4]} / {right[4]}")
    if left[0] != right[0]:
        raise Divergence(f"{where}: observation diverged after {action}")
    if left[4]["events"] != right[4]["events"]:
        raise Divergence(f"{where}: events diverged after {action}")
    left_debug, right_debug = recorded_digest(headless), recorded_digest(visible)
    if left_debug != right_debug:
        raise Divergence(f"{where}: full game state diverged after {action}: {left_debug} != {right_debug}")
    return left[0]


def advance_until(envs: EnvPair, predicate: Callable[[dict[str, Any]], bool], budget: int,
                  where: str) -> tuple[dict[str, Any], bool]:
    """Step one tick at a time, comparing both paths, until ``predicate`` holds."""
    state = envs[0].observe()
    if predicate(state):
        return state, True
    for _ in range(budget):
        if state["terminal"]:
            break
        state = compare_pair(envs, {"type": "wait", "ticks": 1}, where)
        if predicate(state):
            return state, True
    return state, predicate(state)


def check_coverage(observation: dict[str, Any], expected: Coverage) -> None:
    actual = {
        "terrain": observation["terrain"],
        "night": observation["night"],
        "fog": observation["fog"],
        "gravestones": any(item["type"] == GRAVESTONE for item in observation["grid_items"]),
    }
    want = {
        "terrain": expected.terrain,
        "night": expected.night,
        "fog": expected.fog,
        "gravestones": expected.gravestones,
    }
    if actual != want:
        raise Divergence(f"level {expected.level} ({expected.label}) reset coverage {actual} != {want}")
    roster = set(observation["loadout_context"]["zombie_roster"])
    missing = set(expected.roster_contains) - roster
    if missing:
        raise Divergence(f"level {expected.level} ({expected.label}) roster {sorted(roster)} is missing {sorted(missing)}")


def pick_plant(state: dict[str, Any], seeds: tuple[int, ...] | None = None,
               row: int | None = None) -> dict[str, Any] | None:
    """The first legal placement for one of ``seeds``, optionally in a given row."""
    packets = state["packets"]
    for placement in state["legal_actions"]["plants"]:
        if seeds is not None and packets[placement["packet"]]["type"] not in seeds:
            continue
        if row is not None and placement["row"] != row:
            continue
        return placement
    return None


def shooter_placement(state: dict[str, Any]) -> dict[str, Any] | None:
    """A placeable shooter, preferring a lane that already holds a zombie."""
    for row in sorted({item["row"] for item in state["zombies"]}):
        placement = pick_plant(state, SHOOTER_SEEDS, row)
        if placement is not None:
            return placement
    return pick_plant(state, SHOOTER_SEEDS)


def plant_and_check_cooldown(envs: EnvPair, level: int, budget: int,
                             cooldown_probe: int) -> tuple[dict[str, Any], bool, dict[str, Any] | None]:
    """Plant the first affordable card and verify the cooldown contract.

    Returns the state, whether a cooldown was observed (conveyor levels consume the
    card instead) and the card that went on cooldown.
    """
    state, plantable = advance_until(envs, lambda item: bool(item["legal_actions"]["plants"]), budget,
                                     f"level {level} warmup")
    if not plantable:
        raise Divergence(f"level {level}: no card became plantable within {budget} ticks")

    placement = pick_plant(state)
    if placement is None:
        raise Divergence(f"level {level}: warmup reported a plantable card but none was found")
    index = placement["packet"]
    before_type = state["packets"][index]["type"]
    state = compare_pair(envs, {"type": "plant", **placement}, f"level {level} plant")

    packet = state["packets"][index]
    if packet["type"] != before_type:
        # A conveyor-belt bank removes the card and shifts the rest, so this index is
        # no longer the card that was planted.
        return state, False, None
    if packet["active"] or packet["refresh_time"] <= 0:
        raise Divergence(f"level {level}: planted card did not enter cooldown: {packet}")

    refresh_time = packet["refresh_time"]
    counter = packet["cooldown"]
    probed = 0
    for step in range(1, cooldown_probe + 1):
        state = compare_pair(envs, {"type": "wait", "ticks": 1}, f"level {level} cooldown")
        probed = step
        packet = state["packets"][index]
        if packet["refresh_time"] != refresh_time:
            raise Divergence(f"level {level}: refresh_time changed from {refresh_time} to {packet['refresh_time']}")
        if packet["active"]:
            raise Divergence(f"level {level}: card became active after {step} of {refresh_time} cooldown ticks")
        if packet["cooldown"] != counter + step:
            raise Divergence(
                f"level {level}: cooldown advanced by {packet['cooldown'] - counter} over {step} ticks "
                f"(expected {step}), packet={packet}")
        if state["terminal"]:
            break
    return state, True, {"packet": index, "type": before_type, "refresh_time": refresh_time,
                         "ticks_probed": probed}


def _play(env: PvZEnv, task: Any, ticks: int, snapshot_every: int) -> dict[str, Any]:
    """Play a deterministic rollout, optionally rewinding at every ``snapshot_every``.

    With ``snapshot_every`` set, each snapshot point is a *tick-neutral* rewind:

        snapshot()      # tick T
        step(wait 1)    # tick T + 1
        restore()       # tick T     -- a real rewind
        step(wait 1)    # tick T + 1 -- replay the same tick

    so the rewound rollout stays on the same tick schedule as the plain one and any
    difference in the digest sequence is caused by the rewind, not by a shifted clock.
    """
    digests: list[str] = []
    rewinds = 0
    unfaithful: dict[str, Any] | None = None
    planted = 0
    observation = env.reset(deck=DECK, task=task)[0]
    for step in range(ticks):
        if observation["terminal"]:
            break
        placements = observation["legal_actions"]["plants"]
        if placements and planted < STRESS_PLANT_LIMIT:
            observation = env.step({"type": "plant", **placements[0]})[0]
            planted += 1
        elif snapshot_every and step % snapshot_every == 0:
            before = digest(env)
            snapshot_id = env.snapshot()
            observation = env.step({"type": "wait", "ticks": 1})[0]
            observation = env.restore(snapshot_id)
            if digest(env) != before and unfaithful is None:
                unfaithful = {"step": step, "tick": observation["tick"]}
            rewinds += 1
            observation = env.step({"type": "wait", "ticks": 1})[0]
            env.release_snapshot(snapshot_id)
        else:
            observation = env.step({"type": "wait", "ticks": 1})[0]
        digests.append(digest(env))
    return {"digests": digests, "rewinds": rewinds, "unfaithful": unfaithful,
            "planted": planted, "terminal": bool(observation["terminal"])}


def check_rewind_stress(env: PvZEnv, level: int, seed: int, ticks: int,
                        snapshot_every: int) -> dict[str, Any]:
    """Snapshotting must be invisible, and the environment must be reproducible.

    Three rollouts of the same task run in one process:

    * ``plain`` and ``repeat`` are both un-rewound, so they must agree exactly -- that is
      episode-to-episode reproducibility, which object ids break if a generation counter
      survives a reset;
    * ``rewound`` interleaves tick-neutral rewinds, so it must agree with ``plain`` --
      that is restore fidelity.

    Both are checked separately rather than inferred from one comparison, because they
    are different bugs with different fixes: a non-reproducible environment makes the
    rewind look guilty, and blaming the rewind for it wastes a debugging session.  The
    rollouts are also required to be non-empty and to have actually rewound, so a level
    that ends immediately cannot pass by comparing nothing.
    """
    where = f"level {level} rewind stress"
    task = training_task(seed, level)
    plain = _play(env, task, ticks, 0)
    repeat = _play(env, task, ticks, 0)
    rewound = _play(env, task, ticks, snapshot_every)

    for name, run in (("plain", plain), ("repeat", repeat), ("rewound", rewound)):
        if not run["digests"]:
            raise Divergence(f"{where}: the {name} rollout recorded no steps")
    if rewound["rewinds"] == 0:
        raise Divergence(f"{where}: the rewound rollout performed no rewinds, so it proves nothing")
    if rewound["unfaithful"] is not None:
        raise Divergence(
            f"{where}: restore did not reproduce the snapshot state at step {rewound['unfaithful']['step']} "
            f"(tick {rewound['unfaithful']['tick']})")

    def first_difference(one: dict[str, Any], other: dict[str, Any]) -> int | None:
        for index, (left, right) in enumerate(zip(one["digests"], other["digests"])):
            if left != right:
                return index
        return None if len(one["digests"]) == len(other["digests"]) else min(
            len(one["digests"]), len(other["digests"]))

    repeat_difference = first_difference(plain, repeat)
    if repeat_difference is not None:
        raise Divergence(
            f"{where}: two un-rewound rollouts in one process diverge at step {repeat_difference}; "
            f"the environment is not reproducible across episodes, so the rewind cannot be judged")

    rewind_difference = first_difference(plain, rewound)
    if rewind_difference is not None:
        raise Divergence(f"{where}: rewinding changed the trajectory at step {rewind_difference}")

    return {"level": level, "seed": seed, "steps": len(plain["digests"]),
            "rewinds": rewound["rewinds"], "planted": plain["planted"],
            "episode_reproducible": True, "rewind_faithful": True, "snapshot_invisible": True}


def branch_actions(state: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    """A mixed action batch for one branch command, or ``[]`` when the board is idle."""
    actions: list[dict[str, Any]] = [
        {"type": "plant", **placement} for placement in state["legal_actions"]["plants"][:limit - 1]
    ]
    actions.append({"type": "wait", "ticks": 1})
    return actions if len(actions) >= 2 else []


def check_branch_equivalence(env: PvZEnv, level: int, seed: int, rounds: int,
                             step_ticks: int, batch_limit: int) -> dict[str, Any]:
    """``BRANCH_SNAPSHOT_FAST`` must equal "restore, act, snapshot" done by hand.

    The command exists only to collapse three round trips into one, so its results have
    to be bit-for-bit what the explicit sequence produces: the same observation, and a
    child snapshot whose state digest matches.  Each round restores the parent once per
    action for the reference, then branches the whole batch, then advances the board so
    the next round branches from a different state.
    """
    where = f"level {level} branch equivalence"
    task = training_task(seed, level)
    observation = env.reset(deck=DECK, task=task)[0]
    checked = 0
    rounds_run = 0
    for round_index in range(rounds):
        if observation["terminal"]:
            break
        actions = branch_actions(observation, batch_limit)
        if not actions:
            observation = env.step({"type": "wait", "ticks": 1})[0]
            continue

        parent = env.snapshot()
        references: list[tuple[dict[str, Any], str]] = []
        for action in actions:
            env.restore(parent)
            reference, _, _, _, info = env.step(action)
            if not info["ok"]:
                raise Divergence(f"{where}: reference action rejected: {action} {info}")
            references.append((reference, digest(env)))

        env.restore(parent)
        branches = env.branch_snapshot(parent, actions)
        for index, (branch, (reference, reference_digest)) in enumerate(zip(branches, references)):
            if not branch.get("ok"):
                raise Divergence(f"{where}: round {round_index} branch {index} failed: {branch}")
            if branch["observation"] != reference:
                raise Divergence(
                    f"{where}: round {round_index} branch {index} ({actions[index]}) "
                    f"observation differs from restore-then-step")
            child = branch.get("snapshot_id")
            if child is not None:
                env.restore(child)
                if digest(env) != reference_digest:
                    raise Divergence(
                        f"{where}: round {round_index} branch {index} ({actions[index]}) "
                        f"child state differs from restore-then-step")
                env.release_snapshot(child)
            checked += 1
            rounds_run = round_index + 1

        env.restore(parent)
        for _ in range(step_ticks):
            observation = env.step({"type": "wait", "ticks": 1})[0]
            if observation["terminal"]:
                break
        env.release_snapshot(parent)

    if checked == 0:
        raise Divergence(f"{where}: no branch was evaluated, so it proves nothing")
    return {"level": level, "seed": seed, "branches_checked": checked, "rounds": rounds_run,
            "branch_matches_sequential": True}


def run_level(envs: EnvPair, level: int, expected: Coverage | None, args: argparse.Namespace) -> dict[str, Any]:
    task = training_task(args.seed, level)
    observations = [env.reset(deck=DECK, task=task)[0] for env in envs]
    if observations[0] != observations[1]:
        raise Divergence(f"level {level}: reset observation diverged")
    if expected is not None:
        check_coverage(observations[0], expected)
    if digest(envs[0]) != digest(envs[1]):
        raise Divergence(f"level {level}: reset privileged state diverged")

    # Replay-semantics starts already have zombies on the board; assert that so the
    # checks below cannot silently degrade into testing an empty board.
    if not observations[0]["zombies"]:
        raise Divergence(f"level {level}: expected the replay start to have zombies on the board")

    state, cooled, cooldown = plant_and_check_cooldown(envs, level, args.ticks, args.cooldown_probe)

    # Push towards the state the search actually branches on: a shooter in a lane that
    # already holds a zombie, firing.  Peashooters and snow peas are the plants that
    # call Plant::GetPeaHeadOffset -- the function that dereferenced a null reanimation
    # when a restore left a plant holding a dead body reanimation.  Saving up for one
    # costs ~1500 ticks of natural sun on day terrain and never happens at night, so
    # this phase gets the wider budget and the outcome is reported, not asserted.
    state, armed = advance_until(envs, lambda item: shooter_placement(item) is not None,
                                 2 * args.ticks, f"level {level} save-sun")
    projectiles_seen = False
    if armed:
        placement = shooter_placement(state)
        if placement is not None:
            state = compare_pair(envs, {"type": "plant", **placement}, f"level {level} shooter")
            state, projectiles_seen = advance_until(envs, lambda item: bool(item["projectiles"]),
                                                    args.ticks, f"level {level} engage")

    for _ in range(args.ticks):
        state = compare_pair(envs, {"type": "wait", "ticks": 1}, f"level {level} tail")
        if state["terminal"]:
            break

    return {
        "level": level,
        "label": expected.label if expected is not None else "custom",
        "tick": state["tick"],
        "terminal": state["terminal"],
        "gravestones": sum(1 for item in observations[0]["grid_items"] if item["type"] == GRAVESTONE),
        "card_cooldown_checked": cooled,
        "cooldown": cooldown,
        "projectiles_seen": bool(projectiles_seen),
        "busy_plants": len(state["plants"]),
        "busy_zombies": len(state["zombies"]),
        "visible_headless_equal": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--resource-dir", type=Path, required=True)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--levels", type=int, nargs="+",
                        help="override the level list; coverage assertions only apply to levels in the table")
    parser.add_argument("--seed", type=int, default=1701, help="seed for the per-level checks")
    parser.add_argument("--ticks", type=int, default=2000, help="tick budget per waiting phase")
    parser.add_argument("--cooldown-probe", type=int, default=60, help="ticks of cooldown counting to verify")
    parser.add_argument("--stress-ticks", type=int, default=2500,
                        help="rollout length for the rewind stress; 0 skips the stress phases")
    parser.add_argument("--stress-seed", type=int, default=30000,
                        help="seed for the stress phases; the project's dev seed by default")
    parser.add_argument("--stress-snapshot-every", type=int, default=10,
                        help="ticks between rewinds in the stressed rollout")
    parser.add_argument("--branch-rounds", type=int, default=6, help="branch-equivalence rounds per level")
    parser.add_argument("--branch-step-ticks", type=int, default=40,
                        help="ticks the board advances between branch rounds")
    parser.add_argument("--branch-batch", type=int, default=4, help="actions per branch command")
    args = parser.parse_args()
    if args.ticks < 1 or args.cooldown_probe < 1:
        parser.error("--ticks and --cooldown-probe must be positive")
    if args.stress_ticks < 0 or args.stress_snapshot_every < 1:
        parser.error("--stress-ticks must be non-negative and --stress-snapshot-every positive")
    if args.branch_rounds < 1 or args.branch_step_ticks < 1 or args.branch_batch < 2:
        parser.error("--branch-rounds and --branch-step-ticks must be positive and --branch-batch at least 2")

    covered = {item.level: item for item in COVERAGE}
    targets: tuple[tuple[int, Coverage | None], ...] = (
        tuple((level, covered.get(level)) for level in args.levels) if args.levels
        else tuple((item.level, item) for item in COVERAGE))

    results = []
    for level, expected in targets:
        with PvZEnv(args.resource_dir, args.executable, headless=True, debug_replay=True) as headless, \
                PvZEnv(args.resource_dir, args.executable, headless=False, debug_replay=True) as visible:
            report = run_level((headless, visible), level, expected, args)
        results.append(report)
        print(json.dumps(report), flush=True)

    stress = []
    branches = []
    if args.stress_ticks > 0:
        for level, _ in targets:
            # The stress phases read the privileged state directly, so they do not need
            # the per-operation debug digest the visible/headless comparison relies on.
            with PvZEnv(args.resource_dir, args.executable, headless=True) as env:
                try:
                    report = check_rewind_stress(env, level, args.stress_seed, args.stress_ticks,
                                                 args.stress_snapshot_every)
                    branch = check_branch_equivalence(env, level, args.stress_seed, args.branch_rounds,
                                                      args.branch_step_ticks, args.branch_batch)
                except SimulatorExited as exc:
                    # A regression in the snapshot machinery does not have to show up as a
                    # digest difference; the last one could also take the simulator down
                    # with a null dereference.  Either way the run has to fail, and it has
                    # to say so instead of printing a traceback from deep inside the pipe.
                    raise Divergence(
                        f"level {level}: the simulator died during the stress phases ({exc})") from exc
            stress.append(report)
            branches.append(branch)
            print(json.dumps(report), flush=True)
            print(json.dumps(branch), flush=True)

    print(json.dumps({
        "levels": len(results),
        "all_equal": True,
        "cooldown_checked": sum(1 for item in results if item["card_cooldown_checked"]),
        "projectile_states": sum(1 for item in results if item["projectiles_seen"]),
        "stress_levels": len(stress),
        "rewinds": sum(item["rewinds"] for item in stress),
        "episode_reproducible": all(item["episode_reproducible"] for item in stress),
        "branches_checked": sum(item["branches_checked"] for item in branches),
    }, indent=2))


if __name__ == "__main__":
    main()
