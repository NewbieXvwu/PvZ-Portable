"""Check the RESEED command on the real simulator (DESIGN.md §13.3, rule 1).

Plays level 7 with a fixed opening until wave 3, snapshots, then rolls the same idle future out
several ways and asserts:
  1. restore without reseed is deterministic (two rollouts identical) -- the clairvoyance problem;
  2. restore + reseed(s) is reproducible for the same s;
  3. different seeds give different futures (spawns and wave tables);
  4. waves already sent keep their composition, and the wave count is unchanged.

Usage: python scripts/verify_reseed.py [--resource-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from pvz_env import PvZEnv, TaskSpec  # noqa: E402
from scripted_baseline import profile_for_deck  # noqa: E402

DECK = (0, 1, 2, 3, 4, 5, 7)


def wave_table(env: PvZEnv, waves: int) -> list[list[int]]:
    return [env.critic_inputs(w)["wave_zombies"] for w in range(waves)]


def rollout(env: PvZEnv, snap: int, reseed: int | None, steps: int = 150) -> dict:
    obs = env.restore(snap)
    if reseed is not None:
        env.reseed(reseed)
    table = wave_table(env, obs["wave_count"])
    spawned = []
    seen = set()
    for _ in range(steps):
        obs, _r, terminated, truncated, _info = env.step({"type": "wait", "ticks": 60})
        for z in obs.get("zombies") or []:
            if z["id"] not in seen:
                seen.add(z["id"])
                spawned.append((obs["tick"], z["row"], z["type"]))
        if terminated or truncated:
            break
    return {"table": table, "spawned": spawned, "end_tick": obs["tick"], "wave_count": obs["wave_count"]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resource-dir", default="/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN")
    args = p.parse_args()
    env = PvZEnv(args.resource_dir, headless=True)
    try:
        obs, _ = env.reset(deck=DECK, task=TaskSpec(level=7, seed=1400000, playthrough=2,
                                                    profile=profile_for_deck(DECK)))
        # A plain opening so the board survives to wave 3: sunflowers, then peashooters.
        plan = [(1, 0, r) for r in range(5)] + [(0, 2, r) for r in range(5)]
        while obs["wave"] < 3:
            if plan and obs["sun"] >= (50 if plan[0][0] == 1 else 100):
                packet, col, row = plan[0]
                obs, *_ = env.step({"type": "plant", "packet": packet, "col": col, "row": row})
                plan.pop(0)
                continue
            obs, _r, terminated, _t, _i = env.step({"type": "wait", "ticks": 60})
            assert not terminated, "game ended before wave 3"
        sent = obs["wave"]
        snap = env.snapshot()

        a1, a2 = rollout(env, snap, None), rollout(env, snap, None)
        b1, b2 = rollout(env, snap, 7), rollout(env, snap, 7)
        c = rollout(env, snap, 8)
        checks = {
            "1_no_reseed_is_deterministic": a1 == a2,
            "2_same_seed_reproducible": b1 == b2,
            "3a_reseed_changes_spawns": b1["spawned"] != a1["spawned"] and c["spawned"] != b1["spawned"],
            "3b_reseed_changes_future_waves": b1["table"][sent:] != a1["table"][sent:]
                                              and c["table"][sent:] != b1["table"][sent:],
            "4a_sent_waves_kept": a1["table"][:sent] == b1["table"][:sent] == c["table"][:sent],
            "4b_wave_count_kept": a1["wave_count"] == b1["wave_count"] == c["wave_count"],
        }
        report = {"waves_sent_at_snapshot": sent, "wave_count": a1["wave_count"],
                  "spawned_in_rollout": {"no_reseed": len(a1["spawned"]), "seed7": len(b1["spawned"]),
                                         "seed8": len(c["spawned"])},
                  "first_future_wave": {"no_reseed": a1["table"][sent], "seed7": b1["table"][sent],
                                        "seed8": c["table"][sent]},
                  "checks": checks}
        print(json.dumps(report, indent=1))
        if not all(checks.values()):
            raise SystemExit("RESEED verification FAILED")
        print("RESEED verification passed")
    finally:
        env.close()


if __name__ == "__main__":
    main()
