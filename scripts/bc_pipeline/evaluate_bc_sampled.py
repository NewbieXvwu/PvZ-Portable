from __future__ import annotations

import concurrent.futures
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/Users/newbiexvwu/PvZAgent/python")
sys.path.insert(0, "/Users/newbiexvwu/PvZAgent/scripts")
OUT = Path("/tmp/pvz_bc")
RESOURCE = "/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN"
MODEL_PATH = OUT / "scripted_level7_bc_v1.pt"
POLICY_PATH = OUT / "scripted_baseline_cap20.py"
_MODEL = _ENV = _TEACHER = None


def evaluate(job):
    group, seed = job
    import torch
    from pvz_agent_model import GameplayModelV1, select_action
    from pvz_env import PvZEnv, TaskSpec
    global _MODEL, _ENV, _TEACHER
    if _MODEL is None:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        _MODEL = GameplayModelV1(checkpoint["config"])
        _MODEL.load_state_dict(checkpoint["state_dict"])
        _MODEL.eval()
        torch.set_num_threads(1)
    if _ENV is None:
        _ENV = PvZEnv(RESOURCE, headless=True)
    if _TEACHER is None:
        spec = importlib.util.spec_from_file_location("scripted_teacher_eval", POLICY_PATH)
        _TEACHER = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_TEACHER)
    deck = _TEACHER.deck_for_level(7)
    task = TaskSpec(level=7, seed=seed, playthrough=2,
                    profile=_TEACHER.profile_for_deck(deck))
    obs, _ = _ENV.reset(deck=deck, task=task)
    torch.manual_seed(seed)
    previous_action, elapsed, events = None, 0, {}
    action_counts = Counter()
    actions = 0
    with torch.inference_mode():
        while not obs["terminal"] and actions < 4000:
            output = _MODEL.step(obs, None, previous_action, elapsed, events)
            action, _, _ = select_action(_MODEL, output, obs, deterministic=False)
            action_counts[action["type"]] += 1
            obs, _, done, _, info = _ENV.step(action)
            if not info.get("ok"):
                obs, _, done, _, info = _ENV.step({"type": "wait", "ticks": 60})
            previous_action = action
            elapsed = int(info.get("ticks_advanced", 0))
            events = info.get("events") or {}
            actions += 1
            if done:
                break
    return {"group": group, "seed": seed, "won": int(obs["result"]) == 1,
            "terminal": bool(obs["terminal"]), "wave": obs["wave"],
            "actions": actions, "action_counts": dict(action_counts)}


def main():
    from pvz_bc_train import DEV_COUNT, DEV_FIRST, TRAIN_FIRST, WORKERS
    jobs = ([ ("train_seen", TRAIN_FIRST + i) for i in range(DEV_COUNT)]
            + [("development_unseen", DEV_FIRST + i) for i in range(DEV_COUNT)])
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for index, row in enumerate(pool.map(evaluate, jobs, chunksize=1), 1):
            rows.append(row)
            if index % 64 == 0:
                print(f"sampled eval {index}/{len(jobs)}", flush=True)
    groups = {}
    for group in ("train_seen", "development_unseen"):
        cases = [row for row in rows if row["group"] == group]
        action_counts = Counter()
        for row in cases:
            action_counts.update(row["action_counts"])
        groups[group] = {
            "wins": sum(row["won"] for row in cases), "n": len(cases),
            "win_rate": sum(row["won"] for row in cases) / len(cases),
            "truncated": sum(not row["terminal"] for row in cases),
            "mean_actions": round(sum(row["actions"] for row in cases) / len(cases), 1),
            "action_counts": dict(action_counts), "episodes": cases,
        }
    result = {"sampling": "categorical policy sampling; torch seed equals environment seed per episode",
              "seed_sets": {"train_seen": [1_400_000, 1_400_255],
                            "development_unseen": [30_000, 30_255]},
              "evaluation": groups}
    path = OUT / "sampled_evaluation.json"
    path.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({"report": str(path), "evaluation": {
        key: {k: v for k, v in value.items() if k != "episodes"} for key, value in groups.items()
    }}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
