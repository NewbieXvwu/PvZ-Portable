from __future__ import annotations

import concurrent.futures
import gzip
import hashlib
import importlib.util
import json
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path("/Users/newbiexvwu/PvZAgent")
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]
OUT = Path("/tmp/pvz_bc")
OUT.mkdir(exist_ok=True)
RESOURCE = "/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN"
TRAIN_FIRST, TRAIN_COUNT = 1_400_000, 5_000
DEV_FIRST, DEV_COUNT = 30_000, 256
SAMPLES_PER_SHARD = 2_500
WORKERS = 8
CONFIG = {"layers": 1, "width": 64, "heads": 4, "ff_width": 128,
          "gru_layers": 2, "gru_width": 64}


def clean_action(action: dict) -> dict:
    return {key: action[key] for key in ("type", "packet", "row", "col", "ticks", "until")
            if key in action}


def episode_transition(obs, action, previous_action, elapsed, events, model_api):
    tensors, metadata = model_api.observation_tokens(obs, CONFIG.get("input_flags", 0))
    return {
        "tokens": model_api.pack_tokens(tensors, metadata),
        "wave": obs["wave"],
        "legal": model_api.policy_legal_summary(obs, CONFIG),
        "previous_action": previous_action,
        "elapsed_since_previous_observation": elapsed,
        "events": events or {},
        "action": clean_action(action),
    }


def run_shard(args):
    policy_path, first_seed, count = args
    import pvz_agent_model as model_api
    from pvz_env import PvZEnv, TaskSpec

    spec = importlib.util.spec_from_file_location("scripted_teacher", policy_path)
    teacher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(teacher)
    env = PvZEnv(RESOURCE, headless=True)
    deck = teacher.deck_for_level(7)
    rng = random.Random(610_000 + first_seed)
    samples, seen = [], 0
    wins = natural = truncated = 0
    for offset in range(count):
        seed = first_seed + offset
        task = TaskSpec(level=7, seed=seed, playthrough=2,
                        profile=teacher.profile_for_deck(deck))
        obs, _ = env.reset(deck=deck, task=task)
        previous_action = None
        elapsed = 0
        events = {}
        actions = 0
        while not obs["terminal"] and actions < 4000:
            action = clean_action(teacher.choose(obs))
            seen += 1
            if len(samples) < SAMPLES_PER_SHARD:
                keep = True
                index = len(samples)
            else:
                index = rng.randrange(seen)
                keep = index < SAMPLES_PER_SHARD
            if keep:
                item = episode_transition(obs, action, previous_action, elapsed, events, model_api)
                if index == len(samples):
                    samples.append(item)
                else:
                    samples[index] = item
            obs, _, done, _, info = env.step(action)
            if not info.get("ok"):
                obs, _, done, _, info = env.step({"type": "wait", "ticks": 60})
            previous_action = action
            elapsed = int(info.get("ticks_advanced", 0))
            events = info.get("events") or {}
            actions += 1
            if done:
                break
        natural += int(bool(obs["terminal"]))
        truncated += int(not bool(obs["terminal"]))
        wins += int(int(obs["result"]) == 1)
        if (offset + 1) % 125 == 0:
            print(f"collect {first_seed}-{seed}: episodes={offset + 1}/{count} wins={wins}", flush=True)
    return {"samples": samples, "episodes": count, "wins": wins,
            "natural": natural, "truncated": truncated, "decisions": seen}


def expand_legal(summary):
    plants = []
    for packet, mask in zip(summary["packets"], summary["plant_mask"], strict=True):
        for cell in range(54):
            if (mask >> cell) & 1:
                plants.append({"packet": int(packet), "row": cell // 9, "col": cell % 9})
    shovels = [(cell % 9, cell // 9) for cell in range(54)
               if (summary["shovel_mask"] >> cell) & 1]
    return {"legal_actions": {"plants": plants, "shovels": shovels, "wait": summary["wait"]}}


def train(samples, device_name="mps"):
    import torch
    from pvz_agent_model import (GameplayModelV1, configure_torch_threads,
                                 hard_behavior_cloning_loss, resolve_device)

    configure_torch_threads(4)
    device = resolve_device(device_name if torch.backends.mps.is_available() else "cpu")
    model = GameplayModelV1(CONFIG).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    rng = random.Random(7819)
    batch_size, epochs = 64, 5
    history = []
    model.train()
    for epoch in range(epochs):
        order = list(range(len(samples)))
        rng.shuffle(order)
        losses = []
        for start in range(0, len(order), batch_size):
            batch = [samples[index] for index in order[start:start + batch_size]]
            outputs, _ = model.forward_sequences([[item] for item in batch], [None] * len(batch))
            loss = torch.stack([
                hard_behavior_cloning_loss(model, output, expand_legal(item["legal"]), item["action"])
                for output, item in zip(outputs, batch, strict=True)
            ]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        value = sum(losses) / len(losses)
        history.append(value)
        print(f"bc epoch {epoch + 1}/{epochs} loss={value:.5f}", flush=True)
    model.eval()
    model.to("cpu")
    model_path = OUT / "scripted_level7_bc_v1.pt"
    torch.save({"config": CONFIG, "state_dict": model.state_dict()}, model_path)
    return model_path, history


_EVAL_MODEL = None
_EVAL_ENV = None


def eval_case(args):
    group, seed, model_path = args
    import torch
    import pvz_agent_model as model_api
    from pvz_agent_model import GameplayModelV1, select_action
    from pvz_env import PvZEnv, TaskSpec
    global _EVAL_MODEL, _EVAL_ENV
    if _EVAL_MODEL is None:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        _EVAL_MODEL = GameplayModelV1(checkpoint["config"])
        _EVAL_MODEL.load_state_dict(checkpoint["state_dict"])
        _EVAL_MODEL.eval()
    if _EVAL_ENV is None:
        _EVAL_ENV = PvZEnv(RESOURCE, headless=True)
    teacher_spec = importlib.util.spec_from_file_location(
        "scripted_teacher_eval", str(OUT / "scripted_baseline_cap20.py"))
    teacher = importlib.util.module_from_spec(teacher_spec)
    teacher_spec.loader.exec_module(teacher)
    deck = teacher.deck_for_level(7)
    task = TaskSpec(level=7, seed=seed, playthrough=2,
                    profile=teacher.profile_for_deck(deck))
    obs, _ = _EVAL_ENV.reset(deck=deck, task=task)
    previous_action, elapsed, events = None, 0, {}
    actions = 0
    with torch.inference_mode():
        while not obs["terminal"] and actions < 4000:
            output = _EVAL_MODEL.step(obs, None, previous_action, elapsed, events)
            action, _, _ = select_action(_EVAL_MODEL, output, obs, deterministic=True)
            obs, _, done, _, info = _EVAL_ENV.step(action)
            if not info.get("ok"):
                obs, _, done, _, info = _EVAL_ENV.step({"type": "wait", "ticks": 60})
            previous_action = action
            elapsed = int(info.get("ticks_advanced", 0))
            events = info.get("events") or {}
            actions += 1
            if done:
                break
    return {"group": group, "seed": seed, "won": int(obs["result"]) == 1,
            "terminal": bool(obs["terminal"]), "wave": obs["wave"], "actions": actions}


def main():
    import torch
    import pvz_agent_model as model_api

    torch.manual_seed(20261004)
    policy_path = OUT / "scripted_baseline_cap20.py"
    source = (ROOT / "scripts/scripted_baseline.py").read_text(encoding="utf-8")
    old = "if shooters < 14:"
    if source.count(old) != 1:
        raise RuntimeError("teacher cap literal changed unexpectedly")
    policy_path.write_text(source.replace(old, "if shooters < 20:"), encoding="utf-8")
    shards = [(str(policy_path), TRAIN_FIRST + i * (TRAIN_COUNT // WORKERS), TRAIN_COUNT // WORKERS)
              for i in range(WORKERS)]
    started = time.perf_counter()
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for item in pool.map(run_shard, shards, chunksize=1):
            results.append(item)
            print(f"collector shard finished episodes={item['episodes']} wins={item['wins']} "
                  f"samples={len(item['samples'])}", flush=True)
    collection_seconds = round(time.perf_counter() - started, 1)
    samples = [sample for item in results for sample in item["samples"]]
    random.Random(3107).shuffle(samples)
    data_path = OUT / "scripted_level7_bc_samples_v1.pkl.gz"
    with gzip.open(data_path, "wb", compresslevel=1) as stream:
        import pickle
        pickle.dump(samples, stream, protocol=pickle.HIGHEST_PROTOCOL)
    model_path, loss_history = train(samples)
    eval_jobs = ([ ("train_seen", TRAIN_FIRST + i, str(model_path)) for i in range(DEV_COUNT)]
                 + [("development_unseen", DEV_FIRST + i, str(model_path)) for i in range(DEV_COUNT)])
    evaluations = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for index, result in enumerate(pool.map(eval_case, eval_jobs, chunksize=1), 1):
            evaluations.append(result)
            if index % 64 == 0:
                print(f"eval {index}/{len(eval_jobs)}", flush=True)
    groups = {}
    for group in ("train_seen", "development_unseen"):
        rows = [row for row in evaluations if row["group"] == group]
        groups[group] = {"wins": sum(row["won"] for row in rows), "n": len(rows),
                         "win_rate": sum(row["won"] for row in rows) / len(rows),
                         "truncated": sum(not row["terminal"] for row in rows),
                         "episodes": rows}
    payload = {
        "schema_version": 1,
        "purpose": "Plain behavior cloning from 5000 full scripted level-7 demonstrations.",
        "simulator_sha256": hashlib.sha256((ROOT / "build/pvz-portable").read_bytes()).hexdigest(),
        "demonstrations": {"seeds": [TRAIN_FIRST, TRAIN_FIRST + TRAIN_COUNT - 1],
                           "episodes": sum(item["episodes"] for item in results),
                           "wins": sum(item["wins"] for item in results),
                           "decisions": sum(item["decisions"] for item in results),
                           "sampled_transitions": len(samples), "collection_seconds": collection_seconds},
        "model": {"config": CONFIG, "epochs": len(loss_history), "mean_bc_loss": loss_history,
                  "checkpoint": str(model_path), "dataset": str(data_path)},
        "evaluation": groups,
        "comparison": {"scripted_teacher_archived": "206/256 (80.5%)",
                       "original_model_observed": "0/8 (archived small sample)"},
    }
    report = OUT / "result.json"
    report.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({"report": str(report), "demonstrations": payload["demonstrations"],
                      "evaluation": {key: {k: v for k, v in row.items() if k != "episodes"}
                                     for key, row in groups.items()},
                      "checkpoint": str(model_path), "dataset": str(data_path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
