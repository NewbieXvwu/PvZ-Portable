from __future__ import annotations

import json
import pickle
import random
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path.home() / "PvZAgent-gru-bc-level7-v1"
DATA = OUT / "scripted_level7_bc_samples_v2.sqlite"
CHECKPOINT = Path.home() / "PvZAgent-bc-handoff" / "best_model_pipeline.pt"
TEACHER = ROOT / "scripts/bc_pipeline/scripted_baseline_d1.py"
RESOURCE = str(Path.home() / ".cache/pvz-research-resources")
TRAIN_FIRST = 1_400_000
UNSEEN_FIRST = 30_000
UNSEEN_COUNT = 256
SEQUENCE_LENGTH = 32
EPISODES_PER_BATCH = 8
MAX_EPOCHS = 50
PATIENCE = 2
SEED = 281_006
ACTION_TYPES = ("plant", "shovel", "wait")

sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]
import torch
import pvz_agent_model as api
from pvz_agent_model import GameplayModelV1, replay_log_probs, select_action


def split_episode_ids(conn: sqlite3.Connection, split: int) -> list[int]:
    return [row[0] for row in conn.execute(
        "SELECT DISTINCT episode_id FROM samples WHERE split=? ORDER BY episode_id", (split,))]


def load_episodes(conn: sqlite3.Connection, split: int, episode_ids: list[int]) -> dict[int, list[dict]]:
    if not episode_ids:
        return {}
    marks = ",".join("?" for _ in episode_ids)
    result: dict[int, list[dict]] = {episode_id: [] for episode_id in episode_ids}
    query = ("SELECT episode_id,payload FROM samples WHERE split=? AND episode_id IN (" + marks
             + ") ORDER BY episode_id,step_id")
    for episode_id, payload in conn.execute(query, (split, *episode_ids)):
        result[episode_id].append(pickle.loads(payload))
    return result


def sequence_batches(rows: dict[int, list[dict]], offsets: dict[int, int],
                     hidden: dict[int, torch.Tensor | None]):
    active = [episode_id for episode_id, sequence in rows.items()
              if offsets[episode_id] < len(sequence)]
    sequences = [rows[episode_id][offsets[episode_id]:offsets[episode_id] + SEQUENCE_LENGTH]
                 for episode_id in active]
    return active, sequences, [hidden[episode_id] for episode_id in active]


def train_epoch(model, optimizer, conn, episode_ids, rng):
    model.train()
    order = list(episode_ids)
    rng.shuffle(order)
    total_loss = 0.0
    total_samples = 0
    batch_count = 0
    for first in range(0, len(order), EPISODES_PER_BATCH):
        ids = order[first:first + EPISODES_PER_BATCH]
        rows = load_episodes(conn, 0, ids)
        hidden = {episode_id: None for episode_id in ids}
        offsets = {episode_id: 0 for episode_id in ids}
        max_length = max(map(len, rows.values()), default=0)
        for offset in range(0, max_length, SEQUENCE_LENGTH):
            active, sequences, hiddens = sequence_batches(rows, offsets, hidden)
            if not active:
                continue
            outputs, hidden_out = model.forward_sequences(sequences, hiddens)
            flat_rows = [row for sequence in sequences for row in sequence]
            logp, _ = replay_log_probs(model, outputs, flat_rows)
            loss = -logp.float().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = len(flat_rows)
            total_loss += float(loss.detach().cpu()) * count
            total_samples += count
            batch_count += 1
            for index, episode_id in enumerate(active):
                hidden[episode_id] = hidden_out[:, index].detach()
                offsets[episode_id] += len(sequences[index])
        if (first // EPISODES_PER_BATCH + 1) % 100 == 0:
            print(f"train episodes={min(first + EPISODES_PER_BATCH, len(order))}/{len(order)} "
                  f"samples={total_samples}", flush=True)
    return {"loss": total_loss / max(1, total_samples), "samples": total_samples,
            "optimizer_steps": batch_count}


def evaluate_validation(model, conn, episode_ids):
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    type_counts = Counter()
    type_correct = Counter()
    packet_correct = packet_total = 0
    with torch.inference_mode():
        for first in range(0, len(episode_ids), EPISODES_PER_BATCH):
            ids = episode_ids[first:first + EPISODES_PER_BATCH]
            rows = load_episodes(conn, 1, ids)
            hidden = {episode_id: None for episode_id in ids}
            offsets = {episode_id: 0 for episode_id in ids}
            max_length = max(map(len, rows.values()), default=0)
            for offset in range(0, max_length, SEQUENCE_LENGTH):
                active, sequences, hiddens = sequence_batches(rows, offsets, hidden)
                if not active:
                    continue
                outputs, hidden_out = model.forward_sequences(sequences, hiddens)
                flat_rows = [row for sequence in sequences for row in sequence]
                logp, _ = replay_log_probs(model, outputs, flat_rows)
                loss_sum -= float(logp.float().sum().cpu())
                sample_count += len(flat_rows)
                for row, output in zip(flat_rows, outputs):
                    target = row["action"]
                    kind = target["type"]
                    pred, _, _ = select_action(model, output, row["legal"], deterministic=True)
                    type_counts[kind] += 1
                    type_correct[kind] += int(_same_action(pred, target))
                    if kind == "plant":
                        allowed = set(row["legal"]["packets"])
                        candidates = [(float(output["packet_logits"][i].cpu()), packet)
                                      for i, packet in enumerate(output["packet_ids"])
                                      if packet in allowed]
                        packet_total += 1
                        packet_correct += int(max(candidates)[1] == target["packet"])
                for index, episode_id in enumerate(active):
                    hidden[episode_id] = hidden_out[:, index]
                    offsets[episode_id] += len(sequences[index])
    return {
        "loss": loss_sum / max(1, sample_count), "samples": sample_count,
        "exact_action_by_teacher_type": {
            kind: {"correct": type_correct[kind], "total": type_counts[kind],
                   "accuracy": type_correct[kind] / type_counts[kind] if type_counts[kind] else None}
            for kind in ACTION_TYPES
        },
        "plant_packet_conditional_accuracy": {
            "correct": packet_correct, "total": packet_total,
            "accuracy": packet_correct / packet_total if packet_total else None,
        },
    }


def _same_action(pred, target):
    if pred.get("type") != target.get("type"):
        return False
    keys = {"plant": ("packet", "row", "col"), "shovel": ("row", "col"),
            "wait": ("ticks", "until")}[target["type"]]
    return all(pred.get(key) == target.get(key) for key in keys)


def import_teacher():
    import importlib.util
    spec = importlib.util.spec_from_file_location("gru_bc_teacher_d1", TEACHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_unseen_eval(model, teacher):
    from pvz_event_env import EventWaitEnv
    from pvz_env import TaskSpec
    deck = (0, 1, 2, 3, 4, 5, 7)
    env = EventWaitEnv(RESOURCE, headless=True)
    wins = truncated = decisions = 0
    action_counts = Counter()
    episodes = []
    model.eval()
    for index, seed in enumerate(range(UNSEEN_FIRST, UNSEEN_FIRST + UNSEEN_COUNT), 1):
        torch.manual_seed(seed)
        task = TaskSpec(level=7, seed=seed, playthrough=2,
                        profile=teacher.profile_for_deck(deck))
        obs, _ = env.reset(deck=deck, task=task)
        hidden = None
        previous_action = None
        previous_wait_result = None
        elapsed = 0
        events = {}
        count = 0
        with torch.inference_mode():
            while not obs["terminal"] and count < 4000:
                output = model.step(obs, hidden, previous_action, elapsed, events,
                                    previous_wait_result)
                action, _, _ = select_action(model, output, obs, deterministic=False)
                obs, _, done, _, info = env.step(action)
                if not info.get("ok"):
                    raise RuntimeError(f"GRU-BC action rejected seed={seed}: {action}")
                hidden = output["hidden"]
                previous_action = action
                previous_wait_result = info.get("wait_result")
                elapsed = int(info.get("ticks_advanced", 0))
                events = info.get("events") or {}
                action_counts[action["type"]] += 1
                decisions += 1
                count += 1
                if done:
                    break
        terminal = bool(obs["terminal"])
        won = int(obs.get("result", 0)) == 1
        wins += int(won)
        truncated += int(not terminal)
        episodes.append({"seed": seed, "won": won, "terminal": terminal,
                         "wave": obs["wave"], "decisions": count})
        if index % 32 == 0:
            print(f"GRU-BC unseen sampled episodes={index}/{UNSEEN_COUNT} wins={wins} "
                  f"truncated={truncated}", flush=True)
    env.close()
    return {"seeds": [UNSEEN_FIRST, UNSEEN_FIRST + UNSEEN_COUNT - 1],
            "episodes": UNSEEN_COUNT, "wins": wins, "win_rate": wins / UNSEEN_COUNT,
            "truncated": truncated, "decisions": decisions,
            "action_counts": dict(action_counts), "episodes_detail": episodes}


def main():
    if not DATA.exists():
        raise FileNotFoundError(f"demonstration dataset is missing: {DATA}")
    source = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    config = source["config"]
    expected = {"layers": 6, "width": 256, "heads": 8, "ff_width": 1024,
                "gru_layers": 2, "gru_width": 256, "critic_width": 256,
                "critic_layers": 1, "input_flags": 7,
                "wait_mode": "events", "wait_mask": "progress_v1"}
    if config != expected:
        raise RuntimeError(f"BC configuration mismatch: {config}")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GameplayModelV1(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    conn = sqlite3.connect(DATA)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(samples)")}
    if not {"episode_id", "step_id"} <= columns:
        raise RuntimeError("dataset lacks ordered episode_id/step_id fields; recollect with this version")
    conn.execute("CREATE INDEX IF NOT EXISTS sequence_order ON samples(split,episode_id,step_id)")
    conn.commit()
    counts = {int(split): int(count) for split, count in conn.execute(
        "SELECT split,COUNT(*) FROM samples GROUP BY split")}
    if counts.get(0, 0) + counts.get(1, 0) < 1_000_000 or counts.get(1, 0) == 0:
        raise RuntimeError(f"BC dataset too small: {counts}")
    train_ids = split_episode_ids(conn, 0)
    validation_ids = split_episode_ids(conn, 1)
    best_loss = float("inf")
    stale = 0
    curve = []
    best_path = OUT / "gru_sequence_best_model.pt"
    started = time.time()
    rng = random.Random(SEED)
    print(json.dumps({"phase": "gru_sequence_training", "device": str(device),
                      "dataset": str(DATA), "rows": counts,
                      "train_episodes": len(train_ids), "validation_episodes": len(validation_ids),
                      "sequence_length": SEQUENCE_LENGTH, "episodes_per_batch": EPISODES_PER_BATCH},
                     ensure_ascii=False), flush=True)
    for epoch in range(1, MAX_EPOCHS + 1):
        train_metrics = train_epoch(model, optimizer, conn, train_ids, rng)
        validation = evaluate_validation(model, conn, validation_ids)
        row = {"epoch": epoch, "train_loss": train_metrics["loss"],
               "validation_loss": validation["loss"], "train": train_metrics,
               "validation": validation}
        curve.append(row)
        if validation["loss"] < best_loss - 1e-7:
            best_loss = validation["loss"]
            stale = 0
            torch.save({"config": config, "model_state_dict": model.state_dict(),
                        "state_dict": model.state_dict(), "epoch": epoch,
                        "validation_loss": best_loss, "sequence_mode": "ordered episode subsequences",
                        "sequence_length": SEQUENCE_LENGTH}, best_path)
        else:
            stale += 1
        print(f"epoch={epoch} train_loss={train_metrics['loss']:.6f} "
              f"validation_loss={validation['loss']:.6f} best={best_loss:.6f} "
              f"stale={stale} samples={train_metrics['samples']}", flush=True)
        if stale >= PATIENCE:
            break
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    model.eval()
    teacher = import_teacher()
    unseen = run_unseen_eval(model, teacher)
    manifest_path = OUT / "collection_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    report = {
        "schema_version": 1,
        "configuration": config,
        "dataset": manifest["dataset"],
        "training": {"epochs_run": len(curve), "best_epoch": best["epoch"],
                     "best_validation_loss": best_loss, "loss_curve": curve,
                     "optimizer": "Adam", "learning_rate": 3e-4,
                     "early_stop": f"{PATIENCE} consecutive epochs without validation-loss decrease",
                     "sequence_mode": "rows ordered by episode_id/step_id; hidden carried across 32-step chunks",
                     "sequence_length": SEQUENCE_LENGTH, "episodes_per_batch": EPISODES_PER_BATCH,
                     "device": str(device), "seconds": round(time.time() - started, 1)},
        "validation_metrics": curve[-1]["validation"],
        "unseen_sampled_evaluation": unseen,
        "model_path": str(best_path),
    }
    report_path = OUT / "gru_sequence_results.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"phase": "gru_sequence_done", "report": str(report_path),
                      "best_epoch": best["epoch"], "best_validation_loss": best_loss,
                      "unseen_wins": unseen["wins"], "unseen_episodes": unseen["episodes"],
                      "unseen_win_rate": unseen["win_rate"], "truncated": unseen["truncated"]},
                     ensure_ascii=False), flush=True)
    conn.close()


if __name__ == "__main__":
    main()
