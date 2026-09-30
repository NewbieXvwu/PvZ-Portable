"""Full-pipeline necessity audit: measure every stage, then ask if it is needed.

The earlier hotspot work found ``relation_bias`` and the evaluation loop.  This probe
covers the stages that were still never measured, and reports the *memory* cost of each
stored field as well as the time, because the rollout payload is what sets the peak RSS
and the disk traffic.

    python scripts/pipeline_necessity_audit.py --episodes 4
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from pvz_agent_model import GameplayModelV1, configure_torch_threads  # noqa: E402
from pvz_env import PvZEnv  # noqa: E402
from pvz_seed_jobs import atomic_numpy, read_numpy  # noqa: E402
from train_pvz_ppo import _task_spec, add_advantages, collect_task_episode  # noqa: E402
import train_pvz_ppo_task_family as trainer  # noqa: E402

DEFAULT_RESOURCES = Path.home() / ".cache/pvz-research-resources"
LOCAL_RESOURCES = Path("/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN")
ROLLOUT_EPISODES = 2000


def deep_size(value: Any, seen: set[int] | None = None) -> int:
    """Bytes held by *value*, counting each object once."""
    seen = seen if seen is not None else set()
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    total = sys.getsizeof(value)
    if isinstance(value, dict):
        total += sum(deep_size(k, seen) + deep_size(v, seen) for k, v in value.items())
    elif isinstance(value, (list, tuple)):
        total += sum(deep_size(item, seen) for item in value)
    elif isinstance(value, np.ndarray):
        total = max(total, value.nbytes)
    return total


def row(name: str, seconds: float, total: float, note: str = "") -> None:
    share = f"{seconds / total * 100:6.1f}%" if total else "      "
    print(f"  {name:44s} {seconds * 1e3:10.3f} ms  {share}  {note}")


def _zip_manifest(arrays: dict[str, np.ndarray], manifest: bytes) -> bytes:
    """Exactly what ``atomic_numpy`` hands to the filesystem, kept in memory."""
    import io

    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays,
                        __manifest__=np.frombuffer(manifest, dtype=np.uint8))
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-dir", type=Path, default=LOCAL_RESOURCES)
    parser.add_argument("--episodes", type=int, default=4)
    args = parser.parse_args()

    resource_dir = args.resource_dir.expanduser()
    if not resource_dir.is_dir():
        raise SystemExit(f"resource dir not found: {resource_dir}")

    configure_torch_threads(1)
    torch.manual_seed(0)
    model = GameplayModelV1().eval()
    train, heldout = trainer._task_family()
    cap1 = trainer._curriculum_tasks(train["tasks"], "cap1")
    print(f"device=cpu  episodes={args.episodes}  rollout batch={ROLLOUT_EPISODES}\n")

    # ------------------------------------------------------------------ collect
    print("--- A. rollout collection (real env) ---")
    episodes: list[dict[str, Any]] = []
    with PvZEnv(resource_dir) as env:
        for index in range(args.episodes):
            task = cap1[index % len(cap1)]
            episode = collect_task_episode(model, env, task, task["seeds"][index % 64],
                                           index, 4000)
            episodes.append(episode)
    transitions = sum(len(e["transitions"]) for e in episodes)
    decisions = transitions
    print(f"  {args.episodes} episodes, {decisions} decisions, "
          f"{decisions / args.episodes:.1f} decisions/episode")

    # ------------------------------------------------- per-field object memory
    print("\n--- B. per-field memory of one episode (object bytes) ---")
    sample = episodes[0]
    sample_transitions = sample["transitions"]
    total = deep_size(sample)
    # One shared ``seen`` so an object reachable from two fields (``previous_action``
    # is literally the previous transition's ``action`` dict) is billed once, to the
    # first field that reaches it.  A fresh ``seen`` per field double-counts them.
    seen: set[int] = set()
    field_bytes: dict[str, int] = {}
    for field in sorted(sample_transitions[0]):
        field_bytes[field] = sum(deep_size(t[field], seen) for t in sample_transitions)
    envelope = total - sum(field_bytes.values())
    for field, size in sorted(field_bytes.items(), key=lambda kv: -kv[1]):
        print(f"  {field:42s} {size / 1024:9.1f} KiB  {size / total * 100:5.1f}%")
    print(f"  {'(transition dicts + episode envelope)':42s} {envelope / 1024:9.1f} KiB"
          f"  {envelope / total * 100:5.1f}%")
    print(f"  {'TOTAL per episode':42s} {total / 1024:9.1f} KiB")
    print(f"  {'projected for 2000 episodes':42s} {total * ROLLOUT_EPISODES / 1024**2:9.1f} MiB")
    print(f"  {'transition dict alone (x N)':42s}"
          f" {sys.getsizeof(sample_transitions[0]) * len(sample_transitions) / 1024:9.1f} KiB"
          f"  {sys.getsizeof(sample_transitions[0]) / total * len(sample_transitions) * 100:5.1f}%")

    print("\n--- B2. what the big fields actually contain ---")
    transition = sample_transitions[3]
    for field in ("events", "legal", "action", "critic_extra", "previous_action"):
        value = transition[field]
        rendered = repr(value)
        print(f"  {field:16s} {type(value).__name__:8s} "
              f"size={sys.getsizeof(value):5d}  {rendered[:110]}")
    model_event_keys = {"zombies_killed", "plants_eaten", "sun_produced", "sun_spent",
                        "mower_triggered", "waves_started", "level_won", "level_lost"}
    stored_event_keys = set().union(*(set(t["events"]) for t in sample_transitions))
    print(f"  events keys stored but never read by the model: "
          f"{sorted(stored_event_keys - model_event_keys) or 'none'}")

    print("\n--- B3. events keys are re-created by JSON decoding on every step ---")
    populated = [t["events"] for t in sample_transitions if t["events"]]
    first, second = populated[0], populated[1]
    shared = [left is right for left, right in zip(first, second)]
    print(f"  key objects shared between two steps:   {sum(shared)}/{len(shared)}")
    print(f"  keys present in sys.intern table:       "
          f"{sum(key is sys.intern(key) for key in first)}/{len(first)}")
    dict_bytes = sum(sys.getsizeof(events) for events in populated)
    key_bytes = sum(sys.getsizeof(key) for events in populated for key in events)
    print(f"  per episode: {dict_bytes / 1024:.1f} KiB of dicts + "
          f"{key_bytes / 1024:.1f} KiB of duplicated key strings")
    print(f"  interning the 8 keys would drop {key_bytes / 1024:.1f} KiB/episode"
          f" ({key_bytes / total * 100:.1f}% of the payload)")
    print(f"  projected saving over {ROLLOUT_EPISODES} episodes:"
          f" {key_bytes * ROLLOUT_EPISODES / 1024**2:.1f} MiB")

    # ----------------------------------------------------------- advantage pass
    print("\n--- C. add_advantages (pure-Python reverse loop) ---")
    for episode in episodes:
        for transition in episode["transitions"]:
            transition.setdefault("reward", 0.0)
    start = time.perf_counter()
    add_advantages(episodes, 0.95)
    advantage_seconds = time.perf_counter() - start
    per_transition = advantage_seconds / transitions
    row("add_advantages", advantage_seconds, advantage_seconds,
        f"{per_transition * 1e6:.2f} us/transition")
    print(f"  projected for a {ROLLOUT_EPISODES}-episode batch:"
          f" {per_transition * ROLLOUT_EPISODES * (transitions / len(episodes)):.2f} s")

    # vectorised equivalent, to price the cheap alternative
    from pvz_value import DISCOUNT_REFERENCE_TICKS, VALUE_GAMMA
    rewards = np.array([t["reward"] for e in episodes for t in e["transitions"]], dtype=np.float64)
    values = np.array([t["value"] for e in episodes for t in e["transitions"]], dtype=np.float64)
    durations = np.array([t["action_duration_ticks"] for e in episodes
                          for t in e["transitions"]], dtype=np.float64)
    start = time.perf_counter()
    ratio = durations / DISCOUNT_REFERENCE_TICKS
    discount = VALUE_GAMMA ** ratio
    trace = discount * (0.95 ** ratio)
    advantage = np.zeros_like(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        next_value = values[index + 1] if index + 1 < len(values) else 0.0
        running = rewards[index] + discount[index] * next_value - values[index] + trace[index] * running
        advantage[index] = running
    vector_seconds = time.perf_counter() - start
    row("add_advantages (numpy, precomputed pow)", vector_seconds, advantage_seconds,
        f"{advantage_seconds / vector_seconds:.1f}x faster")

    # ------------------------------------------------ per-transition tensor walk
    print("\n--- C2. train_update's per-transition Python loops ---")
    total_transitions = ROLLOUT_EPISODES * int(round(transitions / len(episodes)))
    values = torch.tensor([t["value"] for e in episodes for t in e["transitions"]],
                          dtype=torch.float32)
    values = (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)
    pool = [t for e in episodes for t in e["transitions"]]
    start = time.perf_counter()
    for index, transition in enumerate(pool):
        transition["normalized_advantage"] = values[index]
    tensor_seconds = time.perf_counter() - start
    python_values = values.tolist()
    start = time.perf_counter()
    for index, transition in enumerate(pool):
        transition["normalized_advantage_python"] = python_values[index]
    python_seconds = time.perf_counter() - start
    for transition in pool:
        transition.pop("normalized_advantage_python", None)
        # ``train_update`` leaves a tensor here; ``atomic_numpy`` cannot encode it,
        # so the shard format is only valid for transitions that have not been
        # through the update yet.  Undo the probe's edit before section D.
        transition.pop("normalized_advantage", None)
    scale = total_transitions / len(pool)
    print(f"  storing a 0-dim tensor per transition   {tensor_seconds / len(pool) * 1e6:8.3f} us"
          f"  -> x{total_transitions} {tensor_seconds * scale:6.2f} s")
    print(f"  storing a Python float instead          {python_seconds / len(pool) * 1e6:8.3f} us"
          f"  -> x{total_transitions} {python_seconds * scale:6.2f} s")
    print(f"  ratio                                   {tensor_seconds / python_seconds:8.2f}x")
    print("  (on CUDA each index is a kernel launch, so the gap is wider there)")

    # ---------------------------------------------------------- shard round-trip
    print("\n--- D. rollout shard persistence (disk is used as the pool IPC channel) ---")

    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        payload = {"metadata": {"probe": 1}, "result": sample}
        for compressed in (False, True):
            start = time.perf_counter()
            for index in range(args.episodes):
                atomic_numpy(directory / f"{compressed}_{index}.npz",
                             {"metadata": payload["metadata"],
                              "result": episodes[index]}, compressed=compressed)
            write_seconds = (time.perf_counter() - start) / args.episodes
            size = sum((directory / f"{compressed}_{index}.npz").stat().st_size
                       for index in range(args.episodes)) / args.episodes
            start = time.perf_counter()
            for index in range(args.episodes):
                read_numpy(directory / f"{compressed}_{index}.npz")
            read_seconds = (time.perf_counter() - start) / args.episodes
            label = "savez_compressed" if compressed else "savez (no compression)"
            print(f"  {label:30s} write {write_seconds * 1e3:8.2f} ms  "
                  f"read {read_seconds * 1e3:8.2f} ms  {size / 1024:8.1f} KiB/episode")
            print(f"  {'  projected x2000':30s} write {write_seconds * ROLLOUT_EPISODES:8.2f} s  "
                  f"read {read_seconds * ROLLOUT_EPISODES:8.2f} s  "
                  f"{size * ROLLOUT_EPISODES / 1024**2:8.1f} MiB")

    # ---------------------------------------------- where the round-trip time goes
    print("\n--- D2. decompose the round trip: encoding, not disk bandwidth ---")
    import pickle
    from pvz_seed_jobs import _archive_decode, _archive_encode

    start = time.perf_counter()
    arrays: dict[str, np.ndarray] = {}
    for _ in range(5):
        arrays = {}
        encoded = _archive_encode(payload, arrays)
    encode_seconds = (time.perf_counter() - start) / 5

    start = time.perf_counter()
    for _ in range(5):
        manifest = json.dumps({"schema_version": 1, "value": encoded},
                              separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    manifest_seconds = (time.perf_counter() - start) / 5

    start = time.perf_counter()
    for _ in range(5):
        blob = _zip_manifest(arrays, manifest)
    blob_seconds = (time.perf_counter() - start) / 5

    start = time.perf_counter()
    for _ in range(5):
        json.loads(manifest)
    json_read_seconds = (time.perf_counter() - start) / 5

    start = time.perf_counter()
    for _ in range(5):
        _archive_decode(encoded, arrays)
    decode_seconds = (time.perf_counter() - start) / 5

    start = time.perf_counter()
    for _ in range(5):
        pickled = pickle.dumps(sample, protocol=5)
    pickle_seconds = (time.perf_counter() - start) / 5

    start = time.perf_counter()
    for _ in range(5):
        pickle.loads(pickled)
    unpickle_seconds = (time.perf_counter() - start) / 5

    raw = len(manifest) + sum(a.nbytes for a in arrays.values())
    print(f"  uncompressed payload                    {raw / 1024:8.1f} KiB")
    npz_total = encode_seconds + manifest_seconds + blob_seconds + json_read_seconds + decode_seconds
    print(f"  encode: walk nested dict -> arrays      {encode_seconds * 1e3:8.3f} ms")
    print(f"  encode: json.dumps manifest             {manifest_seconds * 1e3:8.3f} ms"
          f"   ({len(manifest) / 1024:.1f} KiB)")
    print(f"  encode: zlib compress the arrays        {blob_seconds * 1e3:8.3f} ms"
          f"   ({len(blob) / 1024:.1f} KiB compressed)")
    print(f"  decode: json.loads manifest             {json_read_seconds * 1e3:8.3f} ms")
    print(f"  decode: rebuild nested dicts            {decode_seconds * 1e3:8.3f} ms")
    print(f"  npz codec total                         {npz_total * 1e3:8.3f} ms"
          f"   -> x2000 {npz_total * ROLLOUT_EPISODES:6.2f} s")
    print("  (disk write + read measured in D is on top of this)")
    print(f"  alternative: pickle the episode         {pickle_seconds * 1e3:8.3f} ms"
          f"   ({len(pickled) / 1024:8.1f} KiB)")
    print(f"  alternative: unpickle it                {unpickle_seconds * 1e3:8.3f} ms"
          f"   -> x2000 {(pickle_seconds + unpickle_seconds) * ROLLOUT_EPISODES:6.2f} s")

    # ------------------------------------------------------------ env spawn cost
    print("\n--- E. environment spawn cost (paid per worker, per update) ---")
    start = time.perf_counter()
    env = PvZEnv(resource_dir)
    construct_seconds = time.perf_counter() - start
    task = cap1[0]
    start = time.perf_counter()
    env.reset(deck=task["deck"], task=_task_spec(task, task["seeds"][0]))
    first_reset_seconds = time.perf_counter() - start
    start = time.perf_counter()
    for index in range(5):
        env.reset(deck=task["deck"], task=_task_spec(task, task["seeds"][index]))
    later_reset_seconds = (time.perf_counter() - start) / 5
    start = time.perf_counter()
    env.close()
    close_seconds = time.perf_counter() - start
    print(f"  PvZEnv() construct (lazy)         {construct_seconds * 1e3:8.1f} ms")
    print(f"  first reset (spawn + load assets) {first_reset_seconds * 1e3:8.1f} ms")
    print(f"  later resets (warm)               {later_reset_seconds * 1e3:8.1f} ms")
    print(f"  PvZEnv.close()                    {close_seconds * 1e3:8.1f} ms")
    print(f"  18 workers x 10 updates, pool rebuilt each update:"
          f" {(first_reset_seconds + close_seconds) * 10:.1f} s wall (18 in parallel)")
    print(f"  18 workers x 1 persistent pool:"
          f" {first_reset_seconds + close_seconds:.1f} s wall total")

    # ------------------------------------------------- checkpoint / state writes
    print("\n--- F. per-update persistence ---")
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    losses = {"rollout_episode_hashes": [f"{index:032x}" for index in range(ROLLOUT_EPISODES)]}
    with tempfile.TemporaryDirectory() as temporary:
        checkpoint = Path(temporary) / "model.pt"
        start = time.perf_counter()
        torch.save({"state_dict": state_dict, "losses": losses, "provenance": {}}, checkpoint)
        save_seconds = time.perf_counter() - start
        save_bytes = checkpoint.stat().st_size
        start = time.perf_counter()
        torch.load(checkpoint, map_location="cpu", weights_only=False)
        load_seconds = time.perf_counter() - start
        print(f"  torch.save checkpoint             {save_seconds * 1e3:8.1f} ms  "
              f"{save_bytes / 1024 / 1024:6.1f} MiB")
        print(f"  torch.load checkpoint             {load_seconds * 1e3:8.1f} ms")
        print(f"  of which the 2000 episode digests:"
              f" {len(json.dumps(losses)) / 1024:6.1f} KiB of JSON")
        state = {"evaluations": [{"raw_seed_results_path": "x"}] * 5,
                 "learning_curve": [{"row": index} for index in range(10)],
                 "runs": [{"run_number": 1}]}
        start = time.perf_counter()
        (Path(temporary) / "state.json").write_text(json.dumps(state))
        print(f"  training_state.json write         "
              f"{(time.perf_counter() - start) * 1e3:8.1f} ms  "
              f"{(Path(temporary) / 'state.json').stat().st_size / 1024:6.1f} KiB")

    # ---------------------------------------------------- evaluation raw dump
    print("\n--- G. evaluation raw dump size ---")
    sample_record = {
        "seed": 1, "won": False, "result": 0, "terminal_wave": 3, "wave_count": 3,
        "terminal_tick": 1800, "peak_offense": 5,
        "economy_curve": [{"tick": index, "sun": index, "sun_produced": index, "sun_spent": 0}
                          for index in range(0, 1800, 180)],
    }
    per_record = deep_size(sample_record)
    total_records = 1280
    print(f"  one seed record                   {per_record / 1024:8.1f} KiB")
    print(f"  {total_records} records (cap3 + stage0)  "
          f"{per_record * total_records / 1024**2:8.1f} MiB raw, gzipped to disk every eval")

    # -------------------------------------------------------------- memory floor
    print("\n--- H. peak resident memory of the current design ---")
    episodes_bytes = total * ROLLOUT_EPISODES
    model_bytes = sum(v.numel() * v.element_size() for v in state_dict.values())
    print(f"  rollout payload in the parent      {episodes_bytes / 1024**2:8.1f} MiB "
          f"({ROLLOUT_EPISODES} episodes)")
    print(f"  worker model state (18 workers)    {model_bytes * 18 / 1024**2:8.1f} MiB")
    print(f"  one model state pickle             {model_bytes / 1024**2:8.1f} MiB "
          f"(sent to every worker, every update)")

    del episodes
    gc.collect()
    print("\ndone")


if __name__ == "__main__":
    main()
