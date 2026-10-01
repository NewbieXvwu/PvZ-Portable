"""Capture the first divergence of readonly repeated baseline evaluations.

Instrumentation delegates model inference and action selection to the original
evaluation code. It never updates weights or draws additional random numbers.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import pvz_agent_model as agent
from pvz_seed_jobs import atomic_json
import train_pvz_ppo_task_family as family
import t4_capability_profile as profile

TRACE: list[dict] = []
DESTINATION: Path | None = None


def tensor_sha(value: torch.Tensor | None) -> str | None:
    if value is None:
        return None
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(str((value.dtype, tuple(value.shape))).encode()
                          + value.numpy().tobytes()).hexdigest()


def json_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def initialize(resource_dir: str, weights: dict, jobs: dict, threads: int,
               config: dict, destination: str, fusion: bool) -> None:
    global DESTINATION
    family._init_evaluation_worker(resource_dir, weights, jobs, threads, "cpu", config)
    agent.set_relation_bias_fusion(fusion)
    DESTINATION = Path(destination)
    model = family.WORKER_MODEL
    original_step = model.step
    original_select = profile.select_action

    def step(observation, hidden=None, previous_action=None, delta_ticks=0, events=None):
        before = tensor_sha(torch.get_rng_state())
        hidden_before = tensor_sha(hidden)
        output = original_step(observation, hidden, previous_action, delta_ticks, events)
        after = tensor_sha(torch.get_rng_state())
        packed = agent.pack_tokens(observation)
        record = {"decision": len(TRACE), "observation": copy.deepcopy(observation),
                  "observation_sha256": json_sha(observation),
                  "packed_tokens_sha256": {k: hashlib.sha256(v.tobytes()).hexdigest()
                                           for k, v in packed.items()},
                  "previous_action": copy.deepcopy(previous_action), "delta_ticks": delta_ticks,
                  "events": copy.deepcopy(events), "hidden_in_sha256": hidden_before,
                  "rng_before_step": before, "rng_after_step": after,
                  "output_sha256": {key: tensor_sha(value) for key, value in output.items()
                                    if isinstance(value, torch.Tensor)},
                  "logits": {key: output[key].detach().cpu().tolist()
                             for key in ("type_logits", "wait_logits", "packet_logits")}}
        TRACE.append(record)
        return output

    def select(model_arg, output, observation, *args, **kwargs):
        before = tensor_sha(torch.get_rng_state())
        selected, lp, entropy = original_select(model_arg, output, observation, *args, **kwargs)
        after = tensor_sha(torch.get_rng_state())
        TRACE[-1].update({"action": dict(selected), "log_prob": float(lp), "entropy": float(entropy),
                          "rng_before_selection": before, "rng_after_selection": after})
        return selected, lp, entropy

    model.step = step
    profile.select_action = select


def run_job(job_id: int) -> tuple[int, dict]:
    TRACE.clear()
    initial_weights = profile._state_sha256(family.WORKER_MODEL.state_dict())
    returned_id, outcome = family._evaluation_worker(job_id)
    if initial_weights != profile._state_sha256(family.WORKER_MODEL.state_dict()):
        raise ValueError("readonly diagnostic changed weights")
    if len(TRACE) != outcome["actions"]:
        raise ValueError("incomplete action trace")
    path = DESTINATION / f"job_{job_id:03d}.json.gz"
    atomic_json(path, {"job": job_id, "outcome": outcome, "steps": TRACE}, compressed=True)
    return returned_id, {"outcome": outcome, "trace_path": str(path),
                         "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "model_forward_rng_changes": sum(x["rng_before_step"] != x["rng_after_step"] for x in TRACE),
                         "sampling_gap_rng_changes": sum(x["rng_after_step"] != x["rng_before_selection"] for x in TRACE)}


def first_difference(a: list[dict], b: list[dict]) -> dict | None:
    keys = ("observation_sha256", "packed_tokens_sha256", "previous_action", "delta_ticks", "events",
            "hidden_in_sha256", "rng_before_step", "rng_after_step", "output_sha256",
            "rng_before_selection", "action", "log_prob", "entropy", "rng_after_selection")
    for i, (left, right) in enumerate(zip(a, b)):
        changed = [key for key in keys if left[key] != right[key]]
        if changed:
            return {"decision": i, "fields": changed,
                    "tick": [left["observation"]["tick"], right["observation"]["tick"]],
                    "action": [left["action"], right["action"]],
                    "logits": [left["logits"], right["logits"]],
                    "output_tensor_differences": [k for k in left["output_sha256"]
                                                  if left["output_sha256"][k] != right["output_sha256"][k]],
                    "observation_field_differences": [k for k in left["observation"]
                                                      if left["observation"][k] != right["observation"][k]]}
    if len(a) != len(b):
        return {"decision": min(len(a), len(b)), "fields": ["length"], "length": [len(a), len(b)]}
    return None


def main() -> None:
    import gzip
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--resource-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or args.summary.exists():
        raise ValueError("fresh diagnostic output required")
    protocol = json.loads(args.protocol.read_text())
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != protocol["helper_sha256"]:
        raise ValueError("helper differs from preregistered source")
    for relative, expected in protocol["required_fingerprints"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"diagnostic source differs from preregistration: {relative}")
    checkpoint_path = ROOT / protocol["reference_checkpoint"]
    if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != protocol["reference_checkpoint_sha256"]:
        raise ValueError("reference checkpoint hash mismatch")
    agent.configure_torch_threads(protocol["worker_threads"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["experiment_config"]
    manifest_path = ROOT / config["evaluation"]["manifest"]
    manifest = json.loads(manifest_path.read_text())
    task = next(t for t in manifest["tasks"] if t["task_id"] == protocol["task_id"])
    if protocol["environment_seed"] not in task["seeds"] or protocol["modes"] != ["sampled"]:
        raise ValueError("diagnostic must retain the original evaluation job")
    args.output_dir.mkdir(parents=True)
    jobs = {i: {"task": task, "seed": protocol["environment_seed"],
                "action_seed": protocol["action_seed"], "deterministic": False,
                "max_actions": protocol["max_actions"], "allow_truncation": True}
            for i in range(protocol["repeats_per_configuration"])}
    started = time.monotonic()
    records, comparisons = {}, []
    context = multiprocessing.get_context("spawn")
    for fusion in protocol["relation_bias_fusion"]:
        for workers in protocol["workers"]:
            key = f"fusion{int(fusion)}_workers{workers}"
            destination = (args.output_dir / key).resolve()
            destination.mkdir()
            pool = context.Pool(min(workers, len(jobs)), initializer=initialize,
                                initargs=(str(args.resource_dir), checkpoint["state_dict"], jobs,
                                          protocol["worker_threads"], checkpoint["config"],
                                          str(destination), fusion))
            collected = {}
            try:
                iterator = pool.imap_unordered(run_job, sorted(jobs), chunksize=1)
                for _ in jobs:
                    job_id, result = iterator.next(timeout=900)
                    collected[job_id] = result
                pool.close()
            except BaseException:
                pool.terminate()
                raise
            finally:
                pool.join()
            records[key] = [collected[i] for i in sorted(jobs)]
            atomic_json(args.output_dir / "partial.json", {"records": records})
            print(key, [(r["outcome"]["actions"], r["outcome"]["terminal_tick"]) for r in records[key]], flush=True)
    reference = next(iter(records.values()))[0]
    reference_steps = json.loads(gzip.decompress(Path(reference["trace_path"]).read_bytes()))["steps"]
    for key, group in records.items():
        for job_id, record in enumerate(group):
            steps = json.loads(gzip.decompress(Path(record["trace_path"]).read_bytes()))["steps"]
            comparisons.append({"configuration": key, "job": job_id,
                                "first_difference_from_reference": first_difference(reference_steps, steps)})
    summary = {"schema_version": 1, "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
               "checkpoint_sha256": protocol["reference_checkpoint_sha256"],
               "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "model_state_sha256": profile._state_sha256(checkpoint["state_dict"]),
               "records": records, "comparisons": comparisons, "seconds": time.monotonic() - started,
               "interpretation": "Instrumentation and fusion counterfactual on one diagnosed job; no formal learning/gate pass or replacement of original rows.",
               "coexecution": protocol["coexecution"]}
    atomic_json(args.output_dir / "report.json", summary)
    atomic_json(args.summary, summary)


if __name__ == "__main__":
    main()
