"""Profile one real search worker from a saved search_play config; A/B code stays outside production."""
from __future__ import annotations
import argparse
import ast
import cProfile
import importlib.util
import json
import os
from pathlib import Path
import pstats
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--code", type=Path, default=ROOT / "scripts/search_play.py")
    parser.add_argument("--executable", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--noise-every", type=int, default=0)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--task-id")
    parser.add_argument("--seed-offset", type=int)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("profiled_teacher", args.code)
    teacher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(teacher)
    teacher.ROOT = ROOT
    config = json.loads(args.config.read_text())
    for key, value in config.items():
        if isinstance(value, str):
            try:
                config[key] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass
    config.update(executable=args.executable, out=args.out, frames=None, measure_cpu=True)
    for key in ("manifest", "task_id", "seed_offset"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    config = SimpleNamespace(**config)
    from render_episode import load_training_task
    from pvz_env import PvZEnv
    seed = load_training_task(Path(config.manifest), config.task_id)["seeds"][config.seed_offset]
    value_keep = None
    if hasattr(teacher, "_export_values"):
        value_keep, paths = teacher._export_values(config.value, config.static_value)
    else:
        paths = [config.value, config.static_value]
    teacher._init_worker(paths[0], config.rollout_policy, config.rollout_sim_skip, config.cpp_rollouts, paths[1])
    trace, commands = [], {}
    original_command, original_search = PvZEnv._command, teacher._search

    def command(env, text):
        kind = text.split(" ", 1)[0]
        started = time.process_time()
        response = original_command(env, text)
        row = commands.setdefault(kind, dict(count=0, request_bytes=0, python_cpu_s=0.0))
        row["count"] += 1; row["request_bytes"] += len(text)
        row["python_cpu_s"] += time.process_time() - started
        return response

    def search(env, obs, policy, settings, future_seed, stats):
        from copy import deepcopy
        before = stats["searched"]
        saved = deepcopy(stats) if args.noise_every else None
        chosen = original_search(env, obs, policy, settings, future_seed, stats)
        if stats["searched"] != before:
            row = dict(tick=obs["tick"], future_seed=future_seed, policy=policy, chosen=chosen,
                       targets=stats.get("last_targets"))
            if args.noise_every and len(trace) % args.noise_every == 0:
                row["noise_choice"] = original_search(env, obs, policy, settings, future_seed + 7777777, saved)
            trace.append(row)
        return chosen

    teacher._search, PvZEnv._command = search, command
    profile = cProfile.Profile(timer=time.process_time)
    try:
        profile.enable()
        result = teacher._play(dict(seed=seed, arm="fast_search", args=config))
        profile.disable()
        env = teacher._W["env"]
        engine_profile = env._command("ENV_PROFILE")
        def memory(pid):
            path = Path(f"/proc/{pid}/status")
            return {key: value.strip() for key, value in (line.split(":", 1) for line in path.read_text().splitlines())
                    if key in ("VmRSS", "VmHWM", "VmSwap")}
        result.update(engine_profile=engine_profile, memory=dict(worker=memory(os.getpid()), engine=memory(env._process.pid)),
                      commands=commands, noise_points=sum("noise_choice" in row for row in trace),
                      noise_agreement=sum(row.get("noise_choice") == row["chosen"] for row in trace if "noise_choice" in row))
        profile.dump_stats(str(args.out / "worker.prof"))
        with (args.out / "top15.txt").open("w") as stream:
            pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("tottime").print_stats(15)
        (args.out / "trace.json").write_text(json.dumps(trace) + "\n")
        (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        PvZEnv._command, teacher._search = original_command, original_search
        teacher._close_worker_env()
        if value_keep is not None:
            value_keep.cleanup()


if __name__ == "__main__":
    main()
