"""Profile one real search worker from a saved search_play config; A/B code stays outside production."""
from __future__ import annotations
import argparse
import ast
import cProfile
import importlib.util
import hashlib
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
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--shadow-trace", type=Path)
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
    import inspect
    worker_options = dict(static_value_path=paths[1])
    if "cpp_rollouts" in inspect.signature(teacher._init_worker).parameters:
        worker_options["cpp_rollouts"] = True  # Frozen pre-retirement baseline only.
    teacher._init_worker(paths[0], config.rollout_policy, config.rollout_sim_skip, **worker_options)
    # Use task33's actual instrumentation for both implementations. It normally invokes
    # search_play.main at import; install its wrappers around this selected teacher instead.
    os.environ["HOTPATH_OUT"] = str(args.out / "hotpath.jsonl")
    os.environ["HOTPATH_BYTES"] = "0"  # Re-serialising responses would inflate measured worker CPU.
    previous_teacher = sys.modules.get("search_play")
    previous_main = teacher.main
    sys.modules["search_play"] = teacher
    teacher.main = lambda: None
    try:
        hotpath_spec = importlib.util.spec_from_file_location("task37_hotpath", ROOT / "scripts/hotpath_profile.py")
        hotpath = importlib.util.module_from_spec(hotpath_spec)
        hotpath_spec.loader.exec_module(hotpath)
    finally:
        teacher.main = previous_main
        if previous_teacher is None:
            sys.modules.pop("search_play", None)
        else:
            sys.modules["search_play"] = previous_teacher
    trace, commands = [], {}
    noise_cpu = dict(worker=0.0, simulator=0.0)
    clock_ticks = os.sysconf("SC_CLK_TCK")

    def engine_cpu(env):
        # /proc/<pid>/stat includes CPU used by all threads in this process.
        fields = Path(f"/proc/{env._process.pid}/stat").read_text().rsplit(")", 1)[1].split()
        return (int(fields[11]) + int(fields[12])) / clock_ticks
    shadow = json.loads(args.shadow_trace.read_text()) if args.shadow_trace else None
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
        row = dict(tick=obs["tick"], future_seed=future_seed, policy=policy, chosen=chosen,
                   state_sha256=hashlib.sha256(json.dumps(obs,sort_keys=True).encode()).hexdigest(),
                   searched=stats["searched"] != before, targets=stats.get("last_targets"))
        if row["searched"]:
            if args.noise_every and (stats["searched"] - 1) % args.noise_every == 0:
                main_acc, main_count = dict(hotpath.ACC), dict(hotpath.COUNT)
                worker_start, engine_start = time.process_time(), engine_cpu(env)
                row["noise_choice"] = original_search(env, obs, policy, settings, future_seed + 7777777, saved)
                noise_cpu["worker"] += time.process_time() - worker_start
                noise_cpu["simulator"] += engine_cpu(env) - engine_start
                hotpath.ACC.clear(); hotpath.ACC.update(main_acc)
                hotpath.COUNT.clear(); hotpath.COUNT.update(main_count)
        if shadow is not None:
            expected = shadow[len(trace)]
            if row["state_sha256"] != expected["state_sha256"] or future_seed != expected["future_seed"]:
                raise AssertionError(f"shadow root mismatch at decision {len(trace)} tick {obs['tick']}")
            row.update(expected=expected["chosen"], executed=expected["chosen"])
        trace.append(row)
        if shadow is not None:
            return row["executed"]
        return chosen

    teacher._search, PvZEnv._command = search, command
    profile = cProfile.Profile(timer=time.process_time)
    try:
        if not args.no_profile:
            profile.enable()
        result = teacher._play(dict(seed=seed, arm="fast_search", args=config))
        if not args.no_profile:
            profile.disable()
        env = teacher._W["env"]
        # Each invocation owns one fresh worker/simulator pair. Account both
        # implementations here: current collection no longer returns host_cpu.
        from pvz_cpu_usage import usage, process_usage, difference
        zero = {"user_s": 0.0, "system_s": 0.0}
        result["host_cpu"] = {
            "worker": difference(usage(), zero),
            "simulator": difference(process_usage(env._process.pid), zero)}
        result["host_cpu_method"] = "fresh worker getrusage and simulator /proc, same boundary for both arms"
        if shadow is not None and len(trace) != len(shadow):
            raise AssertionError(f"shadow length mismatch: {len(trace)} != {len(shadow)}")
        engine_profile = env._command("ENV_PROFILE")
        def memory(pid):
            path = Path(f"/proc/{pid}/status")
            return {key: value.strip() for key, value in (line.split(":", 1) for line in path.read_text().splitlines())
                    if key in ("VmRSS", "VmHWM", "VmSwap")}
        result.update(engine_profile=engine_profile, memory=dict(worker=memory(os.getpid()), engine=memory(env._process.pid)),
                      commands=commands, noise_points=sum("noise_choice" in row for row in trace),
                      noise_agreement=sum(row.get("noise_choice") == row["chosen"] for row in trace if "noise_choice" in row))
        result["noise_cpu_s"] = noise_cpu
        result["hotpath"] = json.loads((args.out / "hotpath.jsonl").read_text().splitlines()[-1])
        result["host_cpu_excluding_noise_s"] = {
            name: result["host_cpu"][name]["total_s"] - noise_cpu[name] for name in noise_cpu}
        if not args.no_profile:
            profile.dump_stats(str(args.out / "worker.prof"))
            with (args.out / "top15.txt").open("w") as stream:
                pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("tottime").print_stats(15)
        (args.out / "trace.json").write_text(json.dumps(trace) + "\n")
        pending = args.out / "result.json.tmp"
        pending.write_text(json.dumps(result, indent=2) + "\n")
        pending.replace(args.out / "result.json")
        print(json.dumps(result), flush=True)
    finally:
        PvZEnv._command, teacher._search = original_command, original_search
        teacher._close_worker_env()
        if value_keep is not None:
            value_keep.cleanup()


if __name__ == "__main__":
    main()
