"""Check a real native packet's elapsed counter and one-tick readiness boundary."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from pvz_env import PvZEnv, TaskSpec
from pvz_observation_features import packet_refresh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh output required")
    task = TaskSpec(level=7, seed=95600, wave_cap=None, forced_seeds=(3,),
                    preplanted=tuple((kind, row, col) for row in range(5)
                                     for kind, cols in ((0, (4, 5, 6, 7)), (1, (0, 1))) for col in cols))
    report = {"scope": "engineering timer fixture; never a training or evaluation task",
              "sun_start": 50, "seed": task.seed, "level": task.level,
              "preplanted": task.preplanted, "rows": [], "status": "partial"}
    try:
        with PvZEnv(resource_dir=args.resource_dir) as env:
            obs, _ = env.reset(task=task, deck=(0, 1, 2, 3, 4, 5))
            index = next(p["index"] for p in obs["packets"] if p["type"] == 3)
            def record(label):
                packet = next(p for p in obs["packets"] if p["index"] == index)
                status = packet_refresh(packet)
                report["rows"].append({"label": label, "tick": obs["tick"],
                                       "packet": packet, "derived": status})
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                return packet, status
            packet, status = record("reset")
            for _ in range(100):
                if packet["active"]:
                    break
                obs, _, done, _, _ = env.step({"type": "wait", "ticks": min(300, status["remaining"])})
                if done:
                    raise RuntimeError("fixture ended before initial refresh")
                packet, status = record("initial_refresh_wait")
            if not packet["active"]:
                raise RuntimeError("initial refresh did not finish in 100 waits")
            legal = next(a for a in obs["legal_actions"]["plants"] if a["packet"] == index)
            obs, _, done, _, info = env.step({"type": "plant", **legal})
            if not info["ok"]:
                raise RuntimeError("fixture planting was rejected")
            packet, status = record("purchased")
            total = packet["refresh_time"]
            for _ in range(100):
                if status["remaining"] <= 1:
                    break
                obs, _, done, _, _ = env.step({"type": "wait", "ticks": min(300, status["remaining"] - 1)})
                if done:
                    raise RuntimeError("fixture ended before purchased cooldown boundary")
                packet, status = record("cooldown_wait")
            if packet["active"] or packet["cooldown"] != total or status["remaining"] != 1:
                raise RuntimeError("native inactive counter==refresh_time boundary differs")
            obs, _, _, _, _ = env.step({"type": "wait", "ticks": 1})
            packet, status = record("one_tick_to_ready")
            if not packet["active"] or packet["cooldown"] != 0 or status["remaining"] != 0:
                raise RuntimeError("native readiness did not switch exactly one tick later")
        report.update(gate_result="pass", status="complete",
                      simulator_sha256=hashlib.sha256((ROOT / "build/pvz-portable").read_bytes()).hexdigest(),
                      feature_source_sha256=hashlib.sha256((ROOT / "python/pvz_observation_features.py").read_bytes()).hexdigest())
        print("native timer boundary passed", total, "ticks", flush=True)
    except BaseException as error:
        report.update(gate_result="fail", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
