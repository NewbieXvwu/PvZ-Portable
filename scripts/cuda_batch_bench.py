"""Compare CUDA schedules on identical roots/futures, checking result and leaf bytes.

Each workload contains batches * lanes explicit pairs: --repeat is ignored by the CUDA pair-list
door. No other live CUDA process may overlap a timing. Large outputs stay in the supplied directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from pvz_cuda_root import read_cudaroot


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exclusive_gpu() -> None:
    result = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
    live = [int(pid) for pid in result.splitlines() if pid.strip().isdigit()
            and Path(f"/proc/{pid.strip()}").exists()]
    if live:
        raise RuntimeError(f"timing requires an idle GPU; other live CUDA PIDs: {live}")
    mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    if int(mem["MemAvailable"].split()[0]) < 6 * 1024 * 1024:
        raise RuntimeError("timing requires at least 6 GiB available host memory")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--roots", type=Path, required=True)
    parser.add_argument("--net", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--counts", type=int, nargs="+", default=[8, 30, 300])
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--status-file", type=Path, help="append each measured result here")
    parser.add_argument("--label", default="CUDA批次对照", help="experiment label for status entries")
    args = parser.parse_args()
    if args.batches < 1 or not args.counts or min(args.counts) < 1:
        parser.error("counts and batches must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    root_count = sum(bool(r["supported"]) for r in read_cudaroot(args.roots)["roots"])
    if root_count == 0:
        parser.error("the corpus contains no supported root")
    environment = {**os.environ, "PVZ_CUDA_PRIVATE_LIMIT": "1200", "PVZ_CUDA_COOPERATIVE_HEADS": "1"}
    environment.pop("PHASE", None)
    provenance = {key: {"path": str(path.resolve()), "sha256": sha(path)}
                  for key, path in (("baseline", args.baseline), ("candidate", args.candidate),
                                    ("roots", args.roots), ("net", args.net))}
    provenance["environment"] = {"PVZ_CUDA_PRIVATE_LIMIT": "1200", "PVZ_CUDA_COOPERATIVE_HEADS": "1"}
    provenance["label"] = args.label
    rows = []
    for count in args.counts:
        pairs = args.out / f"pairs-{count}.txt"
        pairs.write_text("".join(f"{(i * 17) % root_count} {1600000000 + i}\n"
                                 for i in range(count * args.batches)))
        reference = None
        for name, binary in (("baseline", args.baseline), ("candidate", args.candidate)):
            tag = args.out / f"n{count}-{name}"
            result_path, leaf_path = tag.with_suffix(".bin"), tag.with_suffix(".leaf")
            command = [str(binary.resolve()), "--roots", str(args.roots.resolve()), "--net", str(args.net.resolve()),
                       "--pairs", str(pairs.resolve()), "--lanes", str(count), "--horizon", "2500",
                       "--out", str(result_path.resolve()), "--leaf-out", str(leaf_path.resolve())]
            exclusive_gpu()
            system = subprocess.check_output(["nvidia-smi"], text=True)
            system += Path("/proc/loadavg").read_text() + Path("/proc/meminfo").read_text()
            tag.with_suffix(".system.txt").write_text(system)
            started = time.monotonic()
            run = subprocess.run(command, env=environment, capture_output=True, text=True, check=True)
            process_seconds = time.monotonic() - started
            tag.with_suffix(".txt").write_text(run.stdout + run.stderr)
            exclusive_gpu()
            match = re.search(r"wall (\d+) ms for (\d+) rollouts", run.stdout)
            if match is None or int(match[2]) != count * args.batches:
                raise RuntimeError("the binary did not run every requested batch")
            result, leaf = result_path.read_bytes(), leaf_path.read_bytes()
            records = list(struct.iter_unpack("<33i", result))
            if len(records) != count * args.batches or any(record[16] for record in records):
                raise RuntimeError("a rollout was missing or reported a device error")
            magic, leaves, width = struct.unpack_from("<8sii", leaf)
            if magic != b"PVZLEAF1" or leaves != len(records) or width != 5668 or len(leaf) != 16 + leaves * width * 4:
                raise RuntimeError("the leaf file has an invalid count, width or byte length")
            if reference is None:
                reference = result, leaf
            if (result, leaf) != reference:
                raise RuntimeError(f"{count} lanes: candidate changed results or leaf bits")
            row = {"lanes": count, "variant": name, "batches": args.batches,
                   "wave_ms": int(match[1]) / args.batches, "process_seconds": process_seconds,
                   "result_sha256": sha(result_path), "leaf_sha256": sha(leaf_path), "command": command}
            rows.append(row)
            complete = len(rows) == 2 * len(args.counts)
            (args.out / "summary.json").write_text(json.dumps({"provenance": provenance, "rows": rows,
                "complete": complete, "all_bits_equal": True if complete else None}, indent=1) + "\n")
            print(json.dumps(row), flush=True)
            if args.status_file:
                stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
                with args.status_file.open("a") as handle:
                    handle.write(f"\n- {stamp}（{args.label}）{name}，{args.net.stem}/2500tick，{count}道×"
                                 f"{args.batches}真实独立批次，平均{row['wave_ms']:.2f}ms/波；"
                                 f"完整命令/SHA/原始输出见{args.out}。\n")


if __name__ == "__main__":
    main()
