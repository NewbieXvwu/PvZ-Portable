#!/usr/bin/env bash
# Mixed DAgger round: the same teacher on N CPU workers and M GPU workers, one output directory.
#
# The formal queue runs search_play.py with these teacher parameters and reads OUT/summary.json
# (won) plus the frame archives under --frames.  This script splits the seed range N:M, runs one
# CPU process and one GPU process concurrently, and merges their games.jsonl into exactly the
# layout a single search_play.py run would have produced.
#
# Usage:
#   scripts/mixed_dagger_round.sh --student ~/PvZ-search/artifacts/student/v27/model.pt \
#     --task day_l7_z17_train --seed-offset 12160 --games 16 \
#     --frames ~/PvZ-search/artifacts/student/frames --frames-label z17dag28 \
#     --out ~/PvZ-search/artifacts/hard/z17_dagger28_16 --cpu-workers 2 --gpu-workers 2
#
# The student checkpoint is validated/exported up front (OUT/student.pvzrn); the GPU process
# re-exports into its own temporary directory because that is what the server loads.
set -euo pipefail

# The training queue owns this file; both collection processes inherit the path.
# search_play checks it only for CUDA workers, before each new game.
if [[ -n "${PVZ_GPU_PAUSE_FILE:-}" ]]; then export PVZ_GPU_PAUSE_FILE; fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$HOME/.venvs/ml/bin/python3"
MANIFEST="$HOME/PvZ-search/artifacts/stage4/day_l7_hard.json"
VALUE="$HOME/PvZ-search/artifacts/value_net/v3/model.pt"
STATIC_VALUE="$HOME/PvZ-search/artifacts/hard/value_branch1/model.pt"
RESOURCE_DIR="$HOME/.cache/pvz-research-resources"
EXECUTABLE="$ROOT/build/pvz-portable"
CUDA_BINARY="$ROOT/build/rollout_cuda_small"
DAGGER_BETA=0
VERIFY_SAMPLES=9
SPECULATIVE=1

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

STUDENT=""; TASK=""; SEED_OFFSET=""; GAMES=""; FRAMES=""; LABEL=""; OUT=""; CPU_WORKERS=""; GPU_WORKERS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --student) STUDENT="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --seed-offset) SEED_OFFSET="$2"; shift 2 ;;
    --games) GAMES="$2"; shift 2 ;;
    --frames) FRAMES="$2"; shift 2 ;;
    --frames-label) LABEL="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --cpu-workers) CPU_WORKERS="$2"; shift 2 ;;
    --gpu-workers) GPU_WORKERS="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --no-speculative) SPECULATIVE=0; shift ;;
    -h|--help) usage ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done
[ -n "$STUDENT" ] || { echo "missing --student" >&2; usage; }
[ -n "$TASK" ] || { echo "missing --task" >&2; usage; }
[ -n "$SEED_OFFSET" ] || { echo "missing --seed-offset" >&2; usage; }
[ -n "$GAMES" ] || { echo "missing --games" >&2; usage; }
[ -n "$FRAMES" ] || { echo "missing --frames" >&2; usage; }
[ -n "$LABEL" ] || { echo "missing --frames-label" >&2; usage; }
[ -n "$OUT" ] || { echo "missing --out" >&2; usage; }
[ -n "$CPU_WORKERS" ] || { echo "missing --cpu-workers" >&2; usage; }
[ -n "$GPU_WORKERS" ] || { echo "missing --gpu-workers" >&2; usage; }
[ -f "$STUDENT" ] || { echo "student not found: $STUDENT" >&2; exit 2; }
[ -f "$MANIFEST" ] || { echo "manifest not found: $MANIFEST" >&2; exit 2; }
[ -x "$EXECUTABLE" ] || { echo "executable not found: $EXECUTABLE (build pvz-portable)" >&2; exit 2; }
[ -x "$CUDA_BINARY" ] || { echo "cuda binary not found: $CUDA_BINARY (build rollout_cuda_small)" >&2; exit 2; }
[ -f "$VALUE" ] || { echo "value net not found: $VALUE" >&2; exit 2; }
[ -f "$STATIC_VALUE" ] || { echo "static value net not found: $STATIC_VALUE" >&2; exit 2; }
if [ -e "$OUT/games.jsonl" ]; then
  echo "refusing to mix into an existing run: $OUT/games.jsonl exists" >&2; exit 2
fi


for name in SEED_OFFSET GAMES CPU_WORKERS GPU_WORKERS; do
  value=${!name}
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "$name must be a nonnegative integer" >&2; exit 2; }
  printf -v "$name" '%d' "$((10#$value))"
done
[ "$((CPU_WORKERS + GPU_WORKERS))" -gt 0 ] || { echo "need at least one worker" >&2; exit 2; }
[ "$GAMES" -gt 0 ] || { echo "games must be positive" >&2; exit 2; }
for part in cpu gpu; do
  [ ! -e "$OUT/.parts/$part/games.jsonl" ] || { echo "partial run exists; use a new --out directory" >&2; exit 2; }
done
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p "$OUT/.parts" "$OUT/logs" "$FRAMES"

# 1. Validate student .pt -> .pvzrn, so a checkpoint the CUDA core cannot read fails before any game.
PYTHONPATH="$ROOT/python:$ROOT/scripts" "$PYTHON" - "$STUDENT" "$OUT/student.pvzrn" <<'PY'
import sys
from pvz_rollout_policy import export_student, load_student
net = load_student(sys.argv[1])
export_student(net, sys.argv[2])
print(f"student: obs={net.obs} -> {sys.argv[2]}")
PY

# 2. split the seed range N:M; the remainder goes to the CPU side (the reference backend).
GPU_GAMES=0
if [ "$GPU_WORKERS" -gt 0 ]; then GPU_GAMES=$(( GAMES * GPU_WORKERS / (CPU_WORKERS + GPU_WORKERS) )); fi
if [ "$GPU_GAMES" -eq 0 ] && [ "$GPU_WORKERS" -gt 0 ]; then GPU_GAMES=1; fi
CPU_GAMES=$(( GAMES - GPU_GAMES ))
[ "$CPU_GAMES" -ge 0 ] || { echo "split error: more GPU games than games" >&2; exit 2; }

export PYTHONPATH="$ROOT/python:$ROOT/scripts"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PVZ_CUDA_BINARY="$CUDA_BINARY" PVZ_CUDA_PRIVATE_LIMIT=1200 PVZ_CUDA_COOPERATIVE_HEADS=1
export PVZ_SERVE_TRACE=1 PVZ_EXECUTABLE="$EXECUTABLE" PVZ_RESOURCE_DIR="$RESOURCE_DIR"

COMMON=("$ROOT/scripts/search_play.py" --rollout-policy "$STUDENT" --value "$VALUE" --static-value "$STATIC_VALUE"
  --manifest "$MANIFEST" --task-id "$TASK" --executable "$EXECUTABLE"
  --cuda-binary "$CUDA_BINARY" --resource-dir "$RESOURCE_DIR"
  --arms fast_search --student-candidates 2 --shovel-candidates 0
  --verify-samples "$VERIFY_SAMPLES" --dagger-beta "$DAGGER_BETA"
  --frames "$FRAMES" --frames-label "$LABEL" --measure-cpu)
PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do
    if kill -0 -- "-$pid" 2>/dev/null; then kill -TERM -- "-$pid" 2>/dev/null || true; fi
  done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

started=$(date +%s)

if [ "$CPU_GAMES" -gt 0 ]; then
  echo "cpu: $CPU_GAMES games on $CPU_WORKERS workers (seed-offset $SEED_OFFSET)" | tee "$OUT/logs/cpu.log"
  setsid "$PYTHON" "${COMMON[@]}" --rollout-backend cpp --workers "$CPU_WORKERS" \
    --seed-offset "$SEED_OFFSET" --seeds "$CPU_GAMES" --out "$OUT/.parts/cpu" >> "$OUT/logs/cpu.log" 2>&1 &
  PIDS+=("$!")
fi
if [ "$GPU_GAMES" -gt 0 ]; then
  GPU_OFFSET=$(( SEED_OFFSET + CPU_GAMES ))
  echo "gpu: $GPU_GAMES games on $GPU_WORKERS workers (seed-offset $GPU_OFFSET)" | tee "$OUT/logs/gpu.log"
  GPU_SPEC=()
  if [ "$SPECULATIVE" -eq 1 ]; then GPU_SPEC=(--cuda-speculative-halving); fi
  setsid "$PYTHON" "${COMMON[@]}" --rollout-backend cuda --workers "$GPU_WORKERS" "${GPU_SPEC[@]}" \
    --seed-offset "$GPU_OFFSET" --seeds "$GPU_GAMES" --out "$OUT/.parts/gpu" >> "$OUT/logs/gpu.log" 2>&1 &
  PIDS+=("$!")
fi

for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    echo "a collection process failed; see $OUT/logs; other owned processes will stop" >&2
    exit 1
  fi
done
elapsed=$(( $(date +%s) - started ))

# 3. merge into the single-process layout: games.jsonl (sorted by seed) + summary.json + config.json.
"$PYTHON" - "$ROOT" "$OUT" "$elapsed" "$CPU_GAMES" "$GPU_GAMES" "$CPU_WORKERS" "$GPU_WORKERS" "$TASK" "$SEED_OFFSET" "$LABEL" "$STUDENT" "$MANIFEST" "$FRAMES" <<'PY'
import json, sys
from pathlib import Path
root, out = Path(sys.argv[1]), Path(sys.argv[2])
elapsed, cpu_games, gpu_games = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
cpu_workers, gpu_workers = int(sys.argv[6]), int(sys.argv[7])
task, offset, label, student = sys.argv[8], int(sys.argv[9]), sys.argv[10], sys.argv[11]
sys.path.insert(0, str(root / "scripts"))
import search_play

from render_episode import load_training_task
manifest, frames = Path(sys.argv[12]), Path(sys.argv[13])
expected = load_training_task(manifest, task)["seeds"][offset:offset + cpu_games + gpu_games]
if len(expected) != cpu_games + gpu_games or len(set(expected)) != len(expected):
    raise SystemExit("requested seed interval is incomplete or has duplicates")
rows = []
configs = []
for part in ("cpu", "gpu"):
    path = out / ".parts" / part / "games.jsonl"
    if path.exists():
        part_rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        wanted = expected[:cpu_games] if part == "cpu" else expected[cpu_games:]
        if sorted(r["seed"] for r in part_rows) != sorted(wanted) or any(r["arm"] != "fast_search" for r in part_rows):
            raise SystemExit(f"{part}: unexpected, missing or duplicate games")
        if part == "gpu" and any(r.get("cuda_coverage_version") != 1 or r.get("cuda_lanes", 0) <= 0
                or r["cuda_lanes"] != r["rollout_lanes"] or r["cuda_batches"] != r["rollout_batches"]
                or any(r.get(k, -1) != 0 for k in ("cuda_fallback_lanes", "cuda_fallback_batches", "cuda_error_lanes"))
                for r in part_rows):
            raise SystemExit("GPU coverage failed: refusing formal output")
        configs.append(json.loads(path.with_name("config.json").read_text()))
        rows += part_rows
rows.sort(key=lambda r: (r["seed"], r["arm"]))
if sorted(r["seed"] for r in rows) != sorted(expected):
    raise SystemExit("merged seed set does not match requested interval")
for c in configs:
    for key, value in {"verify_samples": "9", "student_candidates": "2", "dagger_beta": "0.0",
                       "task_id": task, "rollout_policy": student}.items():
        if c.get(key) != value: raise SystemExit(f"teacher parameter mismatch: {key}")
if len(configs) == 2:
    for key in ("rollout_policy", "value", "static_value", "manifest", "task_id", "samples", "horizon",
                "halving", "margin", "min_gap", "rollout_sim_skip", "frames", "frames_label", "frames_format"):
        if configs[0][key] != configs[1][key]: raise SystemExit(f"backend parameter mismatch: {key}")
for seed in expected:
    if not (frames / f"seed_{seed}" / f"{label}_fast_search" / "frames.npz").is_file():
        raise SystemExit(f"missing training archive: seed {seed}")
with (out / "games.jsonl").open("w") as stream:
    for row in rows:
        stream.write(json.dumps(row) + "\n")
search_play.summarize(out / "games.jsonl")

usage = {}
for part in ("cpu", "gpu"):
    path = out / ".parts" / part / "cpu_usage.json"
    if path.exists():
        usage[part] = json.loads(path.read_text())
config = {
    "mode": "mixed", "task": task, "student": student, "label": label, "seed_offset": offset,
    "games": len(rows), "cpu_workers": cpu_workers, "gpu_workers": gpu_workers,
    "cpu_games": cpu_games, "gpu_games": gpu_games, "elapsed_wall_s": elapsed,
    "host_cpu_s_per_game": (round(sum(u["total_host_cpu_s"] for u in usage.values()) / len(rows), 3)
                            if usage else None),
    "parts": usage,
}
(out / "config.json").write_text(json.dumps(config, indent=1) + "\n")
print(json.dumps(config, indent=1))
PY

echo "mixed round done: $OUT ($(wc -l < "$OUT/games.jsonl") games, $elapsed s)"
