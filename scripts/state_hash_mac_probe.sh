#!/usr/bin/env bash
# task82: run the cross-process state-hash probe on a Mac checkout.
#
# The desktop cannot drive the Mac build, so this script is the hand-off.  It
# starts one game process at a time (the probe itself is serial), records eight
# independent resets plus the first 2000 operations of a human demo, and then
# summarises the result.  It does not need the task82 fixes deployed first: the
# point is to measure the current official build, so a varying hash is an
# expected finding rather than an error.
#
# Override any path with the matching environment variable; defaults match the
# Mac checkout used for recording.  Run from the repository root or anywhere.
set -euo pipefail

ROOT="${PVZ_MAC_ROOT:-/Users/newbiexvwu/PvZAgent}"
EXECUTABLE="${PVZ_MAC_EXECUTABLE:-$ROOT/build/pvz-portable}"
RESOURCE_DIR="${PVZ_RESOURCE_DIR:-/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN}"
ARCHIVE="${PVZ_ARCHIVE:-$ROOT/artifacts/human_demo/raw/seed_20261031/demo_fog_31_snow_m200}"
OUT="${PVZ_OUT:-$HOME/pvz-logs/task82/mac}"

for path in "$EXECUTABLE" "$RESOURCE_DIR/main.pak" "$ARCHIVE/ticks.jsonl" \
            "$ROOT/scripts/state_hash_determinism.py" "$ROOT/scripts/state_hash_compare.py"; do
  if [[ ! -e "$path" ]]; then
    echo "missing: $path" >&2
    echo "merge branch hashdet into $ROOT (or set PVZ_MAC_ROOT/PVZ_MAC_EXECUTABLE)" >&2
    exit 2
  fi
done

mkdir -p "$OUT"
PVZ_NO_POPUP=1 python3 "$ROOT/scripts/state_hash_determinism.py" \
  --executable "$EXECUTABLE" \
  --resource-dir "$RESOURCE_DIR" \
  --archive "$ARCHIVE" \
  --out "$OUT" \
  --repeats 8 --operations 2000

python3 "$ROOT/scripts/state_hash_compare.py" "$OUT" --out "$OUT/comparison.json"
echo "hand back $OUT/summary.json and $OUT/comparison.json"
