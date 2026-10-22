#!/bin/bash
# PGO for the engine process (task35).
#
# Two phases in the *same* build directory: GCC names the .gcda after the object file's
# path, so generating in one directory and using the profile from another silently finds
# nothing.
#
#   1. build with -fprofile-generate, run the rollout benchmark (the workload that matters:
#      every decision is a handful of C++ rollouts stepping the engine);
#   2. re-configure the same directory, which now picks up -fprofile-use automatically
#      (CMakeLists.txt looks for .gcda in PVZ_PGO_DATA), and rebuild.
#
# Usage: scripts/build_pgo.sh [--workload bench|search]
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
DATA=${PVZ_PGO_DATA:-$ROOT/artifacts/perf/pgo}
BUILD=$ROOT/build
PYTHON=${PYTHON:-$HOME/.venvs/ml/bin/python3}
WORKLOAD=${1:-bench}

rm -rf "$DATA" "$BUILD"
mkdir -p "$DATA"

echo "== phase 1: instrumented build ($BUILD -> $DATA)"
cmake -S "$ROOT" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_C_FLAGS="-fprofile-generate=$DATA" \
	-DCMAKE_CXX_FLAGS="-fprofile-generate=$DATA" \
	-DCMAKE_EXE_LINKER_FLAGS="-fprofile-generate=$DATA" \
	-DCMAKE_SHARED_LINKER_FLAGS="-fprofile-generate=$DATA" > /dev/null
ninja -C "$BUILD" -j "${JOBS:-8}" pvz-portable

echo "== workload: $WORKLOAD"
cd "$ROOT"
"$PYTHON" scripts/bench_engine.py bench --executable "$BUILD/pvz-portable" \
	--out "$ROOT/artifacts/perf/pgo_workload.json"

COUNT=$(find "$DATA" -name '*.gcda' | wc -l)
echo "== $COUNT profile files in $DATA"
[ "$COUNT" -gt 0 ] || { echo "no profile data was produced" >&2; exit 1; }

echo "== phase 2: profile-use build"
cmake -S "$ROOT" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_C_FLAGS="" -DCMAKE_CXX_FLAGS="" \
	-DCMAKE_EXE_LINKER_FLAGS="" -DCMAKE_SHARED_LINKER_FLAGS="" > /dev/null
ninja -C "$BUILD" -j "${JOBS:-8}" pvz-portable
echo "== done: $BUILD/pvz-portable uses $DATA"
