#!/usr/bin/env bash
# 一台朋友机器的全套安装：micromamba 环境 → 依赖 → 编译 → 自检。
#
#   bash bootstrap.sh                 # 装到 ~/pvz-worker，clone distributed 分支最新提交
#   PVZ_COMMIT=<sha> bash bootstrap.sh
#   PVZ_BRANCH=distributed PVZ_REPO_URL=... bash bootstrap.sh
#
# 可重复执行：已装好的步骤会跳过。不需要 sudo，不需要 brew，不需要任何令牌。
# Linux / WSL / macOS(arm64, x86_64) 同一份脚本。
set -euo pipefail

ROOT=${PVZ_WORKER_ROOT:-$HOME/pvz-worker}
ENV=$ROOT/env
REPO=$ROOT/repo
BRANCH=${PVZ_BRANCH:-distributed}
REPO_URL=${PVZ_REPO_URL:-https://github.com/NewbieXvwu/PvZ-Portable.git}
COMMIT=${PVZ_COMMIT:-}
JOBS=${PVZ_JOBS:-$(( $(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4) ))}
LOM_V=0.8.9
LOM_URL="https://lib.openmpt.org/files/libopenmpt/src/libopenmpt-${LOM_V}+release.autotools.tar.gz"
LOG=$ROOT/logs/bootstrap.log
mkdir -p "$ROOT"/{bin,logs,models,resources,tasks,state,out,frames,manifest}

say() { printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }
die() { say "FAILED: $*"; exit 1; }

OS=$(uname -s); ARCH=$(uname -m)
case "$OS" in
  Darwin) PLAT=osx ;;
  Linux)  PLAT=linux ;;
  *) die "unsupported OS $OS (this script covers macOS and Linux/WSL)" ;;
esac
case "$ARCH" in
  arm64|aarch64) A=arm64 ;;
  x86_64|amd64)  A=x86_64 ;;
  *) die "unsupported arch $ARCH" ;;
esac
say "bootstrap: $OS/$ARCH -> $ROOT (jobs=$JOBS)"

# ---------------------------------------------------------------- 1. micromamba
if [ ! -x "$ROOT/bin/micromamba" ]; then
  say "installing micromamba"
  mkdir -p "$ROOT/bin"
  curl -Ls --retry 5 --retry-delay 3 "https://micro.mamba.pm/api/micromamba/${PLAT}-${A}/latest" \
    | tar -xj -C "$ROOT" bin/micromamba
fi
export MAMBA_ROOT_PREFIX=$ROOT/mamba
"$ROOT/bin/micromamba" --version | tee -a "$LOG"

# ---------------------------------------------------------------- 2. env
PKGS=(python=3.12 pip cmake ninja pkg-config sdl2 libpng jpeg zlib)
if [ "$PLAT" = linux ]; then PKGS+=(openblas cxx-compiler); export BLA_VENDOR=OpenBLAS; fi
if [ ! -x "$ENV/bin/python" ]; then
  say "creating env (${PKGS[*]})"
  "$ROOT/bin/micromamba" create -y -p "$ENV" -c conda-forge "${PKGS[@]}"
else
  say "env exists, ensuring packages"
  "$ROOT/bin/micromamba" install -y -p "$ENV" -c conda-forge "${PKGS[@]}"
fi
export PATH="$ENV/bin:$PATH"
export CMAKE_PREFIX_PATH="$ENV"
export PKG_CONFIG_PATH="$ENV/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
python -V | tee -a "$LOG"; cmake --version | head -1 | tee -a "$LOG"

# ---------------------------------------------------------------- 3. python deps
if ! python -c "import torch, numpy" >/dev/null 2>&1; then
  say "installing torch + numpy"
  if [ "$PLAT" = darwin ]; then
    python -m pip install --no-input "torch>=2.0" "numpy>=1.24" 2>&1 | tee -a "$LOG"
  else
    python -m pip install --no-input --index-url https://download.pytorch.org/whl/cpu "torch>=2.0" 2>&1 | tee -a "$LOG"
    python -m pip install --no-input "numpy>=1.24" 2>&1 | tee -a "$LOG"
  fi
fi
python - <<'PY' | tee -a "$LOG"
import torch, numpy, platform
print("torch", torch.__version__, "numpy", numpy.__version__, "py", platform.python_version())
PY

# ---------------------------------------------------------------- 4. libopenmpt
# The game links libopenmpt (music). conda-forge has no build for it on osx-arm64, so it is built
# here from the upstream release tarball. Shared build on purpose: the game's CMake has a
# static-link check that then demands mpg123/vorbis/ogg as well.
if ! ls "$ENV"/lib/libopenmpt.* >/dev/null 2>&1; then
  say "building libopenmpt $LOM_V"
  TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
  curl -sSL --retry 5 -o "$TMP/lom.tar.gz" "$LOM_URL"
  tar xzf "$TMP/lom.tar.gz" -C "$TMP"
  SRC=$(find "$TMP" -maxdepth 1 -type d -name 'libopenmpt-*' | head -1)
  ( cd "$SRC" && ./configure --prefix="$ENV" --disable-static --enable-shared \
      --without-mpg123 --without-ogg --without-vorbis --without-vorbisfile --without-sdl2 --without-flac \
      --without-portaudio --without-portaudiocpp --without-sndfile \
      --disable-openmpt123 --disable-examples --disable-tests \
      >"$ROOT/logs/libopenmpt-configure.log" 2>&1 \
    && make -j"$JOBS" >"$ROOT/logs/libopenmpt-make.log" 2>&1 \
    && make install >>"$ROOT/logs/libopenmpt-make.log" 2>&1 ) \
    || { tail -25 "$ROOT/logs/libopenmpt-configure.log" 2>/dev/null; tail -25 "$ROOT/logs/libopenmpt-make.log" 2>/dev/null; die "libopenmpt build"; }
  rm -rf "$TMP"; trap - EXIT
fi
say "libopenmpt: $(ls "$ENV"/lib/libopenmpt.* | head -2 | tr '\n' ' ')"

# ---------------------------------------------------------------- 5. repo
if [ -d "$REPO/.git" ]; then
  say "repo exists, updating to $BRANCH ${COMMIT:-HEAD}"
  git -C "$REPO" fetch --depth 1 origin "$BRANCH"
  git -C "$REPO" checkout -q FETCH_HEAD
elif [ -f "$REPO/CMakeLists.txt" ]; then
  say "repo present (pushed tree, no .git) — keeping it"
else
  say "cloning $REPO_URL ($BRANCH)"
  git clone -q --depth 1 --branch "$BRANCH" "$REPO_URL" "$REPO"
fi
if [ -n "$COMMIT" ]; then
  git -C "$REPO" fetch -q --depth 1 origin "$COMMIT" || true
  git -C "$REPO" checkout -q "$COMMIT"
fi
git -C "$REPO" rev-parse --short HEAD 2>/dev/null | sed 's/^/commit /' | tee -a "$LOG" || true

# ---------------------------------------------------------------- 6. build
say "cmake configure + build (-j$JOBS)"
cmake -S "$REPO" -B "$REPO/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$ENV" \
  -DPVZ_ROLLOUT_BLAS=ON \
  ${BLA_VENDOR:+-DBLA_VENDOR=$BLA_VENDOR} >"$ROOT/logs/cmake-configure.log" 2>&1 \
  || { tail -40 "$ROOT/logs/cmake-configure.log"; die "cmake configure"; }
cmake --build "$REPO/build" -j"$JOBS" >"$ROOT/logs/cmake-build.log" 2>&1 \
  || { tail -60 "$ROOT/logs/cmake-build.log"; die "cmake build"; }
[ -x "$REPO/build/pvz-portable" ] || die "pvz-portable not built"
say "built $(du -h "$REPO/build/pvz-portable" | cut -f1) binary"

# ---------------------------------------------------------------- 7. self-check
if [ -f "$ROOT/resources/main.pak" ] && [ -d "$ROOT/resources/properties" ]; then
  if [ -f "$ROOT/models/selfcheck.pt" ] && [ -f "$ROOT/models/value_v3.pt" ] && [ -f "$ROOT/manifest/day_l7_hard.json" ]; then
    say "self-check: 1 student-only game"
    rm -rf "$ROOT/selfcheck"; mkdir -p "$ROOT/selfcheck"
    ( cd "$REPO" && "$ENV/bin/python" scripts/search_play.py \
        --rollout-policy "$ROOT/models/selfcheck.pt" \
        --value "$ROOT/models/value_v3.pt" \
        --manifest "$ROOT/manifest/day_l7_hard.json" --task-id day_l7_z17_train \
        --seed-offset 0 --seeds 1 --arms fast --workers 1 \
        --frames "$ROOT/selfcheck/frames" --frames-label selfcheck \
        --resource-dir "$ROOT/resources" --out "$ROOT/selfcheck/out" ) >>"$LOG" 2>&1 \
      || { tail -30 "$LOG"; die "self-check game"; }
    PYTHONPATH="$REPO/python:$REPO/scripts" "$ENV/bin/python" - "$ROOT/selfcheck" <<'PY' | tee -a "$LOG"
import sys, pathlib
from pvz_frame_store import frame_paths, read_frames
root = pathlib.Path(sys.argv[1])
paths = frame_paths(root / "frames")
assert paths, "no frame archive written"
frames = read_frames(paths[0])
print("self-check OK:", len(paths), "archive,", len(frames), "frames")
PY
  else
    say "self-check skipped: need models/selfcheck.pt, models/value_v3.pt and manifest/day_l7_hard.json (controller push sends them)"
  fi
else
  say "self-check skipped: resources not pushed yet (controller push --resources)"
fi

# ---------------------------------------------------------------- summary
say "disk: $(du -sh "$ROOT" | cut -f1) total; env $(du -sh "$ENV" | cut -f1); repo $(du -sh "$REPO" | cut -f1)"
say "DONE. run: $ENV/bin/python $ROOT/repo/scripts/dist/worker.py --root $ROOT run"
