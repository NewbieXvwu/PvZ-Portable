#!/usr/bin/env bash
# 探路包构建脚本：先报告环境，再检查依赖，缺什么就说清楚，最后尝试编译。
# 用法：bash build.sh
# 结果：成功时 build/pvz-portable 可用；失败时仍可用 build_prebuilt/pvz-portable 兜底。
set -u
cd "$(dirname "$0")"

JOBS="${JOBS:-$(nproc 2>/dev/null || echo 2)}"
# 解压工具不一定会保留可执行位（Python zipfile 就不保留），兜底二进制先补上。
[ -f build_prebuilt/pvz-portable ] && chmod +x build_prebuilt/pvz-portable 2>/dev/null
REPORT=build_report.json
: > "$REPORT.tmp"

log() { printf '%s\n' "$*" | tee -a "$REPORT.tmp"; }
kv()  { printf '%s=%s\n' "$1" "$2" >> "$REPORT.tmp"; }

log "=== 1/4 环境 ==="
OS="$( . /etc/os-release 2>/dev/null; printf '%s' "${PRETTY_NAME:-unknown}" )"
GLIBC="$(ldd --version 2>/dev/null | head -1)"
CPU="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//')"
CORES="$(nproc 2>/dev/null || echo unknown)"
CMAKE_V="$(cmake --version 2>/dev/null | head -1)"
CXX_V="$( (c++ --version 2>/dev/null || g++ --version 2>/dev/null || clang++ --version 2>/dev/null) | head -1)"
log "os:      $OS"
log "kernel:  $(uname -srm)"
log "glibc:   $GLIBC"
log "cpu:     $CPU"
log "cores:   $CORES"
log "cmake:   ${CMAKE_V:-MISSING}"
log "c++:     ${CXX_V:-MISSING}"
kv os "$OS"; kv glibc "$GLIBC"; kv cpu "$CPU"; kv cores "$CORES"
kv cmake "$CMAKE_V"; kv cxx "$CXX_V"

log ""
log "=== 2/4 依赖 ==="
missing=""
have_header() { [ -f "/usr/include/$1" ] || [ -f "/usr/local/include/$1" ]; }
# 注意：不能用 ls 一次列多个路径 —— 任一不存在就整体返回非零，会把装好的库误报成缺失。
have_lib() {
  ldconfig -p 2>/dev/null | grep -q "lib$1\.so" && return 0
  local path
  for path in /usr/lib/*/lib"$1".so /usr/lib/lib"$1".so /usr/local/lib/lib"$1".so; do
    [ -e "$path" ] && return 0
  done
  return 1
}

check() {  # check <名字> <判定命令...>
  local name="$1"; shift
  if "$@" >/dev/null 2>&1; then log "  ok      $name"; else log "  MISSING $name"; missing="$missing $name"; fi
}

[ -n "$CMAKE_V" ] || { log "  MISSING cmake"; missing="$missing cmake"; }
# C++20 <format>：GCC 13+ / Clang 17+ 才有，GCC 11/12 一定没有。
cat > /tmp/pvz_probe_format.cpp <<'EOF'
#include <format>
#include <string>
int main() { return static_cast<int>(std::format("{}", 1).size()); }
EOF
if c++ -std=c++20 /tmp/pvz_probe_format.cpp -o /tmp/pvz_probe_format >/dev/null 2>&1; then
  log "  ok      c++20 <format>"
  kv cxx20_format ok
else
  log "  MISSING c++20 <format>（需要 GCC 13+ 或 Clang 17+）"
  missing="$missing cxx20_format"
  kv cxx20_format missing
fi
check sdl2-header have_header SDL2/SDL.h
check sdl2-lib    have_lib SDL2
check openmpt-header have_header libopenmpt/libopenmpt.h
check openmpt-lib    have_lib openmpt
check zlib-header have_header zlib.h
check zlib-lib    have_lib z
check png-header  have_header png.h
check png-lib     have_lib png
check jpeg-header have_header jpeglib.h
check jpeg-lib    have_lib jpeg
if have_header cblas.h || have_header openblas/cblas.h || have_header x86_64-linux-gnu/cblas.h; then
  log "  ok      blas-header"
else
  log "  MISSING blas-header"
  missing="$missing blas"
fi

if [ -n "$missing" ]; then
  log ""
  log "缺依赖：$missing"
  if command -v apt-get >/dev/null 2>&1; then
    SUDO=""; [ "$(id -u)" -eq 0 ] || { command -v sudo >/dev/null 2>&1 && SUDO="sudo"; }
    log "尝试 apt-get 安装（可能需要网络）..."
    $SUDO apt-get update -qq >/dev/null 2>&1
    $SUDO apt-get install -y -qq cmake build-essential pkg-config \
      libsdl2-dev libopenmpt-dev zlib1g-dev libpng-dev libjpeg-dev libopenblas-dev \
      >/dev/null 2>&1
    log "apt-get 返回 $?"
  fi
fi

log ""
log "=== 3/4 配置 ==="
if [ -n "$missing" ] && [ ! -f /usr/include/SDL2/SDL.h ]; then
  log "依赖仍然不全，跳过编译（用 build_prebuilt/pvz-portable 兜底）"
  kv build ok; kv build_skipped "missing deps:$missing"
  mv "$REPORT.tmp" "$REPORT"; exit 2
fi

rm -rf build
if ! cmake -S src -B build -DCMAKE_BUILD_TYPE=Release \
     -DFETCHCONTENT_SOURCE_DIR_MIMALLOC="$PWD/vendor/mimalloc" \
     -DPVZ_PGO_DATA=/nonexistent \
     -DCMAKE_C_FLAGS=-O2 -DCMAKE_CXX_FLAGS=-O2 > build_configure.log 2>&1; then
  log "cmake 配置失败，见 build_configure.log（末尾 20 行）："
  tail -20 build_configure.log | tee -a "$REPORT.tmp"
  kv build failed; kv build_stage configure
  mv "$REPORT.tmp" "$REPORT"; exit 3
fi
log "  configure ok"

log ""
log "=== 4/4 编译（-j$JOBS）==="
START=$(date +%s)
if cmake --build build --target pvz-portable -j"$JOBS" > build.log 2>&1; then
  END=$(date +%s)
  log "  build ok，用时 $((END-START)) 秒"
  ls -la build/pvz-portable | tee -a "$REPORT.tmp"
  kv build ok; kv build_seconds "$((END-START))"
else
  log "  编译失败，见 build.log（末尾 30 行）："
  tail -30 build.log | tee -a "$REPORT.tmp"
  kv build failed; kv build_stage compile
  mv "$REPORT.tmp" "$REPORT"; exit 4
fi
mv "$REPORT.tmp" "$REPORT"
log ""
log "完成。下一步：python3 selfcheck.py"
