#!/usr/bin/env bash
# 重放 4 局人类示范，逐 tick 核对，确认引擎与录制时一致。
#
#   scripts/llmplay/mac_check.sh [资源目录] [引擎]
#
# 不传参数时用 $PVZ_RESOURCE_DIR 与仓库根的 build/pvz-portable。
# 每局打印一行 PASS/FAIL，最后一行是 MACCHECK OK 或 MACCHECK FAILED。
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RES="${1:-${PVZ_RESOURCE_DIR:-}}"
EXE="${2:-$ROOT/build/pvz-portable}"
RAW="${PVZ_DEMO_RAW:-$ROOT/artifacts/human_demo/raw}"

if [ -z "$RES" ]; then
  echo "用法：$0 <资源目录> [引擎]" >&2
  exit 2
fi
if [ ! -f "$EXE" ]; then
  echo "找不到引擎：$EXE" >&2
  exit 2
fi
if [ ! -d "$RES" ]; then
  echo "找不到资源目录：$RES" >&2
  exit 2
fi

expected="demo_fog_31_snow_m200:39590 demo_fog_32_snow_m200:57596 demo_fog_33_cactus_snow_m200:41107 demo_fog_33_cattail_snow_m200:41831"
failed=0
for item in $expected; do
  task="${item%%:*}"
  want="${item##*:}"
  archive="$(find "$RAW" -maxdepth 2 -type d -name "$task" 2>/dev/null | head -1)"
  if [ -z "$archive" ]; then
    echo "MISSING  $task（$RAW 下没找到存档）"
    failed=1
    continue
  fi
  out="$(python3 "$ROOT/scripts/human_demo.py" --resource-dir "$RES" --executable "$EXE" verify --archive "$archive" 2>/dev/null)"
  rc=$?
  ops="$(printf '%s' "$out" | python3 -c 'import json,sys;print(json.load(sys.stdin)["operations"])' 2>/dev/null)"
  if [ $rc -ne 0 ]; then
    echo "FAIL     $task  verify 报错：$out"
    failed=1
  elif [ "$ops" != "$want" ]; then
    echo "FAIL     $task  操作数 $ops，应为 $want"
    failed=1
  else
    echo "PASS     $task  $out"
  fi
done

if [ $failed -ne 0 ]; then
  echo "MACCHECK FAILED"
  exit 1
fi
echo "MACCHECK OK"
