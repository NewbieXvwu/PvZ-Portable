#!/usr/bin/env bash
# 组装"Mac 自检包"：正式包内容 + 4 局人类示范存档 + 一条命令的 run.sh。
#
#   scripts/llmplay/mac_bundle.sh <正式包 stage 目录> <示范存档根> <输出目录>
#
# 为什么需要它：4 局示范存档在 artifacts/ 下、**不入 git**（AGENTS §1），
# 所以在一台干净的 Mac 上 clone 仓库后直接跑 mac_check.sh 会报 MISSING。
# 这个脚本把包、示范、校验入口装进一个目录，拷到 Mac 上解压即可。
#
# 产物只落本机（不进 git、不上 HF）：
#   <输出目录>/pkg/        正式包内容（含 res/ 与 build/pvz-portable）
#   <输出目录>/demos/      4 局示范（ticks.jsonl / frames.jsonl / meta.json）
#   <输出目录>/run.sh      校验入口
#   <输出目录>/README.txt  两条命令的说明
#   <输出目录>.tar.gz
set -euo pipefail

STAGE="${1:?用法: $0 <正式包 stage 目录> <示范存档根> <输出目录>}"
DEMOS="${2:?用法: $0 <正式包 stage 目录> <示范存档根> <输出目录>}"
OUT="${3:?用法: $0 <正式包 stage 目录> <示范存档根> <输出目录>}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAGE="$(cd "$STAGE" && pwd)"
DEMOS="$(cd "$DEMOS" && pwd)"
OUT="$(mkdir -p "$OUT" && cd "$OUT" && pwd)"

TASKS="demo_fog_31_snow_m200 demo_fog_32_snow_m200 demo_fog_33_cactus_snow_m200 demo_fog_33_cattail_snow_m200"

rm -rf "$OUT/pkg" "$OUT/demos"
mkdir -p "$OUT/pkg" "$OUT/demos"

# 1. 正式包内容（含引擎与资源）
cp -a "$STAGE/." "$OUT/pkg/"

# 2. human_demo.py 及其依赖（verify 要用的那几个模块，包里没有）
cp "$ROOT/scripts/human_demo.py" "$OUT/pkg/scripts/human_demo.py"
for name in pvz_event_env.py pvz_wait_events.py pvz_frame_store.py; do
  cp "$ROOT/python/$name" "$OUT/pkg/python/$name"
done

# 3. 4 局示范（只拷需要的三份文件，frames 是 verify 的比对基准）
for task in $TASKS; do
  src="$(find "$DEMOS" -maxdepth 2 -type d -name "$task" | head -1)"
  if [ -z "$src" ]; then
    echo "找不到示范存档：$task（在 $DEMOS 下）" >&2
    exit 2
  fi
  dst="$OUT/demos/$(basename "$(dirname "$src")")/$task"
  mkdir -p "$dst"
  cp "$src/ticks.jsonl" "$src/frames.jsonl" "$src/meta.json" "$dst/"
done

# 4. 校验入口
cat > "$OUT/run.sh" <<'RUN'
#!/usr/bin/env bash
# 在 Mac 上验四局人类示范。默认走 Docker（包里发的是 Linux 静态 ELF）。
#
#   ./run.sh                       # Docker 模式：自动拉 python:3.11-slim，装 numpy 后重放
#   ./run.sh local <Mach-O 引擎>   # 本地模式：用你自己编的 macOS 引擎
#
# 期望：四行 PASS + 末行 MACCHECK OK。
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODE="${1:-docker}"
EXE="${2:-}"

if [ "$MODE" = docker ]; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "没有 docker。装 Docker Desktop，或 ./run.sh local <Mach-O 引擎>" >&2
    exit 2
  fi
  exec docker run --rm -v "$HERE":/w -w /w python:3.11-slim bash -c \
    "pip install -q numpy && bash /w/run.sh local /w/pkg/build/pvz-portable"
fi

if [ "$MODE" != local ]; then
  echo "用法：$0 [docker | local <引擎>]" >&2
  exit 2
fi

EXE="${EXE:-$HERE/pkg/build/pvz-portable}"
if [ ! -f "$EXE" ]; then
  echo "找不到引擎：$EXE" >&2
  exit 2
fi
# pvz_env 会 chdir 到引擎所在目录再起进程，相对路径会 ENOENT。
EXE="$(cd "$(dirname "$EXE")" && pwd)/$(basename "$EXE")"
if [ "$(uname)" = Darwin ]; then
  magic="$(od -An -tx1 -N4 "$EXE" | tr -d ' \n')"
  if [ "$magic" = "7f454c46" ]; then
    echo "这是 Linux ELF（$EXE），macOS 不能直接执行。" >&2
    echo "用 ./run.sh docker，或 ./run.sh local /path/to/macho-engine" >&2
    exit 2
  fi
fi
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "缺 numpy：pip3 install numpy" >&2
  exit 2
fi

expected="demo_fog_31_snow_m200:39590 demo_fog_32_snow_m200:57596 demo_fog_33_cactus_snow_m200:41107 demo_fog_33_cattail_snow_m200:41831"
failed=0
for item in $expected; do
  task="${item%%:*}"
  want="${item##*:}"
  archive="$(find "$HERE/demos" -maxdepth 2 -type d -name "$task" | head -1)"
  if [ -z "$archive" ]; then
    echo "MISSING  $task"
    failed=1
    continue
  fi
  out="$(python3 "$HERE/pkg/scripts/human_demo.py" --resource-dir "$HERE/pkg/res" \
        --executable "$EXE" verify --archive "$archive" 2>/dev/null)"
  rc=$?
  # 引擎把 INFO 日志混在标准输出里，只取那一行 JSON。
  line="$(printf '%s\n' "$out" | grep -E '^\{"verified"' | tail -1)"
  ops="$(printf '%s' "$line" | python3 -c 'import json,sys;print(json.load(sys.stdin)["operations"])' 2>/dev/null)"
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
RUN
chmod +x "$OUT/run.sh"

cat > "$OUT/README.txt" <<'DOC'
Mac 自检包（task81）
====================

验什么：同一个引擎重放 4 局人类示范，逐 tick 的操作序列必须与录制时一致。
期望输出：四行 PASS，末行 MACCHECK OK。

怎么跑（二选一）：

1) Docker（推荐；包里的引擎是 Linux 静态 ELF，macOS 不能直接执行）
     tar xzf mac_bundle.tar.gz && cd mac_bundle && ./run.sh
   首次会拉 python:3.11-slim 并 pip 装 numpy。

2) 本地 Mach-O 引擎（若你在 Mac 上从同一份源码编过）
     ./run.sh local /path/to/pvz-portable
   需要 numpy：pip3 install numpy

把输出贴回来即可。四局的操作数应为
39590 / 57596 / 41107 / 41831，均 result=1（通关）。
DOC

tar -czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo "bundle: $OUT"
echo "archive: $OUT.tar.gz"
du -sh "$OUT" "$OUT.tar.gz"
