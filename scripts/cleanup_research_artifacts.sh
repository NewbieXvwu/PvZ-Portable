#!/usr/bin/env bash
# 清理 artifacts/research 下陈旧的训练产物。
#
# 判据（2026-10-01 审计）：每个 run 的 80 个检查点里，真正有用的只有 8 个 ——
# evaluated 全部（分析脚本按这些节点取模型）、最新一个 trained（续跑用）、
# initial 与 boundary（起点与冻结边界）。其余 trained 是每个 update 都存一份的中间快照，
# 没有任何引用链指向它们。
#
# 整条删除的线，其结论层（learning_curve.json / evaluations/*.json.gz / provenance.json）
# 已存在于 artifacts/research_evidence/ 归档并已入库，删除不丢结论。
#
# 用法：
#   bash cleanup_research_artifacts.sh          # 空跑，只打印计划
#   DRY_RUN=0 bash cleanup_research_artifacts.sh # 真正执行
set -euo pipefail

DRY_RUN="${DRY_RUN:-1}"
RESEARCH_DIR="${RESEARCH_DIR:-$HOME/PvZ-Portable/artifacts/research}"

# 正在跑的 run，永不触碰。
PROTECTED=(
  reward_r0_seed1_v2
  reward_r1_seed1_v2
  reward_r2_seed1_v2
  reward_r3_seed1_v2
  reward_r0_seed2_v2
  reward_r1_seed2_v2
  reward_r2_seed2_v2
  reward_r3_seed2_v2
)

# 整条删除：第一版奖励对比（已被第二版取代）、冒烟 run（阶段早已结束）、
# 分片交付机制的验证残留（那套机制从未用于生产）。
FULL_DELETE=(
  reward_r0_seed0_v1
  reward_r1_seed0_v1
  t5a_smoke_r0_seed0_v1
  t5a_smoke_r0_seed0_v2
  t5a_smoke_r0_seed0_v3
  t5a_smoke_r0_seed0_v4
  checkpoint_chunk_audit_v1
  checkpoint_chunk_export_v1
  checkpoint_chunk_restored_v1
)

# 已完成的 run：只删中间 trained 检查点。
TRIM_TRAINED=(
  reward_r0_seed0_v2
  reward_r1_seed0_v2
  reward_r2_seed0_v2
  reward_r3_seed0_v2
)

cd "$RESEARCH_DIR"

echo "清理目录: $RESEARCH_DIR"
echo "模式    : $([ "$DRY_RUN" = "1" ] && echo '空跑（不删任何东西）' || echo '*** 实际执行 ***')"
echo
echo "=== 清理前 ==="
du -sh . 2>/dev/null | sed 's/^/  /'

echo
echo "=== 安全检查：正在跑的 run 是否在待删列表里 ==="
for p in "${PROTECTED[@]}"; do
  for d in "${FULL_DELETE[@]}" "${TRIM_TRAINED[@]}"; do
    if [ "$p" = "$d" ]; then
      echo "  !! 冲突：$p 同时出现在保护列表与待删列表，中止"
      exit 1
    fi
  done
done
echo "  通过：$(printf '%s ' "${PROTECTED[@]}")均不在待删列表"

echo
echo "=== 第二组各 run 的完成情况（确认确实跑完再删中间检查点） ==="
for d in "${TRIM_TRAINED[@]}"; do
  [ -d "$d" ] || { printf "  %-24s 不存在，跳过\n" "$d"; continue; }
  python3 - "$d" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
state = json.loads((run / "training_state.json").read_text(encoding="utf-8"))
c = state.get("counters", {})
print("  %-24s status=%-8s decisions=%s" % (run.name, state.get("status"), c.get("decisions")))
PY
done

echo
echo "=== 第一组：整条删除 ==="
total_full=0
for d in "${FULL_DELETE[@]}"; do
  if [ -d "$d" ]; then
    kb=$(du -sk "$d" | cut -f1)
    total_full=$((total_full + kb))
    printf "  %-32s %8s\n" "$d" "$(du -sh "$d" | cut -f1)"
  fi
done
printf "  %-32s %8s\n" "小计" "$(awk -v k=$total_full 'BEGIN{printf "%.1f GB", k/1048576}')"

echo
echo "=== 第二组：删中间 trained 检查点 ==="
total_trim=0
for d in "${TRIM_TRAINED[@]}"; do
  run_dir="$d/runs/run_1"
  [ -d "$run_dir" ] || continue
  mapfile -t trained < <(ls -1 "$run_dir"/*_trained_*.pt 2>/dev/null | sort || true)
  n=${#trained[@]}
  if [ "$n" -le 1 ]; then
    printf "  %-24s 只有 %d 个 trained，跳过\n" "$d" "$n"
    continue
  fi
  keep="${trained[$((n - 1))]}"
  freed=0
  for f in "${trained[@]}"; do
    [ "$f" = "$keep" ] && continue
    freed=$((freed + $(du -k "$f" | cut -f1)))
    if [ "$DRY_RUN" != "1" ]; then rm -f -- "$f"; fi
  done
  total_trim=$((total_trim + freed))
  printf "  %-24s %2d 个 → 保留 1 个，删 %2d 个，释放 %.1f GB\n" \
    "$d" "$n" "$((n - 1))" "$(awk -v k=$freed 'BEGIN{print k/1048576}')"
done

if [ "$DRY_RUN" = "1" ]; then
  echo
  echo "=== 空跑结束，未删除任何文件 ==="
  printf "预计释放：整条删除 %.1f GB + 中间检查点 %.1f GB = %.1f GB\n" \
    "$(awk -v k=$total_full 'BEGIN{print k/1048576}')" \
    "$(awk -v k=$total_trim 'BEGIN{print k/1048576}')" \
    "$(awk -v a=$total_full -v b=$total_trim 'BEGIN{print (a+b)/1048576}')"
  exit 0
fi

# 删除记录：本项目重视可追溯性，删掉的东西留一份清单。
MANIFEST="CLEANUP_MANIFEST_$(date -u +%Y%m%dT%H%M%SZ).txt"
{
  echo "# artifacts/research 清理记录"
  echo "# 时间: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# 机器: $(hostname)"
  echo "# 判据: 每个 run 只保留 evaluated 全部 + 最新一个 trained + initial + boundary；"
  echo "#       其余 trained 是每个 update 都存一份的中间快照，无引用链指向。"
  echo "# 整条删除的线，结论层已存在于 artifacts/research_evidence/ 归档。"
  echo
  echo "## 整条删除"
} > "$MANIFEST"
for d in "${FULL_DELETE[@]}"; do
  if [ -d "$d" ]; then
    printf "%-32s %8s  %s 个文件\n" "$d" "$(du -sh "$d" | cut -f1)" \
      "$(find "$d" -type f | wc -l)" >> "$MANIFEST"
  fi
done
echo >> "$MANIFEST"
echo "## 删除的中间 trained 检查点" >> "$MANIFEST"

for d in "${FULL_DELETE[@]}"; do
  [ -d "$d" ] && rm -rf -- "$d"
done

for d in "${TRIM_TRAINED[@]}"; do
  run_dir="$d/runs/run_1"
  [ -d "$run_dir" ] || continue
  mapfile -t trained < <(ls -1 "$run_dir"/*_trained_*.pt 2>/dev/null | sort || true)
  n=${#trained[@]}
  [ "$n" -le 1 ] && continue
  keep="${trained[$((n - 1))]}"
  for f in "${trained[@]}"; do
    [ "$f" = "$keep" ] && continue
    echo "  $f" >> "$MANIFEST"
    rm -f -- "$f"
  done
  echo "  (保留 $keep)" >> "$MANIFEST"
done

echo
echo "删除记录: $RESEARCH_DIR/$MANIFEST"
echo
echo "=== 清理后 ==="
du -sh . 2>/dev/null | sed 's/^/  /'
echo
echo "=== 剩余 run ==="
du -sh */ 2>/dev/null | sort -hr | head -20
