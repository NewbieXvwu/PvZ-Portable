#!/usr/bin/env bash
# artifacts/research 维护入口（在台式机上跑）。
#
# 这个脚本**不自己实现删除逻辑**，只把工作转交给
# scripts/prune_research_checkpoints.py —— 那里复用了 python/pvz_research.py 的
# prune_trained_checkpoints()，与训练进行中的策略是同一份代码，并且保护
# resume.json / training_state.json 指向的检查点。
#
# 为什么不在这里自己写一遍（本文件 2026-10-01 之前就是这个样子）：
#   旧版自己实现了一份裁剪，只保留 1 个 trained（代码侧策略是 8），而且不保护
#   resume.json 指向的文件。实测四个 reward_r*_v2 的 resume.json 都指向 boundary
#   而不是最新 trained —— 那份实现放到今天再跑一次，就会删掉续跑要用的检查点。
#
# 用法：
#   bash cleanup_research_artifacts.sh            # 空跑，只打印计划
#   DRY_RUN=0 bash cleanup_research_artifacts.sh  # 真正执行
#
# 环境变量：
#   REPO_ROOT  仓库根，默认 ~/PvZ-Portable
#   PYTHON     解释器，默认 ~/.venvs/ml/bin/python（系统 python 没有 numpy）
#
# 整条退役一条实验线**不在本脚本里做**：那是一次性操作，删前必须按 AGENTS.md §4
# 确认结论层（learning_curve.json / evaluations/*.json.gz / provenance.json）
# 已归档到 artifacts/research_evidence/。2026-10-01 那次退役 9 条线的记录在
# artifacts/CLEANUP_MANIFEST_20261001T053523Z_retired_runs.txt。
set -euo pipefail

DRY_RUN="${DRY_RUN:-1}"
REPO_ROOT="${REPO_ROOT:-$HOME/PvZ-Portable}"
PYTHON="${PYTHON:-$HOME/.venvs/ml/bin/python}"

if [ ! -d "$REPO_ROOT" ]; then
  echo "找不到仓库：$REPO_ROOT（用 REPO_ROOT=... 指定）" >&2
  exit 1
fi
if [ ! -x "$PYTHON" ]; then
  echo "找不到解释器：$PYTHON（用 PYTHON=... 指定）" >&2
  exit 1
fi

cd "$REPO_ROOT"

ARGS=()
if [ "$DRY_RUN" != "1" ]; then
  ARGS+=(--apply)
fi

exec "$PYTHON" scripts/prune_research_checkpoints.py ${ARGS[@]+"${ARGS[@]}"}
