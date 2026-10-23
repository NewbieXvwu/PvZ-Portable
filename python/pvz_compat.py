"""One place for the backward-compatibility paths and the experiment switches still in the tree.

Two rules this file exists to enforce (AGENTS.md, task33 2026-10-23):

* a branch kept only for old checkpoints / archives / panels is marked LEGACY(<id>) in the code and
  registered here with **who it is kept for** and **when it may go**;
* an experiment switch that is allowed to stay is marked EXPERIMENT(<id>) next to its add_argument
  and registered with a **decide-by date**, after which the test fails until somebody turns it into
  the only behaviour or deletes it.

python/test_compat_registry.py checks both directions: every marker has an entry, every entry has a
marker, every decide-by date is still in the future, every legacy entry says who it is for and what
would let it go.

Nothing here has side effects: it is data plus two helpers.
"""
from __future__ import annotations

from datetime import date

# id -> {what, kept_for, since, remove_when, where}
LEGACY: dict[str, dict] = {
    "obs-versions": {
        "what": "学生观测版本 v1 / v2lite / v2 与它们冻结的编码器签名",
        "kept_for": "artifacts/hard_mac/frames 下 21 056 个档案（encoder 5c3448262051）、"
                    "以及面板与文档仍在对照的 v34、identity_dw3_s1、dw4i_s1 检查点",
        "since": "2026-10-14",
        "remove_when": "旧档案全部重算、且没有面板或文档再对照带 v2lite/v2 观测的检查点",
        "where": ["python/pvz_student_observation.py", "python/pvz_frame_store.py:encoder_signature"],
    },
    "jsonl-archives": {
        "what": "frames.jsonl 旧档案的读取路径（写入分支只服务它）",
        "kept_for": "2026-10 以前的 JSONL 档案，whatif 重放要读（AGENTS §8 第 4 条）",
        "since": "2026-10-01",
        "remove_when": "artifacts 下不再有 frames.jsonl",
        "where": ["python/pvz_frame_store.py:frame_paths", "python/pvz_frame_store.py:frame_bytes"],
    },
    "slot-packet-head": {
        "what": "按槽位出卡的头（packet_head='slot'）",
        "kept_for": "v34 及更早的学生检查点（现行面板的对照）",
        "since": "2026-10-14",
        "remove_when": "v34 不再出现在任何面板与文档对照里",
        "where": ["python/pvz_rollout_policy.py:RolloutNet"],
    },
    "python-rollout-reference": {
        "what": "_rollout_policy 里的纯 Python 续走实现",
        "kept_for": "scripts/perf_rollout_check.py 用它逐帧对拍 C++ 路径",
        "since": "2026-10-16",
        "remove_when": "C++ 推演的对拍测试改用存档回放",
        "where": ["scripts/search_play.py:_rollout_policy"],
    },
    "conveyor-start-key": {
        "what": "观测里的 conveyor_start='original' 键与它的 free_cards 分支",
        "kept_for": "L10 传送带关任务族（task20）",
        "since": "2026-10-19",
        "remove_when": "传送带任务族退役",
        "where": ["python/pvz_rollout_policy.py:free_cards"],
    },
    "aux-labels": {
        "what": "辅助标签（row_damage_labels / cache_aux / aux_weight）",
        "kept_for": "task16 的辅助头实验数据与训练配方",
        "since": "2026-10-20",
        "remove_when": "辅助头实验判死或转正",
        "where": ["python/pvz_frame_store.py:cache_aux"],
    },
    "pvzrn-versions": {
        "what": "C++ 学生网络导出格式 PVZRN002–005",
        "kept_for": "旧导出文件与 python/test_cpp_rollout.py 的版本矩阵",
        "since": "2026-10-20",
        "remove_when": "测试只保留 PVZRN006",
        "where": ["python/pvz_env.py:load_rollout_net", "src/EnvironmentRollout.cpp"],
    },
}

# id -> {flag, script, owner, since, decide_by, why}
EXPERIMENTAL_FLAGS: dict[str, dict] = {
    "cuda-backend": {
        "flag": "--rollout-backend", "script": "scripts/search_play.py",
        "owner": "codex task4", "since": "2026-10-17", "decide_by": "2026-11-06",
        "why": "GPU 续走从未在正式轮次启用（113 份运行配置全是 cpp）",
    },
    "cuda-prefetch": {
        "flag": "--cuda-prefetch-verify", "script": "scripts/search_play.py",
        "owner": "codex task11", "since": "2026-10-18", "decide_by": "2026-11-06",
        "why": "与推测减半一起等 CUDA 后端定论",
    },
    "cuda-speculative": {
        "flag": "--cuda-speculative-halving", "script": "scripts/search_play.py",
        "owner": "codex task11", "since": "2026-10-18", "decide_by": "2026-11-06",
        "why": "推测减半在 CPU 路径上没有收益，只在 GPU 批次里才有意义",
    },
    "candidate-generator": {
        "flag": "--candidate-generator", "script": "scripts/search_play.py",
        "owner": "codex task13", "since": "2026-10-18", "decide_by": "2026-11-06",
        "why": "学生逐格候选与静态候选的比较还没做完",
    },
    "candidate-coverage": {
        "flag": "--candidate-coverage-model", "script": "scripts/search_play.py",
        "owner": "codex task11", "since": "2026-10-18", "decide_by": "2026-11-06",
        "why": "覆盖候选开关，默认关闭（student-bottleneck-plan.md:307）",
    },
    "shovel-candidates": {
        "flag": "--shovel-candidates", "script": "scripts/search_play.py",
        "owner": "主控", "since": "2026-10-17", "decide_by": "2026-11-06",
        "why": "铲子候选保持关闭（student-bottleneck-plan.md:151），正确形态是替换而不是先铲后种",
    },
    "sparse-tracks": {
        "flag": "--cpu-sparse-tracks", "script": "scripts/search_play.py",
        "owner": "codex task6", "since": "2026-10-17", "decide_by": "2026-11-06",
        "why": "稀疏轨道默认关闭，等内存上限定了再决定",
    },
    "measure-cpu": {
        "flag": "--measure-cpu", "script": "scripts/search_play.py",
        "owner": "task18", "since": "2026-10-19", "decide_by": "2026-11-06",
        "why": "只用于手工量 CPU 账，正式轮次从不设置",
    },
}


def legacy_ids() -> set[str]:
    return set(LEGACY)


def expired(today: date | None = None) -> list[str]:
    """Experiment switches whose decide-by date has passed: decide or delete."""
    today = today or date.today()
    return [key for key, entry in EXPERIMENTAL_FLAGS.items()
            if date.fromisoformat(entry["decide_by"]) < today]
