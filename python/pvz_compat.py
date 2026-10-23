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
        "what": "frames.jsonl 旧档案的读取路径（写入只产 npz）",
        "kept_for": "2026-10 以前的 JSONL 档案，whatif 重放要读（AGENTS §8 第 4 条）",
        "since": "2026-10-01",
        "remove_when": "所有机器与 HF 上都不再有 frames.jsonl，且没有 whatif 存档需要重放"
                        "（Mac 的存量已于 2026-10-23 转成 npz，台式机待转）",
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
        "what": "纯 Python 续走实现 _python_rollout_reference（产品路径已只有 C++）",
        "kept_for": "python/test_cpp_rollout.py 与 scripts/perf_rollout_check.py 逐帧对拍 C++ 路径",
        "since": "2026-10-16",
        "remove_when": "C++ 推演的对拍测试改用存档回放",
        "where": ["scripts/search_play.py:_python_rollout_reference"],
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
# Empty since 2026-10-23: every switch that was registered here was deleted with the main-control
# approval of hotpath-audit.md section 6.1/6.2. A new entry needs a real experiment that may be
# withdrawn, not a way to keep two behaviours alive.
EXPERIMENTAL_FLAGS: dict[str, dict] = {}


def legacy_ids() -> set[str]:
    return set(LEGACY)


def expired(today: date | None = None) -> list[str]:
    """Experiment switches whose decide-by date has passed: decide or delete."""
    today = today or date.today()
    return [key for key, entry in EXPERIMENTAL_FLAGS.items()
            if date.fromisoformat(entry["decide_by"]) < today]
