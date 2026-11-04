"""Assemble the task243 conclusion from completed measurements, without replaying games."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from task239_compare import summarize


def read(path):
    return json.loads(path.read_text())


def main():
    base = ROOT / "artifacts/task243"
    exported = read(base / "features/summary.json")
    offline = read(base / "models/offline.json")
    teacher = read(base / "teacher/report.json")
    panel = read(base / "teacher_panel.json")
    for task in panel["tasks"]:
        task["pairs"] = len(task["seeds"])
    arms = {arm: read(base / f"diag-{arm}/rows.json") for arm in ("production", "v5", "v6")}
    reference = {row["id"]: row for row in arms["production"]}
    maxdiff = 0.0
    for arm in ("v5", "v6"):
        for row in arms[arm]:
            for plan, entry in row["plans"].items():
                if not entry.get("applicable"):
                    continue
                for horizon, values in entry["values"].items():
                    expected = reference[row["id"]]["plans"][plan]["values"][horizon]
                    maxdiff = max(maxdiff, float(np.max(np.abs(np.asarray(values) - expected))))
    errors = Counter(failure["error"].strip().splitlines()[-1] for failure in exported["failures"])
    report = dict(task="task243", title="价值网v6全量重训与老师替换实测", status="完成，未部署",
        inputs=read(base / "inputs.json"),
        data=dict(selection=read(base / "selection.json"), successful=exported["games"],
                  failed=exported["failed"], failure_reasons=dict(errors), by_terrain=offline["data"],
                  frames_per_game=96, feature_encoding="仅C++ InternalFeatures编码v6，v5取保留的旧特征列",
                  sampling="random.Random(243)，按地形与胜负配额，在关卡间轮流抽样，全局种子去重",
                  filter="frame_count>=32，result为终局胜/负，且有frames.npz",
                  selection_manifest=str((base / "selection.tsv").resolve()),
                  selection_sha256=hashlib.sha256((base / "selection.tsv").read_bytes()).hexdigest(),
                  labels="存档终局胜负，新引擎逐操作重放成功后保留",
                  split=offline["split_games"],
                  production_training_overlap={key: offline[key] for key in
                    ("known_production_training_seeds", "production_training_seeds_in_new_train",
                     "production_training_seeds_in_new_validation_or_test")},
                  scope="按胜负均衡抽样的生产语料留出集，Brier/log loss对应这个抽样分布"),
        training={key: offline[key] for key in ("device", "width", "epochs", "lr", "weight_decay",
                  "dropout", "batch_size", "seed", "roof", "seconds", "training")},
        offline={key: offline[key] for key in ("metrics", "bootstrap", "bootstrap_repetitions",
                  "production", "delta_direction")},
        teacher=dict(panel=panel, **teacher),
        shovel=dict(drills=31, horizons=[1200, 2500], shared_future_seeds=[23600000+i for i in range(4)],
                    judge="固定生产v5/v5roof末端叶值，共享task236冻结学生/GPT续走，不用被测模型作裁判",
                    judge_max_difference_between_arms=maxdiff,
                    results={arm: summarize(rows, (1200, 2500)) for arm, rows in arms.items()},
                    limitation="参考叶值仍有估计误差，短视界续走结果不能代表整局救场成功率",
                    production_shovel_enabled=False),
        model_paths={arm: str((base / f"models/{arm}.pt").resolve()) for arm in ("v5", "v6")},
        model_sha256={arm: hashlib.sha256((base / f"models/{arm}.pt").read_bytes()).hexdigest()
                      for arm in ("v5", "v6")},
        production=dict(deployed=False, recommendation=teacher["eligible"],
            steps=["候选先通过老师整局配对非劣验收，本轮两组均未通过，当前不执行后续切换",
                   "主控确认采用的叶值模型与屋顶路由后，通过HF transfer传模型到台式机，取用后删除中转文件",
                   "从vn6full构建并部署支持encoder6的独立台式机引擎，保留旧引擎回滚路径",
                   "组边界切换qc的--executable、--value与--value-roof，静态候选网与搜索参数沿用本次冻结配置",
                   "铲子候选继续关闭"]),
        resources=dict(mac_workers_max=6, desktop="只读生产存档，nice19/ionice-c3/4GB虚拟内存上限，无GPU/pytest",
                       peak_artifacts_gib_observed=6.8,
                       collector="未暂停或停止采集", engine_build="~/pvz-mac-collect/build-vn6，未触碰build2",
                       intermediates="已删除源存档、特征分片、memmap、预测数组与临时日志，仅保留新模型和报告附件；冻结参考模型的本任务副本已删除，哈希保留于inputs"))
    report["training"]["roof"] = "未另训屋顶网：现行v5roof依赖task105扫描起点的老师续走软标签，本轮为整局胜负语料，使用统一模型并单列屋顶验收"
    report["training"]["data_scope"] = "本轮训练使用重放后的整局终局标签，未加入task239建议的额外铲子分支后果行；31题铲子仅作独立验收"
    report["offline"]["production"] = "生产路由：非屋顶v5，屋顶v5roof"
    report["offline"]["delta_direction"] = "候选减对照：log loss/Brier为负表示改善，AUC为正表示改善"
    report["teacher"]["protocol"] = "每关32个新种子，各关种子段独立，组内同种子同未来，学生/GPT先验/静态候选网与搜索参数固定，仅换叶值"
    report["teacher"]["recommendation_rule"] = "每关净胜差至少-2/32且单侧McNemar未显著变差，合并配对95%区间下界至少-2/32"
    report["teacher"]["search"] = dict(verify_samples=9, student_candidates=2, per_card=1,
        samples=3, horizon=2500, min_gap=300, sim_skip=61, dagger_beta=1,
        static="固定生产静态候选网", prior="普通地形固定学生，雾与屋顶固定GPT先验",
        shovel=False)
    if not any(teacher["eligible"].values()):
        report["conclusion"] = dict(
            recommendation="不建议将本轮v6或同数据v5替换生产叶值，保持生产配置与铲子关闭状态",
            offline="两组重训模型的留出集终局预测均显著优于生产，v6相对同数据v5的增益置信区间跨零，先导特征增益未在全量训练中复现",
            teacher="离线预测改善未转化为老师整局胜率提升，两组候选均未通过配对非劣验收",
            shovel="固定参考裁判下，v6铲掉不补的符号判反更多，白铲与人类救场的平均绝对误差也更高，不支持开启铲子",
            cause="本轮未定位整局退化的机制，不能据终局预测指标推断老师动作排序质量")
    target = ROOT / "MEMORY/docs/task243-vn6-full.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=1, allow_nan=False) + "\n")
    print(target)


if __name__ == "__main__":
    main()
