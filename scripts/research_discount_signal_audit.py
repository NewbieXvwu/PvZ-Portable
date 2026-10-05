"""折扣参考（DISCOUNT_REFERENCE_TICKS）对学习信号的影响审计 —— 离线，不训练。

要回答的问题
------------
主线奖励 = 势能塑形 + 终局 ±1。塑形项在回报里**逐项相消**，只剩 −Φ(开局) 这个常数
（见 2026-10-04 记录，实测 6 位小数吻合）。终局项写入最后一个 transition，回报递推
给它的开局权重是 `γ ** ((T − Δ_last) / DISCOUNT_REFERENCE_TICKS)`，其中 `Δ_last`
是最后一步的 tick 时长。下方只用任务固有估算 `T` 近似这一前缀时长，因此 `D` 是粗略代理。

假设：把参考从 300 拉长，能解释"短任务学得动、长任务学不动"。

结论（2026-10-04 实测）：**假设被否**。
- 匹配对照（同关卡/同地形/同倍率/同固有长度，唯一差别是预种 10 株）：
  cap10 无预种 0.500 vs 有预种 1.000；cap15 0.125 vs 1.000；完整 30 波 0.000 vs 0.750。
- 反例：roof_1 固有 6,540 tick（D=0.803）通过率 0.312；
  aid_capfull 固有 52,800 tick（D=0.171，信号最弱）通过率 0.750。
  **信号强 4.7 倍的任务，通过率反而低一半多。**
- 长度只在**同族内**单调（无预种 day 族 0.500→0.125→0.000；有预种 1.0→1.0→1.0→0.75），
  是次要因素；**主导变量是"有没有预种"**。

> **2026-10-04 更正**：本文档早先版本把 `heldout_roof_1` 错配成 `train_roof_1`
> （第 45 关 cap3 vs 第 42 关 cap1），报出的 D=0.930 / 2,180 tick 是错的。
> 更正后 roof_1 的 D 是 **0.803**（不是 0.930），"信号最强"这个说法不再成立，
> 但"信号强度不预测通过率"的结论不变（见上）。`lookup_task()` 已改成拒绝这种回退。

用法
----
    <mise python> scripts/research_discount_signal_audit.py <checkpoint.pt> [曲线节点索引]

`mean_survival_ticks` 是**当前策略**的存活长度，会被性能污染（越差活得越短、D 反而越大），
所以这里用**任务固有长度**（wave_cap × 每波 tick 数）而不是它。
**注意这只是个粗略代理**：出怪推进与清场时间都受策略影响，固有长度不等于真实时长。
"""

from __future__ import annotations

import glob
import json
import os
import sys

GAMMA = 0.99
# 第 7 关实测：aid_cap10 胜局 17,617 tick / 10 波；aid_capfull 52,994 / 30 波 → ≈1,760 tick/波。
# 屋顶每波更长（2180），因为要铺花盆。
TICKS_PER_WAVE_DAY = 1760
TICKS_PER_WAVE_ROOF = 2180
REFERENCES = (300, 600, 1000, 3000)


def load_task_metadata(repo_root: str) -> dict[str, dict]:
    """先收评估清单（键就是评估记录里的键），再收训练清单。

    **顺序很重要**：评估清单必须优先，因为 `heldout_roof_1` 是**第 45 关 / cap 3**，
    而 `train_roof_1` 是**第 42 关 / cap 1** —— 两者只是名字像，不是同一个任务。
    2026-10-04 实测：早先的实现剥掉前缀后回退到 `train_*`，把这两个任务静默拼错，
    导致简报里 heldout_roof_1 的"固有 tick / D"用的是 train_roof_1 的值。
    """
    meta: dict[str, dict] = {}
    for pattern in ("experiments/t7/*/progress_evaluation.json",
                    "artifacts/task_family/heldout.json",
                    "experiments/t7/*/train.json"):
        for path in glob.glob(os.path.join(repo_root, pattern)):
            with open(path) as handle:
                payload = json.load(handle)
            tasks = payload if isinstance(payload, list) else payload.get("tasks", payload)
            for task in tasks:
                meta.setdefault(task["task_id"], task)
    return meta


def lookup_task(meta: dict[str, dict], key: str) -> dict | None:
    """评估记录的键 → 任务定义。**只允许两种别名，其余一律拒绝。**

    - 精确命中（评估清单里的 `heldout_*` / `validation_*`）
    - `validation_X` → `train_X`：已验证两边 level/deck/cap/preplanted 完全一致
      （`experiments/t7/bridge_random_mc_v1/progress_evaluation.json` 的
      `validation_*` 与 `bridge_level7_v1/train.json` 的 `train_*` 同定义）。

    `heldout_X` **不做** `train_X` 回退 —— 历史上正是这一步把 roof_1 拼错了。
    """
    if key in meta:
        return meta[key]
    if key.startswith("validation_"):
        return meta.get("train_" + key[len("validation_"):])
    return None


def intrinsic_ticks(task: dict) -> float:
    """打赢这一关大约需要多少 tick —— 与策略无关，只看任务定义。"""
    waves = task.get("wave_cap")
    if waves is None:
        waves = 30 if task.get("level") == 7 else 1
    per_wave = TICKS_PER_WAVE_ROOF if task.get("terrain") == "roof" else TICKS_PER_WAVE_DAY
    return waves * per_wave


def terminal_weight(ticks: float, reference: int) -> float:
    """用任务固有 tick 估算开局终局权重：1.000 表示开局与终局同等重要。"""
    return GAMMA ** (ticks / reference)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    import torch

    checkpoint_path = argv[0]
    node = int(argv[1]) if len(argv) > 1 else -2  # 默认倒数第二个节点（塌陷之前）
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    curve = checkpoint["training_state"]["learning_curve"]
    point = curve[node]
    meta = load_task_metadata(repo_root)

    header = f"{'task':40} {'通过':>6} {'固有tick':>9} {'cap':>5} {'地形':>5} {'预种':>4} |"
    header += "".join(f" {'D@'+str(r):>8}" for r in REFERENCES)
    print(f"节点 {node}: updates={point['updates']} decisions={point['counters']['decisions']}")
    print(header)

    rows = []
    for key, data in point["summary"]["greedy"]["per_task"].items():
        task = lookup_task(meta, key)
        if task is None:
            print(f"  （清单里没有 {key}，跳过）")
            continue
        ticks = intrinsic_ticks(task)
        rows.append((key, data["pass_rate"], ticks, task.get("wave_cap"),
                     task.get("terrain"), len(task.get("preplanted") or [])))
    rows.sort(key=lambda row: row[2])

    for key, rate, ticks, cap, terrain, preplanted in rows:
        cells = "".join(f" {terminal_weight(ticks, r):>8.3f}" for r in REFERENCES)
        print(f"{key:40} {rate:>6.3f} {ticks:>9.0f} {str(cap):>5} {str(terrain):>5} "
              f"{preplanted:>4} |{cells}")

    print("\n=== 控制变量：固有长度/地形/关卡相同，只差预种 ===")
    for plain, aided in (("ordinary_cap10", "aid_cap10"),
                         ("ordinary_cap15", "aid_cap15"),
                         ("full_level7_v1", "aid_capfull")):
        found_plain = next((r for r in rows if r[0].endswith(plain)), None)
        found_aided = next((r for r in rows if r[0].endswith(aided)), None)
        if not found_plain or not found_aided:
            continue
        print(f"  {plain:16} 通过 {found_plain[1]:.3f} "
              f"(固有 {found_plain[2]:.0f} tick, D@300={terminal_weight(found_plain[2], 300):.3f})"
              f"   vs  {aided:14} 通过 {found_aided[1]:.3f} (预种 {found_aided[5]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
