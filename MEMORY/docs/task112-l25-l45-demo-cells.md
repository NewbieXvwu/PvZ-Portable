# task112：用户新录的 L25 / L45 原生示范建起点格

用户 2026-10-26 录完两局，都通关：

| 任务 | 种子 | 操作数 | 被吃植物 |
|---|---|---|---|
| `demo_pool_25_orig_m100` | 20261025 | 32 648 | 14 |
| `demo_roof_45_orig_m100` | 20261045 | 46 960 | 28 |

存档在 Mac 主检出 `artifacts/human_demo/raw/seed_{20261025,20261045}/`，主控会拷到台式机 cq 的同名目录。

## 要做的

做法完全照 task107（分支 convdemo 的 78ca7023 / 4a0b87a1：L30 / L40 / L50 的起点格）：

1. 每局 verify **一次**，用 Mac 本机引擎即可，两局并行。台式机部署时的 dry-run 会再用生产引擎加载一遍，不要另加复验。
2. 在 `configs/human_demo_tasks.json` 中给两项加上 `demo_start`：
   - 决策阶梯从示范后段开始；
   - `index=0`；
   - 未来种子块从 DESIGN 新登记，与已有块不重叠，各 256 个。
   - L45 蹦极：如果起点格的观测或任务里有和 task110 结论相关的字段，原样沿用现有原生任务的口径，不新增观测字段。
3. DESIGN 在 §14.36 同节追加登记。测试期望同步。
4. 分支 `demo2545`，从 `origin/deploy14d` 开，不合并 pvz-env；推送。由主控随 deploy14e 部署。

最多 6 个游戏进程。遵守 AGENTS.md §6.5。

## 结果

- 2026-10-26 verify（Mac 本机引擎 `/Users/newbiexvwu/PvZAgent/build/pvz-portable`，sha256 `6f7d2c3f…`，与录制时同一二进制 → 逐操作状态哈希）：两局并行、各一个游戏进程。`demo_pool_25_orig_m100` 32,648 操作 / tick 32,567 / wave 30 / result 1；`demo_roof_45_orig_m100` 46,960 操作 / tick 46,835 / wave 30 / result 1。两局 `verified: true`。
- 2026-10-26 起点格登记：`configs/human_demo_tasks.json` 新增两条原生任务（字段照 task111 的 `pvz-env` 版本）并各加 `demo_start`——pool 阶梯 `624..264`（6 档，步距 72，668 帧）、roof 阶梯 `920..380`（7 档，步距 90，966 帧），均 `index=0`；未来种子块 45,600,000 / 45,700,000 各 256，与 44.0M–45.5M、46.0M–47.0M 等既有段机械核对不相交。L45 不新增观测字段（task110 的可见蹦极状态缺口保持原状）。
- 2026-10-26 qc 侧对齐（离线、不起进程）：`load_demo_start_cells` → `make_state` 在两格上生成起点格，档位分别落在 624 / 920（阶梯 0 号 = 示范后段），再次 `reconcile_start_cells` 返回空（幂等）；`command_for` 带 `--demo-decision 624` / `920` 与 `--future-seeds 45600000..45600007` / `45700000..45700007`。
- 2026-10-26 测试与 DESIGN：`test_registered_manifest_start_cells` 期望 34 → 36；`test_curriculum_queue` 63 过 1 skip，`test_student_selection` 的 1 项失败在基线 `cc8cc4ed` 同样失败。DESIGN §14.36 追加 task112 条目与部署清单（不执行，随 deploy14e）。

## 问题

1. **本工作树没有示范存档**，`test_registered_manifest_start_cells` 按既有约定 skip；改用主检出存档根单独加载两个新格（阶梯 / 256 个未来种子 / verified 存档）通过。存档不入 git。
2. **L45 的蹦极状态仍对模型不可见**（task110）：本次只登记起点格，没有新增观测字段；示范里的落点红靶只能靠后续接口改造（估计 2–3 人日）。
3. **阶梯没有接手扫描**：照 task107 的形状取示范后段，具体档位是否落在老师会赢的一侧未实测，由 qc 的 0.85 / 0.5 规则自己收敛。
4. **任务条目在本分支新建**：`origin/deploy14d` 上没有 task111 在 `pvz-env` 登记的两条任务，本分支按同样字段补上并加 `demo_start`；主控合并 `pvz-env` 时按 `task_id` 去重。

