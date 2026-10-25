# task92：课程管线每轮训多个种子、先挑选再进门禁

> 主控 2026-10-26 写。执行者：台式机 Codex（plus 号，tmux `codex-hashdet`；额度用完由 agent_pool 交 CodeBuddy）。工作树 `/home/newbiexvwu/PvZ-search/.claude/worktrees/multiseed`（分支 `multiseed`，主控已从 origin/pvz-env 拉出）。每个任务最多 2 个游戏进程；**不碰 tmux `qc`、cq、流水线状态目录**（只读）。

## 证据（task84，heldout = `day_l7_eval_d7p_holdout1`，192 对）
| 训练 | heldout | 对照 d7 |
|---|---|---|
| r11（现役，seed 1） | 166 | — |
| A：r11 标签与配方，seed 1 重训 | 124 | 172 |
| D2：r11 标签与配方，seed 2 | 169 | 178 |
| B：r13 + 扰动族权重 1.5 | 65 | 169 |
| C：r13，从 r11 续训 3 epoch、学习率 1/4 | 135 | 176 |
| r13 / r14 门禁 | 145 / 151 | — |

同一份数据换个训练种子，heldout 就在 124 到 169 之间摆动，而对照 d7 一直稳定在 169–178。r13–r17 连续五轮被拒，理由都包含 heldout。每轮只训一个种子，结果就是一次抽签。从现役续训（C）也没有解决问题。

## 要做的
1. **读懂现有管线**：
   - `scripts/curriculum_queue.py` 的后台训练（pack → train → evaluate），每轮在 `~/pvz-logs/curriculum_state.pipeline/round_*/`；
   - `scripts/select_student.py`，learn/06 的“挑会打的，而不是模仿得最像的”。
   写清楚一轮训练的耗时构成：打包、训练（GPU）、评估。
2. **设计**：每轮训 K 个种子（K=2 或 3），依据硬件实测决定。
   - 先测清楚：同一块 GPU 上并行训 2 个的显存与时间，台式机内存 25 GB，qc 采集要优先。
   - 挑选面板：
     - 种子与门禁、采集、评估全部不相交，并做机械核对；
     - 覆盖 day / heldout 类（扰动卡组）/ night / pool / fog / roof；
     - 每个任务约 32 局；
     - 用 `select_student.py` 的口径。
   - 选出的那个再进门禁，门禁合同完全不变。
3. **实现**：只在 `multiseed` 分支实现，不加开关；生产默认等主控合并时切换。要有测试：
   - K 个种子的训练目录互不覆盖；
   - 挑选面板的种子不相交；
   - 只有被选中的进门禁，落选的记入轮目录；
   - 某个种子训练失败时，其余照常挑选。B7 的语义保持不变。
4. **离线验证**：用 task84 已经训好的 A / D2 / C 三个模型，在设计好的挑选面板上实测：
   - 挑选面板的排序与 heldout 192 对的排序是否一致（至少选中 D2，不选 A）；
   - 用每局方差估算 K=2 / 3 时的期望增益。
5. 结论写 `MEMORY/docs/multiseed-select.md`，提交并 `git push origin multiseed`。合并与上生产由主控在组边界做。

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；`PVZ_NO_POPUP=1`。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 不进 git；不写字面密钥；不手写打法规则或公式。
- GPU 训练与 qc 冲突时让 qc 先。
- 每完成一小步就提交、push，在“结果”里记一行；有疑问写进“问题”，按最佳判断继续。

## 结果

- 2026-10-26：读完任务与后台 pack/train/evaluate、select_student 和 learn/06 挑选说明；冻结六区各 32 局的独立挑选面板，120M 保留块机械核对无交集，开跑前登记与种子审计已写入。A/D2/C 源模型只读，离线评估将使用复制件；生产流水线状态未写。
- 2026-10-26（CodeBuddy）：步骤 2/3 实现提交（84c6b795）：`_train_selected` 一份打包 → 逐种子独立目录 `seeds/seed_NNNNNN/` 训练（单个种子失败只记 failed.json，不连坐；全失败才走 B7）→ 冻结面板上 `select_student.py` 挑选 → 仅选中者复制为 `round_*/student/model.pt` 进门禁，`candidates.json`/`selection.json`（含 rejected 与每局 games.jsonl SHA）落轮目录。面板种子、门禁池（SEED_POOLS_V2）、human_demo 的机械核对进 enqueue 与测试。测试 72 passed（新增 test_student_selection.py + 3 个流水线测试），已 push origin multiseed。

## 问题
- 2026-10-26 07:1x（CodeBuddy 接手）：步骤 1 完成。一轮耗时构成（round 12–17 complete.json 实测）：pack 0.3–10 min、train 36–130 min（d3 旁证实测 ~170 s/epoch × 12）、门禁评估 2–46 min；后台串行单线程，采集 744 局/轮约 45–105 min，round 16/17 train（7768/6153 s）已成瓶颈——K 种子若串行会进一步拖慢轮节奏，倾向并行 2 训练。训练种子在 recipe 的 "seed" 字段（现值 1），K 种子 = 每轮 K 份配方（不同 seed、不同 out 目录 round_*/student_s*）。select_student.py 口径 = search_play --arms fast 按 wins 总和选，selection.json 已记录全部候选得分可复用为"落选记录"。search_play 无 CUDA 引用（纯 CPU），GPU 只有训练用；RTX 5080 16 GB。GATE_VERSION=3，门禁种子走 seed ledger（allocate_seed_block），挑选面板种子需与之机械核对不相交。
