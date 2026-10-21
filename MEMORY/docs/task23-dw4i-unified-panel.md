# task23（主控 2026-10-22 下发给台式机 Codex，额度用完由 CodeBuddy 接手）：统一面板 + 第一次加宽合训 dw4i

先读仓库根目录 `AGENTS.md`（优先级最高），再读 `MEMORY/docs/student-bottleneck-plan.md` 末尾 10-21、10-22 的几条、
DESIGN §14.7–§14.10、`MEMORY/docs/task19-identity-packets.md`、`task20-conveyor-fidelity.md`、`night-panel.md`。

## 背景
加宽阶段到现在：按身份出卡（identity 头，task19）、原版 1-10（task20，identity 256/256）、夜间卡组 n9（task21，老师 58/80、学生 ~4%）、
扰动卡组 d7p（task19 B）都已合进 pvz-env。台式机 queue101 正用 identity_dw3_s1 采夜间 n9 与 d7p，Mac 在采 d7p。
下一步是**一个模型**把这些都学进去，并且有一张**统一面板**衡量“一个模型打通冒险模式”走到哪。

## 要做的
工作区（台式机）：`git -C ~/PvZ-search worktree add ~/PvZ-search/.claude/worktrees/dw4 -b dw4 origin/pvz-env`，只在这里改代码、单独编译。

1. **统一面板**（脚本进 git，清单不进 git）：一个清单 + 一个脚本 `scripts/unified_panel.py`，给一个学生模型跑完下面全部并出一张表 + 一个 json，以后每个模型都用它：
   - 白天 7 卡：`day_l{1,3,4,6,7,8,9}_eval_d7`；原版 1-10：`day_l10_orig_eval`；
   - 换卡组留出：`day_l{7,8,9,10}_eval_d7p_holdout{1,2,3}`；
   - 夜间 n9：`night_l{11,12,13,14,16,17,18,19}_eval_n9`；2-10 用原版开局另开一族（如 `night_l20_eval_n9_orig`，DESIGN §14.9 补登记）。
   每项局数写成参数（默认白天 / 原版 128，留出每套 64，夜间 128），种子用各自评估段的前 N 个。汇总按区给平均胜率（白天、换卡组、夜间）。
   学生要能是 slot 头（v34）也能是 identity 头；同时最多 4 个游戏进程（台式机 queue101 在跑，别抢它）。
2. **导入 Mac 轮次**：HF `transfer/` 里的 `d9dag4b d10dag4b`（采集 v33）、`d9dag5b d10dag5b`（v34）、`d{7,8,9,10}p{1..4}m`（identity，Mac 还在采，取已上传的）
   解到 `~/PvZ-search/artifacts/student/frames_v2`（与已有轮次同布局，已存在的不覆盖），记进报告。
3. **训练 dw4i**：identity 头 + 按行汇总，配方照 task19 的 identity_dw3_s1（`scripts/run_identity_validation.py` 的参数），
   标签 = identity_dw3_s1 的全部标签 + queue100 的 `d*dag21–28` + Mac 4b/5b + 全部 d7p（`d*p*m`、`d*p*d`）+ 夜间 `n*dag*` + task20 原版两轮。
   等 queue101 至少完成 3 轮（看 `~/pvz-logs/queue101.log`）再开训；先 `pgrep -f train_rollout_policy` 确认没人在训。训 2 个种子（依次训）。
   记录总行数、各来源行数、训练时长与内存峰值。
4. **评估**：统一面板跑 dw4i s1 / s2、identity_dw3_s1、v34。报告写 `MEMORY/docs/unified-panel.md`：每项一行，四个模型并列；
   重点看 (a) 夜间涨了多少、(b) 白天与原版 1-10 有没有掉、(c) 换卡组留出有没有涨。

## 规则
- **不要动任何 tmux 队列（q101 等）与别的代理会话；不要改别的工作树与 `~/PvZ-search` 主工作区**（frames_v2 只增不改）。
- 分支 `dw4` 提交，可以 `git push origin dw4`；不推 pvz-env、不合并。不改默认值；不写手工规则；不写字面密钥（`$HF_TOKEN` / `$PVZ_HF_REPO`）；
  不 `git add -f`；`.pt/.npz/.log` 不进 git。模型训完传 HF `transfer/dw4i_s{1,2}_model.pt`。
- 每完成一小步就提交，并在本文末尾“结果”里记一行做到哪；疑问写“问题”一节，按最佳判断继续，不要停下来等。
  你额度用完时 `agent_switch.sh` 会让 CodeBuddy 接手、额度恢复后交回你，所以进度一定要落在提交和“结果”里。

## 结果

- 2026-10-22，1实现：unified_panel.py统一29项，支持slot/identity、默认2944局/模型、三组任务平均胜率、每任务JSON与表格、按模型/程序/清单/种子哈希拒绝串用续跑、2/3GiB内存守卫与≤4worker。新增L20原版任务族并在DESIGN §14.9/14.10登记；面板成员/原版模式/原留出定义保持检查通过。独立C++编译中，真实双头冒烟随后补做。

## 问题

- 任务书要求读DESIGN §14.10，但初始分支止于§14.9；本次补登记统一面板为§14.10。task20停止旧L10新增评估的约定，与task23明确要求L10三套留出有交叉；遵循本任务明确要求，只保留指定初始牌留出测试，与原版L10分开，不新增旧L10固定初始牌基线。
