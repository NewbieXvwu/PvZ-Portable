# task148：用 GPT Pro 第二次会话的胜局，给屋顶关找门禁卡组（门禁 v6 草案）

主控 2026-10-29 定。分支 `gatev6`，以 `origin/pvz-env` 为底。

## 背景

- 门禁 v5 的面板里屋顶只有 L41、L46：L42–L45、L47–L49 在记分板卡组下，老师和学生一局都没赢过，所以没能进门禁。
- GPT Pro 第二次会话（llmplay3 包，877 局、621 胜，台式机 `/home/newbiexvwu/llm_runs/llmplay3_r2/out/`）在屋顶 L41–L49 原版 1.0 倍下每关赢了 24–38 局，用了 4 套以上卡组。`out/progress.json` 的 `detail.A` 里有每关每套卡组的胜局、稳健种子、选卡理由和评价；`out/report.md` 是可读版。
- C 层 31 套挑战卡组里，有 11 套 5 局最多赢 1 局，附失败诊断（`detail.C[*].diagnosis`）。

## 要做的

1. 对 L42–L45、L47–L49 各挑 1–2 套卡组作门禁候选，按胜局数、稳健种子数排序。要满足 AGENTS §8 的卡组规则（地形覆盖、升级植物所有权、长度 6–10）。只用 progress.json 里的记录，不重放存档（§6.5）。
2. 用现役学生（`/home/newbiexvwu/pvz-logs/curriculum_state.pipeline/` 里的 active student）和搜索老师，在候选卡组上各打少量局（每关每卡组 8 局），量一下现在的胜率，看进门禁后是不是全 0。生产 qc 正在跑 14–16 个游戏进程，所以最多 2 个游戏进程，限内存。
3. 起草 `configs/gate_v6_panel.json` 与 `configs/adventure_scoreboard.json` 的屋顶增补（新文件，不改 v5）。在 DESIGN 的卡组登记处写清来源（GPT Pro r2 run_id `8024414b540c410893a07dc4e853f7e0`）。
4. 把 C 层失败诊断归纳成一页，作为卡组规则修订的输入：哪类卡组、缺什么。写进 `MEMORY/docs/task148-roof-gate-decks.md` 末尾。

## 约束

- 不部署、不改现役门禁、不碰 qc / 状态文件 / 看门狗。
- 不加自校验。每完成一步提交并推送 `gatev6`。
