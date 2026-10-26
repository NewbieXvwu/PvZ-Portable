# 接手扫描与起点格登记自动化（task91）结论

> 2026-10-26。入口 `scripts/takeover_scan.py`（run / adaptive / register），测试
> `python/test_takeover_scan.py`。任务书：[task91-takeover-scan-runner.md](task91-takeover-scan-runner.md)。
>
> 当前状态：工具、两份端到端及**v2全部六份同块回测通过**：人类L46/L48/L49与LLM L46三份档差均0，每份88局。
> 首轮L49漏档已修复且失败证据保留；v2真实CLI复核通过后已自动接上12份生产adaptive，尚待扫描/登记/测试全部完成。
> 生产阶段见ignored的`artifacts/task91/audit/continuation_status.json`，v2复核回执见`guard_v2_comparisons.json`；部分结果不代表生产验收完成。
> 紧凑证据：[takeover-scan-evidence.json](takeover-scan-evidence.json)。

## 用法

```bash
P=/home/newbiexvwu/PvZ-search/.claude/worktrees/llmimport
PY=/home/newbiexvwu/.venvs/ml/bin/python3

# 1) task89 入库（若尚未做）会顺带生成扫描清单 scan_jobs_*.json
# 2) 按清单跑两臂（断点续跑：重跑同一命令只补缺的未来）
$PY $P/scripts/takeover_scan.py run  $P/artifacts/llm_demo/raw/scan_jobs_20261026.json
# 2') 自适应取点（默认）：4 等距老师点 → 跨50%二分/失败区间补探 → 50%线附近补档，
#     学生臂只打 50% 线与顶档两端。所有点都落在 task89 的 12 点网格上，可与 12 点结果直接比档。
$PY $P/scripts/takeover_scan.py adaptive $P/artifacts/llm_demo/raw/scan_jobs_20261026.json
# 3) 判读登记：老师 ≥50% 的存档生成阶梯写 demo_start；共用 task89 加锁账本分配 256 种子块；
#    既有起点格不覆盖（task87 的 _llm 族等），只核对与报告；同步 DESIGN 与测试期望。
$PY $P/scripts/takeover_scan.py register $P/artifacts/llm_demo/scan_out/results --dry-run   # 先看
$PY $P/scripts/takeover_scan.py register $P/artifacts/llm_demo/scan_out/results
```

生产老师参数与 qc 同款：学生 r11（`curriculum_state.pipeline/round_000011/student/model.pt`）、
叶值 v5、静态网 branch1、`--verify-samples 9 --student-candidates 2`；老师臂 β=1，学生臂 β=0（fast 臂）。
结果按存档写 `scan_out/results/<task_id>_s<seed>.json`，每局含胜负、末波、破防行与僵尸类型。

## 实现要点

- Codex 接续修正：断点按实际未来种子集合匹配，禁止用相同行数的其他种子充数；存档/模型/引擎指纹变化时拒绝沿用旧输出。
  CodeBuddy 旧缓存先核对记录的搜索配置，再冻结当前输入（旧运行时的权重哈希不可追溯，不作历史哈希保证）。
  结果原子落盘；全新输出目录可直接执行 adaptive。22项测试通过，真实已完成点续跑0.27s，原始对局文件不变。

- **断点续跑三层**：run 的 per-(存档,点,臂) 输出目录里，search_play 自己按 (arm, future_seed)
  跳过已完局；takeover_scan 再按结果文件里已有的完整点跳过整点。实测重跑 0.08s 零重跑。
- **登记族解析**（`resolve_target_task`）：按存档路径匹配清单里已有的 demo_start 最可靠；
  本机种子 LLM 局与人类示范共用 base task_id 时落 `base_llm` 族（task87 先例）；base 被别的
  存档占用且`base_llm`缺失时，只有老师过线才复制原族配置创建新族并同步DESIGN与测试；
  两个候选都被其他存档占用则只报告冲突。dry-run不建族，不覆盖既有起点格。
- **登记种子账本**：默认与 task89 共用 `artifacts/llm_demo/scan_seed_ledger.json` 及同一个文件锁，
  同时接受扫描96块与登记256块。旧 `register_seed_ledger.json` 只作为兼容占用来源参与避让；
  清单/盘点也参与检查，登记从86M起。按存档绑定分配，失败重试复用已有块。
  隔离验收用显式私有账本，不写生产清单。dry-run仅计算避让结果；底层清单更新也禁止覆盖既有格。
- **阶梯规则**（task71b 同款）：档位全部 ≥50%（4/8 恰过线）；顶档取最晚的 8/8 稳定点，
  没有 8/8 就取最晚 ≥50% 点（报告注明）；顶档以上即使 ≥50% 也不入（不如稳定点）。
- **自适应预算**：最多9个老师点、2个不同学生点，8未来时最多88局（<192/2）。失败区间补探降低漏掉非单调隆起的风险，
  不能保证发现任意未测点的孤立过线；没有发现过线时报告“已测点均<50%”，不冒充全网格皆失败。
  每点原子保存；中断状态不可登记。独立 `adaptive_trace` 保留本次取点顺序、逐局结果和真实预算。
- **测试**：`python/test_takeover_scan.py` 当前38项，另有导入器30项通过（阶梯规则、自适应取点顺序的合成驱动、
  族名解析、清单/测试期望的文件更新）。
  实际课程测试模板里有多个 `set(cells)` 断言，更新器按AST定位 `test_registered_manifest_start_cells`，
  不再误改临时样例。`DemoStartCellsTest` 8项中7项通过、1项因人类示范缺失跳过。

## 回测（自适应 vs 全量 12 点，同一存档同一未来种子块）

对象：task89 包的 L46 三份通关（s20261046 / s20261047 / s20261048，与 task87 扫描的是同一批
存档——网格尾部 996 / 1089 / 1138 与 §6.16 的精确点号一致）。方法：先用 adaptive（84M 块）
跑出自适应线，再 `run --teacher-only` 补齐同块 12 点，`scripts/compare_same_block.py` 对比：

| 存档 | adaptive L50 | 同块 12 点 L50 | 档差 | 自适应续走数 |
|---|---|---|---|---|
| demo_roof_46_doom_m200:20261046 | 634 | 634 | 0 | 88 |
| demo_roof_46_doom_m200_s20261047:20261047 | 693 | 693 | 0 | 88 |
| demo_roof_46_doom_m200_s20261048:20261048 | 622 | 622 | 0 | 88 |

Codex按当前4点起测+补探+预算9的算法复核，三份都实测9个老师点+2个学生点，即88局，
低于全量192局的一半。全部复用原始对局缓存；本次没有重跑游戏。
比较器要求完整12点基线、独立自适应轨迹、同块逐局证据和成本约束，缺点/换块/伪汇总会报错。

**修正**：初版等距 4 点在 s20261048 上漏档——该存档曲线非单调（622 过线 6/8、725 跌回 3/8、
829 再过线 5/8），4 等距（idx 0,4,7,11）的失败段之间测不到 622 的隆起，自适应线停在 829（差 2 档）。
CodeBuddy曾改5点使这三份Δ=0。当前恢复4点起测，在二分后补探失败区间，同样三份Δ=0，
并用9个老师点的硬预算约束成本；这些是指定样本上的实测，不是任意曲线的精度保证。

**跨种子块噪声（不是算法误差）**：同存档的 50% 线跨未来种子块会移动——我们 84M 块与
task87 的扫描块相比，三份的 L50 分别差 1 / 2 / 2 档（如 s20261048 的 622 在我们块下 6/8、
725 只有 3/8；task87 块下 621=4/8、725=8/8）。8 未来是小样本，边界档（4/8 上下）随块翻转。
含义：`demo_start` 阶梯档位只应依赖**同一块**的实测（本工具自洽满足）；不要拿不同块的
两次扫描逐档对比，也不要把 4/8 vs 6/8 的边界差读成趋势。

**人类回测完成**：沿task70部署记录定位到cq工作树中的L46/L48/L49副本，只读复制到本任务
`artifacts/human_demo/raw/`并逐文件SHA核验。当前r11按task89生成的12点网格执行adaptive，
再补齐同一未来块的12点老师对照；v2结果如下。历史Mac胜率表不充作本次逐局证据。
扫描计划与原始证据位于ignored的`artifacts/task91/audit/human_*`；本任务最多2个游戏进程。

当前（v2）人类adaptive与同块12点老师对照：

| 人类存档 | adaptive L50 | 老师点数 | 自适应续走数 | 同块12点验收 |
|---|---|---|---|---|
| L46 s20261046 | 418 | 9 | 88 | full L50=418，档差0，通过 |
| L48 s20261048 | 473 | 9 | 88 | full L50=473，档差0，通过 |
| L49 s20261049 | 1078 | 9 | 88 | full L50=1078，档差0，通过 |

当前六份自适应轨迹单独冻结在ignored的`guard_v2_adaptive_frozen.json`；紧凑证据记录逐份规范JSON哈希与快照哈希。
自适应局数证明成本达标；档差≤1逐份以完整同块逐局证据确认。
六份已完成完整比较，并核对冻结轨迹与原始games逐局内容。首轮L49的d943/d1213均为3/8，
中间唯一未测点d1078为4/8；v1只补探网格间隔>2的失败区间，漏掉了此处，不能据首轮宣布任务通过。
现将补探扩大到间隔>1，覆盖仅剩单个未测点的区间，轨迹标记为`coarse4_guard_budget9_v2`。
仍先4等距、老师最多9点、学生最多2点；六份真实CLI复核档差均0、各88局。
首轮失败证据保留在ignored的`human_v1_failure/`。新CLI只新增此前未跑的L49 d1078学生臂8局（0/8），
99份旧games的SHA与mtime均不变，原始对局未重跑。这不是对任意非单调曲线的精度保证。

## 生产扫描（进行中）

当前10/12完成：前三份L46全部复用缓存；L47 s20261047的L50=760、顶档1392，
s20261048的L50=870、顶档1367，s20261049的L50=913、顶档1434，均88局。
L48 s20261048的L50=684、顶档1253，s20261049的L50=617、顶档1131，
s20261050的L50=630、顶档1154，均88局（第三份学生两端0/8、1/8）；
本机种子局d684=8/8而d798=2/8，补探捕获了更早的非单调过线点。
L49 s20261049的L50=顶档1417，老师4/8、学生0/8，共80局（9老师点+1个学生点）；
本份没有全胜老师点，`top_is_full=false`明确标记为非全胜顶档。
完成项已逐局核对原始games并独立冻结轨迹，哈希写入紧凑证据；当前扫描L49 s20261050。

六份v2回测通过后，监督器串行执行12份导入存档的adaptive→登记预览→register→针对性测试。
前三份LLM L46全部复用缓存；新扫描从L47继续，每任务最多2个游戏进程。
登记前保存清单/DESIGN/课程测试/共享账本四类快照，之后核对已有起点格未覆盖、原始games既有内容未改。
阶段和逐步日志在ignored的`artifacts/task91/audit/continuation_status.json`与`production_*.txt`。
预检发现L49本机LLM的`_llm`族尚不存在；已补齐过线后自动复制原任务配置的新族登记路径，
当前没有预先新增生产族或起点格。完整register测试覆盖dry-run零写入、失败局不建族/不领块、
成功登记与四文件重试幂等；运行中扫描未重启，后续register会加载此修复。

## 端到端（task89 包 2 份，独立 dest/账本/临时清单，未来块 90M 起）

- **import**：全包 102 行走完，12 份通关入 `artifacts/e2e/dest/`，扫描清单 288 组、独立账本
  `artifacts/e2e/scan_seed_ledger.json`（90,000,000 起、每份 1000 步长 96），生产清单零改动。
- **run**：第一份3点、第二份2点，各2未来两臂，共20局（`--decisions 1,906,996,1089` 按网格取交集）。
- **register**（临时清单副本 + 临时 DESIGN/测试/账本）：两族写入 `demo_start`
  （base 被人类示范占用 → 正确回退 `_llm`；阶梯 [996,906]（d906 1/2 恰过线）与 [1089]，
  种子块 86M/87M）；`update_manifest` / `update_test_expectations`（期望集合插入）/
  `update_design`（§14.35 自动编号）产物逐项核对通过；登记账本两条 256 块记录正确。
- run 的 `--decisions` 现按每份存档的网格取交集（否则第二份会 KeyError）。

较早v1代码复核：两份输入重新verify通过，20局扫描全部复用缓存、原始对局mtime/SHA不变；
登记使用同一份隔离账本，实际写入上述两族、DESIGN和正确的测试期望集合。
再次register后清单/DESIGN/测试/账本均不变。生产清单、生产DESIGN、生产测试和生产账本均保持原字节；
本次最多1个游戏进程，收尾无残留游戏进程。原始证据在ignored的 `artifacts/task91/audit/`。

当前登记代码复核（包含缺失`_llm`族自动创建）：导入器及原输入包SHA与上述成功证据相同，
保留原导入/引擎verify证明，不重做；当前run实际复用20局缓存，原始对局mtime/SHA均未改，
任何未缓存续走都会使本次复核失败。隔离清单删除首份已有`_llm`族、移除第二份起点格后，
register真实创建新族并登记两份存档，原族的人类起点格完整保留，DESIGN和实际课程测试期望同步。
私有登记账本只读参考canonical全局账本避让，当前分配87M/88M（初次86M/87M是较早隔离结果）。
dry-run零写入；重试清单/DESIGN/课程测试/私有账本四类文件不变，生产四类文件也保持原字节。
本次新增游戏进程0，未干扰生产扫描；当前源码哈希、新回执及原证明哈希链均写入紧凑证据，
原始回执在ignored的`artifacts/task91/audit/e2e_register_current/`。
