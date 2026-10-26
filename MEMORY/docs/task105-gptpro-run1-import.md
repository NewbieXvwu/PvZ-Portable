# task105：导入 GPT Pro 第一轮通关包

## 输入

- 台式机 `~/llm_runs/gptpro_438cb30a_0446.zip`
  - SHA256 前缀 `676fb355128be674`，61 MB；
  - run_id `438cb30ac4e143368a5cb47a8569066b`。
- GPT Pro 会话跑了约 2 小时后卡死，这是它最后上传到 Drive 的副本。用户会开新会话续跑，同一个 run_id 之后还会交来更新的 zip。
- 内容：
  - 191 份存档；
  - progress.json：A 4/4、B 8/8、C 3/8、D 0/8、E 0/3；
  - 另有约 900 个策略脚本。

## 要做的

1. 按 task89 / task91 定好的链路导入：`scripts/import_llm_run.py` → adaptive → register。
   - 每份存档都在 deploy14 引擎上 verify，不过的标作废并写明原因；
   - 按存档 SHA 去重，后续交来的 zip 要能增量导入、不重复登记。
2. 在本任务书“结果”节写清楚：
   - 每个任务各有几个通关种子、几个稳健种子，被拒的有几个、为什么；
   - 新登记了哪些起点格或接手扫描结果；哪些已经能进 qc，哪些还在等部署。
3. 策略脚本只作参考归档，不进训练，也不进 git。存档与游戏资源同样不进 git、不进 HF。
4. 每个任务最多 2 个游戏进程，qc 优先。分支 `llmimport`，每完成一小步就提交、push。
5. 要进 qc 课程的部分，由主控在组边界部署。写好部署脚本，但不要执行。

## 结果

- 导入阶段已完成：191/191在deploy14上verify通过，拒绝0份（无拒绝原因）。
  191个不同内容SHA，涉及19个任务、87个不同(任务,种子)；49份full、142份half。
  稳健存档按同任务同种子去重后为48个稳健种子，依据随包第10波检查点k=8且≥5胜的记录；
  本机老师接手胜率由后续adaptive另行测量，不混作同一指标。
  真实重导入191/191复用回执，新增族0、复用扫描种子块；191份存档各自保留独立版本族，避免相互覆盖。
  首份adaptive已启动（workers=2），尚未完成接手扫描/起点格登记，暂未向qc部署。
  导入阶段紧凑证据见[task105-import-evidence.json](task105-import-evidence.json)，其中`audit_passed=false`
  明确表示全链路审计未完成。

| 层 | 任务 | 通关种子 | 稳健种子 | 拒绝份数 |
|---|---|---|---|---|
| A | demo_roof_46_doom_m200 | 5 | 3 | 0 |
| A | demo_roof_47_doom_m200 | 5 | 3 | 0 |
| A | demo_roof_48_doom_m200 | 5 | 3 | 0 |
| A | demo_roof_49_doom_m200 | 5 | 3 | 0 |
| B | demo_roof_41_doom_m200 | 5 | 3 | 0 |
| B | demo_roof_42_melon_m200 | 5 | 3 | 0 |
| B | demo_roof_42_gloom_m200 | 5 | 3 | 0 |
| B | demo_roof_42_gloommelon_m200 | 5 | 3 | 0 |
| B | demo_roof_43_melon_m200 | 5 | 3 | 0 |
| B | demo_roof_43_gloom_m200 | 5 | 3 | 0 |
| B | demo_roof_44_melon_m200 | 5 | 3 | 0 |
| B | demo_roof_44_gloom_m200 | 5 | 3 | 0 |
| C | demo_roof_46_doom_m300 | 5 | 3 | 0 |
| C | demo_roof_46_doom_m400 | 4 | 2 | 0 |
| C | demo_roof_47_doom_m300 | 5 | 3 | 0 |
| C | demo_roof_47_doom_m400 | 2 | 0 | 0 |
| C | demo_roof_48_doom_m300 | 6 | 3 | 0 |
| C | demo_roof_48_doom_m400 | 2 | 1 | 0 |
| C | demo_roof_49_doom_m300 | 3 | 0 | 0 |
| C | demo_roof_49_doom_m400 | 0 | 0 | 0 |
| D | fog79_l36_snow_magnet_v2_m200 | 0 | 0 | 0 |
| D | fog79_l36_snow_split_v2_m200 | 0 | 0 | 0 |
| D | fog79_l37_snow_magnet_v2_m200 | 0 | 0 | 0 |
| D | fog79_l37_snow_split_v2_m200 | 0 | 0 | 0 |
| D | fog79_l38_snow_magnet_v2_m200 | 0 | 0 | 0 |
| D | fog79_l38_snow_tall_v2_m200 | 0 | 0 | 0 |
| D | fog79_l39_snow_cactus_v2_m200 | 0 | 0 | 0 |
| D | fog79_l39_snow_blover_v2_m200 | 0 | 0 | 0 |
| E | demo_pool_30_orig_m100 | 0 | 0 | 0 |
| E | demo_storm_40_orig_m100 | 0 | 0 | 0 |
| E | demo_boss_50_orig_m100 | 0 | 0 | 0 |

零份任务表示本包未交付该任务的存档，并非校验拒绝。以上计数按本包所有通过版本的种子并集计算。

- 首份扫描的阶段核对：`demo_roof_48_doom_m200_s81002031_a7adc1574be88`已完的
  5个老师点共40局，逐条对照原始games、7个存档/模型/引擎输入指纹通过；冻结快照SHA已写入阶段证据。
  已核对d1=0/8、d337=0/8、d588=3/8、d756=8/8、d924=8/8。
  这5点只是已完成部分，首份完整adaptive与最终L50尚未验收，没有据此预写起点格或部署。

- 2026-10-26：完整读取原任务书并核实 zip SHA256 为
  `676fb355128be6742d5023f365526396612755d5da6b362144faeaedf5cd0ee7`。
  实际包含191份存档、87个不同的(任务,种子)，348份`.py`和260份`.pyc`；策略只留本机原zip作参考。
  deploy14引擎为cq的`build5/pvz-portable`，SHA256
  `76ff36dc026f67641aefccd5f30f6cdae65eee14ee197dc24497413123a7874a`，与deploy14脚本一致。
- 补充`import_llm_run.py --content-addressed`：数值存档规范内容SHA（只排除验证器写入的两个字段）
  寻址，同种子不同版本分别保留并用SHA后缀建立独立族；逐档原子保存校验回执、输入/引擎/资源/验证器指纹。
  同档重导入复用回执与种子块，不覆盖旧起点格；引擎或验证依赖变化须重新verify。
  初始70项导入器/扫描器测试通过；后续扩到79项（1跳过）。逐档verify现已全部完成。

## 本机续跑入口

- 第二小步：新增`task105_pipeline.py`续跑器（等待原校验进程后重导入检查幂等，再逐档adaptive→register→测试→审计），
  `audit_task105.py`独立逐局/指纹/种子避让验收器，以及`deploy_task105.py`部署脚本。
  部署默认只读，`--apply`要求qc已排空，合并新增项而不覆盖生产既有任务，并保留回滚副本；**未执行部署**。
  当前79项针对性测试通过（1项历史Mac存档缺失跳过；最终审计另用真实课程装载器验证所有新增格）。
  DESIGN自动登记压成两张表，避免逐存档重复建节；课程测试接受本包的`llm_agent`来源。
  以上工具的真实全量扫描/最终审计仍待接手扫描完成，不作为已验收结果。

- 后续验收补充：全量扫描结束后，审计器会实际重试import和register，要求所有校验回执/扫描种子块复用，
  清单、DESIGN、课程测试、种子账本四类文件字节不变，重复登记0；当前尚未执行最终重试验收。
- 增量导入补充：新SHA存档在临时目录复制完整并核验SHA后原子改名，复制中断不会留下正式半份存档。
  模拟中断后重试测试通过；当前相关测试80项通过（1项历史Mac存档缺失跳过）。
- 最终审计另按AST核对课程测试的起点格期望集合，要求等于历史集合加本次所有新增格，
  配合真实loader验证，避免历史存档缺失导致的测试跳过掩盖本次漏登；25个历史期望项已核对保留。

```bash
P=/home/newbiexvwu/PvZ-search/.claude/worktrees/llmimport
PVZ_NO_POPUP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 15 /home/newbiexvwu/.venvs/ml/bin/python3 "$P/scripts/import_llm_run.py" /home/newbiexvwu/llm_runs/gptpro_438cb30a_0446.zip --content-addressed --dest "$P/artifacts/task105/import" --executable /home/newbiexvwu/PvZ-search/.claude/worktrees/cq/build5/pvz-portable --report "$P/artifacts/task105/import_report.md" --scan-out "$P/artifacts/task105/scan_jobs.json"
```

逐档进度与回执在ignored的`artifacts/task105/import/`；运行日志在`~/pvz-logs/task105-import.out`。
尚未向生产qc写入任何任务105课程配置，也未上传HF。

全链路续跑（同样只改本工作树，原进程活着时不要另起第二份）：

```bash
PVZ_NO_POPUP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 15 /home/newbiexvwu/.venvs/ml/bin/python3 "$P/scripts/task105_pipeline.py" /home/newbiexvwu/llm_runs/gptpro_438cb30a_0446.zip
```

状态在`artifacts/task105/pipeline_status.json`；每份完成后写`completed_scans.json`，
扫描/登记与原始逐局数据均在同目录下。每次adaptive最多2个游戏进程，原逐档verify与扫描串行。
验收通过后生成`deployment_plan.json`，主控可先运行`$P/scripts/deploy_task105.py`预览，
在组边界排空qc后再用同一脚本的`--apply`；脚本不重启qc。
