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

- 2026-10-26：完整读取原任务书并核实 zip SHA256 为
  `676fb355128be6742d5023f365526396612755d5da6b362144faeaedf5cd0ee7`。
  实际包含191份存档、87个不同的(任务,种子)，348份`.py`和260份`.pyc`；策略只留本机原zip作参考。
  deploy14引擎为cq的`build5/pvz-portable`，SHA256
  `76ff36dc026f67641aefccd5f30f6cdae65eee14ee197dc24497413123a7874a`，与deploy14脚本一致。
- 补充`import_llm_run.py --content-addressed`：数值存档规范内容SHA（只排除验证器写入的两个字段）
  寻址，同种子不同版本分别保留并用SHA后缀建立独立族；逐档原子保存校验回执、输入/引擎/资源/验证器指纹。
  同档重导入复用回执与种子块，不覆盖旧起点格；引擎或验证依赖变化须重新verify。
  70项导入器/扫描器测试通过。真实逐档verify已串行启动，尚未完成；adaptive/register/部署脚本待后续步骤。

## 本机续跑入口

```bash
P=/home/newbiexvwu/PvZ-search/.claude/worktrees/llmimport
PVZ_NO_POPUP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 15 /home/newbiexvwu/.venvs/ml/bin/python3 "$P/scripts/import_llm_run.py" /home/newbiexvwu/llm_runs/gptpro_438cb30a_0446.zip --content-addressed --dest "$P/artifacts/task105/import" --executable /home/newbiexvwu/PvZ-search/.claude/worktrees/cq/build5/pvz-portable --report "$P/artifacts/task105/import_report.md" --scan-out "$P/artifacts/task105/scan_jobs.json"
```

逐档进度与回执在ignored的`artifacts/task105/import/`；运行日志在`~/pvz-logs/task105-import.out`。
尚未向生产qc写入任何任务105课程配置，也未上传HF。
