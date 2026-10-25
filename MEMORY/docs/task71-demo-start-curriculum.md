# task71：示范起点倒推课程——L33 接手扫描 + qc 的"示范起点"格

> 主控 2026-10-24 写。执行者：Mac CodeBuddy（`cb-demo` 会话，接着 task67b 做）。工作树 `/Users/newbiexvwu/PvZAgent/.claude/worktrees/demo`（分支 `demo`，先 merge origin/pvz-env）。游戏进程最多 4 个；不停 p5d，不碰其他会话。

## 为什么
- task67：L32–L34 换了用户卡组，老师冷启动仍是 0–9%，卡组不是瓶颈。L33/34 输在防空（气球到家），L32 输在开局经济。
- task64：L31 从用户示范第 76 个决策起，老师 8/8；学生要到第 299 个决策才过半。这说明从示范中途接手是可行的课程起点（Salimans & Chen 2018 的倒推课程）。
- 用户已经打通两局 L33：`demo_fog_33_cactus_snow_m200`、`demo_fog_33_cattail_snow_m200`，都是 2.0 倍、verified。

## 要做的
1. **L33 接手扫描**（照 task64 的做法与参数）：
   - 两局各在决策序列上等距取约 12 个点；
   - 每点老师 8 局、学生（β=0）8 局，`--future-seeds` 抽 8 个未来；
   - 报"接手位置 → 老师 / 学生胜率"表，标出老师过 50% 的最早点；
   - 输局归因照 task67（末波、破防僵尸，特别看气球）。
   - 结论写 `MEMORY/docs/demo-start-states.md` 新一节。
2. **qc 的示范起点格**（代码，主控合并部署）：
   - `curriculum_queue` 支持一种新 cell：由（示范存档、起点决策号）定义开局，未来随机流用 RESEED，每组换种子；
   - 阶梯沿"起点决策号"往前推，用现有的 0.85 升档 / 0.5 降档规则，升档 = 起点往前挪一档。档位用第 1 步扫描得到的点，最晚一档取老师已稳定会赢的点；
   - 采集、打标签、打包、门禁全走现有路径；帧里记下起点（存档 SHA + 决策号），打包按普通老师标签处理，起点之前的示范动作照 `label_source: human` 已有的导出办法进训练（不重复导出）；
   - 现有 cell 与清单逐字节不变（照 task68 的验证方式）；新族登记 DESIGN 与 `configs/human_demo_tasks.json`（AGENTS §8）。
   - 先用 L31 `demo_fog_31_snow_m200` 和 L33 两局做清单；L31 的 1.0 倍冷启动格已在 qc 里，示范起点格作为 2.0 倍的补充。
3. **测试**：起点恢复的确定性（同存档同决策号同种子 → 同一局）、阶梯升降、旧 cell 不变。离线 dry run 给出前两轮的分配。
4. 提交并 `git push origin demo`，结果写本文件。

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；`PVZ_NO_POPUP=1`。用户原始存档只读。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 不进 git；不写字面密钥；不手写打法规则或公式。
- 每完成一小步就提交、push，在"结果"记一行；疑问写"问题"，按最佳判断继续。

## 结果

- **第 1 步（L33 接手扫描）**：两局各 12 点 × 8 未来 × 2 臂 = 192 局，另加 L31 `demo_fog_31_snow_m200` 同形扫描 192 局（任务书没要求，但它的档位不能照抄 task64 的 fume 1.0 倍点）。老师 50% 线：L33 cactus **N=458（tick 22428，第 9 波）**、L33 cattail **N=316（tick 15365，第 6 波）**、L31 snow **N=142（tick 7052，第 3 波）**；8/8 线分别是 534 / 473 / 213。L33 两局的破防主力是**气球**（老师 47 局输局里气球到家 42 次），L31 一次气球都没有（普通/路障磨穿）。正文见 [demo-start-states.md](demo-start-states.md) §6。
- **第 2 步（qc 的示范起点格）**：`curriculum_queue` 新增一类格——`demo_start`（清单字段：`archive` / `decisions` 降序阶梯 / `index` / `future_seed_start`+`future_seed_count`）。
  档位值 = 起点决策号，升降档仍是 0.85 / 0.5，升档 = 起点往前挪一档；`command_for` 在起点格上发
  `--demo-archive --demo-decision --future-seeds`（`--seed-offset` 固定 0，每组从种子块里取下一段）。
  帧的 `meta.json` 记 `demo_archive` + `demo_decision` + `demo_archive_sha256`；标签形如 `c1demofog33cactussnowm200s839`；
  打包/打标签/门禁全走现有路径，起点之前的示范动作走 `human_demo.py export`（human，一次导出）。
  三个 cell 登记在 `configs/human_demo_tasks.json`（阶梯 777..142 / 839..458 / 867..316），种子块 44.0M / 44.1M / 44.2M 各 256 个，与现有全部段不相交；DESIGN §14.26。
  旧格不变：`make_multiplier_tasks.py` 未改，同批源清单重建的 420 任务清单与 task68 **逐字节相同**（sha256 `7b72c36d…`）；qc 的 39 个面板格加上三个起点格后逐字段相同（总 39 → 42）。
- **第 3 步（测试与 dry run）**：`python/test_curriculum_queue.py` 54 项通过（新增 8 项：阶梯升降、种子块切分与耗尽、命令确定性、清单校验/存档缺失、旧格逐字节不变、幂等与退休、登记清单自检）。
  起点确定性实测：同存档 + 同决策号 + 同未来种子 → 动作/状态/胜负/决策数逐字节相同，换未来种子则不同（`artifacts/task71/determinism/result.json`）。
  离线 dry run（本地用合成源重建的 420 任务面板，生产代码）：42 个格，每轮 30 组 × 8 局 = 240 局；第一轮就排到三个起点格
  （`s777` / `s839` / `s867`，命令带 `--demo-decision` 与 `--future-seeds 44xx0000..44xx0007`），第二轮覆盖剩余 12 格并轮转重复格。
  全量 `python/`：**540 passed, 56 skipped, 2 failed**，两个失败是 `test_plants_eaten_free`（task65 的引擎字段，本机二进制 `a13f6f59` 早于那次改动；合并前后都失败，与本次无关，见问题 1）。
- **第 4 步**：已 `git push origin demo`（82b38123 → 0c64393b → 本次）。

### 部署清单（主控）

1. **存档必须先到位**：三个 `artifacts/human_demo/raw/<seed>/<task>/` 目录（每个 6.5–7.3 MB，frames.jsonl + ticks.jsonl + meta.json）
   不在 git 里。qc 启动时 `load_demo_start_cells` 会校验存档存在且 `verified`，缺了直接报错退出（不会带着空格跑一轮）。
2. `teacher_rates.json` 追加三个 base（用最晚一档的实测，8 局/格）：
   ```json
   {"bases": {"demo_fog_31_snow_m200": {"wins": 8, "games": 8},
              "demo_fog_33_cactus_snow_m200": {"wins": 8, "games": 8},
              "demo_fog_33_cattail_snow_m200": {"wins": 8, "games": 8}}}
   ```
3. `--demo-start-manifest` 缺省就是 `configs/human_demo_tasks.json`，不用加参数；`curriculum_state.json` 里会多出
   `start cell added: demo_fog_*_m200 (N rungs)` 三行。
4. 回滚：删掉清单里的 `demo_start`（或把 `--demo-start-manifest` 指向一份没有它的清单），三格自动进 `retired_cells`，面板格不受影响。

## 问题

1. **本机二进制早于 task65 的引擎改动**：`test_plants_eaten_free` 的两项在合并 origin/pvz-env 前后都失败
   （`plants_eaten_free` 字段不在引擎回复里）。重建引擎会换掉 `a13f6f59`，而三个示范存档记录的
   `executable_sha256` 正是它；换引擎属于 task70（台式机 redeploy）的范围，本次不动，也不改这两项测试。
2. **帧不是逐字节相同**：同输入两局只有 `verification.seconds`（墙钟）不同，训练行不读它。
   要做到"帧文件逐字节相同"得把这个诊断字段从帧里去掉——那会改所有历史帧的格式，不在本次范围。
3. **human 导出没有"只导前缀"的选项**：`human_demo.py export` 导出整局示范（`label_source: human`），
   起点之后的示范动作也会进训练。它们与老师标签是不同对局（老师那局换了未来种子），不是同一条状态被标两次；
   但同一段局面的示范与老师标签会共存，配比由打包的 `source_weight` 决定。需要"只导到起点"的话，得给
   export 加 `--decision` 截断——本次没加，避免改既有导出语义。
4. **8 个未来种子仍共享前史**（同 task64 问题 1）：同一格 8 局只重抽未开始的波次。档位判据用的是 ≥50% 这种粗线，
   不受影响；单格 7/8 与 8/8 的差不要当趋势读。
5. **L31 的档位来自额外扫描**：任务书只要求 L33 两局。没有 L31 snow_m200 的 12 点扫描，档位只能照抄
   task64 的 fume 1.0 倍点，而 §6.4 的表说明那会错 5–6 个波次。
6. **起点格在 qc 里没有按档位区分的老师胜率**：`teacher_rates` 按 base 记，不按档位记；升档后老师的实际胜率
   会下降（L33 cactus 的 458 档只有 7/8），而 `teacher_rates` 仍写着 8/8。这只影响分配排序（起点格本来就在最前），
   不影响升降档（那用学生窗口）。要按档位区分，得把键扩成 `base@decision`——本次没做。

## 主控追加（2026-10-25，task71b）
用户打通 L32：`demo_fog_32_snow_m200`（磁力菇换寒冰射手，2.0 倍；30 波、1309 个动作、被吃 28 株其中零费 26、0 小推车，verified）。存档在主检出 `/Users/newbiexvwu/PvZAgent/artifacts/human_demo/raw/seed_20261032/demo_fog_32_snow_m200`（只读，先拷进 demo 工作树同路径再用）。
1. 照第 1 步做 12 点接手扫描（老师/学生各 8 局，`--future-seeds`），报老师过 50% 与 8/8 的最早点、输局归因（用户说后期是在雾里布坚果防线、前面垫零费小喷菇）。
2. 在 `configs/human_demo_tasks.json` 给它加 `demo_start`（阶梯取扫描结果，种子块与现有全部段不相交），照第 2 步的测试与 DESIGN 登记；提交并 push origin demo。部署由主控做。
- 追加（同日）：用户又打通 L34 `demo_fog_34_cactus_snow_m200`（30 波、1198 动作、被吃 29 其中零费 28、0 小推车，verified；打法：开局看到有气球就在第三列种仙人掌，再上寒冰射手和大喷菇）。存档 `/Users/newbiexvwu/PvZAgent/artifacts/human_demo/raw/seed_20261034/demo_fog_34_cactus_snow_m200`。L32 做完后对它做同样的扫描与起点格。

## 结果（task71b，2026-10-25）

- **第 1 步（L32 接手扫描）**：12 点 × 8 未来 × 2 臂 = 192 局。老师 50% 线 **N=225**（4/8）、
  8/8 线 **N=560**；学生 50% 线 N=672、8/8 线 N=895。破防**无气球**（老师 26 输局：普通 30、
  路障 19、橄榄 3），早期输局在第 7–13 波——与 task67 的 L32 开局经济归因同型。
  N=336 的 3/8 低于 N=225（8 未来共享前史的噪声带，不入阶梯）。正文见
  [demo-start-states.md](demo-start-states.md) §6.6。
- **第 2 步（登记）**：`configs/human_demo_tasks.json` 给 `demo_fog_32_snow_m200` 加 `demo_start`
  （阶梯 `1231..225` 共 9 档降序，N=336 低于 50% 不入；种子块 44,300,000–44,300,255，256 个，
  与现有全部段不相交）。`test_registered_manifest_start_cells` 期望集合同步加它，
  `python/test_curriculum_queue.py` 54 项通过。DESIGN §14.26 加 task71b 条目。
- **提交**：`git push origin demo`（本步）。
- 追加（同日）：用户打通屋顶 L42 `demo_roof_42_melon_m200`（30 波、1513 动作、被吃 42、零费 0，verified，存档 `/Users/newbiexvwu/PvZAgent/artifacts/human_demo/raw/seed_20261042/demo_roof_42_melon_m200`）。照 L32/L34 做接手扫描与起点格。屋顶老师冷启动全败，最晚档要取老师稳定 8/8 的点；注意 `fog_observation` 不涉及屋顶。
- 追加（同日）：用户打通雾夜 L39 `demo_fog_39_cactus_snow_m200`（30 波、1609 动作、被吃 19 其中零费 18、0 小推车，verified，存档 `/Users/newbiexvwu/PvZAgent/artifacts/human_demo/raw/seed_20261039/demo_fog_39_cactus_snow_m200`）。L39 老师冷启动各卡组 0–1/16（气球破防，task74b），这是该关唯一的成功轨迹。L42 做完后照同样流程扫描与起点格；起点格族带 `fog_observation`（与 B6 的 `_hv2` 口径一致）。
- 追加（同日）：用户打通屋顶 L43 `demo_roof_43_melon_m200`（30 波、1129 动作、仅被吃 2 株、0 小推车，verified，存档 `/Users/newbiexvwu/PvZAgent/artifacts/human_demo/raw/seed_20261043/demo_roof_43_melon_m200`）。L42 之后照同样流程做扫描与起点格。主控定：屋顶示范以西瓜套为主，玉米套不再录（群怪下西瓜溅射远强于玉米的单体控制）。
- 更正（同日）：忧郁菇+大蒜套照录（用户：无衰减穿透对群怪比西瓜更强），L42/43/44 各一局，录完同样做扫描与起点格。只有玉米套不录。
- 追加（同日）：用户打通屋顶 L44 `demo_roof_44_melon_m200`（40 波、1438 动作、被吃 5 株、0 小推车，verified，存档 `/Users/newbiexvwu/PvZAgent/artifacts/human_demo/raw/seed_20261044/demo_roof_44_melon_m200`）。排在 L43 之后做扫描与起点格。
- 追加（同日）：用户打通 L42 忧郁菇+西瓜混合 `demo_roof_42_gloommelon_m200`（30 波、1357 动作、被吃 24 株、0 小推车，verified，存档 `/Users/newbiexvwu/PvZAgent/artifacts/human_demo/raw/seed_20261042/demo_roof_42_gloommelon_m200`；用户说后期被蹦极偷得多）。排在 L44 之后，同样做接手扫描。是否进课程只看扫描：老师接手过 50% 的点才入阶梯，过不了就不登记 `demo_start`，在“结果”里写清老师在哪一段接不住。
- 追加（同日）：用户打通屋顶 L49 `demo_roof_49_doom_m200`（40 波、1577 动作、被吃 17、0 小推车，verified，存档 `/Users/newbiexvwu/PvZAgent/artifacts/human_demo/raw/seed_20261049/demo_roof_49_doom_m200`；用户说毁灭菇基本能清掉整个大波，没带保护伞，西瓜一直被蹦极偷走）。L49 是屋顶最难的一关，**排到 L42 西瓜之后、L39 之前**。
