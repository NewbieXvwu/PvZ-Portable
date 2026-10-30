# task175｜植物大战僵尸搜索教师—模仿学生系统：Q1–Q7 一手文献核查与路线图

**研究截止日期：2026-10-30**　｜　**证据性质：论文/官方技术资料 + 项目既有实验；未在本报告重新运行项目实验**

> **阅读约定**：`[Rxx]` 指文末可点击的一手来源；`[P1–P5]` 指压缩包中的原项目记录及章节。**“实证”只指所述论文在其原实验场景或项目已有实验中的结果；“迁移建议”并不自动构成 PvZ 的实证。** 离线 `cell_acc`、完整动作准确率、对局胜率、教师胜率和根节点墙钟是不同指标，不可混写。文中“置信度”指对**本项目决策**的信心，而非对论文真实性的评价。

## 1. 一页结论

| 问题 | 可执行结论 | 最关键的证据与限制 | 对本项目决策的置信度 |
|---|---|---|---|
| **Q1：实体注意力 vs CNN** | **执行已排队的 E4 等预算、等输入版本、等参数对照，不预言 10M–30M 交叉点。** v3 只在根节点试，续走继续 CNN。 | AlphaStar 消融混入动作头/连接方式等变量，不能推出纯结构交叉点；NetHack 缩放律说明 BC 可随算力改善，但不是实体 vs CNN；本地 152 局原型小且早期有池化错误。[R01–R04][P3 §11.8, §11.13][P4] | **中**（实验设计）；**低**（结构必胜） |
| **Q2：MoE / FiLM** | **不重新开启五地形硬专家/MoE 主线**；保留已入队 `r42_terrain_s` 作一次受控检验。条件化只有在可证明的负迁移和额外 OOD 收益下再讨论。 | Soft MoE 的获益来自价值型 RL，并未验证陌生卡组 BC；本地 E3 统一曝光后专家优势明显缩小，陌生卡组远逊联合网。[R06–R11][P3 §11.9] | **高**（暂缓重 MoE） |
| **Q3：平均 vs 最差地形** | **采用有约束的多指标模型选择**：平均胜率作主目标，分地形非劣化/保底；对地板效应使用存活波数。不要机械改成 `min(五地形胜率)`。 | group DRO 在其他领域可改最差组，但易受小组噪声与正则化影响；本地 roof 双方 0/128、day 离线与真打方向相反；门禁 v6 已实施。[R13–R17][P3 §11.9, §11.12–13] | **中高** |
| **Q4：分歧/遗憾驱动采集** | **先做教师质量分层 + 低成本影子日志，再试“模型—搜索差值”优先查询**，同时保留均匀探索基线。 | PLR、SafeDAgger、RND-DAgger、KataGo 提供不同形式的“对有学习价值状态多投入资源”；不能把 TD-error / PAIRED regret 简化成老师学生动作不同。[R18–R28][P3 §11.12] | **中** |
| **Q5：陌生卡组泛化** | **优先提高教师在未见组合中的可胜性与采集覆盖，而不是只增加网络容量。** 从相同卡牌的“新组合”建立留出评测，确保搜索老师也能赢。 | Cardsformer 在 30 张未训练卡的测试条件下仍有 47.5% 对战胜率，但与本项目的已知植物新组合不同；本地老师在 54 套陌生卡组仅 14.6%。[R05,R29,R30][P3 §12.5] | **高**（先解决教师瓶颈）；**低**（迁移数值） |
| **Q6：旧调研可信度** | **更正 AlphaZero 50 万局归属、温度采样方向、CARE 缩写；区分论文、代码配置、本地结论。** | 原始 AlphaGo 2016 的 24.2%/2µs 已逐字核实；ELF OpenGo 附表把 500,000 写在 AlphaGo Zero 与 ELF 列，AlphaZero 列为 `?`。[R16,R31–R35] | **高**（核实内容） |
| **Q7：长期新技术** | **现在：可靠评估、有限搜索预算分配、主动查询、端到端教师质量审计；半年内：轻量离线 RL 与 Gumbel 搜索；长期：世界模型/无示范自学。** | 单张消费 GPU、约 26GB 主存、CPU 高频 rollout；直接复现 DreamerV3/MuZero 大规模工作不符合性价比。[R17,R24,R36–R45][P4] | **中**（路线图优先级） |

### 最新内部事实基线，避免重复已做实验

项目任务书的“约 2300 万行、60% 历史、82% 默认卡组”等是早期口径；**不应以此覆盖后续冻结统计**：r42 包共有 **23,631,235 行**，排除评估留出后 E13 实际训练 **20,873,919 行**；按**合并卡组签名**口径其头部一套占 **57.6%**，不能与原先“按（关卡，卡组）”45.1% 直接比较。E13 等 4500 步时，陌生卡组 `cell_acc` 为现状 **0.2730**、温度 **0.2209**、卡组封顶 **0.1509**；封顶有效批量约 400，故**结论仅限此等算力实现下的成本/收益**。E1 的历史降权在部分 7.11M 语料上也未改善陌生卡组。E8 的旧/新教师标签全体 top-1 一致 **0.614**、有搜索时 **0.440**、无搜索时 **0.941**，但等价动作歧义 **0.05%**，因此不能把噪声简单解释为“同义动作”。[P3 §11.10–11.12]

**已有产线动作**：deploy17/18 **已将示范起点从 75% 降至 25%、四类采集均为 25%、随机卡组池增至 54（排除 26 套评估留出）、obs v3 随帧写入、门禁 v6 加入最少 4 个评测单元及屋顶存活波数**；r52 **09:03 晋升、门禁 1,489 秒**，B 的历史权重 1.0 本轮未胜。E4 的 token 预取已测得 **95,850 行/秒**，结构候选已入队；deploy19 的整局原生路径仍待 **45 种子 A/B** 验证。**这里提出的是验收/增量实验，不是重复部署。** [P3 §11.8, §11.12–13]

## 2. Q1｜1000 万–3000 万行为克隆行：实体/Transformer 何时反超 CNN？

**查到的最接近研究**：

| 一手来源（实验设置） | 明确可追的数字/发现 | 与本项目相似度 | 对 PvZ 的解释 |
|---|---|---|---|
| **AlphaStar**，Vinyals 等，2019，*Nature*，图 3 消融与扩展方法。[R01] | 主网络 **139M** 参数（其中 **55M** 用于推理），训练使用海量 SC2 人类重放并继续大规模强化学习；图 3f 比较**累积加入多组件**后的 Elite Bot 胜率。 | **中低**：有实体注意力和复合动作；没有 18 万参数/单 GPU/纯 CNN 配平实验。 | **不应**把图 3 中 Transformer、pointer、连接方式的累积增益解释为“10–30M 标签足以让注意力超过 CNN”。 |
| **NetHack/Atari 模仿缩放律**，Tuyls 等，TMLR 2024，§实验。[R02] | 训练损失与回报随算力/模型、数据规模呈可拟合趋势；发表版最佳 NetHack 离线结果约为既有水平 **1.7 倍**（早期预印本摘要为 1.5 倍）。 | **中**：都是游戏 BC；但任务、输入/动作编码与老师来源不同。 | 同时画“行数 × 参数量 × 更新步数”的**学习曲线**，不要仅比较最终检查点。 |
| **AlphaStar Unplugged**，Mathieu 等，2023，离线 SC2 强化学习。[R03] | 展示大规模离线序列学习/离线 RL 可行；**未提供与本项目同数据同预算的 CNN–实体 Transformer 胜负阈值**。 | **中低**：SC2、离线数据；模型/算力巨大。 | 可参考使用离线固定语料评价，但不能外推结构临界点。 |
| **Gato**，Reed 等，2022，多模态多任务序列 Transformer。[R04] | **1.2B** 参数，跨 **604** 类任务/环境；不是同数据量 BC 与 CNN 对照。 | **低**。 | 证明统一序列表征可能，不是当前规模的部署证据。 |
| **Procgen**，Cobbe 等，ICML 2020，16 个程序生成游戏。[R05] | 研究中**更大模型改善样本效率与泛化**；训练机制为 RL，未比较实体 vs CNN。 | **中低**：游戏与跨配置泛化相似，学习范式不同。 | 反证“容量增加必然损害 OOD”的普遍说法。 |
| **本地 E4 前史**。[P3 §11.8–11.9][P4] | 152 局原型在修复卡槽均值池化后完整动作准确率 **91.0% 对等参 CNN 的 91.3%**；full r42 部署候选 v3 根节点 **4.3ms（128/2）/14ms（192/4）**，主控记录中 task161 的端到端局/小时约 **−4%（已测）**，但生产版本及网络成本比例改变后仍须复测。 | **高**：本项目，但小样本单种子。 | 原型并未否证全量 v3，也没有证明其必胜；CPU 成本要区分根节点与 rollout。 |

**回答交叉点**：截至此次核对，**没有可严谨引用的“在 1000 万、2000 万或 3000 万教师行上实体注意力由劣转优”的数值**。更严格地说，**相同游戏、同一个观察版本、同一动作头、同一训练行与优化步数、同参数量、至少多种子**的公开比较未找到。ViT/DeiT 图像分类曲线不能代替游戏搜索老师 BC 曲线。[R01–R05]

**实验落实**：对已入队 `r42_cnn64_s`、`r42_cnn128_s`、`r42_resnet6_s`、`r42_v3_small_s`、`r42_v3_medium_s`、`r42_cnn_entity_s` 与条件化臂，报告**固定样本预算、固定有效梯度步数、相同合法动作 mask 与三头 loss、2–3 个随机种子**的纵向曲线。新 obs v3 若 CNN 也能读取，应增加“**同输入版本 CNN**”对照以识别**信息新增**与**架构增益**。主要终点为**真实未见卡组对局胜率、难地形存活/胜率、根节点耗时**；`cell_acc` 只作诊断，不作最终裁决。若 E4 占用显卡预算过大，优先完成已排队的模型，不重新开一轮重复对照。**置信度：反对预设交叉点为高；v3 会胜过 CNN 为低。**

## 3. Q2｜小规模 MoE、FiLM、任务条件化与 OOD

| 一手文献 | 模型与实际验证场景 | 本项目相似度 | 可以/不可以推出什么 |
|---|---|---|---|
| Obando Ceron 等 2024，**Soft MoE**。[R06] | ICML：在**价值型深度 RL** 中，Soft MoE 使增大参数规模更有效；涉及 DQN/Rainbow 系。 | **中低**：游戏与预算受限相关，但并非搜索老师 BC 或陌生卡组。 | 可探索**共享软路由**；**不能**声称分地形专家能改未知卡组。 |
| Yang 等 2020，**Soft Modularization**。[R07] | NeurIPS：连续控制多任务 RL，学习任务条件化的软模块连接。 | **低**：已知机器人任务与 PvZ 组合外推不同。 | 若存在可验证任务间负迁移，可做小条件化臂。 |
| Hendawy 等 2023/2024，**MOORE**。[R08] | 正交专家多任务 RL，MiniGrid/MetaWorld 任务，强调表征模块化。 | **中低**：部分网格任务相近，数据/动作/搜索不同。 | 不证明按五地形分开的 hard expert 会泛化新卡组。 |
| Sodhani、Zhang、Pineau 2021，**CARE**。[R09] | ICML：**任务元数据上下文 + k 个表征编码器**，用于 Meta-World 多任务连续控制。论文实际展开缩写为 *Contextual Attention-based REpresentation learning*。 | **中低**：有多任务已知上下文，但不测试 PvZ 未见组合。 | 可在**共享 CNN** 上用关卡模式/可见卡组做轻量条件化；不可认为上下文可代替数据。 |
| Pérez 等 2018，**FiLM**。[R10] | AAAI：以输入条件产生通道尺度/偏移，在视觉问答/推理中成功；不是 PvZ 策略 RL。 | **低**：机制简单可移植，效果证据远。 | 如果同输入 CNN 存在明确条件依赖，则可低成本加条件仿射变换，先测 OOD。 |
| Chi 等 2022，**MoE 表征坍缩**；Geirhos 等 2020，**捷径学习**。[R11,R12] | 前者研究稀疏 MoE 表征退化，后者讨论模型抓取伪相关特征。 | **中低**：机制警示，不是本项目因果诊断。 | 需监控路由负载和组间性能；不能断言本地模型已出现“路由塌缩”或“容量必伤泛化”。 |

**本地更有判别力的结果**：[P3 §11.9] 五地形 E3 起初 4.35M 行、每个专家约 1M 行，专家 4500 步对联合网 4500 步，导致专家**每行更高曝光**。把专家降到相同曝光的 **1086 步**后，day 仅约 **+1.6pp**、night/pool/roof 近乎持平，fog 反向约 **−6.2pp**；把联合 CNN64 增至 **19000 步**，屋顶离线 `cell_acc` **0.4999** 超过屋顶专家 **0.4738**。随机卡组探针联合 64 **0.2699**、联合 128 **0.3150**，五个单地形专家 **0.0656–0.2252**。day 真打联合与专家 **104/144 vs 97/144，McNemar p=0.39**，不可称显著；roof **0/128 vs 0/128** 是地板效应。以前另一轮全量加宽 128 的陌生卡组胜率由 **60.0%→53.8%**，夜间 **51.4%→42.6%**；两组数据集和指标不同，不能拼成单一“128 一定更差”的结论。[P2, P3]

**决策**：不以“专家原始每地形 +1.8–5.8pp”开 MoE 主线。维护已经排队的 M1 地形条件化作确认，不增加独立 5 专家部署、训练、门禁成本。只有共享网在固定种子真打存在可重复的负迁移（且新条件化**在未见卡组真打**改善、CPU 根节点成本可控）才重开。MoE 学术普遍性：**置信度低**；当前暂不投入重工程：**置信度高**。

## 4. Q3｜平均胜率、最差组与晋升协议

| 一手来源 | 论文实际结论及数字 | 相似度与应用界限 |
|---|---|---|
| Sagawa 等，ICLR 2020，**group DRO**。[R13] | 最差组泛化可提高；在人类语言推断/视觉任务配合强正则化、早停得到 **10–40pp** 最差组准确率提升；**朴素 group DRO 在过参数模型中可失败**。 | **中**：最差地形是组，但模型选择不同于损失优化，且 PvZ 屋顶稀样本。 |
| Gulrajani & Lopez-Paz，ICLR 2021，**DomainBed**。[R14] | 在多个领域泛化基准中重做**训练、模型选择协议**；许多复杂 DG 算法未稳定超过充分调优的 ERM。 | **中高**：强烈支持“选择协议本身是实验变量”，不支持固定选择哪一套。 |
| Xie 等，NeurIPS 2023，**DoReMi**。[R15] | 用 **280M** 参数代理域权重改善 **8B** 语言模型训练；相对原配方平均下游 **+6.5pp**、达到基线性能所需步数约少 **2.6 倍**。其目标是域的**超额损失最小最大化**，非直接按最差胜率选 PvZ 模型。 | **低**：语言模型、训练数据混合而非 PvZ 在线晋升。E13 反例要求慎用。 |
| Tian 等，ICML 2019，**ELF OpenGo** 补充材料表 S1。[R16] | **AlphaGo Zero 原报告 55% 胜率晋升准则**；表 S1 AlphaZero 列则为 **none**（该表所列版本）；ELF 列 **none**。训练“有没有历史门禁”因实现与版本不同。 | **中**：有棋类搜索/自我训练，但不能以“AlphaZero 都要 55%”作固定论据。 |
| Agarwal 等，NeurIPS 2021，**rliable**。[R17] | 少量运行时点估计很不稳，应报告置信区间、性能剖面、IQM，并重视种子与任务间变化。 | **高**：恰好有多地形、小样本、配对评测和预算受限。 |

**推荐制度**：把“选择（selection）”与“晋升（promotion gate）”分开。选择阶段保留主目标**按部署时关卡/卡组权重的配对平均胜率**，加明确可审计的**约束**：夜间、泳池、雾和屋顶等预先定义组不能有超容忍度退步；在屋顶胜率长期为零的短面板同时报告**存活波数、生存到第 k 波的概率**，不在事后挑有利波数。对每个组用固定**局/种子簇 bootstrap**，并报告训练种子变异。对少于四个独立评价单元的结果避免做强晋升。平均与最差组用**历史 11 轮候选池回放**比较其做出的“晋升/拒绝决定”，再决定是否改选择排序；不能用同一评价组既挑模型又宣称无偏泛化。[P3 §11.12–13]

**关键反事实**：若 roof 两方都是 **0/128**，`min` 只会产生并列并把其它地形信号全抹掉；若某一地形仅 8 个未见卡组，`min` 又易被噪声支配。因此**没有充分文献理由现在把选择器机械替换成最小组胜率**。现有 v6 门禁已做了最低评估单元和屋顶波数，先回测并验证其**总体墙钟、历史决定不变性**。**置信度中高。**

## 5. Q4｜用教师—学生分歧、遗憾、搜索惊讶度驱动采集

这里最重要的概念区别是：**“老师和学生选不同动作”不等于“学生后悔了”；“教师搜索的价值差”也未必是无偏 regret**。真正的策略遗憾通常需要比较环境中两策略的**实际回报**；搜索值差只能作为代理，且搜索老师本身在陌生卡组表现不佳。[R18–R25][P3 §11.12]

| 一手文献与原实验 | 如何定义或近似“值得再学” | 本项目相似度／可复用部分 |
|---|---|---|
| **PLR**，Jiang、Grefenstette、Rocktäschel，ICML 2021，Procgen。[R18] | 用 **TD error 等学习潜力信号**决定重放哪一程序生成关卡；与其他方法结合，论文摘要报告对基线**超过 76%** 的测试回报提升。 | **中高**：卡组×关卡×倍率能当作“关卡实例”；但当前主要是 BC，无训练 TD error 时不要冒充论文的同一指标。 |
| **Robust PLR / PLR⊥**，Jiang 等，NeurIPS 2021。[R19] | 重放引导的对抗环境设计；对**未经筛选**的关卡不更新策略，理论与实验关注零样本迁移。 | **中**：适合带合法配置生成器的后期 RL；筛选出“不会赢”的卡组很可能反而变成无效课程。 |
| **ACCEL**，Parker-Holder 等，ICML 2022；**PAIRED**，Dennis 等，NeurIPS 2020。[R20,R21] | ACCEL 通过编辑环境逐步增加复杂度，PAIRED 用**主角/对手的回报差**给生成器提供 regret 信号。 | **中低**：卡组可编辑，但两套策略对照、配置可解性校验和额外 rollout 成本高。不能以教师和学生 top-1 不同代替 PAIRED 正式 regret。 |
| **SafeDAgger**，Zhang、Cho，AAAI 2017。[R22] | 学习安全/偏差判别器，只在学生接近偏离老师时查询昂贵专家；赛车模拟器中减少专家查询。 | **高**：项目的搜索老师正是昂贵查询源；要验证判别器错误漏报不会把关键失败留在样本外。 |
| **RND-DAgger**，Biré 等，**ICLR 2025**。[R23] | 以随机网络蒸馏的**状态新颖度**触发教师调用，在 3D 赛车、第三人称导航和机器人运动实验减少专家查询。 | **高**（方法结构）：可基于现有观测向量训练低成本新颖度模型，检测未见组合；与 PvZ 的奖励目标并非同实验。 |
| **KataGo**，Wu，AAAI 2020 研究论文。[R24] | **playout-cap randomization**：论文中主训练约 **25%** 的回合决策做全搜索，初期分别约 **600/100** 次访问，后期约 **1000/200**；主论文主要把**全搜索位置**写作策略训练样本；并用搜索策略目标修剪。 | **高（算力分配）/中（采集）**：可对“少数有学习价值的根节点”加大搜索，不必每一步等成本。该比例不是 PvZ 应当照抄的常数。 |
| **KataGoMethods.md**，官方仓库，`Policy Surprise Weighting` 小节。[R25] | **g170（2020）起**，对完整搜索后网络先验与搜索策略目标的 KL 分歧加重样本权重；大致**一半均匀 + 一半按 KL 惊讶度**分配；“快搜索”若足够意外也可能有权重。 | **高（日志与权重结构）**：分歧可以用来找信息增量，但教师若误判，惊讶样本会放大误导。 |
| **单示范回放**，Salimans、Chen，2018；**Go-Explore**，Ecoffet 等，*Nature* 2021。[R27,R28] | 从演示状态/已探索状态重启，先“返回已有进展”再探索；前者报告在 Montezuma's Revenge 学到高分回报，后者依赖可恢复状态档案。 | **中**：C++ 快速保存/加载状态可移植；但目标是**真实冒险模式起点通关**，不可用从近胜利状态出发的成功率偷换整局胜率。 |

**教师质量比采样花样更靠前。** [P3 §11.12] 已完成的 E8 发现教师搜索点重放一致性只有 **0.440**（非搜索点反而 **0.941**），教师自身自复现 **0.77–0.85**。由此，“分歧越大越值得学”的前提并不总成立：分歧可能是**搜索不稳定**，而非学生没学会。应将**教师能否赢、搜索值差、搜索重试一致率**三者作为独立轴记录；把同一随机种子、不同候选动作从**同一冻结状态**续走，避免强迫老师和学生走不同随机环境时带来的假差异。

**推荐两阶段小实验**：先影子记录每个根节点的学生提案、搜索首选、候选价值差、置信度/方差、观察到的可见卡组、搜索预算和后续对局结果，不影响现有 25% ×4 采集配额；再在**固定新增老师 rollout 预算**内，以“随机查询”“动作不一致”“搜索价值差大/小”“RND 新颖度高”四种选择器并列，比较每新增 **1,000 次搜索调用**换来的独立未见卡组真实胜率增量以及教师胜局率，按教师搜索可靠/不可靠层报告。若价值差选择器只提高离线准确率而不提高整局胜率，不晋升。**支持力度：中**；PLR、RND-DAgger 的效果数字**不能**直接移植到教师标注效率。

## 6. Q5｜卡组/组合泛化：有哪些直接证据？

| 一手文献 | 实验和可核数字 | 相似度及限制 |
|---|---|---|
| **Cardsformer**，Xia 等，ECAI 2023，论文 §5–6、**表 1**。[R29] | 在《炉石传说》用**卡牌描述编码 + 状态转移预测 + 自我对战 RL**。与 Dynamic Lookahead 交手，卡组含 **0/10/20/30 张未训练卡**时胜率依次 **78.0±2.5% / 70.7±2.2% / 64.0±4.0% / 47.5±2.6%**；**每设置 1,000 场**，标准差由每 100 场分组计算。论文承认新卡仅涵盖**较简单效果**，复杂触发机制未覆盖。 | **中**：同为卡牌游戏及未见配置，但 PvZ 主要是**全部已知植物的新组合**；Cardformer 还要学未见卡**身份/描述/效果**，推理与训练范式也不同。 |
| **Learning With Generalised Card Representations for Magic: The Gathering**，Bertram、Fürnkranz、Müller，2024，摘要/实验。[R30] | 用数值、类别、文本、图像、使用元数据表示卡牌；报告在**完全未见卡牌**上预测约 **55% 人类选牌决策**。 | **中低**：预测的是**选牌/构筑过程中的人的选择**，不是陌生卡组**对战胜率**；不可把 55% 当 PvZ 通关率。 |
| **Procgen**，Cobbe 等，ICML 2020，§3–4。[R05] | **16 个**程序生成游戏，展示足够不同训练关卡和测试关卡可揭示**过拟合和模型规模效应**；扩大模型在其设置中提高泛化。 | **中**：直接支持要有严格未见配置评测，但训练的是 RL 程序生成**关卡**，不是固定植物组合的 BC。 |
| **本项目**。[P3 §11.2, §11.11–13] | 历史默认卡组占比极高；E13 仅改变已收集样本的重权/封顶反而伤陌生卡组；已把生产随机卡组池扩到 **54 套并隔离 26 套留出**；老师在此前 **54 套陌生卡组测得约 14.6% 胜率**。 | **最高**：这是对当前资源分配最直接的证据，但要标清老师测评版本、卡组来源和评测 seeds。 |

**核心推理**：一张植物的能力已在训练数据出现，并不意味着网络学到了“这张牌与另外 7 张牌在某地形、某出怪倍率下的协同/替代关系”；**组合泛化不能等同卡牌词义泛化**。Cardsformer 更接近“卡牌描述带来属性级迁移”，而此项目应先问**观测有没有给出合法卡池、每张牌当前可用性、效果状态与环境动态**；不是立刻加大语言模型。也不能从 Procgen 得到“需要恰好 X 万不同卡组”的通用采样门槛。[R05,R29,R30][P4]

**建议评测矩阵**：按 **地形 × 训练卡组签名是否出现 × 单卡在训练中是否出现 × 关卡模式 × 难度倍率** 划分独立层；固定卡组多重集 hash、防止跨关/跨轮拆分泄漏，并单列禁用/非法卡组；教师、CNN 学生、根节点搜索和候选 v3 在**同一引擎构建、同卡组、同随机种子**配对。用教师是否能通关把局分为**有可靠正例 / 教师失败 / 教师结论不稳定**，同时记录“教师成功局中的学生成功率”与整个测试分布的学生成功率，前者仅是**条件指标**，不得替换总体指标。比较新四类各 25% 配额带来的**净新增卡组签名、教师成功轨迹、未见真正胜率**；勿只追总行数。[P3 §11.12–13]

**“多样性还是数量”与“随机卡组样本代价”的答案**：论文有“环境多样性重要”的定性证据，但**本项目一局新卡组比一局固定卡组多耗多少标签/需多少局才能泛化**仍查不到直接可移植数字。E13 说明**压低旧数据权重**不等于“凭空制造多样性”；由固定资源下的**真打配对学习曲线**决定新增采集价值。**置信度：先扩有用数据覆盖高，量化样本代价低。**

## 7. Q6｜逐项核验上一份调研 §12 的 12 条陈述

> 状态为**已核实**／**被推翻或需改写**／**尚查不到**。这里既核实数字，也核实把数字**归属于哪篇文章、哪个实验设置**。引用位置指正式论文页/图/表/小节，或官方代码文件。

| # | 原待核查说法 | 结论、原文定位与正确读法 |
|---:|---|---|
| **1** | AlphaStar **1.39 亿参数** | **已核实**。Vinyals 等 [R01] 论文详细架构/扩展方法说明 **139M** 参数，**推理只需 55M**；这是完整 AlphaStar 体系，不是本项目 180k 学生的同规模对照。不能把其图 3f 当单变量注意力消融。 |
| **2** | **AlphaZero** 回放窗 **50 万局** | **归属被推翻/AlphaZero 本身查不到相同明示值**。Tian 等 [R16] 的**补充材料 p.2、Table S1** 三列明列：**AlphaGo Zero (AGZ)=500,000 games；AlphaZero (AZ)=`?`；ELF OpenGo=500,000 games**。`?` 表示所据公开论文细节未明，不等于 AZ=0；AGZ 对应 Silver 等 2017 [R32]，AZ 对应 Silver 等 2018 [R33]。不能把**局数**误写为**状态行数**。 |
| **3** | AlphaGo 2016 快 rollout **约 24% / 2µs** | **已核实且可更精确**。Silver 等 [R31] **正式 Nature 2016 p.485（PDF 第 2 页）**：“rollout policy”是**线性 softmax + 局部模式特征**，行动预测准确率 **24.2%**、每次选动作 **2µs**；监督 CNN 策略 **57.0%**、约 **3ms**。注意两模型不是同能力/同输入；这是**快走子与强策略分工**的案例，不是 PvZ 的耗时保证。 |
| **4** | KataGo 近期窗口/样本权重公式 | **部分已核实**。Wu [R24] 原论文 **Playout Cap Randomization / Policy Target Pruning** 小节核实 **25% 全搜索**的代表运行；官方 KataGo [R25] **`Policy Surprise Weighting`** 说明 g170 起全搜索状态权重**约一半均匀、一半按 KL 惊讶度**，且快搜索特殊位置可记录。官方样例脚本 [R26] 出现 `taper-window-scale 50000` 和最小行数参数，但这些是**脚本/版本配置，不是跨版本固定的“50,000 局窗口”**。**完整生产默认窗口规模不能一概而论**。 |
| **5** | Leela Zero 回放窗和抽样细节 | **查不到可安全归一的版本无关数字**。原项目/代码 [R34] 有训练数据与工作流，但训练端细节及线上窗口依时期、分支变化；**Leela Zero（围棋）不要与 Leela Chess Zero（国际象棋）混为一谈**。本报告不用它支持精确窗口结论。 |
| **6** | “Transformer 在 **10M–30M** 行不如/等于 CNN” | **查不到、不可断言**。AlphaStar、NetHack 缩放律、Gato [R01–R04] 都没有所要求的同 PvZ 标签规模、动作空间、算力/参数的交叉实验。既不能确认为真，也不能凭视觉分类经验视作定律；由 E4 测。 |
| **7** | “MoE 对 OOD 能否帮助” | **没有统一方向；旧绝对化推论不成立**。Soft MoE [R06] 是价值型 RL 参数缩放；CARE [R09] 是已知任务上下文；两者都没有测 PvZ 式“已知牌新组合”。本地 E3 [P3 §11.9] **实测反对当前五专家方案**，但不是推翻所有 MoE 或 FiLM 的论文结论。 |
| **8** | 等价动作标签噪声是关键瓶颈 | **通用文献未直接核实；本地假说被实测弱化**。[P3 §11.12] 等价动作矛盾仅 **0.05%**、卡槽矛盾 **0**；旧/新教师搜索时 top-1 不一致更显著。**不应优先再做 top-k 软标签**，除非新数据表现发生质变。 |
| **9** | DoReMi/data mixing laws 可用于 PvZ 的 RL/BC | **只属外推，不能直接借用公式与效果**。[R15] 代理训练的是语言模型预训练的域混合，比较的是相对参考损失；本地 E1/E13 [P3 §11.10–11.11] 降历史/封卡组显著伤本项目离线未知卡组，**不得把 2.6×/6.5pp 挪用**。 |
| **10** | “温度采样 τ=5，Johnson 等 **2016**；应以指数 **α>1** 提升少数域” | **数值出处查不到，数学方向明确错误**。Johnson 等公开的 Google Multilingual NMT 论文发表于 **2017** [R35]，不是可据此确认 τ=5 的 2016 一手出处。若记总行数 `n_i`、以温度 τ 采样，则 `p_i ∝ n_i^(1/τ)`，**τ>1 时指数 1/τ<1 才会提高少数域相对占比**；若写 `p_i∝n_i^α`，须 **0<α<1**。而 E13 使用的 `clip((200k/n)^0.5,0.3,1)` 是**逐行权重下调规则**，不是相同采样公式，不能混用。 |
| **11** | CARE 的缩写和机制 | **已核实并细化**。[R09] 原论文题目 *Multi-Task Reinforcement Learning with Context-based Representations*；论文正文实际命名 **Contextual Attention-based REpresentation learning (CARE)**，用**任务元数据编码器 + k 个可学习编码器、上下文控制其共享**，连续控制多任务 RL（Meta-World）；不能仅把 CARE 展开为“Context-based Representations”。 |
| **12** | CNN128 伤未知卡组/夜间 | **项目记录可核实、不是外部论文实证**。[P2/P3] 较早测试同语料：陌生卡组 **60.0→53.8%**，夜间 **51.4→42.6%**。但不同 E3 语料/指标中，联合128 未知卡组 `cell_acc` **0.3150** 高于联合64 **0.2699**；**不构成矛盾**——不同实验条件与“真打 vs 离线准确率”。准确说法为“**加宽没有在已测方案中稳定兑付未见卡组真打收益**”。 |

**Q6 三条必须修订进原调研的勘误**：①把“AlphaZero 500k”改为“**AlphaGo Zero 500k；AlphaZero 未指定**”；②删除“**α>1 可增大小域相对权重**”，改为“**温度>1 对应幂指数<1**”；③ CARE 释义写成原文使用的 **Contextual Attention-based REpresentation learning**，并注明论文完整题目不同于缩写。其余不确定内容继续显式保留为“查不到/跨任务外推”。

## 8. Q7｜长远发展：19 篇值得精读的论文／官方技术报告

**如何读这份清单**：论文效果首先来自它们自己的基准。下文“成本”均指**在现有 1 张消费 GPU、Mac、约 26GB RAM、CPU 主导推理的条件下尝试相应思想**的相对工程/计算成本，**不是原论文精确复现时长**；“高风险”包括计算不合算、目标错配、对数据/标签质量敏感。为了避免反复抄表，表格用 `[Rxx]` 链至文末完整书目信息。

### 8.1 现在就值得用的 9 篇

| 优先阅读 | 它是什么／关键证据 | 具体接入项目哪个环节 | 预计成本与主要风险 | 时间 |
|---|---|---|---|---|
| **1. Agarwal 等：Deep RL at the Edge of the Statistical Precipice** [R17] | 少训练运行下应报告 **bootstrap 区间、IQM、performance profiles**，不要只看均值。 | 扩充 `bench_play.py` / 门禁产物：按**同关卡–卡组–种子**配对的胜负差、置信区间、屋顶波数、各地形样本数；给每一条晋升记录可复核的引擎 hash。 | **低**：统计脚本与历史回放；风险是少样本组过于宽的区间导致长期无法决策。 | **现在·最高** |
| **2. Wu：Accelerating Self-Play Learning in Go** [R24] | 搜索**高/低预算分流**，主运行约四分之一决策投入更深搜索，训练重点用高质量策略目标。 | 改**根节点搜索预算调度**而非 rollout CNN；先根据日志按来源/搜索稳定性做分层、配固定总搜索次数对照。 | **低–中**：主要 C++/评测工作；风险是省了算力却降低教师提案质量，不能照搬 25%。 | **现在** |
| **3. Jiang 等：Prioritized Level Replay** [R18] | 用潜在学习价值而不是均匀概率重新选择训练配置。 | 把“关卡、卡组 hash、倍率、出怪种子”定义为 level，生成候选**学习价值分位数**；保留原 25% ×4 采样配额作 A/B 基线。 | **中**：需 level 索引及不同策略下独立留出；高分可能只是不可赢或异常搜索。 | **现在** |
| **4. Biré 等：RND-DAgger** [R23] | 新颖度模型控制昂贵专家查询，2025 年在 3D 游戏与机器人场景验证。 | 从 obs v3 中已可见的卡槽、实体、局面编码训练小 RND；只作为**老师是否加搜/是否入训练池**的评分器，不代替最终结果评测。 | **低–中**：小 GPU/CPU 模型；“新颖”≠“有教学价值”，不得排除普通但关键的困难样本。 | **现在** |
| **5. Cobbe 等：Procgen** [R05] | 16 类程序生成游戏揭示过拟合和不同训练分布的重要性。 | 维护**固定不泄漏的卡组×地形×倍率**测试轴，专门测试**组合新颖度**，不再把固定默认卡组总行数当泛化证明。 | **低**：主要评测设计；风险是测试分布被反复调参污染。 | **现在** |
| **6. Xia 等：Cardsformer** [R29] | 卡牌描述 + 学习动态 + 策略；在复杂卡牌 OOD 场景有明确对照，见 Q5。 | **先不接大语言编码器**；审查目前卡槽、植物状态、资源、敌人阶段是否都在 obs 中可见，必要时补可观测属性后做实体–格子混合表示。 | **低（观测审查）/高（完整论文复现）**；效果主要来自未见**新牌**，不必然帮已知牌新搭配。 | **现在研究，架构半年内** |
| **7. Bertram 等：Magic 卡表示** [R30] | 按卡数值/类别/文本等泛化未知卡，报告 **55%** 新卡选择预测。 | 可把植物 ID **旁路为结构化可见属性嵌入**（冷却、资源消耗、射程/作用类型等须来源于游戏状态而非手写战术评分），对照纯 ID；勿把 MTG 构筑准确率当对局胜率。 | **中**：要定义字段、输入版本与 ablation；属性编码可能隐含人工知识，需守住“不手写策略规则”界线。 | **现在审查，半年内对照** |
| **8. Salimans & Chen：Learning Montezuma's Revenge from a Single Demonstration** [R27] | 从演示中间状态开始强化学习，解决稀疏奖励，论文报告 **74,500 分**。 | 利用引擎可复现状态做**失败点附近的训练子课程**；但晋升必须回到真实关卡起点；已部署示范格仅 25% 别回滚成 75%。 | **中**：状态保存与回放协议；主要风险是“从半路成功”掩盖真实通关失败。 | **现在只审计现行配额** |
| **9. Ecoffet 等：First Return, Then Explore（Go-Explore）** [R28] | 先返回可恢复的已访问状态，再拓展探索，针对长时序稀疏奖励。 | 为 L45/屋顶等难点建**状态档案与可控重启**，明确区分训练起点与正式评测起点；可作为无示范探索的基础设施。 | **中**：大量可复现存档和环境状态一致性；不能以重启胜率报“通关”。 | **现在准备、半年内实试** |

### 8.2 半年内可能值得投入的 6 篇

| 优先阅读 | 它是什么／原研究证据 | 具体怎么接入 | 预计成本与风险 | 时间 |
|---|---|---|---|---|
| **10. Danihelka 等：Policy Improvement by Planning with Gumbel** [R36] | 用 Gumbel 采样与序贯资源分配，在**较小搜索预算**改进搜索策略。 | **仅根节点**试 Gumbel 风格候选筛选/搜索改进，对固定搜索调用数比较老师本身胜率与后续学生收益；三头合法动作的组合空间需明确定义。 | **中**：改搜索核心、需严格同预算单元测试；不保证适合当前多步候选 rollout。 | **半年内优先** |
| **11. Hester 等：Deep Q-learning from Demonstrations（DQfD）** [R37] | 演示监督项加 TD 学习；42 个 Atari 游戏中**41 个**在前 100 万步优于 DQN 基线，**14 个**超越示范成绩。 | 将现有 `(s,a)` 教师标签扩为**完整过渡 `(s,a,r,s',done)`**，用小支线试 Q/优势权重联合 BC，不动生产 CNN。 | **中**：游戏奖励/合法动作分解、离线数据行为策略偏差；老师很弱时 RL 也可能放大错估。 | **半年内** |
| **12. Kostrikov 等：Offline RL with Implicit Q-Learning（IQL）** [R38] | 在离线数据分布内做 Q 更新，尽量避免直接查询 OOD 动作，之后用**优势加权 BC**改善策略。 | 充分保存每局轨迹和终局收益后，在难关实验**价值加权学生**，与纯 BC 在未见卡组真打配对比较。 | **中–高**：多头大动作空间、长时延/稀疏终奖、状态可观察性和 OOD 数据问题。 | **半年内** |
| **13. Wang 等：EfficientZero V2** [R39] | ICML 2024，在**66 个任务的 50 个**优于论文比较的 DreamerV3；重视受限样本的模型式 RL。 | 选**一个小型非平凡关卡**，评估在固定新采样局数下，学动态/价值是否真正超过 BC+现有精确引擎搜索。 | **高**：整体框架昂贵，且 CPU 续走速度很可能不如原 C++ 引擎。 | **半年内小样验证，不做整套移植** |
| **14. Dohare 等：Loss of Plasticity in Deep Continual Learning** [R43] | 长期持续训练可能失去学习新任务的可塑性，Nature 2024 研究连续反向传播与单元更新。 | 若生产以后改为**不断 warm-start**，日志比较“从头训”与“持续微调”的新卡组适应性、遗忘和神经元活性；**目前每轮从头训练，优先级较低**。 | **低（监控）/中（方法改动）**；贸然重置单元可能伤短期稳定性。 | **半年内条件触发** |
| **15. Jacob 等：Integer-only Quantization** [R44] | 在 ARM 设备上 8 位量化使参数存储相对 32 位约为 **1/4**，但速度增益依赖实现。 | 对 **CNN rollout 核心**先做 INT8 推理基准、精度和对局匹配，再决定是否扩展到根节点；确保量化后的 C++ 标量路径、SIMD 真有提升。 | **中**：需要 int8 核、校准和网络一致性验证；批量 1 的 CPU 上**不能保证提速**。 | **半年内工程验证** |

### 8.3 长期观察的 4 篇（目前不适合直接复刻）

| 优先阅读 | 它是什么／来源可核的关键结论 | 真正可能引入的位置 | 成本与风险 | 时间 |
|---|---|---|---|---|
| **16. Schrittwieser 等：MuZero** [R40] | 学**奖励、价值、策略**的内在动力学，并用树搜索改进；在 57 个 Atari 游戏及围棋、国际象棋、将棋展示效果。 | 如果未来目标是不依赖已移植 C++ 规则的学习系统，先用精确模拟器数据检验**短时 latent transition**是否能维持动作排序。 | **非常高**：训练与搜索复杂；当前原始精确引擎便宜，世界模型误差会逐步累积。 | **长期** |
| **17. Hafner 等：DreamerV3** [R41] | *Nature* 2025，统一世界模型+演员/评论家，在**超过 150 个任务**中用同一配置取得广泛结果。 | 真正从观测/交互学习的远期支线；先比较当前引擎**真实每步模拟时间**与世界模型的**推理/训练总时间**，再谈替换。 | **极高**：算力/内存、长局面有效预测；高配置大模型完全不适合直接投入生产。 | **长期** |
| **18. Micheli 等：IRIS（Transformers are Sample-Efficient World Models）** [R42] | Atari 100k，约两小时**等价游戏交互**，平均人类归一化分数 **1.046**，26 个 Atari 中 **10 个**超人类，**无前瞻搜索**。 | 用稀疏交互日志训练**离线影子模型**，先预测下一观测/奖励，而不用于控制；对低频根节点难关逐步验证。 | **高**：原结果在像素 Atari，不等同复杂 PvZ 几百/几千步的高保真模拟；Transformer 推理仍贵。 | **长期** |
| **19. Chen 等：Decision Transformer** [R45] | 把高回报轨迹学习转为**目标回报条件的自回归动作序列**，在 Atari、Gym 等离线任务有可比结果。 | 仅在已经积累**大量能通关的多地形轨迹**后，作为无需搜索的长时间上下文学生对照。 | **中–高**：当前老师 OOD 胜率低、长序列 CPU 开销大；数据缺正例时返还条件不可信。 | **长期** |

**未单列进 19 篇但应保持追踪的技术**：ACCEL/PAIRED 对将来的**无规则、无需人类示范的环境课程**有价值 [R20,R21]，但初期需确保生成的卡组/倍率**合法且可解**；DoReMi 是数据混合方法而非已证明的 PvZ 经验法则 [R15]；AlphaStar/Gato 提供的是更大规模多任务系统启发 [R01,R04]，不是单显卡施工图；语言模型可用来**归纳失败日志、提出待检验假说、读论文和辅助生成不泄漏的实验配置**，但目前**找不到一手实验可证明让 LLM 直接指挥每一步会优于当前毫秒级老师搜索**。不建议把 LLM 嵌在每一帧 CPU 推理热路径。所谓“不手写规则”在本项目中应落实为**不写按游戏知识定动作的策略条件/轨迹公式**；合法动作过滤、实验分组、统计估计与真实引擎提供的可见状态不是替代学习策略的手写解题法。

### 8.4 三段路线图：从“学会模仿”到“不用人类演示”

| 阶段 | 明确阶段成果 | 必须通过的检验／退出条件 |
|---|---|---|
| **近期可试（当前至约 6–8 周）** | ①已入队 **E4** 跑完，区分 CNN、v3 和富观测的增益；② `r52+` 的四类采集、54 卡组池真实覆盖效果；③ teacher 在未见卡组按搜深分层的胜率和失败恢复率；④现有门禁 v6、deploy19 原生路径 A/B 与同构建配对评估；⑤加搜索惊讶度的**影子日志**。 | 真正未见卡组**整局胜率**、最弱地形存活/胜率、教师可赢局比例、CPU 吞吐，至少 2–3 训练种子和配对 seed；任何方法若只赢离线 `cell_acc` 或以未可比硬件速度胜出不予晋升。 |
| **中期（约 2–6 个月）** | ①按教师可靠性决定搜索预算及主动查询；②在非生产支线试 **Gumbel 根搜索、BC+优势权重/IQL**；③针对 L5/L45/屋顶做富实体观测和困难状态档案；④压测量化/蒸馏是否使续走速度更快；⑤正式划分**难度倍率及跨关组合 OOD**基准。 | 以**相同新增采集局数、同老师搜索总 CPU 时间、同训练更新预算**比较策略；如果质量不升或老师搜不赢，应先改善教师/覆盖而非再扩模型。 |
| **长期（6 个月以后，按预算触发）** | ①在已有精确引擎上验证**学习动力学**与短视规划；②降低对示范的依赖，逐步提高**在线 RL/自主探索比例**；③跨种子、跨卡组、跨倍率验证超越人类；④在可靠自学后才考虑无引擎黑箱的世界模型。 | 每项都需比**同预算精确模拟器+当前搜索**更有收益。**“超过人类”须预先指定人类高手样本、关卡/卡组分布、操作约束、计分协议和区间**；不能以挑选有利关卡的单次胜利宣布完成。 |

**真实的长期难点**不是把 18 万参数换成亿级 Transformer，而是：**老师在没见过的卡组是否能发现获胜策略、能否用有限 CPU 时间稳定指导训练、学生能否从非独立且偏旧的训练分布上学出对新组合有效的决策，最后如何从模仿转为按游戏结果自我改善**。这些必须按里程碑逐个解锁。[P3 §11.9–13][R17,R24,R37–R42]

## 9. 对项目的行动建议（排好顺序，含可证伪判据）

> **以下阈值若标为“建议预登记”，是本报告的实验设计提案，不是文献中存在的普适数值。** 优先使用项目已预登记的门槛/已有队列，不能为了提高通过率在看完测试结果后修改评测集合或阈值。每次报告应明确 `engine_sha`、student/teacher checkpoint、数据快照、采样协议、评测面板及其固定 seeds、原始分母。

| 排序 | 要做什么（与目前完成状态核对） | 实验设计／是否成功的判据 | 证据强度和成本 |
|---|---|---|---|
| **P0-1　完成并解释 E4** | **先消化 task162 已排队的 8 组×2 种子**，不重复建模；增加报表，区分新 obs、参数与结构。 | 相同冻结 r42 包/同划分/固定 12 epoch、有效 batch 2048 比较 CNN64、CNN128、ResNet、v3、小混合、terrain、MoE。**沿用项目 E4 硬门槛**：v3 在未见卡组以及屋顶/雾不能比配平 CNN 退步，并且**真实对局**优于强 CNN 基线，根节点墙钟增加 **<30%**；额外建议公布配对区间，并加入足够训练种子复核获胜臂。若离线好、真打差或吞吐越界，**停止部署**。 | **文献支持：中**（探索结构）/ **没有预测胜负的同规模直证**；**成本中–高**，GPU 队列是主瓶颈。[R01,R02,R05][P3 §11.7–13] |
| **P0-2　观察 r52+ 的真实数据闭环** | **不要再次部署**“四类 25%”或“增加 38 套卡组”，它们已经上线；跟踪 2–3 个新轮次的独立局分母、教师成功、被选择臂和实际门禁。 | 把 [P3 §12.6] **既有预登记**当主判据：非示范格教师胜率 **≥70%**、**8 套评估卡组 ≥12/64**，同时打印各卡组完整置信区间、难关失败分布和训练新增有效独立配置；若不达标优先诊断教师弱点，不增设手写赢法。若历史权重 1.0 组仍被拒且胜率不改善，维持现状，不再开已关闭的 E1/E13 对照。 | **项目实证高 / 方法迁移中**；**成本低**（现有生产日志与评测）。[R05,R17,R29][P3 §11.12–13] |
| **P0-3　明确教师上限，先增加可学习的“胜局”** | 对 `task129` 原先 **54 套×8 局 = 432 局、63 胜**测例冻结版本，并对新老师/搜索深度重复**同配对卡组 seeds**；将教师在搜索预算、动作候选分歧、根节点搜索重现性与实际终局结果联表。 | **建议预登记**：同搜索调用数或同 CPU 秒，未见卡组**教师整局成功率**相对 14.6% 旧基线改善至少**5pp**，配对 bootstrap 95% 下界 >0，且五地形无已定义严重退步；该门槛是业务决策提议，不是论文证明。因旧样本可能版本不同，基线需同构建重测；若搜索胜率仍低，**暂缓大规模标注失败局面**，优先搜索预算/候选质量的诊断。 | **项目动机高 / 文献直接力度中**；**成本中**（CPU 搜索）。[R22–R25,R36][P3 §1.3, §11.12] |
| **P0-4　把模型选择与晋升分开评估** | 现有 v6 门禁已覆盖**最少 4 单元+屋顶波数**；对最近 **11 轮**存档**同引擎**回测“平均优先”“平均+分地形下界”“最差组优先”三种选择器，先离线只读不改生产。 | 预注册不超过 **1 个**排序方案的可更改参数集；核查**历史通过/否决决定、最弱地形真实损失、总体晋升成功率、墙钟**。**建议优先**满足项目 [P3 §12.6] 的既有回归要求：旧 11 轮门禁结论不变、墙钟降低或不明显变坏。roof 双方 0 胜时用生存波数区分，不以单次 `min` 取胜。 | **文献支持中高**；**成本低–中**（已优化的 Mac C++ 对局工具）。[R13–R17][P3 §11.1, §11.12–13] |
| **P1-1　验证 deploy19 原生完整对局的正确性** | 已合并部署，不需重做优化；完成待执行的 **45 个种子 A/B**，用完全一致的引擎构建和行为日志比较新旧模式。 | 预注册允许的 `state_hash` 差异清单、最终赢输和搜局轨迹一致性，**不能只看平均速度**；若语义不一致或改变晋升结论，原生路径回退/保持可选。速度以每小时有效独立对局及整套门禁秒数计。 | **纯项目工程实证**，文献不支持直接数值；**成本低**。[P3 §11.12–13] |
| **P1-2　主动查询与优先搜索的影子试验** | 记录学生提案、老师搜索分布、同状态 value 差、搜索重试一致率、RND 新颖度、终局；先只读日志，再与随机策略对照。 | **建议预登记**：在**完全相同额外搜索 CPU 秒**、固定未见评测面板和相同训练更新步数下，分歧/新颖度采集后**未见卡组整局胜率**比随机查询至少 **+3pp 且 95% 配对区间下界 >0**，弱地形不显著退步。若查询数下降但老师质量/胜率同步下降，判负。**切勿以 KL 大等价于教师正确**。 | **文献方法证据中**，本项目定量增益**待测**；**成本中**。[R18,R22–R25] |
| **P2-1　离线 RL/少预算搜索研究支线** | E4、采集观测、门禁可靠之后再考虑 Gumbel 搜索和 BC+IQL；在**一个已知可通关但非固定 L7 的难关**上开始。 | 同一批转移、同等老师查询时间/训练 GPU 时间，分别报告**离线 loss、教师胜率、学生真打、未见相近卡组**；若根节点搜索**不比现搜索更强**，或 RL 只在训练卡组增益，停止此支线。| **文献支持中**（游戏 RL 有用），PvZ 直接证据**低**；**成本中高**。[R36–R39] |

### 评测与统计的最低交付协议

1. **锁定层级**：`(terrain, level, deck_multiset_hash, difficulty, seed, engine_hash)`；测试集按**整卡组与整局**隔离，而不是随机打散 2300 万行。任何数据从测试流入训练都要重新划分。
2. **双层指标**：生产目标为**真实初始状态通关率**；同时报告生存波数、每局 CPU 总耗时、训练与推理内存；辅助报告 `cell_acc`、三头 loss、分歧与 teacher success，但不可冒充通关。
3. **公平预算**：固定**优化器更新步数、实际有效 batch、训练读到的独立行数、等输入字段、教师搜索次数或 CPU 秒**，多种训练种子；用 **配对局/随机种子簇 bootstrap CI**，必要时 McNemar 检验配对输赢，并明示多指标多次尝试的比较范围。
4. **测胜率以前测可测性**：如果两个屋顶模型都是 0/128，“不显著”**不是**“证明等效”；提高种子数或用预注册波数分层；如果某组只有 8 套卡组，不承诺 ±1pp 精度。[R17][P3 §11.9]
5. **从研究到生产**：候选模型离线通过→独立真打与吞吐通过→原生产门禁通过→保留回滚 checkpoint；不得用学生在老师胜局条件下的成功率代替所有卡组整体分母。

## 10. 单列：查不到、来源脆弱或只能外推的说法

| 待确定问题 | 当前证据边界 | 正确标注/下一步 |
|---|---|---|
| **10M–30M BC 的 Transformer/CNN 交叉样本阈值** | 未找到条件匹配的一手同数据消融。 | **查不到**；只靠 E4 在本项目实证。 |
| **实体注意力理论上“数据越多终会超过 CNN”** | 大模型多任务和图像结果不能推出架构排序的必然关系。 | **外推**；没有必然交叉的证明。 |
| **MoE 对陌生 PvZ 卡组一定更好/一定更差** | 一手 MoE 论文多为已知多任务 RL、语言模型；本地实验只说明特定五专家配置性价比差。 | **未证普遍性**；不重新投入当前主线。 |
| **单纯提高参数就能学会卡牌组合泛化** | Procgen 的增大模型改善泛化，与本地部分加宽真打下降并存；训练分布、预算、输入不同。 | **无统一单调结论**；按同设置真打裁决。 |
| **通过数据重权/封顶即可提高本项目 OOD** | E1/E13 三方向实测负面，但 cap 有有效 batch 与卡组签名定义混淆，单训练种子。 | **本地此实现判负**，不是证明所有数据混合都无效。 |
| **DoReMi 的样本混合函数可直接搬进 BC/DAgger** | 原工作是语言模型基于 reference 的超额损失重加权。 | **只能外推**，且本项目近邻 E13 负向。 |
| **AlphaZero 固定 500,000 局/55% 晋升门槛** | ELF OpenGo Table S1 原文对 AZ 分别记 `?`、`none`；AGZ 才是 500k/55%。 | **纠正旧归属**。 |
| **KataGo/Leela Zero 存在跨版本固定“近期回放 N 局”** | KataGo 有可核的特定版本脚本与 surprise weighting；Leela 不同分支/时间段不同。 | **没有版本号与训练代码 SHA 就不报精确 N**。 |
| **Johnson 等 2016 论文推荐 τ=5 与 α>1** | 已核的可公开原文是 Johnson 等 **2017** 多语言 NMT；没有核实其 τ=5 段；且 α>1 的方向数学上相反。 | **数值不可引、方向必改**。 |
| **在未见 54 套卡组老师胜率 14.6% 意味学生不可超越 14.6%** | 教师低胜率是**标签支持与当前搜索的强瓶颈**；RL 学生理论上可超越教师，正例也可能来自其它策略。 | **不能作严格上界**；报告它为当前数据获取与 imitation ceiling 的诊断信号。 |
| **LLM 可直接给 PvZ 根节点可靠行动策略** | 缺本项目同搜索调用预算、合法三头动作、CPU 延迟的原始对照。 | **外推未证**；只建议非热路径辅助实验设计/失败归纳。 |
| **世界模型推理一定比 C++ 引擎快** | MuZero/Dreamer/IRIS 论文不是本项目 1:132、CPU 小 CNN 的成本对照。 | **未证且可能恰好相反**；先做 shadow model 预测误差与真实吞吐比较。 |
| **各地形最差组排名应完全主导晋升** | roof 0/128 地板、某些组卡组数很少；group DRO 是训练目标研究而非固定选择器定理。 | **没有普适论文结论**；以历史回放和预登记多目标门禁做局部判决。 |

---

## 11. 引用索引：公开一手文献与官方源

**文献索引格式**：`编号｜作者｜年份｜完整题名｜发表场所／arXiv／DOI｜直达原文`。作者超过数人者列出关键作者并加 “et al.”；原网页/正式论文的作者列表为规范全量书目，避免将长达十余人的作者串影响检索。**下列来源均为论文原文、论文官方页面或官方代码，不把百科/转述博客作为核心证据。** 少数论文官方网站禁止自动打开全文时，参考了其**正式论文原始 PDF 可检索正文/摘要**；没有因此编造未读到的表格数值。

### Q1／Q2：表征、缩放与条件专家

- **[R01]** Oriol Vinyals, Igor Babuschkin, Wojciech Czarnecki, et al. **2019**. *Grandmaster level in StarCraft II using multi-agent reinforcement learning*. **Nature** 575, 350–354. DOI [10.1038/s41586-019-1724-z](https://doi.org/10.1038/s41586-019-1724-z). [官方扩展 PDF（神经结构、参数量、Fig.3）](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf).
- **[R02]** Jens Tuyls, Dhruv Madeka, Kari Torkkola, Dean P. Foster, Karthik Narasimhan, Sham Kakade. **2024**. *Scaling Laws for Imitation Learning in Single-Agent Games*. **Transactions on Machine Learning Research**. [TMLR 正式论文 PDF](https://openreview.net/pdf?id=qGVchjFj3a) · [arXiv:2307.09423](https://arxiv.org/abs/2307.09423). **1.7× 引自 TMLR 2024 版本而不是 2023 早期 arXiv 摘要。**
- **[R03]** Michaël Mathieu, Sherjil Ozair, Srivatsan Srinivasan, Caglar Gulcehre, Shangtong Zhang, et al. **2023**. *AlphaStar Unplugged: Large-Scale Offline Reinforcement Learning*. [arXiv:2308.03526](https://arxiv.org/abs/2308.03526) · [DeepMind 官方实现/实验协议](https://github.com/google-deepmind/alphastar/blob/main/alphastar/unplugged/README.md).
- **[R04]** Scott Reed, Konrad Żołna, Emilio Parisotto, et al. **2022**. *A Generalist Agent*. [arXiv:2205.06175](https://arxiv.org/abs/2205.06175)（Gato，1.2B/604）。
- **[R05]** Karl Cobbe, Chris Hesse, Jacob Hilton, John Schulman. **2020**. *Leveraging Procedural Generation to Benchmark Reinforcement Learning*. **ICML**, PMLR 119:2048–2056. [论文全文及附录](https://proceedings.mlr.press/v119/cobbe20a.html).
- **[R06]** Johan Samir Obando Ceron, Ghada Sokar, Timon Willi, Clare Lyle, Jesse Farebrother, et al. **2024**. *Mixtures of Experts Unlock Parameter Scaling for Deep RL*. **ICML**, PMLR 235:38520–38540. [官方论文](https://proceedings.mlr.press/v235/obando-ceron24b.html).
- **[R07]** Ruihan Yang, Huazhe Xu, Yi Wu, Xiaolong Wang. **2020**. *Multi-Task Reinforcement Learning with Soft Modularization*. **NeurIPS 33**. [官方论文](https://papers.neurips.cc/paper_files/paper/2020/hash/32cfdce9631d8c7906e8e9d6e68b514b-Abstract.html).
- **[R08]** Ahmed Hendawy, Jan Peters, Carlo D'Eramo. **2024**（预印本 2023）. *Multi-Task Reinforcement Learning with Mixture of Orthogonal Experts*. **ICLR 2024**. [ICLR 正式论文页面](https://proceedings.iclr.cc/paper_files/paper/2024/hash/19c145aaad40927c51f4d10eaa339c20-Abstract-Conference.html) · [arXiv:2311.11385](https://arxiv.org/abs/2311.11385).
- **[R09]** Shagun Sodhani, Amy Zhang, Joelle Pineau. **2021**. *Multi-Task Reinforcement Learning with Context-based Representations*. **ICML**, PMLR 139:9767–9779. [正式论文 PDF](https://proceedings.mlr.press/v139/sodhani21a/sodhani21a.pdf) · [实验/代码补充](https://proceedings.mlr.press/v139/sodhani21a/sodhani21a-supp.pdf). **CARE 机制与全称见正文方法段和补充 §A.3。**
- **[R10]** Ethan Pérez, Florian Strub, Harm de Vries, Vincent Dumoulin, Aaron Courville. **2018**. *FiLM: Visual Reasoning with a General Conditioning Layer*. **AAAI 2018**. [论文/DOI 页面](https://ojs.aaai.org/index.php/AAAI/article/view/11671).
- **[R11]** Zewen Chi, Li Dong, Shaohan Huang, Damai Dai, Shuming Ma, Barun Patra, et al. **2022**. *On the Representation Collapse of Sparse Mixture of Experts*. **NeurIPS 35**. [官方原文](https://proceedings.neurips.cc/paper_files/paper/2022/hash/df4f371f1f89ec8ba5014b3310578048-Abstract.html).
- **[R12]** Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, Felix A. Wichmann. **2020**. *Shortcut Learning in Deep Neural Networks*. **Nature Machine Intelligence**. [DOI 10.1038/s42256-020-00257-z](https://doi.org/10.1038/s42256-020-00257-z).

### Q3：最差组、数据混合、晋升与统计

- **[R13]** Shiori Sagawa, Pang Wei Koh, Tatsunori B. Hashimoto, Percy Liang. **2020**. *Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization*. **ICLR 2020**. [arXiv:1911.08731](https://arxiv.org/abs/1911.08731) · [原会议论文 PDF](https://openreview.net/pdf?id=ryxGuJrFvS).
- **[R14]** Ishaan Gulrajani, David Lopez-Paz. **2021**. *In Search of Lost Domain Generalization*. **ICLR 2021**. [arXiv:2007.01434](https://arxiv.org/abs/2007.01434).
- **[R15]** Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, Percy Liang, et al. **2023**. *DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining*. **NeurIPS 36**. [官方论文](https://papers.neurips.cc/paper_files/paper/2023/hash/dcba6be91359358c2355cd920da3fcbd-Abstract-Conference.html).
- **[R16]** Yuandong Tian, Jerry Ma, Qucheng Gong, Shubho Sengupta, Zhuoyuan Chen, James Pinkerton, Larry Zitnick. **2019**. *ELF OpenGo: An Analysis and Open Reimplementation of AlphaZero*. **ICML**, PMLR 97. [正文](https://proceedings.mlr.press/v97/tian19a.html) · [**附录 Table S1，PDF 第 2 页**](https://proceedings.mlr.press/v97/tian19a/tian19a-supp.pdf).
- **[R17]** Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron C. Courville, Marc G. Bellemare. **2021**. *Deep Reinforcement Learning at the Edge of the Statistical Precipice*. **NeurIPS 34**. [官方论文及补充](https://papers.nips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html) · [rliable 官方代码](https://github.com/google-research/rliable).

### Q4：课程、查询、搜索与再探索

- **[R18]** Minqi Jiang, Edward Grefenstette, Tim Rocktäschel. **2021**. *Prioritized Level Replay*. **ICML**, PMLR 139:4940–4950. [官方论文](https://proceedings.mlr.press/v139/jiang21b.html).
- **[R19]** Minqi Jiang, Michael Dennis, Jack Parker-Holder, Jakob Foerster, Edward Grefenstette, Tim Rocktäschel. **2021/2022**. *Replay-Guided Adversarial Environment Design*. **NeurIPS 2021**；[arXiv:2110.02439](https://arxiv.org/abs/2110.02439)（PLR⊥ 变体）。
- **[R20]** Jack Parker-Holder, Minqi Jiang, Michael Dennis, Mikayel Samvelyan, Jakob Foerster, Edward Grefenstette, Tim Rocktäschel. **2022**. *Evolving Curricula with Regret-Based Environment Design*. **ICML**, PMLR 162:17473–17498. [官方论文](https://proceedings.mlr.press/v162/parker-holder22a.html).
- **[R21]** Michael Dennis, Natasha Jaques, Eugene Vinitsky, Alexandre Bayen, Stuart Russell, Andrew Critch, Sergey Levine. **2020**. *Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design*. **NeurIPS 33**. [官方论文](https://papers.nips.cc/paper_files/paper/2020/hash/985e9a46e10005356bbaf194249f6856-Abstract.html)（PAIRED）。
- **[R22]** Jiakai Zhang, Kyunghyun Cho. **2017**. *Query-Efficient Imitation Learning for End-to-End Simulated Driving*. **AAAI 2017**. [DOI/原论文](https://doi.org/10.1609/aaai.v31i1.10857)（SafeDAgger）。
- **[R23]** Emilien Biré, Anthony Kobanda, Ludovic Denoyer, Rémy Portelas. **2025**. *Efficient Active Imitation Learning with Random Network Distillation*. **ICLR 2025**. [ICLR 正式论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/3f9f3159318fd946e1ddf510e059a3f6-Abstract-Conference.html) · [arXiv:2411.01894](https://arxiv.org/abs/2411.01894).
- **[R24]** David J. Wu. **2020**（arXiv 2019）. *Accelerating Self-Play Learning in Go*. **AAAI 2020**. [原论文 PDF](https://rlg.mlanctot.info/2020/papers/AAAI20-RLG_paper_36.pdf) · [arXiv:1902.10565](https://arxiv.org/abs/1902.10565).
- **[R25]** David J. Wu / KataGo 团队. **官方维护文档，持续更新**. *Other Methods Implemented in KataGo*, **`Policy Surprise Weighting`** 小节；[KataGo 官方 GitHub](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md). **工程方法文档，不是经同行评审的单独论文。**
- **[R26]** KataGo 官方维护者. **版本化工程脚本**. `python/selfplay/synchronous_loop.sh`, `-taper-window-scale`/`-max-train-bucket-per-new-data` 等样例配置. [官方 GitHub](https://github.com/lightvector/KataGo/blob/master/python/selfplay/synchronous_loop.sh). **具体值必须注明代码版本；不能替代整个项目的通用回放窗口说明。**
- **[R27]** Tim Salimans, Richard Chen. **2018**. *Learning Montezuma's Revenge from a Single Demonstration*. [arXiv:1812.03381](https://arxiv.org/abs/1812.03381) · [作者机构一手技术说明](https://openai.com/index/learning-montezumas-revenge-from-a-single-demonstration/).
- **[R28]** Adrien Ecoffet, Joost Huizinga, Joel Lehman, Kenneth O. Stanley, Jeff Clune. **2021**. *First Return, Then Explore*. **Nature** 590, 580–586. [DOI 10.1038/s41586-020-03157-9](https://doi.org/10.1038/s41586-020-03157-9) · [作者原文 PDF](https://adrien.ecoffet.com/files/go-explore-nature.pdf).

### Q5：卡牌游戏中的新配置

- **[R29]** Wannian Xia, Yiming Yang, Jingqing Ruan, Dengpeng Xing, Bo Xu. **2023**. *Cardsformer: Grounding Language to Learn a Generalizable Policy in Hearthstone*. **ECAI 2023**, DOI **10.3233/FAIA230581**. [原出版论文 PDF](https://ebooks.iospress.nl/pdf/doi/10.3233/FAIA230581) · [出版社文章](https://journals.sagepub.com/doi/10.3233/FAIA230581). **数字见表 1：未见卡数 0/10/20/30，1,000 场/条件。**
- **[R30]** Timo Bertram, Johannes Fürnkranz, Martin Müller. **2024**. *Learning With Generalised Card Representations for “Magic: The Gathering”*. **IEEE Conference on Games 2024**. [arXiv:2407.05879](https://arxiv.org/abs/2407.05879) · [作者论文 PDF](https://sites.ualberta.ca/~mmueller/ps/2024/COG24_Magic_Drafting.pdf).

### Q6：历史事实原文

- **[R31]** David Silver, Aja Huang, Chris J. Maddison, Arthur Guez, Laurent Sifre, et al. **2016**. *Mastering the game of Go with deep neural networks and tree search*. **Nature** 529, 484–489. DOI **10.1038/nature16961**. [DeepMind 原论文 PDF（**PDF 第 2 页 / 印刷 p.485**：24.2%、2µs、57%、3ms）](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf).
- **[R32]** David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Arthur Guez, et al. **2017**. *Mastering the Game of Go without Human Knowledge*. **Nature** 550, 354–359. DOI [10.1038/nature24270](https://doi.org/10.1038/nature24270)（AlphaGo Zero）；**500,000 局亦可对照 [R16] 补表**。
- **[R33]** David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, et al. **2018**. *A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play*. **Science** 362:1140–1144. DOI [10.1126/science.aar6404](https://doi.org/10.1126/science.aar6404)（AlphaZero）。
- **[R34]** Gian-Carlo Pascutto、Leela Zero 开源开发者社区. **持续维护的项目**. *Leela Zero: Go engine with no human-provided knowledge*. [Leela Zero 官方仓库](https://github.com/leela-zero/leela-zero). **此处未验证统一的精确回放 K，不应引用不存在的单一参数。**
- **[R35]** Melvin Johnson, Mike Schuster, Quoc V. Le, Maxim Krikun, Yonghui Wu, Zhifeng Chen, Nikhil Thorat, et al. **2017**. *Google's Multilingual Neural Machine Translation System: Enabling Zero-Shot Translation*. **Transactions of the Association for Computational Linguistics 5**. [ACL Anthology Q17-1024 原文](https://aclanthology.org/Q17-1024/). **仅核对论文年份/归属；不声称此原文证明 τ=5。**

### Q7：进阶搜索、RL、世界模型、部署

- **[R36]** Ivo Danihelka, Arthur Guez, Julian Schrittwieser, David Silver. **2022**. *Policy Improvement by Planning with Gumbel*. **ICLR 2022**. [ICLR 原论文 PDF](https://openreview.net/pdf?id=bERaNdoegnO).
- **[R37]** Todd Hester, Matej Vecerik, Olivier Pietquin, Marc Lanctot, Tom Schaul, et al. **2018**. *Deep Q-learning From Demonstrations*. **AAAI 2018**. DOI [10.1609/aaai.v32i1.11757](https://doi.org/10.1609/aaai.v32i1.11757).
- **[R38]** Ilya Kostrikov, Ashvin Nair, Sergey Levine. **2022**（预印本 2021）. *Offline Reinforcement Learning with Implicit Q-Learning*. **ICLR 2022**. [arXiv:2110.06169](https://arxiv.org/abs/2110.06169).
- **[R39]** Shengjie Wang, Shaohuai Liu, Weirui Ye, Jiacheng You, Yang Gao. **2024**. *EfficientZero V2: Mastering Discrete and Continuous Control with Limited Data*. **ICML**, PMLR 235:51041–51062. [官方论文](https://proceedings.mlr.press/v235/wang24at.html).
- **[R40]** Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, et al. **2020**. *Mastering Atari, Go, chess and shogi by planning with a learned model*. **Nature** 588, 604–609. DOI [10.1038/s41586-020-03051-4](https://doi.org/10.1038/s41586-020-03051-4)（MuZero）。
- **[R41]** Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap. **2025**. *Mastering diverse control tasks through world models*. **Nature** 640, 647–653. DOI [10.1038/s41586-025-08744-2](https://doi.org/10.1038/s41586-025-08744-2)（DreamerV3）。
- **[R42]** Vincent Micheli, Eloi Alonso, François Fleuret. **2022/2023**. *Transformers are Sample-Efficient World Models*. **ICLR 2023 / arXiv 2022**. [arXiv:2209.00588](https://arxiv.org/abs/2209.00588)（IRIS）。
- **[R43]** Shibhansh Dohare, J. Fernando Hernandez-Garcia, Qingfeng Lan, Parash Rahman, A. Rupam Mahmood, Richard S. Sutton. **2024**. *Loss of plasticity in deep continual learning*. **Nature** 632, 768–774. DOI [10.1038/s41586-024-07711-7](https://doi.org/10.1038/s41586-024-07711-7).
- **[R44]** Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, Dmitry Kalenichenko. **2018**. *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*. **CVPR 2018**, 2704–2713. [CVF 官方论文](https://openaccess.thecvf.com/content_cvpr_2018/html/Jacob_Quantization_and_Training_CVPR_2018_paper.html).
- **[R45]** Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Misha Laskin, Pieter Abbeel, Aravind Srinivas, Igor Mordatch. **2021**. *Decision Transformer: Reinforcement Learning via Sequence Modeling*. **NeurIPS 34**. [官方原文](https://papers.neurips.cc/paper/2021/hash/7f489f642a0ddb10272b5c31057f0663-Abstract.html).
- **[R46]** Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Mohammadamin Barekatain, Simon Schmitt, David Silver. **2021**. *Learning and Planning in Complex Action Spaces*. **ICML**, PMLR 139:4476–4486. [官方论文](https://proceedings.mlr.press/v139/hubert21a.html)（Sampled MuZero；**本项目三头组合动作若上 Gumbel/MuZero 应一并阅读**）。

### 项目附件（并非同行评审论文；内部证据需注明版本与实验条件）

- **[P1]** `attachments/task160-a-history.md`，*task160-A：仓库历史综述（架构/输入、数据/配方/课程/采集）*，2026-10-29。按编号查实验来源、模型/动作架构演变和早期任务号；其历史记录**不能覆盖后面 P3 的新增结果**。
- **[P2]** `attachments/task160-b-literature.md`，*task160-B：历史文献综述*；特别是**§12“未核实/二手/外推的说法”**为本报告 **Q6 的逐条审计母表**，不是可直接当作一手事实的来源。
- **[P3]** `attachments/task160-decision-memo.md`，*项目架构/采集/门禁决策备忘*，截至 **2026-10-30 §11.13**。直接核对 **§11.2（未见卡组留出）、§11.7（C++ 推理）、§11.8（E4 队列）、§11.9（E3 等曝光/真打）、§11.10（E1）、§11.11（E13）、§11.12（E8/部署）、§11.13（r52 门禁与 E4 预取）**。
- **[P4]** `attachments/student-architecture.md`，*学生/价值网架构的远期升级：个体 + 注意力*，2026-10-26、后续补录至 10-29。重点为 1:132 根/rollout 前向比例、旧 PPO 实体网络的历史教训、L5/L45 观测输入限制与不允许手写物理路径/策略规则的要求。
- **[P5]** `attachments/teacher.md`，*教师行为与搜索架构说明*。用于解释根节点、候选搜索和轻量续走，不包含可对抗一手文献的独立实验结果。

**终局判断**：对当前项目，**最有胜率/成本杠杆的工作不是泛泛地“换个更大的模型”，而是让**（1）观察真正包含可学信息，（2）教师在陌生组合上能可靠发现胜局，（3）训练从可靠、覆盖广的状态学习，（4）晋升由独立真打和公平计算预算决定。**文献支持上述实验原则；具体结构、权重与搜索策略是否增益只能由已锁定的本地对照给出。**
