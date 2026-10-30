

# task175：联网文献调研报告（Q1–Q7）

- 日期：2026-10-29
- 角色：在 task160-B 与后续实测（E1/E3/E13/E8、task23、task129、task161）之上核实、纠正、补充。不重复已经由本地配对实验闭合的结论，只在文献能改变解释或优先级时展开。
- 证据分级：正文中的数字均来自本次打开的论文 PDF、官方技术报告或官方仓库文档。博客/二手综述单独标注。查不到的不补。

---

## 1. 一页结论

| 问题 | 建议 | 依据（最短） | 置信度 |
|---|---|---|---|
| Q1 实体/Transformer 相对 CNN 的数据效率 | **不要指望 2e7 行上注意力更省数据。** v3 只为关系偏置（保龄球/蹦极/卡序）服务，且只放根节点；必须赢过等参宽 CNN，分「未见卡组」和「空间关」。 | 10M–30M 行的同任务对照**查不到**。Tuyls 等 2024 的 BC 缩放律在 NetHack 上用到约 55B 样本、2 亿参数仍够不到专家；Procgen 上反而是**更大的卷积**更泛化。 | 中高（“没有交叉点文献”）；中（“v3 在空间关上是否值”） |
| Q2 MoE / FiLM / 加宽对 OOD | **MoE 维持关闭。** 加宽伤未见卡组，文献机制是伪相关下的过参数化（Sagawa 2020），不是“Transformer 的数据门槛”。FiLM 只对**训练时就已知的地形**有弱正证据，对未见卡组没有。不要用文献把 E1/E13 再打开。 | Soft MoE 在 Atari 上是分布内缩放（8 专家约 +20%），要过全部专家。CARE 里 FiLM 吃掉多任务增益的大头，但是 50 个**见过的**机器人任务。Sagawa：多数组伪相关时，加宽改善平均、恶化最差组；他们建议的“下采样多数组”在我们 E1/E13 上已经伤了陌生卡组。 | 高（不上 MoE）；中（FiLM 以后再测） |
| Q3 挑选用平均还是最差组 | **挑选不要改成纯最差地形。** 门禁继续做“显著退步才否决”；挑选去预测能不能过这道门（task136 的方向）。 | Gulrajani & Lopez-Paz 2021：选择协议是方法的一部分；留一域比池化验证更悲观。DoReMi 优化的是**训练配比的最差域超额损失**，不是在已训好的候选里做 minimax，而且作者结果报的是平均下游 +6.5 点。AlphaGo Zero 的晋升是全局 55% 胜率，不是按开局分组。 | 高 |
| Q4 用师生分歧驱动采集 | **行权上的 spread 已经在做“分歧加权”。** 值得试的是**格子配额**：教师还不会、且学生更不会的格子；教师胜率已经很高的示范格应退场。不要因 KataGo 的近期窗口去删历史行。 | PLR 用 \|GAE\| 而不是动作分歧。Salimans & Chen 2018 只在后缀还没稳定超过示范时才把起点往前移。KataGo 窗口是**样本** 25 万→约 2200 万、窗内均匀，且那是自博弈策略在变；我们 E1 删历史伤泛化。 | 中高（机制）；中（我们身上的效应量） |
| Q5 未见卡组 | **缺的是“教师能赢的新卡组行”，不是再分配，也不是卡牌语言模型。** 植物集合是封闭的，identity 头已经是对的归纳偏置。 | Procgen：500 关训练、泛化缺口要到约 1 万关才合上；但那里环境会给奖励。我们教师在 54 套未见卡组上只有 14.6%，多采的若全是输局，克隆的是输法。MTG 草稿 55% 是“未见**卡**”的拣选，不是“已知卡的未见组合”的对局策略。 | 高 |
| Q6 二手说法 | 见第 7 节逐条。主要纠正：50 万局窗口是 **AlphaGo Zero** 不是 AlphaZero；温度 τ=5 **不是** Johnson 2017，而是 Arivazhagan 2019，且指数是 \(1/T\) 不是 \(>1\)；AlphaGo 快速走子 24.2% / 2 µs **已核实**；AlphaStar **1.39 亿**权重已在 Nature 正文核实。 | 原文位置见 Q6 表 | 高（已逐条打开的）；其余标“查不到” |
| Q7 长期 | 近期不引入新学习算法。半年内只考虑：根节点少模拟的策略改进（Gumbel）、教师目标剪枝的小审计、把“超过人类”的评测协议写下来。**不要用学到的世界模型替换引擎。** | 引擎一局模拟约 35 s、学生推理 2.7 s；MuZero/IRIS 的前提是没有或很贵的模拟器。 | 中高 |

一句话：文献没有给“现在换大结构或换最差地形选拔”提供同规模正面证据；它解释了加宽为什么伤泛化，并支持“多样性必须采出来、而且教师得能赢”。这和 E1/E3/E13 的本地否决同向，不构成翻案。

---

## 2. Q1. 千万级模仿学习里，实体/Transformer 相对 CNN 的数据效率

### 2.1 文献

| 文献 | 设置 | 关键数字 | 和我们的相似度 |
|---|---|---|---|
| Dosovitskiy et al. 2021, *An Image is Worth 16x16 Words*, ICLR, [arXiv:2010.11929](https://arxiv.org/abs/2010.11929) | 图像分类，不是决策 | ImageNet-1k（约 1.3M）上 ViT 不如同规模 ResNet；JFT-300M 上才超过 | **低**。监督是类别，不是动作；规模差两个数量级。只说明“无偏置注意力吃数据”，不能外推交叉点。 |
| Tuyls et al., *Scaling Laws for Imitation Learning in Single-Agent Games*, TMLR 2024-12, [arXiv:2307.09423](https://arxiv.org/abs/2307.09423) | Atari：CNN 行为克隆，约 **1B** 样本，参数 1k–5M。NetHack：Transformer BC，NLD-AA 约 **3B** 样本，扩展到约 **55B**，参数 200k–200M | Atari 上损失/回报的幂律**一直到专家**。NetHack 专家分约 17k，论文写明还要更多 FLOPs 才到得了；最佳离线模型 Human Monk **7784**，次佳离线方法 **4504**（约 1.7 倍，摘要亦写 1.5 倍量级）。**没有** Transformer 对 CNN 的对照 | **中**。同是单智能体 BC、有程序生成和长时程；但 NetHack 是文本网格、数据是十亿级、教师是脚本 AutoAscend。对我们的用途是：BC 可以有缩放律，**覆盖不到的行为不会被参数补上**。 |
| Vinyals et al., *Grandmaster level in StarCraft II*, Nature 575:350–354, 2019. [doi:10.1038/s41586-019-1724-z](https://doi.org/10.1038/s41586-019-1724-z) | 实体 Transformer + 小地图 ResNet + scatter + 指针 + LSTM；先模仿再联赛 RL | 正文：AlphaStar 有 **1.39 亿**权重，推理只需 **5500 万**。监督学习用 **971,000** 局回放（MMR>3500，约顶尖 22%），再用 MMR>6200 的 **16,000** 局胜局微调。每个智能体 **32** 块三代 TPU、**44** 天。Fig. 3f 监督学习（神族对全体）打内置精英电脑的胜率，按柱读数为：基线 0%，加动作延迟 7%，加指针网络 36%，加 Transformer 71%，加 scatter 87%（累积消融，不是各自独立增量） | **中**（表示法）/ **低**（规模与算力）。可迁移的是“实体一路 + 空间一路 + 指针”，不是“所以 18 万参数该上注意力”。Fig. 3f 是**大数据监督学习**上的架构消融，不是小数据交叉点。 |
| Zambaldi et al., *Relational Deep RL*, ICLR 2019, [arXiv:1806.01830](https://arxiv.org/abs/1806.01830) | SC2LE 迷你游戏，A2C；关系自注意力 vs 去掉该模块的对照，**两边都很大** | 全动作集均值（30 局）：关系模块在 6/7 个迷你游戏上不低于对照。例：CollectMineralShards 196 vs 187；DefeatRoaches 62 vs 61；DefeatZerglingsAndBanelings **736 vs 602**。OpenReview 版对后三个报告相对提升 9、8、134，标准误 1.5、6.4、18.2。四个迷你游戏超过人类宗师 | **中低**。有实体关系、有星际，但是 RL 不是 BC，没有扫数据量，对照不是“等参 CNN”。收益集中在少数需要关系的任务，其余接近天花板。 |
| Lee et al., *Multi-Game Decision Transformers*, NeurIPS 2022, [arXiv:2205.15241](https://arxiv.org/abs/2205.15241) | 附录 D：同一 UDRL 配方，主干换成 Impala CNN | 论文称 Transformer 相对 Impala 有明显优势；未给出“小数据 CNN 更好、大数据反超”的交叉样本量 | **低**。像素、多游戏、DQN 回放大、动作是离散 Atari。不能当成 2e7 行网格 BC 的交叉点。 |
| Cobbe et al., *Procgen Benchmark*, ICML 2020, [arXiv:1912.01588](https://arxiv.org/abs/1912.01588) | PPO，程序生成关卡 | “更大的架构显著提高样本效率和泛化”；Nature-CNN 几乎训不动。这是**卷积加宽有利于泛化**，方向与 task23 相反 | **中**（泛化协议）/ 对 Q1 的架构结论是**反例**，见下。 |
| Pearce et al. 风格的 *Scaling Laws for Pre-training Agents and World Models*, [arXiv:2411.04434](https://arxiv.org/abs/2411.04434) | BC-Token（Transformer）vs BC-CNN | 同参数量下 token 模型每个动作吃进几百个 token，算力大约差两个数量级，损失平台来得更晚。这是**算力口径**不是样本效率口径 | **低**。提醒：把“参数量相同”当成“对照公平”会漏掉前向算力。我们的公平对照必须同时报参数、CPU 微秒和固定优化步数。 |

### 2.2 有没有“小数据 CNN 更好、大数据注意力反超”的交叉点？

**在游戏或决策的行为克隆里，本次没有找到同数据量、同任务、扫过 1e7–3e7 样本的实体/Transformer 对 CNN 交叉曲线。** ViT 的交叉发生在图像分类、约 1e8–3e8 张图，不能写成我们的结论。

能说的只有三句，都带边界：

1. **无关系偏置的纯注意力吃数据**（ViT）。我们的 v3 不是纯 ViT，它带种类对、行差、列差、同格偏置，更接近 Zambaldi / AlphaStar 的关系模块。这类偏置的论文收益出现在“空间关系决定成败”的任务上，而不是“同样的格子 CNN 只要数据够就输给注意力”。
2. **BC 的缩放律到不了行为覆盖之外**（Tuyls：NetHack 55B 样本仍远离 17k 专家分）。这和“老师在 54 套未见卡组上 14.6%”是同一类上限：参数和注意力都补不了教师没走过的策略。
3. **网格泛化上，已发表的正面证据是把卷积做大，不是换成注意力**（Procgen）。我们 task23 加宽到 128 却让留出卡组 60.0%→53.8%、夜间 51.4%→42.6%。两件事不矛盾：Procgen 的分布外是“同一生成过程的新关”，奖励来自环境；我们的分布外是“训练行被一两套卡组主导之后的新组合”，加宽更容易记住主导卡组（见 Q2 的 Sagawa）。所以 task23 否决的是**在这份语料上加宽**，不是“卷积家族在泛化上总是差”。

task153 修订后实体注意力 91.0% 对等参 CNN 91.3%（152 局、仅白天）与上面一致：这个数据量下结构不是杠杆。富输入 + 生产规模 + 空间关的四臂（E4）仍然是唯一能回答“关系偏置值不值”的实验，文献不能代替它。

**对我们：** 维持“根节点才允许 v3、续走 CNN、必须赢过等参宽 CNN、未见卡组与空间关分开报”。不要把 AlphaStar Fig. 3f 的 87% 读成“该上 Transformer”。

**置信度：** 中高（交叉点文献不存在）；中（v3 在 L5/L45 上是否因关系偏置赢过 CNN——文献只有类比，没有测量）。

---

## 3. Q2. 小规模 BC/RL 里，MoE 与条件化对没见过的配置是帮还是伤？

### 3.1 文献

| 文献 | 设置 | 关键数字 | 相似度 |
|---|---|---|---|
| Obando-Ceron et al., *Mixtures of Experts Unlock Parameter Scaling for Deep RL*, ICML 2024, [arXiv:2402.08609](https://arxiv.org/abs/2402.08609) | DQN/Rainbow 倒数第二层换 MoE；Atari | 论文的出发点就是：**RL 里加大 dense 参数常常伤最终表现**。Soft MoE（所有 token 对所有专家加权，不是稀疏激活）能随专家数上升。Rainbow、Impala、8 专家：专家数 1→8 约 **+20%**。随机路由矩阵不如学出来的。60 个游戏上 Soft MoE 仍高于基线 | **中低**。分布**内**的价值学习缩放，不是未见卡组；推理要过所有专家，和 CPU 续走冲突。它支持“不要用 Top-1 稀疏门控指望缩放”，不支持“上 MoE 修 OOD”。 |
| Sodhani, Zhang, Pineau, *Multi-Task RL with Context-based Representations* (CARE = Contextual Attention-based Representation learning), ICML 2021, [arXiv:2102.06177](https://arxiv.org/abs/2102.06177) | Meta-World MT10/MT50，任务元数据（语言）→ 上下文，注意力混合多个编码器。对照含 FiLM | 每环境 2M 步成功率：MT10 上 CARE **0.84±0.051**，多任务 SAC **0.49±0.073**，PCGrad **0.72±0.022**，FiLM **0.75±0.037**。MT50：CARE **0.54±0.031**，SAC **0.36±0.013**，PCGrad **0.50±0.017**，FiLM **0.40±0.012**。100k 步时 MT10：CARE 0.36、SAC 0.13、FiLM 0.27 | **中**（已知离散上下文）/ **低**（未见配置）。FiLM 已经拿走大部分增益，混合编码器是增量。这些任务**都在训练集里**。地形是这种“已知上下文”；未见卡组不是。 |
| Sagawa, Raghunathan, Koh, Liang, *An Investigation of Why Overparameterization Exacerbates Spurious Correlations*, ICML 2020, [arXiv:2005.04345](https://arxiv.org/abs/2005.04345) | 多数组里伪特征比核心特征更预测标签 | 加宽**降低平均测试误差、提高最差组误差**。CelebA ResNet-10：宽度 1 的最差组测试误差 **25.6%**（训练误差还有 12.4%）；零训练误差的过参数模型最差组测试误差到约 **60%**。Waterbirds：90 个随机特征的最差组测试误差 **26.6%**，过参数模型最好也有 **42.4%**。上调少数组权重在过参数区失败；他们有效的补救是**下采样多数组** | **高**（机制）/ **中**（补救能不能照搬）。机制直接对上 task23：白天变好、夜间和留出卡组变差。补救不要照搬，见下。 |
| Hendawy et al., MOORE, ICLR 2024, [arXiv:2311.11385](https://arxiv.org/abs/2311.11385)；Yang et al., Soft Modularization, NeurIPS 2020, [arXiv:2003.13661](https://arxiv.org/abs/2003.13661)；Li et al., GMoE, ICLR 2023, [arXiv:2206.04046](https://arxiv.org/abs/2206.04046) | 多任务 RL 或视觉域泛化 | 本次只重读了摘要级结论：正结果依赖任务身份或“专家与潜在域对齐”，且规模不小。没有一篇是 2e7 行、5 个地形、CPU 前向的 BC | **低–中**。不能当作“MoE 改善卡组 OOD”的证据。GMoE 是视觉域，不是决策。 |

### 3.2 和本地证据怎么拼

- **分专家伤陌生卡组**（E3：联合在随机卡组探针上高 4–20 个百分点，专家优势在同曝光下基本消失）与 CARE 的“共享 + 上下文 > 拆开”同向，也和“每个专家分到的行更少”同向。文献没有给出翻案所需的反例。
- **加宽伤 OOD** 的最接近机制是 Sagawa，不是 ViT 的数据门槛。主导卡组占行 45% 量级时，宽网络有能力把“这套卡的白天习惯”记进权重，平均变好、少数组变差。Procgen 里加宽**帮助**泛化，是因为那里没有这样一个可记忆的伪特征，分布外仍由同一生成过程抽出来。两篇文献都不要单独拿来指挥下一轮。
- Sagawa 的补救（下采样多数组）在图像伪相关上有效，在我们这里 **E1（历史降权/减半）和 E13（按卡组封顶/温度）已经判负**：陌生卡组 −0.9 到 −12 个百分点。差别在于：CelebA 的少数组里核心特征是在场的，下采样只是逼模型去用它；我们的屋顶和随机卡组是**绝对行数不够，而且教师经常不会赢**。下采样删掉的是目前唯一大量、标签还算能学的行，不是把缺失的组合变出来。

**对我们：** MoE、地形硬路由、再分配，都不因这篇调研重新打开。FiLM（按地形的 γ、β）只保留为“如果将来在**新卡组覆盖上来之后**仍测到地形干扰”的便宜臂，现在不排期。条件化解决不了倍率外推（本地：告诉学生倍率 2.0/1.7/1.5 得到 57/126/65 胜每 256 局）——没见过的连续值没有专家可路由。

**置信度：** 高（维持不上 MoE）；中（Sagawa 机制解释 task23）；中低（FiLM 以后会有正收益——只有类比，没有我们规模的测量）。

---

## 4. Q3. 候选该按平均还是按最差组挑？

### 4.1 文献

| 文献 | 选择/优化的对象 | 关键结论 | 相似度 |
|---|---|---|---|
| Gulrajani & Lopez-Paz, *In Search of Lost Domain Generalization*, ICLR 2021, [arXiv:2007.01434](https://arxiv.org/abs/2007.01434) | 域泛化的**模型选择**。三种协议：训练域验证集池化（接近平均）、留一域交叉验证、测试域上的 oracle | 没有选择协议的域泛化算法是不完整的。DomainBed 上，协议固定之后 ERM 不输给专门的域泛化算法。留一域比池化更难、oracle 用了测试域所以不能当方法的一部分 | **高**（“选拔口径就是方法的一部分”）/ **中**（他们的域是图像风格，不是胜率）。直接含义：用平均胜率选拔、用最差地形否决，是**两个不同的算法**。task136 把选拔改成“预计能否过门禁”是在修补这个裂缝，不是可选的整洁化。 |
| Xie et al., *DoReMi*, NeurIPS 2023, [arXiv:2305.10429](https://arxiv.org/abs/2305.10429)；作者博客 [crfm.stanford.edu/2023/09/14/doremi](https://crfm.stanford.edu/2023/09/14/doremi.html) | 用 280M 代理 + group DRO 学**域权重**，再训 8B | The Pile 上平均 one-shot 准确率 **+6.5** 个百分点，达到基线准确率快 **2.6 倍**。目标函数是最差域的超额损失，**报告的成功指标仍是平均**。二手笔记里的“ArXiv 被下调、网页被上调”本次不当作已核实数字 | **中**（最差域思想）/ **低**（语言模型、代理比目标小 30 倍、域是语料桶不是地形）。它**不是**“在 4 个已训好的学生里改用 minimax 打分”的证据，也不是“最小的地形应该加权”的证据。 |
| Sagawa et al., *Distributionally Robust Neural Networks for Group Shifts*, ICLR 2020, [arXiv:1911.08731](https://arxiv.org/abs/1911.08731)（与 2020b 过参数化论文是同一组工作的两篇；本次 2020b 读了正文数字，group DRO 原论文沿用其公开结论：最差组目标在组定义正确且每组有足够样本时改善最差组） | 训练目标，不是选拔 | 组必须定义对。组太碎或组内几乎全是噪声时，最差组目标会去拟合噪声组 | **中**。屋顶真打 0/128 的地板、雾的高方差，正是“噪声组”。 |
| Silver et al., AlphaGo Zero, Nature 2017, [doi:10.1038/nature24270](https://doi.org/10.1038/nature24270) | 自博弈里的**版本晋升** | “新棋手若以 55% 的优势获胜才成为最佳棋手，并用于此后的自博弈。”这是**全局**胜率门槛，用来避免噪声，不是按开局或布局类型做最差组。AlphaZero（[arXiv:1712.01815](https://arxiv.org/abs/1712.01815)）改成单一网络持续更新，**取消**了这道 55% 迭代门 | **中**。同是“搜索教师 + 网络学生”的晋升。他们用一个总体非劣/优势门槛，不用分域 minimax。我们门禁的 Newcombe 下界（总体不能显著差）比 55% 更松，但精神相近；分区 Holm 否决是我们多出来的约束。 |
| Vinyals et al. 2019, Fig. 3a | AlphaStar 联盟 | 联盟组成消融：FSP、PFSP、SP、PFSP+SP 的测试 Elo 与对历史的最低胜率不同。PFSP 是按**对历史的胜率**优先匹配，不是按地图或种族的最差组挑 checkpoint | **低**。说明多对手评估有用，不说明选拔函数该改成 min_terrain。 |
| Wu, KataGo, [arXiv:1902.10565](https://arxiv.org/abs/1902.10565) | 强度用 Elo，消融表主运行 1329 Elo | 没有多地形门禁。样本窗口见 Q6 | **低**（选拔）/ **中**（数据窗口，见 Q4、Q6）。 |

### 4.2 对我们意味着什么

把挑选从“平均胜率”改成“最差地形优先”**没有**被这些文献支持到可以做的程度，风险写得清楚：

1. **口径必须和部署一致**（Gulrajani）。部署标准已经是“总体不显著退步，且没有地形显著退步”。选拔去近似这个标准是对的；改成 min(地形胜率) 会把地板效应的屋顶（0/128 分不出模型）和雾的抽样噪声变成选拔器。
2. **最差组优化不是免费的鲁棒**。DoReMi 在语料上有效，是因为代理模型能估计“这个域再加数据还降得了损失”。我们 B 配方（按关卡平衡）多轮没被挑中；E13 按卡组把权重从主导卡组挪走之后，屋顶 cell 准确率只动了约 +1 个百分点，其余地形和陌生卡组全线下降。文献里的 group DRO 不能为再开 B 配方背书。
3. **AlphaGo Zero 的 55%** 提醒的是另一件事：晋升门槛要高于噪声，而不是要覆盖每一个子集。我们 64 对时同等强度候选只有 8–10% 通过率，已经是门槛偏紧；再把选拔改成最差地形，会把假拒再放大。

**建议：** 维持门禁 v5/v6 的否决式最差组（显著退步才杀），挑选继续按“预计门禁结果”而不是纯平均、也不是纯 min。不要为了贴近 DoReMi 的名字去改。

**置信度：** 高。

---

## 5. Q4. 用师生分歧驱动课程和采集

### 5.1 文献里的“regret”其实不是同一种东西

| 文献 | regret / 分歧怎么估计 | 关键数字 | 相似度 |
|---|---|---|---|
| Jiang, Grefenstette, Rocktäschel, *Prioritized Level Replay*, ICML 2021, [arXiv:2010.03934](https://arxiv.org/abs/2010.03934) | 关卡分数 = 轨迹上 GAE 绝对值的平均，即价值网还不准的地方。不是师生动作差 | 与此前最强方法合用，Procgen 测试回报相对标准 RL 提升 **超过 76%**（是组合系统，不是 PLR 单独）。单独使用亦提高样本效率并形成由易到难的课程 | **中**。关卡 = 我们的格子。但分数依赖价值网；v6 已被否决、v5 冻结，用陈旧价值的 \|TD\| 当采集分，可能只是在追价值网的偏置。 |
| Parker-Holder et al., *ACCEL*, ICML 2022, [arXiv:2203.01302](https://arxiv.org/abs/2203.01302)；Dennis et al., *PAIRED*, ICML 2020 | ACCEL 在高 regret 关卡上做编辑。PAIRED 的 regret ≈ 对抗智能体回报 − 主角回报，要额外训练对手 | 本次未重抽他们的表。机制上：没有“更强的教师”时，只好用学习到的对手来估计潜力 | **中低**。我们有搜索教师，可以直接用“教师胜、学生不胜”或根节点推翻，比再训一个对抗僵尸生成器便宜，也更可解释。不建议先上 PAIRED。 |
| Zhang & Cho, SafeDAgger, 2016, [arXiv:1605.06450](https://arxiv.org/abs/1605.06450)；Menda et al., *EnsembleDAgger*, 2019, [arXiv:1807.08364](https://arxiv.org/abs/1807.08364) | SafeDAgger：新手与专家动作差超过阈值才把控制权交给新手（查询专家）。EnsembleDAgger：用集成方差当“没把握”，对照就是这条 discrepancy 规则 | 机器人倒立摆与 HalfCheetah。没有游戏胜率表可直接搬 | **中**（查询规则）/ **低**（安全约束的形式）。我们不缺专家查询——搜索教师每步都在；缺的是**哪些局值得开**。discrepancy 的可操作版本就是已有的 spread（候选价值 max−min）和根节点推翻。 |
| Ross et al., DAgger, AISTATS 2011, [arXiv:1011.0686](https://arxiv.org/abs/1011.0686)；Anthony et al., Expert Iteration, NeurIPS 2017, [arXiv:1705.08439](https://arxiv.org/abs/1705.08439) | DAgger 聚合所有轮次，理论是 no-regret，**没有**按时间丢数据。ExIt 用搜索改进后的动作当目标，也不筛选“只留分歧状态” | 与本地教训一致：只留纠正行 −41.7 个百分点 | **高**。支持继续聚合；反对把历史 DAgger 整段删掉。 |
| Wu, KataGo, arXiv:1902.10565v5 | **策略目标剪枝**删的是根节点上“被迫探索”的访问，不是选哪些局面来下。playout cap：约 25% 的步做满搜索（起初 600/100，两天后 1000/200），只有满搜索的步写入训练。消融：去掉剪枝 Elo 1276 vs 1329（因子 1.25×）；去掉 cap 随机化 1242（1.37×） | 自博弈、策略在变，所以用增长的近期窗口（Q6） | **中**。剪枝可迁移到“不要把 Dirichlet/强制访问写进学生目标”。cap 随机化可迁移到“不是每步都做深搜索”。它们不决定采集哪些关。 |
| Ecoffet et al., Go-Explore, Nature 2021；Salimans & Chen, [arXiv:1812.03381](https://arxiv.org/abs/1812.03381)，NeurIPS 2018 Deep RL Workshop | 从示范/存档状态开局，而不是模仿示范动作。Salimans：先从示范末尾开局；**后缀上至少 20% 的回合打平或超过示范者**，才把起点往回移 | Montezuma 最高分 **74,500**；他们的评估口径下平均分 10,000，动作噪声 0.01 时平均 8,400（OpenAI 官方博客与论文摘要） | **高**（起点课程）/ 配额数字**不可照搬**。20% 规则的方向和我们的现状相反：屋顶示范格教师胜率约 88.7%，早该停止把预算留在那里。 |

### 5.2 对我们

已经在做的，不要再当新想法：行级 spread 加权就是一种“教师有分歧才加重”的标签权重。E8 显示等价动作硬冲突只有 0.05%，所以上 top-k 软标签没有文献以外的理由，也没有本地理由。

文献真正还没被本地做掉的只有两件，且都应小步：

1. **格子配额用“教师相对学生的胜率差”，并设退场条件。** 分数高 = 教师还能赢、学生经常不赢。教师已 ≥80% 或 ≥90% 的格子降配额（对齐 Salimans 的 20% 前进规则，方向反过来用）。每地形设下限，避免分数只挑白天。判据沿用已部署的 E10′：非示范格教师胜率不低于 70%，未见卡组真打有增益。文献支持到“机制”，效应量没有同规模数字，置信度中。
2. **教师目标剪枝做一次审计再决定写不写进打包。** 先统计根节点访问分布里，非argmax 的质量有多少是探索噪声。若质量小，KataGo 的 1.25× Elo 因子不会在我们这里出现——他们的噪声来自 Dirichlet 和强制 playout，我们的标签噪声结构不同（约六成推翻是价值噪声，等价动作矛盾 0.05%）。不要把 50×（相对 ELF 的约 74 GPU 年，KataGo 约 1.4 GPU 年）当成“加上剪枝就加速 50 倍”。

不要做的：按 AlphaGo Zero 的 50 万局窗口丢历史（那是自博弈且策略已换代；E1 已证删历史伤陌生卡组）；不要上集成 DAgger（多一个学生集成，CPU 采集更慢，而搜索教师已经给了比集成方差更强的信号）。

**置信度：** 中高（方向）；中（值得花的工程量不大，但增益可能很小）。

---

## 6. Q5. 卡组 / 组合泛化

### 6.1 文献

| 文献 | 任务 | 关键数字 | 相似度 |
|---|---|---|---|
| Cobbe et al., Procgen, ICML 2020, [arXiv:1912.01588](https://arxiv.org/abs/1912.01588)；CoinRun, ICML 2019, [arXiv:1812.02341](https://arxiv.org/abs/1812.02341) | 训练关卡数 → 留出关卡回报 | 建议硬难度训 **500** 关、易难度 **200** 关，在全分布上测。几乎所有环境都会过拟合小训练集；**要把泛化缺口合上，需要多至约 10,000 关**。更大的模型在这里同时提高样本效率和泛化 | **高**（“配置数太少就会过拟合”）/ **中**（奖励来自环境，策略是 PPO 不是 BC）。82% 行来自 32 套默认卡组、随机卡组约 0.5% 行，就是“训练关卡数远小于 500”的同类失败。 |
| Tobin et al., *Domain Randomization*, 2017, [arXiv:1703.06907](https://arxiv.org/abs/1703.06907) | 模拟器参数随机化后迁到真实机械臂 | 随机化的**范围**比“开了随机化”更重要。本次未重抽他们的成功率表 | **中**。支持随机卡组要按地形分层、覆盖会输的组合，而不是只覆盖教师已经会赢的组合。 |
| Bertram, Fürnkranz, Müller, *Learning With Generalised Card Representations for Magic: The Gathering*, 2024, [arXiv:2407.05879](https://arxiv.org/abs/2407.05879) | **草稿拣选**，不是对局。约 1 亿次拣选 | 预训练后在完全没见过的 BRO 系列上预测人类拣选 **55.44%**。Features+Meta+Image 在另一对照里对未见卡约 43%，同时保住已知系列。卡的使用率统计（meta）很有用，但**新系列还没有这些统计** | **低–中**。迁移的是“卡要有可泛化的特征，不能只背卡 ID”。我们的植物约 40 种，是封闭集，identity 头已经共享植物嵌入；不存在“下一张新植物”的开放词表。55% 不能读成“上文本编码器就能打未见卡组”。 |
| Xia, Fang, Guo, Du, Xu, *Integrating LLMs with RL for Generalization in Strategic Card Games*, AAMAS 2025 扩展摘要, [ACM](https://dl.acm.org/doi/10.5555/3709347.3744015) | 炉石传说，T5 编码卡牌文本，自博弈 RL。测试卡组里未见卡比例最高到 50% | 摘要称胜率高于树搜索、GPT-4o 和 Cardsformer。**表 1 的具体胜率本次没从 PDF 文本里抽出，不引用百分比** | **低**。他们的泛化是开放卡池里的新卡文本；我们是已知植物的新组合。用 LLM 动作当标签已被 task104 否掉（泳池 −35 个百分点量级）。 |
| LOCM / ByteRL 一线（如 [arXiv:2404.16689](https://arxiv.org/abs/2404.16689) 对 ByteRL 的可利用性实验） | 程序生成卡的小型 CCG；先 BC 再 RL 微调 | BC 能接近强对手，微调后在数百套卡组上打赢或打平 ByteRL。对手本身会玩 | **中低**。说明“先克隆再强化”在卡牌上可行，但前提是对手/教师在那些卡组上不是 14.6%。我们缺的是这前提。 |

本次**没有**找到：在炉石或杀戮尖塔上，用行为克隆、在千万级状态上，把策略泛化到教师也不会赢的未见卡组，并给出可信胜率的一手论文。查不到就不确定论。

### 6.2 对我们

Procgen 的曲线告诉我们多样性的数量级：几十套训练卡组不够，500 是他们用来**测量**泛化的难度点，合上缺口要到数千至一万。以每局约 177 个监督行、每轮最多几百局计，随机卡组行份额不会在一两轮里从 0.5% 变成“500 套 × 足够的局”。这是采集月尺度的事，不是配方权重能完成的事（E13 已从反方向证过）。

更硬的约束是教师：54 套上 14.6%，以及 task154 里把续走加深后 continuation 从 1/48 到 5/48。Procgen 智能体在新关上仍能从环境奖励学习；我们新卡组上的行，标签来自一个经常全输的教师。全输局面的动作标签分辨率为零（价值网 v5 第一轮的教训：94.5% 的决策所有候选都输，标签是常数）。所以随机卡组配额要和“这套卡上教师胜率 > 0”绑在一起，或者先把搜索加深用在教师现在会输、但局能打到中后段的地形上。只加局数、不看教师能不能赢，是在扩大“怎么输”的数据集。

卡的表示：封闭植物集用身份嵌入 + 集合/指针头（已有 identity 头；AlphaStar 的指针是同一族）即可。不要为了 Bertram 或 Xia 的数字去上卡牌文本编码器。

**置信度：** 高（覆盖不足是主因、再分配解决不了）；中（“教师胜率先抬到某个阈值再扩池”的阈值本身没有文献可抄，14.6% 只说明现在太低）。

---

## 7. Q6. 上一份调研第 12 节逐条核实

上一份把这些标成“未在原文核实”。本次能打开原文或官方 PDF 的，改判如下。

| 说法 | 判决 | 原文 |
|---|---|---|
| AlphaStar 约 1.39 亿参数 | **已核实，并补上推理量** | Nature 2019 架构段：“AlphaStar has 139 million weights, but only 55 million weights are required during inference.” 32 块三代 TPU、44 天，同文。971,000 局（MMR>3500）+ 16,000 局高分胜局微调，同文监督学习段。Fig. 3f 累积胜率见 Q1 表。 |
| AlphaZero 回放窗口约 50 万局 | **归属被纠正** | [arXiv:1712.01815](https://arxiv.org/abs/1712.01815) **没有**回放窗口。它写明与 AlphaGo Zero 的差别之一是：单一网络持续更新，不再等迭代、不再用 55% 门槛换“最佳棋手”。50 万局在 **AlphaGo Zero**（Nature 2017）Methods：“Each mini-batch of data is sampled uniformly at random from all positions of the most recent 500,000 games of self-play.” 同段还有晋升规则：新棋手须以 **55%** 优势取胜才替换最佳棋手。自博弈温度：**前 30 手 τ=1**，此后 τ→0。训练规模（同文，另一处）：3 天约 490 万局；40 天约 2900 万局。 |
| AlphaGo 快速走子 24% / 2 µs，慢策略 57% / 3 ms | **已核实，数字比二手更精确** | Silver et al., Nature 2016。SL 策略网络：KGS **3000 万**局面，留出集 **57.0%**（只用棋盘与历史 55.7%），约 **3 ms**。快速走子策略 \(p_\pi\)：线性 softmax 小模式特征，**24.2%**，**2 µs**。二手里的“约 24%、约 2 µs、约 57%、约 3 ms”可以改成这些确定值。 |
| KataGo 回放窗口与样本权重 | **论文里的窗口已核实；“样本权重公式”要拆开** | Wu, arXiv:1902.10565v5：从最近数据的增长窗口里**均匀**抽。窗口从 **250,000 个样本**增到主运行结束时约 **2200 万**。公式 \(N_{\text{window}}=c\left(1+\beta\frac{(N_{\text{total}}/c)^\alpha-1}{\alpha}\right)\)，\(c=250000\)，\(\alpha=0.75\)，\(\beta=0.4\)。单位是**样本不是局**。损失权重在论文里是辅助目标的系数，不是回放采样权重：对手策略辅助 \(w_{\text{opp}}=0.15\)，其余 ownership/score 系数见论文附录（\(w_o=1.5/b^2\)，pdf/cdf 各 0.02 等）。**没有**按对局结果再加权的那条“样本权重公式”；上一份把两件事混在一起了。另：官方仓库 [KataGoMethods.md](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md)（2022 末实验、2023-03 进主分支，**不是 2019 论文**）写辅助软策略头，温度 **T=4**，名义权重为普通策略目标的 **8 倍**。 |
| Leela Zero 的窗口和采样 | **查不到单一常数** | 随版本变。本次没有打开某一提交的配置并核对。不当作证据。 |
| 温度采样 τ=5，并归于 Johnson et al. | **被推翻（归错论文，指数方向也写反了）** | Johnson et al., TACL 2017 / [arXiv:1611.04558](https://arxiv.org/abs/1611.04558) 只比较两种做法：把各语言对**过采样到与最大者一样大**，或**按原比例混合**。正文**没有**温度 T。温度采样在 Arivazhagan et al., *Massively Multilingual NMT in the Wild*, 2019, [arXiv:1907.05019](https://arxiv.org/abs/1907.05019)：抽到语言对 \(l\) 的概率正比于 \(p_l^{1/T}\)。**T=1** 为真实分布，**T=100** 接近均匀，**T=5** 是他们的平衡设定。En→Any 低资源 25 对的 BLEU：双语 11.72，T=1 为 **6.24**，T=100 为 12.87，**T=5 为 12.75**；高资源：双语 29.34，T=1 为 28.63，T=5 为 **28.03**。上一份写“\(p_i\propto n_i^\alpha\)，\(\alpha>1\) 提高低资源权重”是**反的**：上调低资源对应的是 \(\alpha=1/T<1\)。 |
| CARE 的缩写和细节 | **已核实，缩写比上一份更精确** | 论文标题是 Context-based Representations；方法缩写 CARE 展开为 **Contextual Attention-based Representation learning**。多个编码器 + 元数据（RoBERTa）产生的上下文，对编码器做软注意力。数字见 Q2。不是 MoE 稀疏门控。 |
| “Transformer 在 10M–30M 行不如/等于 CNN” | **维持：查不到，不能当结论** | Q1。上一份的外推判断保留。 |
| “MoE 对 OOD 是帮还是伤” | **仍然没有 RL/BC 的直接测量** | Q2。能核实的是：分布内 Soft MoE 帮缩放；伪相关下加宽伤最差组；我们自己的专家臂伤陌生卡组。不要把 GMoE 的视觉结论写成决策结论。 |
| “等价动作标签噪声”没有直接论文 | **维持查不到** | Implicit BC 等仍只是代理。本地审计 0.05% 比任何代理论文都更相关。 |
| “Data mixing laws 适用于 RL/BC” | **维持未验证** | Ye et al. 2024 是语言模型。本次不把他们的函数形式套到行权上。 |
| “宽 128 伤未知卡组/夜间” | **不是文献，是本地实验** | task23：留出 60.0%→53.8%，夜间 51.4%→42.6%，C++ 慢 3.11×。继续当本地证据，不要写进引用索引充论文。 |
| KataGo 约 50× | **已核实口径** | 相对 ELF 约 74 GPU 年，KataGo 主运行约 **1.4 GPU 年**（19 天、<30 GPU），约 **50 倍**。另写“比 AlphaZero 的 41 TPU 年至少低一个数量级，比 Leela Zero 的多年分布式也至少低一个数量级”。不是单一技巧。 |
| AlphaZero 训练规模（上一份未单列，顺便核对） | **已核实** | 700,000 步、mini-batch 4096；5,000 块一代 TPU 生成自博弈，64 块二代 TPU 训练。棋类约 9 小时 / 4400 万局，将棋约 12 小时 / 2400 万局，围棋约 34 小时 / 2100 万局（论文表；思考时间列是 800 次模拟对应的毫秒预算，不在此复述以免抄错列）。 |

---

## 8. Q7. 值得长期看的技术（12 篇，加路线图）

硬约束重申：不手写策略公式；两台机器；推理以 CPU 为主；引擎是资产（一局模拟约 35 s，学生推理约 2.7 s），不是要被模型替换的对象。

| # | 文献 | 它是什么 | 为什么相关 | 落到哪一环 | 成本与风险 | 优先级 |
|---|---|---|---|---|---|---|
| 1 | Sagawa et al. 2020, arXiv:2005.04345 | 过参数化在伪相关下改善平均、恶化最差组 | 给 task23 一个机制，避免下次把“加宽失败”误读成“容量永远没用” | 只改解释，不改训练。等卡组覆盖上来后再考虑等参加宽 | 无实验成本。风险是把他们的“下采样多数组”再试一遍（已否决） | **现在（当作阅读结论，零实验）** |
| 2 | Gulrajani & Lopez-Paz 2021, arXiv:2007.01434 | 域泛化里选择协议决定结论 | 平均选拔 + 最差组门禁是两个算法 | 维持 task136：选拔分去预测门禁，不改成 min | 零。风险是再改一次口径导致无法和 r20–r36 比较 | **现在** |
| 3 | Cobbe et al. 2020 Procgen；Salimans & Chen 2018 | 训练配置数量；从“还没会的后缀”开局 | 卡组数与示范格配额 | 产线已在做的 E10′/E2′：示范格 ≤25%，且优先教师还没稳赢的格子；随机卡组只保留教师胜率不是零的池 | 采集月。风险：教师 14.6% 的池若不过滤，行都是输局 | **现在（沿已部署开关观察，不新开算法）** |
| 4 | Danihelka, Guez, Schrittwieser, Silver, *Policy improvement by planning with Gumbel*, ICLR 2022, [OpenReview](https://openreview.net/forum?id=bERaNdoegnO) | 不放回抽样的根节点策略改进；少模拟次数时明显强于 AlphaZero 的启发式。Atari 上 n=2 次模拟仍能学（Ms. Pac-Man） | 教师的模拟预算是夜间长局的短板（加深续走已有 1/48→5/48 的苗头）。Gumbel 是“少模拟也能改进策略”的搜索侧方法，不必换网络 | 教师根节点的访问/选择规则，续走网络不动。先在 6 套教师曾输的卡组上，与“加 horizon/samples”做一对一，而不是两个一起开 | 要改 C++ 搜索。风险：我们的动作是分解的（种/等 × 卡 × 格），不是 18 个 Atari 动作，Sequential Halving 的实现容易错。先读附录再决定值不值得移植 | **半年内**，排在 E4 和产线观测之后 |
| 5 | Wu, KataGo 2019 的 policy target pruning | 把探索访问从策略目标里减掉再归一化 | 若教师目标里探索质量很大，学生在学噪声 | 先审计访问分布；过阈值再改打包。不要和软标签绑在一起 | 审计约一天搜索。风险低。预期增益可能接近零（等价动作矛盾只有 0.05%） | **半年内，审计先行，默认可不做** |
| 6 | Hinton et al. 2015 蒸馏, arXiv:1503.02531；Rusu et al. 2015 policy distillation, arXiv:1511.06295 | 把大策略压回小网络 | 只有 E4 证明根节点 v3 真打更好之后，才需要把根节点行为蒸馏回续走 CNN | 续走仍是 1:132 的那一侧。蒸馏数据用根节点 v3 的分布，而不是硬标签 | 多一次训练。风险：蒸馏一个还没赢过等参 CNN 的 v3，白做 | **半年内，以 E4 为门** |
| 7 | Jacob et al. 2018, 纯整数推理, arXiv:1712.05877 | int8 量化 | 根节点 4.3 ms 若成为采集的可见部分，量化是比换稀疏注意力便宜的加速 | C++ 根节点前向。续走 CNN 已是约 150 µs，量化它的收益要单测，不一定划算 | 工程。风险是数值对拍失败（项目已有逐位对拍习惯，可挡住） | **半年内，仅当 E4 胜出** |
| 8 | Silver et al. 2016 的双网络分工（已核实 57.0%/3 ms 与 24.2%/2 µs） | 慢策略选点、快策略走子 | 我们已经决定根/续走拆分。这篇只是确认“快的一方允许差一截” | 不必新做实验。用来给 A13 的 +22% 墙钟一个外部先例：AlphaGo 的快策略准确率只有慢策略的四成，仍然进了搜索 | 无 | **现在（先例，不是待办）** |
| 9 | Kostrikov et al., IQL, 2021, [arXiv:2110.06169](https://arxiv.org/abs/2110.06169)；Nair et al., AWAC, 2020, [arXiv:2006.09359](https://arxiv.org/abs/2006.09359) | 离线 RL：在不额外交互的前提下，用价值把策略从示范分布里推出去 | 长期目标 5（不依赖人类示范）。短期不适合：97% 标签是学生自己的动作，奖励是稀疏胜负，离线 RL 会把学生的错放大。价值网 v6 刚被否决 | 若要做，必须用**搜索分叉的成对标签**当优势，而不是整局胜负（本地已证明整局标签不能给候选排序）。这更接近“用现成的叶子价值做加权 BC”，不是换一个离线 RL 库 | 高风险、易悄悄退步。先等价值网校准真的转化为教师胜率（task154：换叶子 1/48→1/48） | **长期观察** |
| 10 | Schrittwieser et al., MuZero, Nature 2020, [arXiv:1911.08265](https://arxiv.org/abs/1911.08265)；Micheli et al., IRIS, 2023, [arXiv:2209.00588](https://arxiv.org/abs/2209.00588)；Hafner et al., Dreamer 系 | 学一个可搜索/可想象的模型，替代或加速环境 | 相关的只有一句话：**我们有正确、够快的引擎**。这些工作的收益来自“没有模拟器”或“模拟器极贵”。用学到的模型替换 6×10 规则引擎，是用近似误差换掉零误差 | 不替换续走。若夜间需要更长视野，优先加真引擎 horizon（已有苗头），而不是训练世界模型 | 数据、GPU、以及模型在雾/随机出怪上的系统性错误。两台机器承担不起 | **长期观察，默认不做** |
| 11 | Ecoffet et al., Go-Explore, Nature 2021；Dennis et al., PAIRED 2020 | 开放式课程：存档返回，或用对抗者生成“刚好会输”的环境 | 卡组生成器可以变成“最大化师生胜率差”的采样器，而不是均匀随机。这是长期的课程，不是本月的开关 | 在随机卡组池上，用最近一轮的教师胜率 − 学生胜率给卡组加权，每地形封顶，避免只生成白天怪卡组 | 生成器若离开黑白名单会造出结构性输局（任务设计约束）。对抗课程容易只产生不可赢卡组，标签又变成常数 | **长期**。近期用分层随机 + 教师胜率过滤就够 |
| 12 | Vinyals et al. 2019 的人类评估协议（Grandmaster、分种族、接口限制、对战官方天梯） | 怎么宣称“超过人类” | 目标 4 需要事先写死：同一信息集（不给倍率以外的隐藏未来波次）、同一卡组规则、局数、人类水平的定义、置信区间。AlphaStar 单独报告了相机接口 vs 非相机（Fig. 3h：监督学习胜率 87% vs 96%）——接口会改变结论 | 写一页评测协议，不训练。在 50 关还没打通前不要做人类对战 | 协议写松了，以后的“超过人类”不可比 | **半年内写协议；对战本身长期** |

不单列、但明确不优先的：对比学习/自监督表征（输入已是结构化通道，不是像素）；持续学习专用算法（每轮从头训累积包，遗忘不是当前机制；问题是标签过时，用抽检重标而不是 EWC）；LLM 稀疏注意力（token 中位约 121，上一份架构笔记已论证不适合）；把 LLM 整局动作灌进训练集（task104 负结果，Xia 2025 的设定也不匹配）。

### 路线图

**近期可试（不新开算法，只守已部署的开关和已排队的 E4）**

- 结构：E4 四臂，固定步数，未见卡组 + 屋顶/雾 + 至少一关空间关系（保龄球或蹦极，若富输入已有字段）。赢的定义是胜过等参宽 CNN，不是胜过 CNN64。
- 数据：观察示范格降到 25% 和随机卡组扩池。停止条件用教师胜率，不用行数。
- 选拔：不改成最差地形 minimax。
- 阅读结论落地：加宽/MoE/再分配维持关闭。

**中期（E4 和产线有结论之后）**

- 若 E4 胜：根节点 v3 + 蒸馏回续走 + 可选 int8；续走路径禁止换大模型。
- 若 E4 负：把 v3 停在“空间关输入仍缺字段”的记录上，不追加层数。
- 教师搜索：Gumbel 根节点或加深续走，**一次只做一个**，用 task154 的那类配对卡组。
- 写“超过人类”的评测协议。
- 访问分布审计，决定要不要 policy target pruning。

**长期**

- 卡组课程从均匀随机过渡到遗憾加权，仍受黑白名单约束。
- 离线 RL 仅当搜索分叉标签能稳定排序候选、且 BC 已平台。
- 世界模型默认不替代引擎；只有引擎在某类关卡上贵到采集等待 GPU 时再重新评估。
- 不依赖人类示范：前提是教师搜索在未见卡组上本身能赢。在 14.6% 上谈“去掉教师”没有意义。

---

## 9. 行动建议（按优先级）

| # | 做什么 | 判据 | 文献支持到哪 |
|---|---|---|---|
| P1 | **不要**因本报告改结构路线。E4 按原预登记跑完：CNN64、等参宽 CNN、v3 只放根节点。固定优化步数。 | 未见卡组与空间关同时不差于等参宽 CNN，且根节点墙钟 < +30%，才进 B 槽。否则停。 | 中。文献只说明“没有理由提前相信注意力更省数据”，否决权在实验。 |
| P2 | **不要**把挑选改成最差地形优先。保持门禁否决 + 选拔预测门禁。 | 若有人提案改 min(地形)，用 r20–r36 回溯：结论翻转则拒绝。 | 高（Gulrajani；AlphaGo Zero 的全局 55%）。 |
| P3 | **不要**重开 E1/E13/MoE/FiLM 排期。Sagawa 只用于解释 task23，不用于恢复下采样。 | 无新实验。 | 高（本地）+ 中（机制）。 |
| P4 | 产线观察随机卡组与示范格配额。给随机卡组加一条**教师胜率过滤**（近几轮该格子教师胜率为 0 就停在更低倍率或换池），不要只加局数。 | 8 套评估卡组胜率上来，且不是靠全输标签堆行数；非示范格教师胜率 ≥70%。 | 中高（Procgen 的数量级 + Salimans 的退场规则）。“过滤阈值取多少”没有文献数字。 |
| P5 | 教师侧：在已登记的“加深续走”与“Gumbel 根节点”里先做加深（改动更小，已有 1/48→5/48）。Gumbel 作为设计文档，不与加深绑在同一轮。 | 夜间/长局净胜为正、墙钟可接受、屋顶不期待改善。 | 加深：本地证据强于文献。Gumbel：文献高、迁移中，故靠后。 |
| P6 | 写一页人类对战协议（信息集、卡组规则、局数、区间），不打。 | 协议被固定下来，以后不能用事后口径。 | 中（AlphaStar 的接口消融说明协议会改变结论）。 |
| P7 | 可选审计：根节点访问分布里非目标动作的质量占比。低于一个事先写下的阈值就不做剪枝。 | 审计报告本身是交付；不自动改训练。 | 中（KataGo 1.25× 是围棋自博弈的因子，不是我们的预期收益）。 |

---

## 10. 查不到，或只能外推

下列句子**不能**出现在后续设计文档里充当已证实证据：

1. “大约 1e7–3e7 条状态–动作上，实体 Transformer 开始超过 CNN。”没有这篇对照。
2. “MoE 能提高未见卡组胜率。”没有 RL/BC 测量。Soft MoE 的 +20% 是 Atari 分布内。
3. “按最差地形挑选候选会提高最终通关。”DoReMi 的 +6.5 点是语言模型的平均下游，且优化的是训练权重不是 checkpoint 选拔。
4. “温度 α>1 可以提高屋顶权重。”公式方向是错的；正确形式是 \(p\propto n^{1/T}\)，T=5 即指数 0.2，且来自 Arivazhagan 2019 不是 Johnson 2017。即便公式用对，E13 已经显示在我们语料上压主导卡组会伤泛化。
5. “AlphaZero 使用 50 万局回放窗口。”AlphaZero 论文没写；50 万局是 AlphaGo Zero。
6. “Leela Zero 的窗口是某个确定局数。”查不到单一官方常数。
7. “学一个 MuZero/IRIS 模型就能加速续走。”在已有正确引擎时没有支持性实验，反有模型误差的风险。
8. “卡牌文本编码器能解决我们的未见卡组。”Bertram 的 55.44% 是草稿拣选；Xia 2025 的表内胜率本次未能逐格核对，且任务是开放新卡。
9. “KataGo 的 50 倍可以靠加上 policy target pruning 复现。”50 倍是整组技巧相对 ELF 的 GPU 年；剪枝单项的消融因子是约 1.25×。
10. Data mixing laws（Ye et al. 2024）的函数形式适用于我们的行权。未验证。

---

## 11. 引用索引

1. Vinyals, O. et al. Grandmaster level in StarCraft II using multi-agent reinforcement learning. *Nature* 575, 350–354 (2019). https://doi.org/10.1038/s41586-019-1724-z
2. Silver, D. et al. Mastering the game of Go with deep neural networks and tree search. *Nature* 529, 484–489 (2016). https://doi.org/10.1038/nature16961
3. Silver, D. et al. Mastering the game of Go without human knowledge. *Nature* 550, 354–359 (2017). https://doi.org/10.1038/nature24270
4. Silver, D. et al. A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. *Science* 362, 1140–1144 (2018). 预印本：https://arxiv.org/abs/1712.01815
5. Dosovitskiy, A. et al. An Image is Worth 16x16 Words. ICLR 2021. https://arxiv.org/abs/2010.11929
6. Tuyls, J. et al. Scaling Laws for Imitation Learning in Single-Agent Games. TMLR (2024). https://arxiv.org/abs/2307.09423
7. Zambaldi, V. et al. Relational Deep Reinforcement Learning. ICLR 2019. https://arxiv.org/abs/1806.01830
8. Lee, K.-H. et al. Multi-Game Decision Transformers. NeurIPS 2022. https://arxiv.org/abs/2205.15241
9. Cobbe, K. et al. Quantifying Generalization in Reinforcement Learning (CoinRun). ICML 2019. https://arxiv.org/abs/1812.02341
10. Cobbe, K. et al. Leveraging Procedural Generation to Benchmark Reinforcement Learning (Procgen). ICML 2020. https://arxiv.org/abs/1912.01588
11. Obando-Ceron, J. S. et al. Mixtures of Experts Unlock Parameter Scaling for Deep RL. ICML 2024. https://arxiv.org/abs/2402.08609
12. Puigcerver, J. et al. From Sparse to Soft Mixtures of Experts. 2024. https://arxiv.org/abs/2308.00951
13. Sodhani, S., Zhang, A., Pineau, J. Multi-Task Reinforcement Learning with Context-based Representations. ICML 2021. https://arxiv.org/abs/2102.06177
14. Hendawy, A., Peters, J., D'Eramo, C. Multi-Task Reinforcement Learning with Mixture of Orthogonal Experts. ICLR 2024. https://arxiv.org/abs/2311.11385
15. Yang, R. et al. Multi-Task Reinforcement Learning with Soft Modularization. NeurIPS 2020. https://arxiv.org/abs/2003.13661
16. Li, B. et al. Sparse Mixture-of-Experts are Domain Generalizable Learners. ICLR 2023. https://arxiv.org/abs/2206.04046
17. Sagawa, S. et al. An Investigation of Why Overparameterization Exacerbates Spurious Correlations. ICML 2020. https://arxiv.org/abs/2005.04345
18. Sagawa, S. et al. Distributionally Robust Neural Networks for Group Shifts. ICLR 2020. https://arxiv.org/abs/1911.08731
19. Gulrajani, I., Lopez-Paz, D. In Search of Lost Domain Generalization. ICLR 2021. https://arxiv.org/abs/2007.01434
20. Xie, S. M. et al. DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining. NeurIPS 2023. https://arxiv.org/abs/2305.10429 作者博客：https://crfm.stanford.edu/2023/09/14/doremi.html
21. Johnson, M. et al. Google's Multilingual Neural Machine Translation System. TACL 2017. https://arxiv.org/abs/1611.04558
22. Arivazhagan, N. et al. Massively Multilingual Neural Machine Translation in the Wild. 2019. https://arxiv.org/abs/1907.05019
23. Ye, J. et al. Data Mixing Laws. 2024. https://arxiv.org/abs/2403.16952 （未在 RL/BC 验证，索引中保留只为标成不可外推）
24. Jiang, M., Grefenstette, E., Rocktäschel, T. Prioritized Level Replay. ICML 2021. https://arxiv.org/abs/2010.03934
25. Parker-Holder, J. et al. Evolving Curricula with Regret-Based Environment Design (ACCEL). ICML 2022. https://arxiv.org/abs/2203.01302
26. Dennis, M. et al. Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design (PAIRED). ICML 2020. https://arxiv.org/abs/2012.02096
27. Zhang, J., Cho, K. Query-Efficient Imitation Learning for End-to-End Autonomous Driving (SafeDAgger). 2016. https://arxiv.org/abs/1605.06450
28. Menda, K., Driggs-Campbell, K., Kochenderfer, M. J. EnsembleDAgger. 2019. https://arxiv.org/abs/1807.08364
29. Ross, S., Gordon, G., Bagnell, J. A. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (DAgger). AISTATS 2011. https://arxiv.org/abs/1011.0686
30. Anthony, T., Tian, Z., Barber, D. Thinking Fast and Slow with Deep Learning and Tree Search (Expert Iteration). NeurIPS 2017. https://arxiv.org/abs/1705.08439
31. Wu, D. J. Accelerating Self-Play Learning in Go (KataGo). 2019. https://arxiv.org/abs/1902.10565 官方方法文档：https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md
32. Ecoffet, A. et al. First return, then explore (Go-Explore). *Nature* 590, 580–586 (2021). https://arxiv.org/abs/1901.10995
33. Salimans, T., Chen, R. Learning Montezuma's Revenge from a Single Demonstration. 2018. https://arxiv.org/abs/1812.03381 官方博客：https://openai.com/index/learning-montezumas-revenge-from-a-single-demonstration/
34. Tobin, J. et al. Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World. 2017. https://arxiv.org/abs/1703.06907
35. Bertram, T., Fürnkranz, J., Müller, M. Learning With Generalised Card Representations for Magic: The Gathering. 2024. https://arxiv.org/abs/2407.05879
36. Xia, W. et al. Integrating Large Language Models with Reinforcement Learning for Generalization in Strategic Card Games. AAMAS 2025 扩展摘要. https://dl.acm.org/doi/10.5555/3709347.3744015
37. Danihelka, I. et al. Policy Improvement by Planning with Gumbel. ICLR 2022. https://openreview.net/forum?id=bERaNdoegnO
38. Schrittwieser, J. et al. Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (MuZero). *Nature* 588, 604–609 (2020). https://arxiv.org/abs/1911.08265
39. Micheli, V., Alonso, E., Fleuret, F. Transformers are Sample-Efficient World Models (IRIS). ICLR 2023. https://arxiv.org/abs/2209.00588
40. Hinton, G., Vinyals, O., Dean, J. Distilling the Knowledge in a Neural Network. 2015. https://arxiv.org/abs/1503.02531
41. Rusu, A. A. et al. Policy Distillation. 2016. https://arxiv.org/abs/1511.06295
42. Jacob, B. et al. Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. CVPR 2018. https://arxiv.org/abs/1712.05877
43. Kostrikov, I., Nair, A., Levine, S. Offline Reinforcement Learning with Implicit Q-Learning (IQL). 2021. https://arxiv.org/abs/2110.06169
44. Nair, A. et al. AWAC: Accelerating Online Reinforcement Learning with Offline Datasets. 2020. https://arxiv.org/abs/2006.09359
45. Pearce, T. et al. 一线的智能体预训练缩放（BC-Token 与 BC-CNN 的算力口径）. https://arxiv.org/abs/2411.04434 （仅用于 Q1 的算力提醒，未当作样本效率结论）
