# task160-B：文献调研（架构 / MoE / 数据配比 / 采集）

- 日期：2026-10-29
- 范围：`task160-architecture-decision-study.md` B 节列出的 9 个主题
- 方法：以 arXiv 摘要页、会议页、项目内文档为主；本次未逐篇读全文，凡只来自二手总结、博客或推断的数字/说法，均在文中标注“中/低”或列入 §12。
- 可信度分级：
  - **高**：同行评审、任务/规模与我们有可比性，或结论在多个独立来源重复出现。
  - **中**：同行评审但领域/规模差异大，或二手总结。
  - **低**：博客/社区/工程经验/推断，只作方向参考。

## 0. 结论先行

| 问题 | 建议 | 一句话理由 | 置信度 |
|---|---|---|---|
| 网络结构 | **先保持 CNN64，修数据配比与输入；再试 FiLM/地形硬路由；实体+注意力 v3 放最后，且只先用于根节点** | r36 的瓶颈证据指向覆盖（L7 一关 46.7%、屋顶 1.8%、随机卡组 0.2%、未知卡组 14.6%），不是参数不够；task23 加宽反而伤未知卡组和夜间 | 中高 |
| 是否用 MoE | **现在不用**；若 task158 证实地形干扰，先 FiLM，再共享主干+按地形硬路由；学出来的门控和 Soft MoE 放最后 | Top-1 MoE 在小规模 RL 上失败，Soft MoE 需要规模和算力；地形在输入里已知，硬路由比学路由便宜且可解释 | 中高 |
| 数据配比 | **先修来源与地形配比**：L7 历史行封顶/下采样，提高随机卡组格、示范起点、屋顶/雾/泳池；目标用 gate v5 的“最差地形”而不是平均 | DoReMi 的 group DRO 与门禁“每地形不能显著退步”同构；B 配方从没赢，若选拔仍按平均口径则很可能输给 A（待核），不代表它本身没价值 | 中 |
| 采集方向 | **师生不一致/regret 驱动**，优先未知卡组、屋顶/雾、空间关系关卡；保留示范起点和随机卡组格；旧标签降权并抽检重标注 | PLR/ACCEL/DAgger/KataGo 的共同点：把采集预算给“当前最不会、且还能学会”的状态 | 中高 |
| 大/小网络分工 | **根节点 v3，续走仍用 CNN**；必要时蒸馏或 int8 | 实测根节点前向只占 0.75%（1:132）；AlphaGo 的快速走子先例支持这种分工 | 高 |

一句话总判断：**在我们 20M 行、18 万参数、CPU 前向、门禁按地形否决的约束下，未来一两轮最该做的是“数据覆盖 + 条件化”，不是“更大的网络或 MoE”。** 结构升级值得做，但必须等数据规模和输入完整度跟上，并且用等参 CNN 对照、按未知卡组/地形分开判。

---

## 1. 实体 + 空间输入的游戏网络

### 1.1 AlphaStar

- 引用：Vinyals et al., *Grandmaster level in StarCraft II using multi-agent reinforcement learning*, Nature 575, 350–354 (2019). DOI: [10.1038/s41586-019-1724-z](https://doi.org/10.1038/s41586-019-1724-z)。官方博客：<https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/>。
- 关键结论：
  - 输入是“实体列表 + 空间地图”两路：单位/建筑按实体编码，空间信息用卷积；两路通过 **scatter connections** 合并（把空间特征散射到对应实体，或反之）。
  - 动作里“选哪个单位”用 **pointer network** 式的指针头，直接在实体列表上打分，而不是把单位 ID 当成固定类别。
  - 网络含 LSTM 和 self-attention；训练用 32 块 TPU、44 天（DeepMind 原文 PDF 可查），参数约 1.39 亿（**二手来源，未在 Nature 正文核实**）。
- 可信度：**架构描述高**（Nature 正文/方法 + 官方博客 + mini-AlphaStar 复现）；**参数量中**（二手）。
- 对我们意味着什么：
  - “实体 token + 指针头”在真实多实体游戏里被验证过，值得作为 v3 的参考；我们旧 `GameplayModelV1` 的“种类/类别/变体嵌入 + 行列嵌入 + 关系偏置 + 卡包/格子指针头”就是同一族设计。
  - 但 AlphaStar 是 **1.39 亿参数 + 32 TPU×44 天 + 海量人类回放/联赛自博弈**。我们 18 万参数、20M 行、CPU 前向，不能把“它用了实体注意力所以我们也该用”当成论据。可迁移的是**输入表示和指针头**，不是规模。

### 1.2 OpenAI Five / Gato / MuZero 的网络规模

- 引用：
  - Berner et al., *Dota 2 with Large Scale Deep Reinforcement Learning*, 2019, [arXiv:1912.06680](https://arxiv.org/abs/1912.06680)。
  - Reed et al., *A Generalist Agent* (Gato), 2022, [arXiv:2205.06175](https://arxiv.org/abs/2205.06175)。
  - Schrittwieser et al., *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model* (MuZero), 2019, [arXiv:1911.08265](https://arxiv.org/abs/1911.08265)；Nature 588, 604–609 (2020)。
- 关键结论：
  - **OpenAI Five**：单层 4096 单元 LSTM，约 **1.59 亿参数**；训练时每 2 秒约 200 万帧。它不是实体 transformer，靠的是规模 + LSTM + 大量自博弈。
  - **Gato**：1.2B 参数、24 层、embedding 2048 的 decoder-only transformer，覆盖 604 个任务；是“通才”路线，不是样本高效路线。
  - **MuZero**：Go 上用 16 个 residual block；网络本身不算巨大，强在**学习模型 + MCTS 搜索**。搜索可以在一定程度上补偿网络容量。
- 可信度：**高**（论文/官方 PDF 明确给出这些数字）。
- 对我们意味着什么：
  - 顶级游戏 agent 的参数在 **1.5 亿–12 亿**，数据是“每秒百万帧/几十天 TPU”级别；我们在 **18 万参数、20M 行**。两者的“架构最优解”不可直接类比。
  - MuZero 的经验更重要：**搜索能补网络**。我们的老师就是“学生 + 搜索”，所以学生不必追 AlphaStar 的容量；把根节点做强、续走保持快，是更符合我们算力结构的路线。

### 1.3 Transformer 与 CNN 在 1000 万–3000 万行规模下的数据效率

- 引用：
  - Dosovitskiy et al., *An Image is Worth 16x16 Words* (ViT), 2020, [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)。
  - Touvron et al., *Training data-efficient image transformers & distillation through attention* (DeiT), 2020, [arXiv:2012.12877](https://arxiv.org/abs/2012.12877)。
  - Liu et al., *A ConvNet for the 2020s* (ConvNeXt), 2022, [arXiv:2201.03545](https://arxiv.org/abs/2201.03545)。
  - Micheli et al., *Transformers are Sample-Efficient World Models* (IRIS), 2022, [arXiv:2209.00588](https://arxiv.org/abs/2209.00588)。
- 关键结论：
  - ViT 在 ImageNet-1k（约 128 万张）上不如同规模 ResNet；预训练到 **JFT-300M（3 亿张）** 后才全面超过 CNN。纯 transformer 对数据量敏感。
  - DeiT 用 **ImageNet-1k + 蒸馏 + 强增强** 把纯 transformer 拉到与 CNN 可比，但蒸馏老师本身是 CNN；这说明“数据和正则/先验”仍然关键。
  - ConvNeXt 把 CNN 现代化后，在同等规模下与 transformer 打平；架构家族不是决定性的。
  - IRIS 在 Atari 100k（约 2 小时游戏）上用 transformer 世界模型做到样本高效，但它是**世界模型 + 想象轨迹**，不是“直接 BC 一个 transformer 策略”。
- 可信度：**高**（视觉/世界模型领域结论扎实）；**中**（迁移到我们的 20M 行状态-动作 BC 是外推，没有直接同规模对照）。
- 对我们意味着什么：
  - 20M 行正好落在“transformer 可用但不自动更好”的区间。实体注意力的优势应来自**关系归纳偏置**（卡组顺序无关、L5 滚动坚果的精确空间关系、L45 蹦极落点），而不是“数据效率更高”。
  - task153 的负结果（未见卡组 84.7% vs CNN64 91.2%）不能证明“注意力不行”，更像**实现缺陷 + 数据小 + 输入缺字段**；但它也提醒：在这个数据规模，结构收益很容易被正则/实现细节吃掉。
  - 如果要上 v3，必须同时做：卡包 token 不进盘面池化、group 嵌入、多头 + 可学习行/列桶偏置、随机遮卡增强、多种子、等参 CNN81/128 对照。否则测到的是实现 bug，不是架构。

### 1.4 帧堆叠/记忆（顺带）

- 引用：DQN 的 frame stacking 是工程惯例；PPO 时代我们的 GRU 让每次更新 9 s → 110 s（项目记忆 `MEMORY/notes/student-architecture.md`）。
- 可信度：**中**（项目实测）。
- 对我们意味着什么：目前没有证据表明“记不住历史”是主要瓶颈；雾夜/屋顶的短板更可能是可见信息/数据覆盖。先做“加 1–2 帧历史”的廉价消融，再谈 GRU/记忆。

---

## 2. 强化学习 / 模仿学习中的 MoE

### 2.1 Soft MoE 的参数缩放证据

- 引用：Obando-Ceron et al., *Mixtures of Experts Unlock Parameter Scaling for Deep RL*, ICML 2024 (Spotlight), [arXiv:2402.08609](https://arxiv.org/abs/2402.08609)。
- 关键结论：
  - 在 DQN/Rainbow 的倒数第二层替换成 MoE；**dense 网络加参数不涨点，Soft MoE 能随参数增加而提升**。
  - **Top-1 MoE 只在 Rainbow 上有帮助，且不随规模提升**；加负载均衡损失也不足以救它。
  - Soft MoE 是“所有 token 对所有专家做加权组合”，不是稀疏激活；计算上要过所有专家。
- 可信度：**高**（ICML spotlight，结论直接）。
- 对我们意味着什么：
  - 若一定要用 MoE，**Soft MoE 是唯一有正面缩放证据的选项**；但它在我们 CPU 单样本前向上会显著变慢，且每个专家分到的数据更少。
  - Top-1/Top-2 稀疏门控在我们这个规模**没有证据支持**，不要先试它。

### 2.2 多任务 RL 的 MoE：MOORE、CARE、soft modularization

- 引用：
  - Hendawy et al., *Multi-Task Reinforcement Learning with Mixture of Orthogonal Experts* (MOORE), ICLR 2024, [arXiv:2311.11385](https://arxiv.org/abs/2311.11385)。
  - Sodhani et al., *Multi-Task Reinforcement Learning with Context-based Representations* (CARE), ICML 2021, [arXiv:2102.06177](https://arxiv.org/abs/2102.06177)。
  - Yang et al., *Multi-Task Reinforcement Learning with Soft Modularization*, NeurIPS 2020, [arXiv:2003.13661](https://arxiv.org/abs/2003.13661)。
- 关键结论：
  - MOORE 用 Gram–Schmidt 让专家表示正交，鼓励共享子空间里的多样性；对多任务迁移有帮助。
  - CARE 用任务元数据（context）学习多个状态编码器，再用注意力组合；本质是“上下文条件化的专家”。
  - Soft modularization 在每层学路由权重，而不是硬分任务。
  - 三者都依赖**任务身份/上下文**；我们的“地形”正好是已知的离散上下文。
- 可信度：**高**（同行评审）；**中**（基准是机器人/控制任务，不是 PvZ 这种 6×10 网格 + 卡组组合）。
- 对我们意味着什么：
  - 如果目标是“地形之间不互相拖累”，**按地形条件化**（FiLM/小 adapter/硬路由头）比学一个通用 MoE 更贴合问题结构；CARE 的“context → 多表示”可以视为 FiLM/adapter 的复杂版。
  - MOORE 的正交约束在专家数少、数据少时未必能学出有意义的分工；不要指望它自动解决屋顶 1.8% 的数据饥饿。

### 2.3 小规模下的路由塌缩

- 引用：
  - Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*, 2017, [arXiv:1701.06538](https://arxiv.org/abs/1701.06538)。
  - Fedus et al., *Switch Transformers*, 2021, [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)。
  - Chi et al., *On the Representation Collapse of Sparse Mixture of Experts*, 2022, [arXiv:2204.09179](https://arxiv.org/abs/2204.09179)。
- 关键结论：
  - 稀疏门控天然倾向“赢家通吃”；Shazeer 用 noisy top-k + 负载均衡辅助损失；Switch 用 aux loss（常用权重约 $10^{-2}$）+ capacity factor。
  - Chi et al. 指出稀疏 MoE 还有**表示塌缩**：路由把 token 映射到低维子空间，专家表示趋同；不是简单的负载不均衡。
  - 负载均衡损失能缓解“用量不均”，但不能保证“每个专家学到不同东西”；Obando-Ceron 也发现加负载损失救不了 Top-1。
- 可信度：**高**。
- 对我们意味着什么：
  - 我们只有约 20M 行、5 个地形、屋顶 1.8%；如果开 4–8 个专家，每个专家实际看到的有效数据更少，塌缩风险高。
  - 若做 MoE，必须报告：路由分布是否退化成地形分类器、每个专家的有效样本数、专家输出的余弦相似度/表示方差；只看总 loss 会漏掉塌缩。

### 2.4 MoE 对分布外泛化：帮还是伤？

- 引用：
  - Li et al., *Sparse Mixture-of-Experts are Domain Generalizable Learners* (GMoE), ICLR 2023, [arXiv:2206.04046](https://arxiv.org/abs/2206.04046)。
  - Li et al., *Can Mixture-of-Experts Surpass Dense LLMs Under Strictly Equal Compute?*, 2025, [arXiv:2506.12119](https://arxiv.org/abs/2506.12119)（二手摘要）。
- 关键结论：
  - GMoE 在视觉 domain generalization 上显示：**当专家与潜在域对齐、且规模足够时，稀疏 MoE 可以提升 OOD**；关键是 backbone 设计与域结构的对齐，而不是“MoE 本身自动更鲁棒”。
  - 另一侧证据：MoE 通常需要**更多数据**才能追平 dense；数据少时验证集质量更差（LLM/视觉经验，二手总结，非 RL 直接证据）。
- 可信度：**中**（视觉 DG 是同行评审；迁移到 RL/BC 是外推；LLM 那条更弱）。
- 对我们意味着什么：
  - 不能把 MoE 当成“未知卡组泛化”的药。我们的 task23 已经看到：**参数更多（宽 128）反而让未知卡组 60.0%→53.8%、夜间 51.4%→42.6%**。在没有增加卡组/地形覆盖的情况下，加容量（无论 dense 还是 MoE）都可能伤 OOD。
  - 如果一定要用 MoE，**硬路由到已知地形**比学路由更安全：不会出现“路由器把未知卡组分到错误专家”的二次泛化问题。

---

## 3. 多任务之间的干扰及对策：PCGrad、FiLM、适配层

- 引用：
  - Yu et al., *Gradient Surgery for Multi-Task Learning* (PCGrad), NeurIPS 2020, [arXiv:2001.06782](https://arxiv.org/abs/2001.06782)。
  - Perez et al., *FiLM: Visual Reasoning with a General Conditioning Layer*, AAAI 2018, [arXiv:1709.07871](https://arxiv.org/abs/1709.07871)。
  - Houlsby et al., *Parameter-Efficient Transfer Learning for NLP* (adapters), ICML 2019, [arXiv:1902.00751](https://arxiv.org/abs/1902.00751)。
- 关键结论：
  - PCGrad：当两个任务的梯度冲突（负余弦）时，把一个梯度投影到另一个的法平面；能缓解多任务优化的“跷跷板”。
  - FiLM：用条件向量生成逐通道的 $\gamma,\beta$，对特征做仿射调制；参数极少，适合离散上下文（地形）。
  - Adapter：在层间插入小瓶颈模块，只训练少量任务专属参数；新任务可增量加入，不破坏旧任务。
- 可信度：**高**（三者都是成熟方法）。
- 对我们意味着什么：
  - 地形是**已知的离散上下文**，FiLM 是最便宜的“条件化专家”：一次前向只多几个乘加，C++ 实现简单（`y = γ(terrain) * x + β(terrain)`）。
  - 如果 task158 显示“单独网络在屋顶/雾上明显更好”，先试 FiLM/小 adapter，而不是一上来就 5 个完整专家；前者参数少、数据共享，后者数据稀释。
  - PCGrad 需要按地形/任务拆梯度，训练器改动较大；只有在能测到明显梯度冲突、且 FiLM 不够时再做。

---

## 4. 模仿学习的缩放律：怎么判断“数据不够”还是“容量不够”

- 引用：
  - Tuyls et al., *Scaling Laws for Imitation Learning in Single-Agent Games*, 2023, [arXiv:2307.09423](https://arxiv.org/abs/2307.09423)。
  - Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla), 2022, [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)（跨领域类比）。
- 关键结论：
  - 在 Atari/NetHack 上，BC 的性能/交叉熵随模型大小和数据量呈幂律；**NetHack 的回报幂律会在专家水平之前平台化**——说明存在“数据/行为覆盖”造成的上限，不是无限加参数就能到专家。
  - Chinchilla：固定算力下，参数和数据应同比增加；只加参数不加数据是次优。这个结论来自语言模型，但“两个轴要一起扫”的方法是通用的。
- 可信度：**高**（Tuyls 论文）；**中**（Chinchilla 跨领域）。
- 对我们意味着什么：
  - 用二维扫描判：**模型宽度（64/128）× 数据比例（25/50/100%）**，在每个地形/卡组切片上拟合学习曲线。
    - 小模型随数据饱和、大模型继续涨 → **容量不够**。
    - 两个模型都随数据涨、差距不变 → **数据不够**。
    - 两个模型都饱和、加数据也不动 → **标签/行为上限**（老师噪声、等价动作、部分可观测）。
  - 现有证据不是“容量不够”的干净信号：宽 128 只在白天早关 +13 局，却在未知卡组/夜间退步。更像是**覆盖/正则问题**。在没做这个二维扫描前，不要用“加宽/上 MoE”当默认答案。
  - 成本：CNN64/128 在 Mac 上跑 3 个数据比例 × 2 个宽度 × 2–3 种子，是小时级，不需要台式机。

---

## 5. 对没见过的组合怎么泛化

### 5.1 Procgen / CoinRun：训练关卡数的影响

- 引用：
  - Cobbe et al., *Quantifying Generalization in Reinforcement Learning* (CoinRun), 2018, [arXiv:1812.02341](https://arxiv.org/abs/1812.02341)；ICML 2019。
  - Cobbe et al., *Leveraging Procedural Generation to Benchmark Reinforcement Learning* (Procgen), 2019, [arXiv:1912.01588](https://arxiv.org/abs/1912.01588)；ICML 2020。
- 关键结论：
  - CoinRun：agent 会**过拟合到很大的训练关卡集**；加深卷积能改善泛化，但不够。
  - Procgen：推荐**训练 500 个关卡**、在留出关卡上测；训练关卡越少，泛化 gap 越大；过拟合是系统性的，不是个别 seed。
- 可信度：**高**。
- 对我们意味着什么：
  - 卡组就是我们的“关卡参数”。r36 里随机卡组格只占 **0.2%**，task153 只有 23 套训练卡组 / 32 套未见卡组，task129 在没见过的卡组上只赢 **14.6%**（54 套 v5 卡组）。这和 Procgen 的“训练关卡太少 → 泛化 gap”是同一类问题。
  - 先把**卡组覆盖**做上去：更多随机卡组格、按地形分层的卡组留出、至少 50+ 套未见卡组再谈“架构是否更会泛化”。32 套未见卡组的方差太大，不足以做结构结论。

### 5.2 卡组/配置随机化

- 引用：Tobin et al., *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*, 2017, [arXiv:1703.06907](https://arxiv.org/abs/1703.06907)（类比）。
- 关键结论：训练时随机化环境参数，能显著提升对未见过配置的鲁棒性；随机化的**范围和分布**比“有没有随机化”更重要。
- 可信度：**中**（sim-to-real，不是卡组）。
- 对我们意味着什么：
  - 随机卡组格是正确方向，但 0.2% 太低；而且如果随机卡组只覆盖“能赢的卡组”，分布仍然偏。应按地形/关卡分层随机，并记录“卡组 × 地形”的覆盖矩阵。
  - 训练时可加**卡包 token dropout / 卡序打乱**（task153 修订 1 的第 4 条），这比只增加卡组数量更便宜地教模型“不要依赖卡序”。

### 5.3 卡牌游戏里的组合泛化

- 引用：
  - You et al., *Combinational Q-Learning for Dou Di Zhu*, 2019, [arXiv:1901.08925](https://arxiv.org/abs/1901.08925)。
  - Zha et al., *RLCard: A Platform for Reinforcement Learning in Card Games*, IJCAI 2020, [论文页](https://www.ijcai.org/proceedings/2020/0764.pdf)。
- 关键结论：卡牌游戏的核心难点是**动作空间组合爆炸**（出牌组合），常见做法是做动作抽象/组合 Q 学习；没有看到“直接端到端泛化到未见卡组”的强证据。
- 可信度：**低–中**（领域相关，但不是我们的“卡组组合”问题；我们不是出牌组合，而是卡槽集合）。
- 对我们意味着什么：
  - 我们的“组合”主要是**卡槽集合 + 地形 + 关卡**，不是手牌组合；更接近“集合输入”而不是“组合动作空间”。因此 **Deep Sets / Set Transformer 的置换不变性**比卡牌游戏的组合 Q 学习更相关。
  - 不要从卡牌游戏文献直接推断“MoE/组合模块能解决卡组泛化”；没有直接证据。

---

## 6. 大小两个网络的分工：快速走子、蒸馏、int8

- 引用：
  - Silver et al., *Mastering the game of Go with deep neural networks and tree search*, Nature 529, 484–489 (2016). DOI: [10.1038/nature16961](https://doi.org/10.1038/nature16961)。
  - Hinton et al., *Distilling the Knowledge in a Neural Network*, 2015, [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)。
  - Rusu et al., *Policy Distillation*, 2015, [arXiv:1511.06295](https://arxiv.org/abs/1511.06295)。
  - Jacob et al., *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*, 2017, [arXiv:1712.05877](https://arxiv.org/abs/1712.05877)。
- 关键结论：
  - AlphaGo 有两套策略：慢的 SL policy（约 57% 准确率、约 3 ms/步）和快的 rollout policy（约 24% 准确率、约 2 µs/步）；搜索用慢策略选点、用快策略快速走子。**快网络不需要准，只需要快且不太偏**。
  - 蒸馏可以把大教师/集成压到小网络；policy distillation 在 RL 里也可行。
  - int8 量化能在 CPU 上显著加速，精度损失可控（配合量化感知训练）。
- 可信度：**高**（AlphaGo 数字来自 Nature/二手技术总结；蒸馏/int8 是成熟结论）。
- 对我们意味着什么：
  - 我们实测根节点前向 882 次/局、续走 116,190 次/局（约 1:132，根节点 0.75%）。**根节点用 v3 大模型、续走用 CNN64 是完全合理的**；续走速度不会变。
  - 如果续走也需要更强，优先“把 v3 蒸馏回 CNN”或 int8，而不是把注意力塞进续走。AlphaGo 的快 rollout 比慢策略差很多但依然有用，说明“快而略弱”在搜索里是可接受的。
  - 门禁/采集/续走都走 C++ 前向的约束下，这条分工还能避免为 v3 写高频 C++ 路径。

---

## 7. 数据配比：DoReMi、data mixing laws、温度采样

- 引用：
  - Xie et al., *DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining*, NeurIPS 2023, [arXiv:2305.10429](https://arxiv.org/abs/2305.10429)。
  - Ye et al., *Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance*, 2024, [arXiv:2403.16952](https://arxiv.org/abs/2403.16952)。
  - Johnson et al., *Google's Multilingual Neural Machine Translation System: Enabling Zero-Shot Translation*, TACL 2017, [arXiv:1611.04558](https://arxiv.org/abs/1611.04558)。
- 关键结论：
  - DoReMi：先用一个小代理模型（论文用 280M） + **group DRO** 学域权重，再用于大模型（论文用 8B）；在 The Pile 上平均 one-shot 准确率 +6.5 个点，达到 baseline 准确率快 2.6×。group DRO 的目标是**最大化最差域的收益**，不是平均。
  - Data mixing laws：把“混合比例 → 性能”拟合成可预测的函数，再用嵌套 scaling law 外推；主要用于语言模型。
  - 温度采样：多语言 NMT 里用 $p_i \propto n_i^\alpha$（$\alpha>1$ 提高低资源语言权重）来平衡；简单、有效、无需训练代理。
- 可信度：**高**（DoReMi/温度采样）；**中**（data mixing laws 是 LM 结论，未在 RL/BC 验证）。
- 对我们意味着什么：
  - **DoReMi 的 group DRO 与 gate v5 同构**：门禁要求“每个地形都不能显著退步”，这正是最差域目标。用平均 loss/平均胜率去调配方，会系统性偏向白天（66.2%），让屋顶/雾继续被淹没。B 配方（屋顶 10.3%）从没赢；如果选拔口径仍以平均为主，B 会输给 A。**建议改用“最差地形”或“各地形相对 A 的 margin”作为配方目标**。
  - 具体做法：用 CNN64 做代理，扫 3 个配方：
    1. 现行 A；
    2. 现行 B（关卡平衡）；
    3. **floor 约束配方**：L7 历史行封顶（如 25–30%）、屋顶/雾/泳池设下限、随机卡组格从 0.2% 提到 2–5%、示范起点保持 3% 左右。
     用 gate v5 面板的分地形胜率评估，取“最差地形不降、平均不显著降”的那个；不要只选平均最高的。
  - 温度采样可作为轻量替代：按地形/来源行数的 $\alpha$ 次方采样（$\alpha$ 略大于 1），比重新打包整个数据集便宜；但要防止屋顶/雾被过度重复导致过拟合，需配合留出。

---

## 8. 专家迭代的数据管理：KataGo / Leela、旧标签、DAgger

- 引用：
  - Wu, *Accelerating Self-Play Learning in Go* (KataGo), 2019, [arXiv:1902.10565](https://arxiv.org/abs/1902.10565)；KataGo 方法文档：<https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md>。
  - Silver et al., *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm* (AlphaZero), 2017, [arXiv:1712.01815](https://arxiv.org/abs/1712.01815)；Science 2018。
  - Ross et al., *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* (DAgger), AISTATS 2011, [arXiv:1011.0686](https://arxiv.org/abs/1011.0686)。
  - Anthony et al., *Thinking Fast and Slow with Deep Learning and Tree Search* (Expert Iteration), NeurIPS 2017, [arXiv:1705.08439](https://arxiv.org/abs/1705.08439)。
  - Schaul et al., *Prioritized Experience Replay*, 2015, [arXiv:1511.05952](https://arxiv.org/abs/1511.05952)。
- 关键结论：
  - KataGo 通过一组技巧把自博弈学习加速约 **50×**：policy target pruning（把低概率目标剪掉、重归一化，避免 Dirichlet 噪声的“薄雾”污染）、playout cap randomization、辅助目标（ownership/score）、global pooling、nested bottleneck 等。**不是单一技巧**。
  - AlphaZero 系使用**近期对局回放窗口**（二手资料常引“约 50 万局”；**未在原文核实**），旧对局逐步淘汰；这是“旧老师标签会过时”的工程回应。
  - Leela Zero 是 AlphaZero 的开源复现，同样使用近期自博弈对局的回放缓冲并从中采样；具体窗口大小和采样权重随版本/配置变化（**未核实**，只作为工程参考）。
  - DAgger 的核心是**数据集聚合**：每轮用当前策略跑、请专家标注，把所有轮次数据合起来训练；理论上是 no-regret。原论文没有“按时间加权”的公式，时间加权是后续工程实践。
  - Expert Iteration：把“搜索（慢、强）”和“策略网络（快、弱）”分开，反复用搜索改进策略；和我们的老师-学生结构一致。
  - Prioritized replay：按 TD/advantage 等优先级采样，比均匀采样更有效；可作为“样本加权”的一般框架。
- 可信度：
  - KataGo/AlphaZero/DAgger/ExIt/Prioritized replay：**高**（论文结论）。
  - AlphaZero 具体窗口大小、KataGo 具体采样权重：**中/低**（二手或工程实现，未逐篇核实）。
- 对我们意味着什么：
  - **旧标签确实可能过时**：老师配置从 v3/v4 价值网、旧学生一路换到现在；60.3% 的行来自课程前历史 DAgger（主要是 L7）。这些行不是“错”，但它们的策略目标反映的是旧老师的偏好，可能和新老师/新关卡分布不一致。
  - 建议的数据管理组合（按性价比）：
    1. **近期加权**：按采集轮次/老师版本给样本权重（例如半衰期式），但保留旧数据做正则；不要一刀切删旧数据。
    2. **policy target pruning**：对老师搜索的访问分布，剪掉低于阈值的目标并重归一化；保留根节点噪声的探索质量，但别让噪声进入学生目标。
    3. **按 regret/不一致加权**：学生与老师分歧大、且老师有把握的状态优先；这比单纯“按时间”更贴近“哪里还能学”。
    4. **DAgger 聚合但分桶**：按地形/关卡/卡组/老师版本分桶，训练时分层采样；避免 L7 历史桶淹没其他桶。
    5. **旧标签抽检重标注**：定期抽一小批旧行，用当前老师重跑，测“旧标签与当前老师的一致率”；一致率低的桶降权。
  - 注意：KataGo 的 50× 是一组技巧的合力；不要指望只加“policy target pruning”就复制它的加速。

---

## 9. 课程 / 无监督环境设计（PLR、ACCEL）、示范起点、等价动作噪声

### 9.1 PLR / ACCEL

- 引用：
  - Jiang et al., *Replay-Guided Adversarial Environment Design* (PLR), NeurIPS 2021, [arXiv:2110.02439](https://arxiv.org/abs/2110.02439)。
  - Parker-Holder et al., *Evolving Curricula with Regret-Based Environment Design* (ACCEL), ICML 2022, [arXiv:2203.01302](https://arxiv.org/abs/2203.01302)。
- 关键结论：
  - PLR：按“当前策略在这些关卡上的 regret（学习潜力）”优先重放关卡；把“优先级”与“新鲜度/编辑”结合，避免只刷旧关卡。
  - ACCEL：在 PLR 基础上对高 regret 关卡做变异/编辑，持续生成“刚好在能力边界”的关卡；在 Procgen 上显著提升零样本泛化。
  - 共同点：**采集预算给“当前最不会、但还可学”的状态/关卡**，而不是均匀覆盖或固定课程。
- 可信度：**高**。
- 对我们意味着什么：
  - 我们的课程已经有“地形 × 倍率格”“示范起点格”“随机卡组格”“开局面板”等格子；可以把 PLR 的 regret 信号接进去：用**学生 vs 老师的胜率差/动作分歧**给格子打分，优先采集高 regret 格。
  - 但要注意门禁的“每地形不退步”：如果只按平均 regret 采，会继续偏向白天/L7；应按地形分层做 regret 排序，给屋顶/雾设最低配额。

### 9.2 示范状态当作起点（Go-Explore、Salimans 2018）

- 引用：
  - Ecoffet et al., *Go-Explore: a New Approach for Hard-Exploration Problems*, 2019, [arXiv:1901.10995](https://arxiv.org/abs/1901.10995)；Nature 2021 *First return, then explore*。
  - Salimans & Chen, *Learning Montezuma's Revenge from a Single Demonstration*, 2018, [arXiv:1812.03381](https://arxiv.org/abs/1812.03381)。
- 关键结论：
  - Go-Explore：先“回到”有希望的已发现状态（存档/恢复），再从那里探索；对硬探索任务极有效。
  - Salimans & Chen：把人类示范的中间状态作为 episode 起点（checkpoint replay），让 agent 从“接近成功”的状态学，而不是从开局硬探索。
- 可信度：**高**。
- 对我们意味着什么：
  - 我们的“示范起点格”（3.2%）就是同一思路；它应继续保留，并**按地形/关卡分层**，不要只在容易的白天关卡用。
  - 起点状态会改变状态分布：从示范中段开始，学生看到的是“老师已经处理过的盘面”，可能缺少早期布局的学习信号。建议同时保留“开局面板”（5.3%）和“示范起点”两类，按难度/地形配比。
  - 起点状态的标签来自老师在该状态的动作；如果同一状态有多个等价好动作，硬标签会引入噪声（见 9.3）。

### 9.3 等价动作多时的标签噪声

- 引用（代理，不是直接研究我们这个问题）：
  - Florence et al., *Implicit Behavioral Cloning*, CoRL 2021, [arXiv:2109.00137](https://arxiv.org/abs/2109.00137)。
  - Zaheer et al., *Deep Sets*, NeurIPS 2017, [arXiv:1703.06114](https://arxiv.org/abs/1703.06114)。
  - Lee et al., *Set Transformer*, ICML 2019, [arXiv:1810.00825](https://arxiv.org/abs/1810.00825)。
  - Müller et al., *When Does Label Smoothing Help?*, NeurIPS 2019, [arXiv:1906.02629](https://arxiv.org/abs/1906.02629)。
- 关键结论：
  - 当示范存在多模态/多峰动作时，显式回归/单标签 BC 会把多个好动作平均成坏动作；Implicit BC 用能量模型表达多模态。
  - Deep Sets / Set Transformer 给出置换不变/等变的集合建模；对“卡槽顺序无关”“等价卡组”这类对称性有用。
  - Label smoothing 能改善校准/泛化，但会损失一些区分度；它是对“硬标签可能有噪声”的通用缓解，不是等价动作的专门解法。
- 可信度：**中**（方法本身高；但**没有找到直接研究“PvZ 等价动作标签噪声”的论文**）。
- 对我们意味着什么：
  - 我们的动作空间里，等价/近似等价动作不少：同一行不同列、卡槽顺序、多个可种位置在当下等价；老师的搜索可能选其中一个，硬标签把其他等价动作当负样本。
  - 低成本缓解：
    1. **动作抽象/去重**：训练目标里把“同一结果”的动作合并（如果引擎可判定）；至少不要把“卡槽顺序不同但同一张卡”当成不同动作。
    2. **软标签**：如果老师能给出多个候选的访问分布/价值，保留 top-k 软目标，而不是只取 argmax。
    3. **置换不变输入**：卡包用集合编码（不带固定槽位坐标），配合卡 token dropout；这正是 task153 修订 1 的方向。
    4. **标签平滑/温度**：只在确认标签有噪声的头上用，别全局开。
  - 先做**标签审计**：抽样统计“同一状态下老师多次搜索的 top-1 一致率”“不同等价动作的标签分布”，再决定要不要上软标签。没有审计就上软标签，可能只是把老师的不确定性当成了真信号。

---

## 10. 排序与取舍

### 10.1 架构选项排序（含“保持 CNN、修数据”）

| 排名 | 选项 | 优点 | 缺点 | 建议/判据 |
|---:|---|---|---|---|
| 1 | **保持 CNN64，修数据配比 + 补输入 + 卡组增强** | 不改生产/C++；61 µs；直接针对已测瓶颈（L7 46.7%、屋顶 1.8%、随机卡组 0.2%、未知卡组 14.6%）；成本最低 | 不解决 L5/L45 的空间关系上限；如果数据修完仍差，还是要上结构 | **先做**。用 gate v5 分地形 + 未知卡组判，不看平均 |
| 2 | **CNN64 + 地形条件化（FiLM / 小 adapter）** | 参数少、单次前向几乎不变；利用已知地形；比 MoE 简单；C++ 实现容易 | 屋顶/雾数据少，条件化可能过拟合；不一定解决卡组泛化 | 与 CNN64 等参对照；task158 之后做 |
| 3 | **等参 CNN128 / 加宽** | 实现最简单；task23 白天早关 +13 局；容量确有一部分收益 | task23：未知卡组 60.0%→53.8%、夜间 51.4%→42.6%、C++ 慢 3.1×；数据不足处更伤 OOD | 只在数据配比修正后重测；必须带未知卡组/分地形 |
| 4 | **共享主干 + 按地形硬路由的专家头** | 地形已知，不用学路由；推理只算一个专家，速度≈CNN64；门禁可按地形晋升；直接验证“干扰”假设 | 每个地形数据更少（屋顶 1.8%、雾 3.7%），专家头可能欠拟合；需要 C++ 路由/导出；未知卡组仍可能差 | 先看 task158；专家臂须在所有地形不差于 CNN128 且推理成本低 |
| 5 | **实体 + 注意力 v3（根节点先用，续走 CNN）** | 卡组顺序不敏感、精确坐标/关系；L5/L45/屋顶天花板最高；根节点只占 0.75% 前向；可复用旧 GameplayModelV1 设计 | 数据饥渴；task153 已暴露实现陷阱（卡包池化偏置、缺 group、单头偏置）；C++ 通用前向工作量大；20M 行可能不够 | 等参 CNN81/128 对照、多种子、未知卡组 + 分地形；赢过等参 CNN 才算结构收益 |
| 6 | **学出来的 MoE / Soft MoE** | 理论上参数效率高；多任务 RL 有成功案例 | Top-1 小规模失败；Soft MoE 要过所有专家，CPU 变慢；每专家数据更少；OOD 证据混合 | 最后；必须带等参 dense 对照、路由分布/专家多样性检查 |
| 7 | **帧堆叠 / 记忆 / GRU**（不推荐现在做） | 雾夜/部分可观测可能受益 | 没有证据表明记忆是当前瓶颈；PPO 时代 GRU 让更新 9 s→110 s；复杂度高 | 先做“加 1–2 帧历史”的廉价消融；雾夜/屋顶差距若主要来自不可见信息再考虑 |

**判读顺序建议：** 先做 1（数据/输入），同时做 2（FiLM）和 4（task158 硬路由）的小规模对照；3 只在数据修正后重测；5 需要等“富 token 输入 + 多卡组数据 + C++ 通用前向”就绪；6 最后。不要跳过 1 直接上 5/6。

### 10.2 MoE 选项排序

| 排名 | 选项 | 优点 | 缺点 | 建议 |
|---:|---|---|---|---|
| 1 | **不用 MoE；先用 FiLM/地形条件化** | 最便宜；保留共享数据；单次前向不变；直接利用已知地形 | 条件化容量有限；极端地形可能仍欠拟合 | **默认**。先证明 FiLM 不够，再谈专家 |
| 2 | **共享主干 + 按地形硬路由的专家头/末层** | 不做路由器；速度≈单专家；可按地形晋升；可解释 | 5 份头参数；屋顶/雾数据薄；共享主干仍可能干扰 | task158 的主对照；通过“所有地形不差 + 成本低”才采纳 |
| 3 | **Soft MoE（共享 + 路由专家，整盘路由）** | 唯一有 RL 参数缩放证据的 MoE；软路由不塌缩 | 过所有专家 → CPU 慢；需要规模；每专家数据少；难做 C++ | 数据上来后再试；带等参 dense 对照 |
| 4 | **Top-1/Top-2 稀疏门控** | 推理只算少数专家；理论省算力 | 小规模 RL 证据差（Top-1 不随规模提升）；负载均衡损失救不了；路由易塌缩 | **不推荐**先试 |
| 5 | **每地形完全独立网络** | 无干扰；可单独晋升/回滚 | 参数最多、每网络数据最少；屋顶/雾几乎不可训；维护 5 套 | 只在“单独训练明显更好且数据足够”时考虑屋顶/雾两个小地形 |
| 6 | **按关卡/卡组路由** | 粒度最细，理论上最贴合 | 路由会退化成关卡分类器；未知卡组无法路由；数据更碎 | **不推荐** |

**关键提醒：** 任何 MoE 方案都必须报告“路由分布 vs 地形”的混淆矩阵。如果路由几乎等价于地形分类器，那它没有带来超出“硬路由”的价值，还多了训练不稳定；如果路由与地形不一致，则要解释它在学什么，否则可能是过拟合关卡 ID。

### 10.3 数据配比与采集策略排序

| 排名 | 选项 | 优点 | 缺点 | 建议/判据 |
|---:|---|---|---|---|
| 1 | **修来源配比：L7 历史行封顶/下采样；提高随机卡组格、示范起点、屋顶/雾/泳池** | 最高 ROI；直接针对已测偏差（L7 46.7%、屋顶 1.8%、随机卡组 0.2%）；不改模型/C++ | 可能损失 L7 上的既有优势；需要重新打包/采样；要防屋顶/雾过拟合 | **先做 3 臂配方对照**（A / B / floor 约束），用 gate v5 最差地形判 |
| 2 | **按地形/卡组分层采样 + 温度/上限；目标用最差地形（group DRO 思路）** | 与门禁同构；避免白天淹没小地形；比 DoReMi 代理便宜 | 权重需要调；可能牺牲平均；卡组×地形组合稀疏 | 用 CNN64 代理扫权重；记录每地形有效样本数 |
| 3 | **数据质量：policy target pruning + 近期加权 + on-policy DAgger 份额 + 旧标签降权/抽检** | 直接改善标签质量；KataGo/DAgger 有依据；不需要新数据 | 需要改打包/训练器；权重不当会丢旧知识；需要标签年龄统计 | 先做标签审计（旧标签 vs 当前老师一致率），再定权重 |
| 4 | **主动采集：师生不一致/regret 驱动（PLR/ACCEL 思路）** | 把采集预算给“最不会且可学”的状态；对未知卡组/屋顶/雾最有效 | 需要 regret 估计与采集调度；可能过拟合高 regret 格；工程量大 | 先在小面板做离线 regret 排序，验证与胜率提升的相关性 |
| 5 | **示范起点 + 随机卡组格作为课程；去重/软标签处理等价动作** | 已有基础设施（3.2% / 0.2%）；对探索和组合泛化都有效 | 起点会改变状态分布；等价动作标签噪声；随机卡组太少 | 提高配额 + 分层；先做标签审计再上软标签 |
| 6 | **DoReMi 式 proxy 权重优化** | 有理论/实证支持（最差域目标）；可自动化 | LM 结论外推；需要代理训练；我们的“域”相关性强，权重可能不稳定 | 在 1/2 不够时做；代理用 CNN64，目标用 gate 分地形 |
| 7 | **扩大采集规模（更多局/更多老师搜索）** | 直接增加覆盖；对数据饥渴的结构（v3/MoE）必要 | 最贵；台式机产能有限；不修配比的话只是放大偏差 | 放在 1–5 之后；先修配比再扩量 |
| 8 | **按专家分配数据（MoE 专属）** | 每个专家数据更纯 | 总数据不变时每专家更少；小地形更糟 | 只有 MoE 被证明有效后再做 |

---

## 11. 本调研回答不了、必须实测的问题

1. **v3 实体注意力在 20M 行 + 富 token 输入下是否赢过等参 CNN？** task153 只有 152 局、23 套训练卡组、旧存档缺 y/滚动坚果，不能回答。
2. **地形干扰是否真实存在？** task158 的四/五臂对照是直接证据；在它出结果前，FiLM/硬路由都只是假设。
3. **MoE 在我们这个规模是否有效？** 没有直接文献；Obando-Ceron 是 Atari + 更大网络，GMoE 是视觉；只能靠 task158 的学门控臂 + 等参 dense 对照回答。
4. **最优配比权重是多少？** DoReMi/data mixing laws 不能直接给我们的数；需要 CNN64 代理扫 A/B/floor 三臂，并用 gate v5 分地形判。
5. **旧标签到底有多过时？** 需要按老师版本/轮次统计“旧标签 vs 当前老师 top-1 一致率”，以及降权后的增益。
6. **等价动作噪声有多大？** 需要标签审计（同一状态多次搜索的 top-1 一致率、等价动作的标签分布）；没有审计就上软标签可能引入新偏差。
7. **屋顶/雾的数据下限是多少？** 需要逐步增加配额，观察门禁通过率与过拟合；1.8%/3.7% 显然太低，但“够”是多少没有文献可抄。
8. **Transformer/注意力的 CPU 前向在根节点频率下是否可接受？** task153 只测了 Python 端比例；需要 C++ v3 前向的实测 µs，再决定是否值得。

---

## 12. 未核实 / 二手 / 外推的说法（明确标注）

- **AlphaStar 1.39 亿参数**：来自二手摘要（academia.edu 等），未在 Nature 正文逐字核实；32 TPU×44 天来自 DeepMind 原文 PDF 摘要。
- **AlphaZero 回放窗口约 50 万局**：来自二手博客/技术总结，未在原文核实；不同实现差异很大。
- **AlphaGo 快 rollout “约 24% 准确率、约 2 µs/步”**：来自二手技术总结（原始 Nature 2016 对 SL policy 给出约 57% 准确率和约 3 ms/步；rollout 的 24%/2 µs 未逐字核实）。
- **KataGo 的具体回放窗口大小、样本权重公式**：KataGo 论文强调 policy target pruning / playout cap randomization / 辅助目标等；具体窗口与权重是工程实现，未在论文中逐条核实。
- **Leela Zero 的回放窗口/采样细节**：开源实现，随版本变化；本文只把它作为“近期回放 + 采样”的工程参考，未核对其具体参数。
- **“Transformer 在 10M–30M 行不如/等于 CNN”**：没有找到同规模、同任务的直接对照；本文是由 ViT/DeiT/ConvNeXt/IRIS 的结果外推。**这是本报告最大的不确定性之一。**
- **“MoE 对 OOD 是帮还是伤”**：RL/BC 领域没有直接证据；GMoE 是视觉 DG，LLM 的“MoE 需要更多数据”是另一领域；本文按“混合/偏保守”处理。
- **“等价动作标签噪声”**：没有找到直接研究 PvZ 式等价动作的论文；Implicit BC / Deep Sets / Set Transformer / label smoothing 是代理方法。
- **“Data mixing laws 适用于 RL/BC”**：未验证；DoReMi 的 group DRO 思路可迁移，但具体函数形式不能直接套。
- **“温度采样 τ=5”**：Johnson et al. 2016 使用温度采样，但具体 τ 值本文未逐字核实；只建议“用 $\alpha>1$ 提高低资源域权重”这一原则。
- **CARE 的缩写/细节**：通过 MOORE 的相关工作段确认 CARE = Sodhani et al. 2021 *Context-based Representations*；论文细节未逐篇读。
- **“宽 128 伤未知卡组/夜间”**：来自项目 task23 实测（二手项目记录），不是文献；但它是我们最重要的本地证据。

---

## 13. 引用索引

| # | 文献 | 年份 | 链接 | 可信度 |
|---:|---|---|---|---|
| 1 | Vinyals et al., Grandmaster level in StarCraft II (AlphaStar) | 2019 | [DOI](https://doi.org/10.1038/s41586-019-1724-z) | 高（架构）/中（参数） |
| 2 | Berner et al., Dota 2 with Large Scale Deep RL (OpenAI Five) | 2019 | [arXiv:1912.06680](https://arxiv.org/abs/1912.06680) | 高 |
| 3 | Reed et al., A Generalist Agent (Gato) | 2022 | [arXiv:2205.06175](https://arxiv.org/abs/2205.06175) | 高 |
| 4 | Schrittwieser et al., MuZero | 2019/2020 | [arXiv:1911.08265](https://arxiv.org/abs/1911.08265) | 高 |
| 5 | Dosovitskiy et al., ViT | 2020 | [arXiv:2010.11929](https://arxiv.org/abs/2010.11929) | 高 |
| 6 | Touvron et al., DeiT | 2020 | [arXiv:2012.12877](https://arxiv.org/abs/2012.12877) | 高 |
| 7 | Liu et al., ConvNeXt | 2022 | [arXiv:2201.03545](https://arxiv.org/abs/2201.03545) | 高 |
| 8 | Micheli et al., IRIS | 2022 | [arXiv:2209.00588](https://arxiv.org/abs/2209.00588) | 高（世界模型）/中（外推） |
| 9 | Obando-Ceron et al., MoE Unlock Parameter Scaling for Deep RL | 2024 | [arXiv:2402.08609](https://arxiv.org/abs/2402.08609) | 高 |
| 10 | Puigcerver et al., From Sparse to Soft MoE | 2023 | [arXiv:2308.00951](https://arxiv.org/abs/2308.00951) | 高 |
| 11 | Hendawy et al., MOORE | 2023/2024 | [arXiv:2311.11385](https://arxiv.org/abs/2311.11385) | 高 |
| 12 | Sodhani et al., CARE | 2021 | [arXiv:2102.06177](https://arxiv.org/abs/2102.06177) | 高 |
| 13 | Yang et al., Soft Modularization | 2020 | [arXiv:2003.13661](https://arxiv.org/abs/2003.13661) | 高 |
| 14 | Shazeer et al., Sparsely-Gated MoE | 2017 | [arXiv:1701.06538](https://arxiv.org/abs/1701.06538) | 高 |
| 15 | Fedus et al., Switch Transformers | 2021 | [arXiv:2101.03961](https://arxiv.org/abs/2101.03961) | 高 |
| 16 | Chi et al., Representation Collapse of Sparse MoE | 2022 | [arXiv:2204.09179](https://arxiv.org/abs/2204.09179) | 高 |
| 17 | Li et al., GMoE (Sparse MoE are Domain Generalizable) | 2022/2023 | [arXiv:2206.04046](https://arxiv.org/abs/2206.04046) | 中 |
| 18 | Yu et al., PCGrad | 2020 | [arXiv:2001.06782](https://arxiv.org/abs/2001.06782) | 高 |
| 19 | Perez et al., FiLM | 2017/2018 | [arXiv:1709.07871](https://arxiv.org/abs/1709.07871) | 高 |
| 20 | Houlsby et al., Adapters | 2019 | [arXiv:1902.00751](https://arxiv.org/abs/1902.00751) | 高 |
| 21 | Tuyls et al., Scaling Laws for Imitation Learning | 2023 | [arXiv:2307.09423](https://arxiv.org/abs/2307.09423) | 高 |
| 22 | Hoffmann et al., Chinchilla | 2022 | [arXiv:2203.15556](https://arxiv.org/abs/2203.15556) | 高（LM）/中（外推） |
| 23 | Cobbe et al., CoinRun | 2018/2019 | [arXiv:1812.02341](https://arxiv.org/abs/1812.02341) | 高 |
| 24 | Cobbe et al., Procgen | 2019/2020 | [arXiv:1912.01588](https://arxiv.org/abs/1912.01588) | 高 |
| 25 | Tobin et al., Domain Randomization | 2017 | [arXiv:1703.06907](https://arxiv.org/abs/1703.06907) | 中 |
| 26 | You et al., Combinational Q-Learning for Dou Di Zhu | 2019 | [arXiv:1901.08925](https://arxiv.org/abs/1901.08925) | 低–中 |
| 27 | Zha et al., RLCard | 2020 | [IJCAI](https://www.ijcai.org/proceedings/2020/0764.pdf) | 低–中 |
| 28 | Silver et al., AlphaGo | 2016 | [DOI](https://doi.org/10.1038/nature16961) | 高 |
| 29 | Hinton et al., Distilling the Knowledge | 2015 | [arXiv:1503.02531](https://arxiv.org/abs/1503.02531) | 高 |
| 30 | Rusu et al., Policy Distillation | 2015 | [arXiv:1511.06295](https://arxiv.org/abs/1511.06295) | 高 |
| 31 | Jacob et al., Integer-Arithmetic-Only Inference | 2017 | [arXiv:1712.05877](https://arxiv.org/abs/1712.05877) | 高 |
| 32 | Xie et al., DoReMi | 2023 | [arXiv:2305.10429](https://arxiv.org/abs/2305.10429) | 高 |
| 33 | Ye et al., Data Mixing Laws | 2024 | [arXiv:2403.16952](https://arxiv.org/abs/2403.16952) | 中 |
| 34 | Johnson et al., Google Multilingual NMT | 2017 | [arXiv:1611.04558](https://arxiv.org/abs/1611.04558) | 高 |
| 35 | Wu, Accelerating Self-Play Learning in Go (KataGo) | 2019 | [arXiv:1902.10565](https://arxiv.org/abs/1902.10565) | 高 |
| 36 | Silver et al., AlphaZero | 2017/2018 | [arXiv:1712.01815](https://arxiv.org/abs/1712.01815) | 高 |
| 37 | Ross et al., DAgger | 2011 | [arXiv:1011.0686](https://arxiv.org/abs/1011.0686) | 高 |
| 38 | Anthony et al., Expert Iteration | 2017 | [arXiv:1705.08439](https://arxiv.org/abs/1705.08439) | 高 |
| 39 | Schaul et al., Prioritized Experience Replay | 2015 | [arXiv:1511.05952](https://arxiv.org/abs/1511.05952) | 高 |
| 40 | Jiang et al., PLR | 2021 | [arXiv:2110.02439](https://arxiv.org/abs/2110.02439) | 高 |
| 41 | Parker-Holder et al., ACCEL | 2022 | [arXiv:2203.01302](https://arxiv.org/abs/2203.01302) | 高 |
| 42 | Ecoffet et al., Go-Explore | 2019 | [arXiv:1901.10995](https://arxiv.org/abs/1901.10995) | 高 |
| 43 | Salimans & Chen, Montezuma from One Demo | 2018 | [arXiv:1812.03381](https://arxiv.org/abs/1812.03381) | 高 |
| 44 | Florence et al., Implicit BC | 2021 | [arXiv:2109.00137](https://arxiv.org/abs/2109.00137) | 中（代理） |
| 45 | Zaheer et al., Deep Sets | 2017 | [arXiv:1703.06114](https://arxiv.org/abs/1703.06114) | 高 |
| 46 | Lee et al., Set Transformer | 2018/2019 | [arXiv:1810.00825](https://arxiv.org/abs/1810.00825) | 高 |
| 47 | Müller et al., When Does Label Smoothing Help? | 2019 | [arXiv:1906.02629](https://arxiv.org/abs/1906.02629) | 中（代理） |
