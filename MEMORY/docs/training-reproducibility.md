# task96：同配方、同 seed 的训练复现

2026-10-26，repro 分支。**已确认现行训练存在 CUDA embedding 梯度非确定性；未发现训练输入漂移，训练器重写的旧归因不成立。建议开启确定性配置，再让 task92 比较不同 seed。** 本任务没有修改生产或操作 qc；所有诊断顺序运行，没有启动游戏进程。

历史 r11→A 的 heldout 从166/192降至124/192，不能把这42局全部归因于某个算子或某次显存状态：历史训练缺少 GPU/确定性配置回执，亦未在本任务中重新评估泛化。

## 数据与代码

r11/A 都有348标签、16,982,931行、6,605,039株、55,773个seed，验证1,704,229行。两份 `source_rows.json` 逐标签计数相同，文件SHA256同为 `fd0f9c12901b4e4c7015cdf40515b3167c6ba39c791d86145e441d93f1500039`。CodeBuddy已对348个引用段做过内容SHA比较，全部相同；本次沿用该结果，没有再读9.6GB重算。补充元数据显示大小全同、独立inode，r11段mtime都早于训练。[逐标签证据](task96-data-metadata.json)保留行数与mtime；原内容检查未提交逐段摘要清单，这是历史证据限制。

两轮名义pack均为 `7d15b915a2192805.npz`；r11侧索引已被prune，A侧81KB文件是正常段索引。**pack名字不是内容指纹**：`archive_fingerprint`依据相对路径、大小和可选训练行清单；pack名再哈希行选项，不哈希实际数据内容。同名等长重标可能逃过它。

部署日志表明r11训练时部署的是 **4f4d217b**，后续评估的三份源码SHA与该提交吻合。把主检出的930d776d当成r11运行版本是错误归因：它连r11已用的recipe/identity入口都没有。4f4d217b与A的bec2b9e6相比：

| 训练链 | 差异 |
| --- | --- |
| train_rollout_policy、pvz_recipe、pvz_training_pack、pvz_value_net | 字节完全相同 |
| RolloutNet、features、encode_frames/encode_student_frames/target_values | 对应AST相同 |
| pack_training_rows / pvz_training_rows | 归档seed解析兼容示范起点目录；本次读取的是相同已编码段 |
| pvz_rollout_policy | 导出版本号增加v2size，不涉及训练网络 |
| pvz_student_observation / pvz_frame_store | 新增观测版本、兼容追加字段与签名登记；已有段未重编码 |

[源码与部署证据](task96-code-provenance.json)保留完整diff。部署版本得到交叉支持，但未记录的训练时单文件覆盖无法事后绝对排除。`encoder_signature`仅覆盖部分特征/目标编码函数和常量，v2还有冻结语义签名；它不覆盖训练循环、优化器、整个加载链或运行库，不能充当训练源码指纹。

## 已有完整pack双跑

任务书要求的同pack、seed1、各1epoch双跑已由CodeBuddy完成，未重跑。除输出路径外配方相同；实际比较21个权重张量，**21/21不同**。

| 项 | run1 | run2 |
| --- | ---: | ---: |
| train loss | 4.2911643558 | 4.3041004138 |
| val loss | 3.7510763180 | 3.7643170536 |
| epoch秒 | 890.387 | 334.052 |
| 含加载总秒 | 1224.317 | 591.676 |
| 规范张量SHA前缀 | 5d52761abc8b… | 148fa6bef2ee… |

[双跑证据](task96-existing-runs.json)含完整张量摘要和差异。第一轮epoch在15分钟内，但含加载超过15分钟；如实记录，不为补齐时限而重训。两次耗时不能作为确定性开销比。

**比较必须分组。** r11与A均为12epoch；上述诊断均为1epoch。训练器的cosine调度周期依赖总步数，所以诊断第1epoch与历史第1epoch的学习率轨迹不同；四个run的loss不能跨组当作同配置对照。有效比较是r11↔A、run1↔run2各自组内。

训练入口先 `torch.manual_seed(seed)`，每epoch使用CPU `torch.randperm`；验证使用排序seed集合与独立 `random.Random(validation.seed)`。**没有DataLoader**；precompute_workers不是其worker数。后续控制实测初始化、排列及输入相同，分歧发生在反向梯度。

## 分歧来源与确定性开关

在原pack固定30段、1,046,710行、batch2048、seed1、1epoch/461步上，使用冻结bec2b9e6训练器。已有的9例未重跑；本次仅补2例未设workspace的确定性配置，总训练墙时约50秒。完整结果见 [中尺度控制](task96-scale-probe.json)。

| 配置 | 次数 | 结果 |
| --- | ---: | --- |
| 确定性配置 + workspace=:4096:8 | 5 | 最终张量与逐步loss全同，含此前显存压力用例 |
| 默认配置、未设workspace | 4 | 最终张量两两不同；其中两次报告相同空闲显存仍不同 |
| 确定性配置、未设workspace（本次补充） | 2 | 相互相同，且与上述5例完全相同 |

七次确定性配置的张量SHA均为 `78cfef54734572ec67ff05d7ea8923b99b04e791a534819a41a30ea18734d399`，逐步loss摘要也相同。补充两例epoch17.53/20.52秒、val loss均4.7820728496，无确定性报错。“确定性配置”在此为 `use_deterministic_algorithms(True)`、cuDNN deterministic=True、benchmark=False；不是只改变workspace。

定位得到的第一处分叉：

- 同一批数据、相同初始化下，第1步forward/loss相同；反向后唯独 `packet_embedding.weight` 梯度不同，第一次更新后也只有它的参数不同。证据：[模块定位](task96-localize.json)、[梯度定位](task96-grad-probe.json)。
- `[B,10]` 的packet查表反向要把重复id梯度归并到表行。独立算子控制中，`[2048,10]` 默认10次得到10种梯度，开启确定性后只剩1种且不报错；裸index_add默认5次5种、确定性1种。证据：[查表形状](task96-embedding-shapes.json)、[归并算子](task96-embedding-probe.json)。因此**没有报错不等于没有非确定路径**：PyTorch可以直接选择确定性实现。
- 原来的小段控制另显示workspace从4096改为16也能改变后续数值。[小段证据](task96-determinism-probe.json)。它是另一条配置轴，不能解释embedding梯度本身，也不能从“确定性+workspace”组合对照声称workspace单独修好了问题。本次两例证明：本中尺度路径未固定workspace也能在确定性配置下复现。

未采kernel trace，不声称已证明具体atomicAdd或cuBLAS选核过程。当前环境为torch2.14.0+cu132、CUDA13.2、cuDNN92400、RTX5080。结果限定于实际运行的样本与环境，不外推跨库版本、跨设备或完整1698万行都逐位一致。

## 成本与建议

已有中尺度交替对照 default→strict→default→strict，epoch均值17.22秒→18.34秒，**观察比值1.065**；但default自身13.51/20.93秒相差1.55倍，宿主噪声大于6.5%。因此这是带负载噪声的测量，不能当作精确生产开销，也不能据此做“无显著开销”的统计断言。

[算子计时](task96-det-cost.json)测得上述查表backward中位数0.297→0.368毫秒，**1.24倍、每次约多0.071毫秒**。完整训练15,278,702行、batch2048，7461步/epoch；若只累加这一处，约多0.53秒，相对历史221秒约0.24%。这只是单算子外推，未包含其他确定性实现、I/O和调度成本，**不是整轮实测**。

建议采用确定性配置作为同seed复现与task92种子比较的基线，并固定/记录workspace。task92仍可比较不同初始化与排列产生的种子方差；它不能代替消除或记录同seed数值漂移。历史42/192的泛化落差仍需由多种子训练与行为评估量化，本任务不把它定性为“r11幸运”。生产未改；建议入口设置为：

```python
# 在首次CUDA上下文/库初始化前固定环境；随后设置训练开关。
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

冻结方案：每轮保存resolved recipe、实际源码版本/dirty diff摘要、训练链源码SHA、环境与上述开关、初始化与排列摘要；保存标签/pack索引和引用段的**内容指纹清单**，复用现有不可变段，不复制整库囤积。配方来源revision与运行源码revision分开记录。文件SHA负责完整性，张量摘要负责比较实际权重。

## 范围与保留限制

完整pack的额外strict/default/strict控制曾因缺页退化至约10秒/步，并触及900秒上限；另一次速率尝试亦失败。按7461步估算约21小时/epoch，继续重跑会争用采集/训练资源。按用户新规不再追加这项额外全量验收：原任务的完整双跑已有结果，原因定位与开关控制已有独立证据；**完整pack在严格模式下逐位复现及生产整轮开销仍未证明**，如实作为限制，不用中尺度结果冒充。

[历史并发记录](task96-concurrency.json)恢复了worker数和MemAvailable，未恢复GPU配置。run1/run2宿主负载相近而耗时差2.7倍，不能用epoch时间反推workspace或GPU争用。确定性机制已在现行训练路径复现；具体历史run受到哪种配置/负载影响无法回溯。

复核原始数值可直接读上述JSON及 `artifacts/task96/` 回执，无需再次训练、全量重哈希或重放游戏。诊断脚本保留在本分支；不会部署或自动启动生产训练。
