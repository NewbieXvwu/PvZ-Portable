# llmplay-decks：GPT Pro 扩卡组包

解压全部 `llmplay_pkg.7z.*` 分卷；原样的 `llmplay_res.zip` 放在生成的
`llmplay_pkg/` 旁。进入接口目录，阅读 `PROMPT.md`，先选择分片，再开始对局。

```python
import os
os.environ["LLMPLAY_SHARDS"] = "3"
os.environ["LLMPLAY_SHARD"] = "0"  # 三个会话分别填0、1、2
os.environ["LLMPLAY_OUT"] = "/mnt/data/out"  # 共用文件系统时，各会话指定不同目录
import llmplay
print(llmplay.next_task())
print(llmplay.progress()["coverage"])
```

默认片0为白天＋夜间，片1为泳池＋雾，片2为屋顶＋Boss。
可传 `next_task(shard=0, shards=N)`（0起编号，N=1–41）；N不是3时按关卡轮转分配。
每次新进程都保持原片号。每关有独立百万种子窗口，用完顺延，窗口不重叠。

任务是40个普通选卡冒险关＋接口支持的L50，原版1.0倍，不截波。
每项=(关卡,种子)，自选新卡组，与本关全部已登记胜利卡组至少差2张。
一胜登记并换组；同组最多三败，第三败存档的note写一句失败原因。
不做稳健复验、打法层或加压。门禁/挑选留出签名跨关拒绝。

白天/泳池每关目标12套，夜间/雾/屋顶/Boss16套，共592套基础目标。
每关至少3套≤7张、3套8张；每地形每张合法非黑名单卡至少进入2套不同胜利卡组。
卡池与大小配额看`next_task()["item"]`，每套用`reason`写清经济/主力/防守/应急/地形卡，
存档字段仍为`deck_reason`。达到目标后继续增长并优先补覆盖缺口；按≥10小时持续工作，
只有用户说停才停。

进度与报告逐局写入`llmplay.out_dir()`，`out.zip`自动更新。
占位、`Game.save/load/load_archive`、`resume(out.zip)`和存档格式均沿用上一版。
包内carry只读结论，不复演；旧存档始终使用其中记录的卡组和环境。

覆盖与重点卡配置在`configs/llmplay_seeds.json`的`decks`中：
`coverage_per_card`为默认次数，`coverage_priority`可设`{"fog":{"27":4}}`这样的重点目标。
黑名单在同一配置，留出签名来自原样的`configs/deploy17_deck_holdouts.json`。

在仓库重打接口（不重建引擎或资源）：

```bash
python3 scripts/llmplay/build_package.py \
  --reuse-package ~/pvz-logs/llmplay5/llmplay_pkg \
  --resource-zip ~/pvz-logs/llmplay5/llmplay_res.zip \
  --out ~/pvz-logs/llmplay-decks
```

产物不进git。引擎、资源、存档、占位、断点续跑与自动out.zip机制沿用llmplay5。
