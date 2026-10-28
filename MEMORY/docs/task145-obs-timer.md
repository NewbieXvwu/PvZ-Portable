# task145：可见计时进学生输入（v2timer，默认不启用）

分支 `obstimer`，以 `origin/obsup2`（task143，8d18b4e5，v2sleep + build7）为底。
结论另见 `/home/newbiexvwu/pvz-logs/task145-brief.md` 末尾。

## 1. 新版本是什么

`v2timer` = `v2sleep` + 4 个棋盘通道（总 25，值编码器 v5 对应 +12）。前缀与 `v2sleep` 逐位相同。

| 通道 | 内容 | 归一化 |
|---|---|---|
| 0 | 引爆 / 唤醒进度 | `1 - do_special_countdown / 100`，任意 `special > 0` 的植物 |
| 1 | 状态进度 | `1 - state_countdown / 该状态满值`，只覆盖大嘴花、窝瓜、玉米加农炮 |
| 2 | 传送带卡序 | 行 0、列 = 卡槽位置，`(type + 1) / 48`；0 = 空槽 |
| 3 | 传送带就绪 | 该位置这张卡现在能不能种（active 且冷却 0） |

通道 0 与基础通道 6 在樱桃/辣椒上重复是有意的：通道 6 只认 `kind in (2, 20)`，
毁灭菇、寒冰菇、咖啡豆正是它漏掉的那三种；新通道自成一套，不再按种类开白名单。

状态满值表（`python/pvz_value_net.py` 的 `_timer_state`，数值取自 `src/Lawn/Plant.cpp`）：
大嘴花咬 70 / 咀嚼 4000；窝瓜看目标 80 / 起跳前 45 / 上升 50 / 下落 10 / 落地 100；
玉米炮装填 500。土豆雷的武装已经在通道 6 里，不重复。

雾里被遮住的植物在 v2 段里 `state` / `state_countdown` 是 -1，新通道保持 0（未知），与通道 6 一致。
传送带卡序读的是 `frame["packets"]` 的 `index` —— 那正是屏幕上从左到右的位置，
用掉一张后 `SeedBank::RemoveSeed` 会把后面的卡整体前移，所以同一张卡的位置会变。

## 2. 旧版本一位没动

- 新函数 `timer_features` 用**小写**模块常量（`_timer_state` / `_timer_special`）：
  `encoder_signature` 会把模块里的大写 int/float 常量算进指纹，加一个大写常量就会
  把 v1/v2 这些冻结版本的签名一起改掉。
- `v2sleep` 的签名在改动前实测为 `309266015e12…`，这次把它写进 `encoder_signature` 的冻结表，
  于是新增 `v2timer` 分支不会让已有 v2sleep 档案失效。实测改完仍是同一串。
- 新版本签名 `813d6f021be5…`，与 v2sleep 不同，不会命中旧缓存。

## 3. 重打训练行的估算（只读推算，没跑）

参考 task143 在 `~/pvz-logs/curriculum_state.pipeline` 上的实测（`task143-deploy15-refresh.md` §3）：
874 个标签段、11.2 GiB、21.1M 行；打包速率 25 s/段/worker（热缓存）、57–123 s/段/worker（冷缓存）。

**v2timer 与 v2sleep 的关键差别：没有零填充复用路径。**
v2sleep 能复用 `cache/v2` 的 74,933 个条目（只补零通道视图，约 73 MB）；
v2timer 的 4 个新通道要从同一帧现算，补零是错的，所以整份重编码 —— 走的是冷缓存速率。

| 项 | 估算 |
|---|---|
| 段重建 | 874 段 × 57–123 s ÷ 2 worker ≈ **7.0–15.0 小时**（4 worker ≈ 3.5–7.5 h） |
| 磁盘（段） | 净增约 11.2 GiB（旧段不回收） |
| 磁盘（缓存） | 新命名空间，量级同 `cache/v2` 的 8.0 GiB；每行 25 通道 vs 21，约 +19% → 约 9.5 GiB |
| 峰值 | 约 +21 GiB（数据目录 32.9 → 54 GiB 量级，829 GB 空闲无压力） |

想缩短的话有一个不改语义的办法：写一个「读 v2sleep 缓存行 → 现算 4 个新通道 → 拼宽 → 重压」的
复用路径，把重编码降到只剩 LZMA 写入（约 0.63 s/档案，74,933 份 ≈ 13 h 单线程 / 3–6 h 多 worker）。
这次没做（任务书写明不训练不部署，先把版本和配方立住）。

## 4. 原生（C++）侧还没有这个版本

`export_student` 对 `v2timer` 直接报错。引擎侧输入版本号只到 8（v2sleep）；
若现在导出，C++ 学生会按 v2sleep 的布局读一个更宽的输入，特征错位不会报错、只会静默变差。
要上线原生 rollout 得先在 C++ 侧加第 9 号输入版本（通道布局同上）并做 Python/C++ 对拍 —— 不在本任务范围。

训练与评估走 Python 学生（`StudentHistory`）与价值编码器 v5，不受这条限制。

## 5. 测试

`ulimit -v 8000000`，同一时间一个 pytest，只跑相关用例：

| 范围 | 结果 |
|---|---|
| `test_timer_inputs.py`（新增 10 条：前缀一致、四个通道取值、雾遮蔽、卡序前移、编码器 v5、签名、配方、热启动、导出拒绝） | 10 passed |
| `test_bungee_inputs.py test_sleep_inputs.py test_observation_upgrade.py test_student_observation.py` | 21 passed, 4 skipped（跳过的是需要引擎二进制的原生对拍） |

没有加任何启动自检或哈希校验；`encoder_signature` 是既有的缓存命名机制，不是新加的自证。
