# task166：门禁与挑选的评估单元预留

## 依据与范围

已读`origin/gateslim:MEMORY/docs/task166-gate-slim.md`：同一v5完整门禁，1评估单元
约90分钟，4评估单元约31分钟。保留完整面板、种子协议、胜负判断与模型训练配方。
从`origin/deploy17`的`310150dc`建立独立`gateunits`分支，只写代码、提交推送，不部署。

## 实现

`scripts/curriculum_queue.py`增加`--min-eval-units N`，默认4，正整数。
有待处理后台任务时，从采集组的并发上限提前扣掉N个名额；没有后台任务时采集可用全部名额。
标准`--workers 16`配置默认采集最多12、门禁/挑选4。N是资源分配设置，控制器重启后的
存量任务也使用当前设置，不改变其冻结配方、面板或种子块。

门禁与挑选共用相同准入逻辑，移除挑选原先固定最多2单元的限制。该实现分配预留宽度N，
不主动抢占额外采集名额。若入队时已有一组占满16名额，后台等该组完成，下组按12名额启动；
若内存不足，后台等`MemAvailable >= 3 GiB + N × 实测单worker RSS`，不会静默缩成1单元。
等待每30秒记原因与需求，内存需求超过整机总量时明确失败，避免不可能满足的静默等待。

采集内存准入使用`max(打包/训练预估峰值, N × 单worker RSS) - 后台已驻留匿名内存`的
剩余预留；单worker探针结果在第一组准入前就传给预留计算。Linux有实测RSS时使用上述估算；
未知内存/尚无探针值时保持原有退化语义，不声称能保证未测工作负载的真实峰值。

采集允许在组内由初始宽度自动扩容，因此评估计算剩余CPU名额时按该组扩容上限计占用，
避免把采集将要增长的名额分给评估。保留现有小机器退化：有效N最多workers-1，至少1；
workers=1沿用一采集加一评估。启动日志同时记录requested/effective，避免误认配置生效宽度。

后续主控批准部署时，原控制器命令不加参数即使用4；可显式设置`--min-eval-units 4`，
或`--min-eval-units 1`恢复单名额预留。此次没有修改/执行生产启动脚本。

## 相关小测试与结论（2026-10-29）

`python/test_curriculum_pipeline.py -k 'min_eval_units or memory_forecast_counts_only_background_headroom
or candidate_transition_keeps_collection_workers or collection_overlaps_both_phases_and_switches_only_between_groups'`
共11项通过，约0.63秒。先看free，nice10、`ulimit -v 8000000`、timeout90；仅临时夹具和模拟
子进程，没有开引擎、训练或扫描生产帧。覆盖默认/配置/小机器宽度、先等旧组再等内存、
第一组的内存预留及扩容上限、挑选和门禁命令实际收到4单元，以及原有后台/采集重叠生命周期。

改动文件：控制器、上述相关测试、本说明、当日日志。没有改变门禁的证书/合同、面板、样本数、
挑选排名或采集任务语义。约31分钟是原task166诊断中的4单元实测；本分支未部署，
不把它当作这次代码的生产耗时。部署后的连续3轮门禁与采集耗时由主控安排验收。
