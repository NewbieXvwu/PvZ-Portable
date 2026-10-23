# 冒险模式50关通关缺口（task55）

状态：事实盘点进行中，尚未形成最终推进建议。

本次只读源码、报告与qc日志；使用既有deploy3的单个引擎进程核对50次reset及每关1 tick，不运行策略、不采批量对局、不操作qc。50关均可reset并推进1 tick；这不证明每关动作完整或能正常通关。引擎查询、源码判定与qc快照SHA见 [adventure-gap.json](adventure-gap.json)。

关卡类型从`LawnApp::Is*Level`、`Board::HasConveyorBeltSeedBank`与`PickBackground`现解析，保留原Adventure模式。L5保龄球、L15打地鼠、L25 LittleTrouble、L35砸罐、L45蹦极、L50僵王，L10/20/30为mini-boss传送带，L40暴风雨传送带。L35是夜间草地，L40虽使用雾区背景但`StageHasFog=false`，L50为夜间Boss屋顶；均以实测及源码为准。

僵尸分开记录引擎允许表与固定核对seed下的已生成波表，不将其中一个冒充实际整局出现的全集；Boss、旗帜及其他脚本增援还需在机制审查中说明。探针卡组由现有`deck_for_level`与所有权接口生成，原生传送带用空手开局；探针不改变生产卡组族。L15/L35当前reset仍有覆盖原生卡槽的语义限制，后续按特殊机制单列。

qc数据冻结到只读快照，当前38个cell覆盖26个level；这里只计“在控制器状态中”，不把已登记/被分配/已采集混为一件事。最新学生独立面板、现役学生、beta1老师及beta0采集执行胜率将分列，未测写缺证，不填0。

待完成：50行完整胜率/卡组/倍率表；特殊关动作与观测缺口；普通关瓶颈归类；依赖任务与推进顺序。
