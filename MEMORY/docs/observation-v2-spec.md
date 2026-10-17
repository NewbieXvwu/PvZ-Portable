# 观测 v2 规格草案

草案日期：2026-10-17。仅定义数据契约与编码，不改正式观测、模型或训练路径。目标是让学生与价值网络获得和老师续走一致的决策状态，同时把玩家可见信息与已生成但未出场的特权信息区分开。

## JSON 字段

保留现有字段含义，增加 `observation_version: 4`。Lean 与完整观测对新增公共字段保持同名、同单位；Lean 可以不返回诊断用非决策字段。

|对象|新增字段|单位与范围|用途|
|---|---|---|---|
|根对象|`sun_countdown`|剩余 tick，无天降太阳时为 `null`|直接读取 `Board::mSunCountDown`，让网络预估自然收入窗口。|
|根对象|`wave_timer_start`、`wave_refresh_threshold`、`wave_refresh_early`、`huge_wave_timer`|tick；生命阈值比例 0–1；提前触发布尔值|区分正常计时、低血量提前刷新和旗帜大波等待。现有 `wave_timer` 保留。|
|根对象|`pending_wave_zombies`|34 项非负计数数组，仅当前波和下一波|描述已抽好但未出场的类型构成。标注 `information_scope: "privileged"`，不得混入玩家可见字段训练。|
|植物实体|`id`、`state`、`state_countdown`、`launch_counter`、`launch_rate`、`shooting_counter`、`wake_up_counter`、`asleep`、`target_zombie_id`|整数/布尔，保留现有完整 JSON 字段命名|由当前完整 JSON 字段直接沿用；新增稳定 `id` 用于跨帧跟踪。`target_zombie_id` 在没有目标时为 0。|
|僵尸实体|`id`、`phase`、`phase_counter`、`velocity_x`、`body_health`、`helm_health`、`shield_health`、`flying_health`、`chilled`、`buttered`、`ice_trap`、`has_object`、`target_plant_id`|整数/浮点/布尔；生命非负、计时 tick 非负|标准 JSON 已含主要字段，需确保 Lean 或帧归档不丢失；目标 ID 仅用于实体关系，不作为类型特征。|
|弹道实体|`id`、`motion`、`x/y/z`、`vx/vy/vz`、`ax/ay/az`、`age`、`target_zombie_id`、`damage`|浮点/整数；tick 与引擎单位|已有完整观测字段保留，补充加速度和唯一 ID，使弹道可跨帧追踪。|
|掉落实体|`id`、`type`、`x/y`、`vx/vy`、`age`、`pickup_state`、`disappear_countdown`、`value`|浮点/整数/布尔|替换目前仅有 type/x/y 的信息，区分正在落下、可收取和即将消失。|
|割草机/格物|割草机增加 `id`、`state_countdown`；格物增加 `id`、`state`、`counter`、内容物类型|当前类型范围内枚举和 tick|Lean 可保留简单格位字段，完整 JSON 输出完整状态，供跨关任务特征编码。|

`id` 只需在一局内稳定，不要求跨重置稳定。`pending_wave_zombies` 是教师可用的特权信息；任何学生实验必须把该字段排除，除非训练和部署明确选择相同信息权限。
