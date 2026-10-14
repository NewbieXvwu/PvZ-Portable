---
name: emergency-wallnut-containment
description: 当僵尸深入阵地内部突破火力线或造成高压逼近时，严禁在僵尸跟前盲目补种脆弱射手充当肉盾；必须优先在僵尸行进路径前方部署坚果墙建立阻挡，利用其高耐久为后方射手争取持续歼敌时间。
probe:
  action_type: plant
  plant_role: [shooter]
  lane_front_zombie_x_max: 480
  plant_life_ticks_max: 1200
---

# 突破高压下的紧急坚果墙拦截机制

## 机制描述

在《植物大战僵尸》中，直射型射手（豌豆射手、双发射手、寒冰射手）的基础血量仅为 300 点，且射击方向严格向右。僵尸啃食速率为 100 点/秒（每 4 tick 造成 4 点伤害）。
当某条路线的僵尸突破至防御纵深内侧（最前僵尸横坐标 $x \le 480$，即已进入第 5 列及更靠内的内侧列）时：

1. **射手填线必然秒破**：如果在僵尸跟前、同格或后方补种直射射手，单只普通僵尸仅需 300 tick（3 秒）即可将其啃光，更难以阻挡多只僵尸或路障/撑杆等高威胁单位。在 300 tick 的短暂存活时间内，豌豆射手（射击间隔 150 tick）最多只能射出 2 颗豌豆（40 点伤害），双发射手最多射出 4 颗（80 点伤害）。若落点在僵尸后方，造成的伤害更是为 0。这种行为不仅白白耗费 100~200 点阳光，而且完全无法延缓僵尸的推进步伐。
2. **坚果墙的高耐久延时价值**：相比之下，坚果墙（Wall-nut，花费仅 50☀）具备高达 4000 点血量，是常规植物的 13.3 倍。将其部署在僵尸行进必经之路上，即使面对单只僵尸持续啃食也能阻挡 4000 tick（40 秒）。这为该路后排已有的射手（如 c2–c3 的射手）提供了充足的无伤输出时间，足以歼灭总血量数千的高血量僵尸集群。
3. **策略缺陷**：本局 PPO 模型在全盘 842 次决策中，坚果墙使用次数为 0（0/842）。每次局部阵地受压或植物被啃食时，策略均在接触位机械重复补种脆弱射手，导致植物秒死并连续送掉割草机，最终在第 29 波第 4 路彻底失守进屋。

## 正确应对规则

当监测到某一路最前僵尸已逼近内侧防御线（$x \le 480$），且该路面临突破威胁时：
1. **禁止肉身补射手**：严禁在正在推进的僵尸身前接触格或其后方盲区种植常规射手充当肉盾。
2. **优先紧急坚果墙拦截**：只要坚果墙卡牌就绪且阳光 $\ge 50$，必须优先在僵尸必经之路前方（取能阻挡僵尸的最靠右合法格，如僵尸到达前的下一列）部署坚果墙。
3. **协同后排火力清场**：坚果墙落地后，依靠后排原有射手输出；若后排火力不足或僵尸集群过于密集，可进一步补充后排火力或配合即时爆发植物（如樱桃炸弹）清场。

```pvz-evidence
kind: mechanism
claim: 当僵尸突破逼近内侧（x <= 480）时，严禁在接触位补种仅有 300 血的脆弱射手送死，必须优先在僵尸前方种植 4000 血的坚果墙建立阻挡，依靠其高耐久为后方火力争取歼敌时间。
falsifier: 当深入僵尸残血即将被后排现有飞行弹道击杀且无法接触植物时；或场上阳光不足 50 且坚果墙处于冷却中时；或该路已有健康坚果墙有效阻挡时，本机制不成立。
counterpoint: /home/newbiexvwu/PvZ-Portable/artifacts/episode_samples/seed_30974/v3_1m | 132
counterpoint: /home/newbiexvwu/PvZ-Portable/artifacts/episode_samples/seed_30974/v3_1m | 250
point: /home/newbiexvwu/PvZ-Portable/artifacts/episode_samples/seed_30974/v3_1m | 273 | plant:0:4:3 | plant:3:4:4 | 通关
point: /home/newbiexvwu/PvZ-Portable/artifacts/episode_samples/seed_30974/v3_1m | 287 | plant:0:4:3 | plant:3:4:4 | 通关
point: /home/newbiexvwu/PvZ-Portable/artifacts/episode_samples/seed_30974/v3_1m | 745 | plant:6:4:5 | plant:3:4:5 | 通关
point: /home/newbiexvwu/PvZ-Portable/artifacts/episode_samples/seed_30974/v3_1m | 766 | plant:6:4:5 | plant:3:4:5 | 通关
```
