---
name: avoid-rear-shooters-blind-spot
description: 当僵尸已深入阵地内侧时，严禁在其后方（右侧列）种植直射射手，避免陷入射击盲区导致火力几何无效。
probe:
  action_type: plant
  plant_role: [shooter]
  action_col_min: 3
  lane_front_zombie_x_max: 260
---

# 严禁在深入僵尸后方部署直射射手（避免几何盲区）

## 机制描述

在 PvZ 中，豌豆射手、寒冰射手等直射类射手的弹道方向是严格向右（朝向僵尸出生方向）发射。
当某一路的来犯僵尸已经突破并推进到阵地内侧（其横坐标 $x \le 260$，即已进入第 2 列或更靠近房子的内侧列）时：
如果在该僵尸行进方向的后方（即第 3 列及更右侧外侧列，$\text{col} \ge 3$）种植直射射手，该射手的全部弹道均朝右射击，处于其身后的僵尸完全位于射击盲区之内，该射手产生不了任何伤害输出。

这种“盲区种植”不仅彻底浪费了宝贵的阳光（如 100 阳光），使僵尸毫无阻碍地继续啃食更靠左的向日葵并突破防线，而且剥夺了策略在其他受威胁的裸路上建立有效火力线或部署坚果墙阻挡的机会，是导致中前期防御雪崩的核心缺陷。

## 正确应对规则

当监测到某路僵尸已经越过拟种植列推进到腹地（$x \le 260$）：
1. **严禁在此僵尸右侧列部署直射射手**。
2. **紧急阻挡优先**：若该路必须守住，应在僵尸行进必经之路的前方（即僵尸左侧列，如 $\text{col} \le \text{zombie\_col}$）部署高血量阻挡植物（坚果墙），将其拦截在防线外。
3. **跨路火力转移**：若阳光充足且其他路同样存在远距离来犯僵尸（裸路受压），应将火力投入到其他能在有效距离内形成击杀的受威胁路上，而不是向已被突破的内侧盲区倾倒无效火力。
4. **有效射界补种**：若必须在该路补充直射火力，必须种植在僵尸当前位置的左侧（后方），确保直射弹道能够覆盖僵尸。

```pvz-evidence
kind: mechanism
claim: 直射射手只能朝右射击，当目标路最前僵尸已推进至内侧（x <= 260）时，在后方外侧列（col >= 3）种植直射射手无法击中该僵尸且白白浪费阳光；改在僵尸前方拦截或向其他裸路投入火力可扭转战局。
falsifier: 当目标路僵尸后方（右侧）紧跟大量后续高血量僵尸集群需要火力压制，且前排已有坚果墙等坚固掩体进行持久拖延时；或者所种植植物为穿透、向后或全向攻击类植物（如大喷菇、杨桃、裂荚射手）时；或者该深入僵尸即将被武装完成的土豆雷或灰烬植物秒杀时，本规则不成立。
counterpoint: /home/newbiexvwu/PvZ-Portable/artifacts/episode_samples/seed_30059/v3_1m | 119
counterpoint: /home/newbiexvwu/PvZ-Portable/artifacts/episode_samples/seed_30059/v3_1m | 145
point: /home/newbiexvwu/PvZ-Portable/artifacts/episode_samples/seed_30059/v3_1m | 170 | plant:0:2:3 | plant:0:3:2 | 通关
point: /home/newbiexvwu/PvZ-Portable/artifacts/episode_samples/seed_30059/v3_1m | 224 | plant:0:3:3 | plant:0:4:2 | 通关
point: /home/newbiexvwu/PvZ-Portable/artifacts/episode_samples/seed_30059/v3_1m | 301 | plant:0:1:3 | plant:0:1:1 | 通关
```
