# task143：把 deploy15（v2sleep 输入 + build7）刷新到 prodint 之上

主控 2026-10-28 派给 Mac Qoder。只准备，不部署。

## 背景

- task131（`origin/obsup` 5423811f）做了 v2sleep 输入的上线准备：加宽热启动、旧缓存补零复用、台式机上的 `build7`、`qc_deploy15.sh`。它的基线是 deploy14i。
- 生产之后又换了几次代码：14j / 14k，以及即将上线的 14p（`origin/prodint` 4beab441 起）。deploy15 的基线已经过时。
- task135 核对过：生产学生每轮都是**从头训练**（`recipe.resolved.json`）。如果属实，task131 的“加宽热启动”在生产里根本用不上，切换输入版本只需要新配方 + 新引擎 + 新训练行。请先确认这一点，它决定 deploy15 要不要带热启动代码。

## 要做的

1. 新建分支 `obsup2`，以 `origin/prodint` 为底，合入 `origin/obsup`，解决冲突。
2. 确认生产训练是不是从头训。是的话，把热启动从部署路径里去掉，或者保留但不启用，写明理由。
3. 写 `scripts/qc_deploy15.sh`：基线为 14p 换上之后的文件；start_qc 照抄 `scripts/qc_deploy14p.sh`，再加 15 的激活分支（build7、`obs=v2sleep` 的新配方）；stop_qc 不排空；首组超时不回滚；预检要便宜。
4. 估算切换后第一轮重打训练行的耗时和磁盘占用，只读台式机数据，不跑训练。
5. 测试严格遵守 AGENTS §6.5 第 8 条。结论写在本文件末尾，推送 `obsup2`。
