# 示范起点格未来种子续期（task99）

当前阶段：打包根因修复与实际标签重打包验证完成；按需种子账本与qc接入仍在实施。未修改线上cq、tmux qc或生产段。

## 起点短局的plane_of修复

`CompactBoards.append`此前在首次出现常量平面前不记映射。ce3ce400标签的前4份存档各保留3行，直到第5份存档才有平面，前12行因此漏记，出现84行/72个id。现首次出现平面时回填先前行的-1，后续行继续按平面编号追加。

平面值也转为store声明的dtype，修正无缓存float32平面与缓存float16板面之间的精度差；CSR和常量平面还原的训练输入遵守同一精度。新段的board/scalar plane_of均逐行写入；老的“无平面且空映射”格式仍可读取。复用已有段时只查npz头，长度或平面精度不符就重建。采样、PACK_VERSION=2和源指纹不变。

对实际`c18demoroof48doomm200s1100_fast_search`的24份存档，使用相同选行契约在本树ignored目录重打包，验证84行、两份映射均84、前12个board id为-1；缓存/无缓存全部payload逐字节一致。源段与存档逐一SHA回验未变。详见[start-cell-pack-check.json](start-cell-pack-check.json)。

## 组边界上线：打包部分

与task98一并合入后，在主控安排的组边界使用部署检出的新代码，仅重打包该标签。旧段的错误形状会被新packer自动识别，无需重建全语料；原全量index对该段的引用继续有效。

```bash
PVZ_NO_POPUP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/newbiexvwu/.venvs/ml/bin/python3 /home/newbiexvwu/PvZ-search/.claude/worktrees/cq/scripts/pack_training_rows.py --data /home/newbiexvwu/pvz-logs/curriculum_state.pipeline/frames --labels c18demoroof48doomm200s1100_fast_search --wait-keep 0.1 --obs v2 --cells --packet-head identity --aux-horizon 0
```

随后检查段`ce3ce4006691a0e1.npz`的`rows`、`board_plane_of.shape[0]`和`scalar_plane_of.shape[0]`均为84，再允许失败训练重试。读取端的570f5e7d止血补丁不能代替重新打包：它避免崩溃，却无法从漏掉的映射恢复被剥离的平面值。
