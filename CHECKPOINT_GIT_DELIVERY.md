# 完整检查点的 Git 交付与恢复

完整AdamW/RNG检查点达到100000000字节时，归档器自动导出默认64MiB的逐字节分片。
单片上限95000000字节；清单记录每片和整文件的字节数/SHA256。没有删除权重、优化器
或随机状态字段。所有分片和清单通过Git提交到pvz-env，原本地检查点保持原样。

```bash
/home/newbiexvwu/.venvs/ml/bin/python scripts/archive_research_evidence.py \
  --experiment-dir artifacts/research/<candidate> \
  --archive-dir artifacts/research_evidence/<candidate>/<new-snapshot> \
  --log logs/t5_research/<candidate>.log
```

小文件仍保留原相对路径；大文件在`<原路径>.gitparts/`，归档清单版本2列出映射。
含分片的归档先恢复到新目录：

```bash
/home/newbiexvwu/.venvs/ml/bin/python scripts/research_checkpoint_chunks.py restore-archive \
  --archive-dir artifacts/research_evidence/<candidate>/<snapshot> \
  --destination-dir artifacts/research/<candidate-restored-new-directory>
```

恢复核对全部归档文件、片序、片SHA及整文件SHA，并核对resume.json指针。原代码、模拟器、
资源和任务指纹满足原实验身份后，才可用原配置加--resume。恢复得到的大文件保留在本地，
Git交付仍使用分片，避免重新提交超限原文件。失败留下新目录和partial，不覆盖已有证据。

验证记录为`artifacts/t5/perf/checkpoint_chunk_audit_v1.json`：真实未训练检查点14826171字节、
真实训练恢复检查点44502059字节经强制4MiB分片后完整复原，SHA与已发布原件一致；
重复恢复不改mtime。101000123字节合成数据测试大文件边界，不冒充已训练512宽模型。
错误SHA、缺片、乱序和不同已有目标均被拒绝，失败现场保留。导出清单、完整恢复及首次
辅助导入失败的日志一并保留。当前训练核心未因该工具改变。
