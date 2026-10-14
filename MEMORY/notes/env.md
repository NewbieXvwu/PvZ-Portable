# 运行环境 / 测试陷阱

- **必须用 mise 的 Python**：`/Users/newbiexvwu/.local/share/mise/installs/python/3.14/bin/python3`（有 numpy + torch）。
  WorkBuddy 托管的 `~/.workbuddy-ai/binaries/python/...` 下**没有** numpy/torch。
  （`~/.workbuddy-ai/...` 是 **WorkBuddy 的用户级托管路径**，其它 Agent 平台没有这个目录，照抄无效。）
- 测试：`cd python && <mise python> -m unittest discover -s . -p "test_*.py"`（**无 pytest**）。
- 真实模拟器资源：`/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN`。
- `test_env_protocol.py` 的 `_resource_dir()` 是**假资源目录**（1 字节 main.pak），拿它调真实 `env.reset()`
  → 子进程卡死、父进程永久等。
- `multiprocessing.Pool` initializer 抛异常时 `imap_unordered` **不报错**，父进程永久挂起；
  `run_seed_jobs` 的 `stall_timeout`（默认 900 s）把它变成 `RuntimeError`。
- 长跑探针**不要** `| tail -N`（丢流式输出）→ 写日志文件。
