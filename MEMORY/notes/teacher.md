# 教师流水线（dsh/pvz-teacher）——**两台机器都能跑**

- **本机（Mac）**：`dsh` 在 `~/.local/share/mise/installs/node/24/bin`；`DSH_HOME=~/.local/share/pvz-agent/dsh-home`；
  mcp-venv `~/.local/share/pvz-agent/mcp-venv`；CLIProxyAPI `/opt/homebrew/bin/cliproxyapi`（v8.0.10）+
  `/opt/homebrew/etc/cliproxyapi.conf` + auth `~/.cli-proxy-api/`。**`GEMINI_API_KEY`** 是 CLIProxyAPI 的
  api-keys（**不是 Google 官方 key**），**值只放环境变量、不写入仓库**；Gemini 凭据是 **`antigravity` 类型**
  的 OAuth（带 refresh_token）；**`gemini-3.8-flash-high` 是网关唯一认的模型 id**。
- **台式机（WSL）**：全部**不用 Docker**；Node 24 `~/.local/opt/node-v24.21.0-linux-x64`；CLIProxyAPI
  `~/.local/opt/cliproxyapi/`；**`~/.pvz-teacher-env.sh`**（任务书 source 它，`GEMINI_API_KEY` 也放这里）。
- **★ 装机 5 坑（换机器必看；详见 `2026-10-12.md`、`docs/codex-night-plan-v5.md`）**：① `~/.venvs/ml` 的 pip
  配了 `bfsu` 镜像 → 装 mcp 报 **403**，要 `--index-url https://pypi.org/simple`；② **npm 11 默认跳过 install
  script** → `npm config set allow-scripts=... --location=user` 再重装；③ **`dsh plugin add` 建的 profile 会漏掉
  `@deepseek-ai/dsh-headless`** → **dsh 启 web 应用并永久卡住**（0 字节输出、`do_epoll_wait`），只需手工写进
  `package.json` 的 `dsh.profile.bundles`；④ **`$DSH_HOME/.credentials.yaml` 必须 600**；⑤ **bundle 里是绝对
  路径**，换机器改**部署副本**（`$DSH_HOME/profiles/<profile>/node_modules/@local/pvz-teacher/cordis.patch.yml`），
  **不要改仓库里的 `bundle/cordis.patch.yml`**（git 跟踪，会弄脏仓库）。
- **用法**：**会话必须带 `--task-manifest`**（否则诊断的是另一副卡槽）；**一局诊断 9 分钟 ~ 2.5 小时 /
  20~40 轮**（只挑少量局，且**把已有发现当"假设"交给它验证**）。**起会话前自检**：① `which dsh`
  ② bundles 必须是 `['@deepseek-ai/dsh-base','@deepseek-ai/dsh-headless','@local/pvz-teacher']`
  ③ `curl -H "x-goog-api-key: $GEMINI_API_KEY" http://localhost:8317/v1beta/models` → 200。
