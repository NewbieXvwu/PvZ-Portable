#!/bin/bash
# Installs only the independent research service. Does not touch qc or its files.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
state=${1:-$HOME/pvz-logs/gpu-window}
python_bin=${PYTHON_BIN:-$HOME/.venvs/ml/bin/python3}
mkdir -p "$HOME/.config/systemd/user" "$state/queue" "$state/mac-handoff"
cat > "$HOME/.config/systemd/user/pvz-gpu-window.service" <<EOF
[Unit]
Description=PvZ GPU window research queue (production first)

[Service]
Type=simple
WorkingDirectory=$repo
ExecStart=$python_bin $repo/scripts/gpu_window_runner.py --queue $state/queue --handoff $state/mac-handoff --production-log $HOME/pvz-logs/curriculum.log
Nice=10
CPUWeight=1
IOWeight=1
CPUQuota=150%
MemoryHigh=4G
MemoryMax=5G
MemorySwapMax=0
OOMScoreAdjust=800
OOMPolicy=kill
TasksMax=64
NoNewPrivileges=yes
KillMode=control-group
TimeoutStopSec=5
Restart=on-failure
RestartSec=10
Environment=OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload
systemctl --user enable pvz-gpu-window.service
systemctl --user restart pvz-gpu-window.service
