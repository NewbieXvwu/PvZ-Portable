#!/bin/bash
# Prepared only. The main script owns the race-free boundary wait and backup.
set -euo pipefail
exec /bin/bash "$HOME/pvz-logs/qc_deploy17.sh" "$@"
