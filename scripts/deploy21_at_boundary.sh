#!/bin/bash
# Prepared only. The reviewed deploy21 script owns the boundary and backups.
set -euo pipefail
exec /bin/bash "$HOME/pvz-logs/qc_deploy21.sh" "$@"
