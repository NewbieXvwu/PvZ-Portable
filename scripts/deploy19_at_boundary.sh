#!/bin/bash
# Prepared only. The reviewed deploy19 script owns the boundary and backups.
set -euo pipefail
exec /bin/bash "$HOME/pvz-logs/qc_deploy19.sh" "$@"
