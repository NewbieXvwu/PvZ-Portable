#!/bin/bash
# Prepared only. The reviewed deploy20 script owns the boundary and backups.
set -euo pipefail
exec /bin/bash "$HOME/pvz-logs/qc_deploy20.sh" "$@"
