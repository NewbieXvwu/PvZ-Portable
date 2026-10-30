#!/bin/bash
# Prepared only; main controller reviews and executes the deploy22 package.
set -euo pipefail
exec /bin/bash "$HOME/pvz-logs/qc_deploy22.sh" "$@"
