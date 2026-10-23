#!/bin/bash
# task36: wait until no other trainer is running and enough RAM is free before starting.
# The recipe worktree's parity check (another session) loads the same corpus, so starting a
# second full-corpus trainer early would OOM both.
set -u
MIN_GB=${MIN_GB:-16}
log() { echo "$(date -Is) $*"; }
while true; do
  busy=$(pgrep -af 'train_rollout_policy\.py|train_value\.py|verify_recipe_migration\.py' | grep -v pgrep | grep -v task36_wait_idle || true)
  avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  if [ -z "$busy" ] && [ "$avail" -ge "$MIN_GB" ]; then
    log "idle: no trainer running, MemAvailable ${avail}GiB >= ${MIN_GB}GiB"
    exit 0
  fi
  log "waiting: MemAvailable ${avail}GiB (need ${MIN_GB}), trainers: $(echo "$busy" | wc -l)"
  sleep 60
done
