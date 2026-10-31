#!/usr/bin/env bash
# task98: prepared only. Start this watchdog at a group boundary after deployment.
# Deployment and watchdog call qc_launch.py; no deployment source extraction.

LOG_DIR=$HOME/pvz-logs
REPO=$HOME/PvZ-search
WORKTREE=$REPO/.claude/worktrees/cq
PYTHON=$HOME/.venvs/ml/bin/python3
QC_LAUNCH=$WORKTREE/scripts/qc_launch.py
WATCHDOG_INTERVAL=120
WATCHDOG_STALE_SECONDS=1800
WATCHDOG_RESTART_WINDOW=1800
WATCHDOG_RESTART_LIMIT=2

watchdog_log() {
    local message="$(date -Iseconds) qc_watchdog: $*"
    printf '%s\n' "$message" >> "$LOG_DIR/qc_setup.log" || return 1
    printf '%s\n' "$message"
}

watchdog_deploying() {
    pgrep -f "$LOG_DIR/qc_deploy[^ /]*[.]sh([[:space:]]|$)" >/dev/null 2>&1 && return 0
    [[ -e "$LOG_DIR/qc_deploy.lock" ]] && ! flock -n "$LOG_DIR/qc_deploy.lock" true
}

watchdog_blocked() {
    [[ -e "$LOG_DIR/stop_curr" ]] || watchdog_deploying
}

watchdog_check_progress() {
    local now=$1 inode lines modified old_inode='' old_lines=0 changed=$1
    local file="$LOG_DIR/curriculum.log" progress="$LOG_DIR/qc_watchdog.progress"
    if [[ ! -f "$file" ]]; then
        watchdog_log 'WARNING curriculum.log missing; no progress evidence'
        return
    fi
    inode=$(stat -c %i "$file") || return
    modified=$(stat -c %Y "$file") || return
    lines=$(wc -l < "$file") || return
    changed=$modified
    if [[ -f "$progress" ]]; then
        read -r old_inode old_lines changed < "$progress" || changed=$modified
        if [[ ! "$changed" =~ ^[0-9]+$ || ! "$old_lines" =~ ^[0-9]+$ ]]; then
            changed=$modified
        elif [[ "$inode" != "$old_inode" || "$lines" -lt "$old_lines" ]]; then
            changed=$modified
        elif [[ "$lines" -gt "$old_lines" ]]; then
            changed=$now
        fi
    fi
    printf '%s %s %s\n' "$inode" "$lines" "$changed" > "$progress" || return
    if (( now - changed > WATCHDOG_STALE_SECONDS )); then
        watchdog_log "WARNING curriculum.log has no new line for $((now - changed))s; live qc is left running"
    fi
}

watchdog_tick() {
    local now recent count archive rc
    now=$(date +%s) || return
    if watchdog_blocked; then
        watchdog_log 'restart suppressed: stop_curr or deployment in progress'
        return
    fi
    if tmux has-session -t =qc 2>/dev/null; then
        watchdog_check_progress "$now"
        return
    fi
    if [[ -e "$LOG_DIR/qc_watchdog.halted" ]]; then
        return
    fi
    recent="$LOG_DIR/qc_watchdog.restarts"
    if [[ -f "$recent" ]]; then
        # Do not discard an unreadable budget and accidentally allow more restarts.
        count=$(awk -v low="$((now - WATCHDOG_RESTART_WINDOW))" '
            !/^[0-9]+$/ {bad=1} $1 >= low {n++} END {if (bad) exit 1; print n+0}' "$recent") || {
            watchdog_log 'CRITICAL restart budget unreadable/corrupt; restart refused'; return;
        }
    else
        count=0
    fi
    if (( count >= WATCHDOG_RESTART_LIMIT )); then
        watchdog_log 'CRITICAL HALTED: qc died after 2 restarts within 30 minutes; controller intervention required' || return
        printf '%s\n' "$now" > "$LOG_DIR/qc_watchdog.halted"
        return
    fi
    archive=$(mktemp "$LOG_DIR/qc_crash_$(date +%Y%m%dT%H%M%S)_XXXXXX.out") || return
    if [[ -f "$LOG_DIR/qc.out" ]]; then
        if ! tail -n 50 "$LOG_DIR/qc.out" > "$archive"; then
            return
        fi
    else
        printf 'qc.out unavailable\n' > "$archive"
    fi
    if watchdog_blocked; then
        return
    fi
    # Reserve the attempt before tmux: watchdog crashes cannot reset the budget.
    if ! printf '%s\n' "$now" >> "$recent"; then
        return
    fi
    watchdog_log "qc dead; restarting via $QC_LAUNCH; crash tail $archive" || {
        return;
    }
    "$PYTHON" "$QC_LAUNCH" --repo "$REPO" --worktree "$WORKTREE" --logs "$LOG_DIR" 9>&- >> "$LOG_DIR/qc_setup.log" 2>&1
    rc=$?
    watchdog_log "restart command exit=$rc"
    if (( rc != 0 && count + 1 >= WATCHDOG_RESTART_LIMIT )); then
        watchdog_log 'CRITICAL HALTED: 2 restart commands failed within 30 minutes; controller intervention required' || return
        printf '%s\n' "$now" > "$LOG_DIR/qc_watchdog.halted"
    fi
}

watchdog_main() {
    mkdir -p "$LOG_DIR" || return 1
    exec 9> "$LOG_DIR/qc_watchdog.lock" || return 1
    flock -n 9 || return 0
    watchdog_log 'watchdog started: check=120s stale=1800s restart limit=2/1800s' || return 1
    while :; do
        watchdog_tick
        sleep "$WATCHDOG_INTERVAL" || break
    done
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    watchdog_main
fi
