#!/usr/bin/env bash
# task98: prepared only. Start this watchdog at a group boundary after deployment.
# No deployment script is sourced or executed. Tests source the functions with a private LOG_DIR.

LOG_DIR=$HOME/pvz-logs
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
    pgrep -f "$LOG_DIR/qc_deploy[^ /]*[.]sh([[:space:]]|$)" >/dev/null 2>&1
}

watchdog_blocked() {
    [[ -e "$LOG_DIR/stop_curr" ]] || watchdog_deploying
}

watchdog_latest_deploy() {
    python3 - "$LOG_DIR" <<'PY'
import sys
from pathlib import Path
paths = list(Path(sys.argv[1]).glob('qc_deploy*.sh'))
if not paths:
    raise SystemExit('no qc_deploy*.sh')
print(max(paths, key=lambda p: (p.stat().st_mtime_ns, p.name)))
PY
}

watchdog_extract_start() {
    # Restrict extraction to start_qc and its variable dependency closure. In
    # particular, never run stop_qc, checkout, backups, rollback, or deployment IO.
    python3 - "$1" <<'PY'
import re
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text()
match = re.search(r'(?m)^start_qc\s*\(\s*\)\s*\{', source)
if not match:
    raise SystemExit('latest deployment has no start_qc function')
depth, quote, escaped, end = 1, None, False, None
for index in range(match.end(), len(source)):
    char = source[index]
    if escaped:
        escaped = False
    elif char == '\\' and quote != "'":
        escaped = True
    elif quote:
        if char == quote:
            quote = None
    elif char in ("'", '"'):
        quote = char
    elif char == '{':
        depth += 1
    elif char == '}':
        depth -= 1
        if depth == 0:
            end = index + 1
            break
if end is None:
    raise SystemExit('unterminated start_qc function')
function = source[match.start():end]
prefix = source[:match.start()]
assignment = re.compile(r'''(?m)(?:^|;)[ \t]*([A-Za-z_]\w*)=("(?:\\.|[^"\\])*"|'[^']*'|[^;\n \t]+)''')
bindings = {m[1]: m[2] for m in assignment.finditer(prefix)}
references = lambda s: set(re.findall(r'\$(?:\{)?([A-Za-z_]\w*)', s))
calls = re.findall(r'(?m)^start_qc(?=[ \t;\n]|$)([^\n]*)', source[end:])
if not calls:
    raise SystemExit('cannot identify top-level start_qc invocation')
invocation = 'start_qc' + calls[-1]
quote, escaped = None, False
for index, char in enumerate(invocation):
    if escaped:
        escaped = False
    elif char == '\\' and quote != "'":
        escaped = True
    elif quote:
        if char == quote:
            quote = None
    elif char in ("'", '"'):
        quote = char
    elif char == ';':
        invocation = invocation[:index]
        break
if any(token in invocation for token in ('$(', '`', '&&', '||', '|', '<', '>')):
    raise SystemExit('unsupported compound start_qc invocation')
needed = references(function) | references(invocation)
done, ordered, visiting = set(), [], set()
def visit(name):
    if name in done or name in ('HOME', 'USER'):
        return
    if name in visiting or name not in bindings:
        raise ValueError(f'unresolved deployment variable: {name}')
    value = bindings[name]
    if any(token in value for token in ('$(', '`', '\n', '<(', '>(')):
        raise ValueError(f'nonliteral deployment assignment: {name}')
    visiting.add(name)
    for dependency in sorted(references(value)):
        visit(dependency)
    visiting.remove(name)
    done.add(name)
    ordered.append(f'{name}={value}')
for name in sorted(needed):
    visit(name)
print('set -eu')
print('\n'.join(ordered))
print(function)
print(invocation)
PY
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
    local now recent count deploy launcher archive rc
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
    deploy=$(watchdog_latest_deploy) || {
        watchdog_log 'ERROR cannot choose latest deployment; restart refused'
        return
    }
    launcher=$(mktemp "$LOG_DIR/qc_watchdog_start_XXXXXX.sh") || return
    if ! watchdog_extract_start "$deploy" > "$launcher" || ! bash -n "$launcher"; then
        watchdog_log "ERROR cannot extract start_qc from $deploy; restart refused"
        rm -- "$launcher"
        return
    fi
    # A deployment/stop request may have arrived during extraction.
    if watchdog_blocked; then
        rm -- "$launcher"
        return
    fi
    archive=$(mktemp "$LOG_DIR/qc_crash_$(date +%Y%m%dT%H%M%S)_XXXXXX.out") || {
        rm -- "$launcher"; return;
    }
    if [[ -f "$LOG_DIR/qc.out" ]]; then
        if ! tail -n 50 "$LOG_DIR/qc.out" > "$archive"; then
            rm -- "$launcher"; return
        fi
    else
        printf 'qc.out unavailable\n' > "$archive"
    fi
    if watchdog_blocked; then
        rm -- "$launcher"
        return
    fi
    # Reserve the attempt before tmux: watchdog crashes cannot reset the budget.
    if ! printf '%s\n' "$now" >> "$recent"; then
        rm -- "$launcher"; return
    fi
    watchdog_log "qc dead; restarting via $deploy start_qc; crash tail $archive" || {
        rm -- "$launcher"; return;
    }
    bash "$launcher" 9>&- >> "$LOG_DIR/qc_setup.log" 2>&1
    rc=$?
    rm -- "$launcher"
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
