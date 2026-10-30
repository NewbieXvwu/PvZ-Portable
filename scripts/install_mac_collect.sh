#!/bin/bash
# Run by the main controller at an idle group boundary on deploy21/deploy22.
# This installs code only; controller stop/restart remains with the main controller.
set -euo pipefail
W=${1:?usage: install_mac_collect.sh CQ_WORKTREE [REVISION]}
REV=${2:-origin/maccollect}
R=/home/newbiexvwu/PvZ-search
file=scripts/mac_collect_import.py
git -C "$R" show "$REV:$file" > "$W/$file.mac197-building"
mv "$W/$file.mac197-building" "$W/$file"
# Apply only the eight collection-source lines, preserving other production edits.
git -C "$R" diff 41bcc4a1 "$REV" -- scripts/curriculum_queue.py | git -C "$W" apply -
