#!/usr/bin/env bash
# isolated-shell.sh — workspace-agent-owned PTY helper
# Guarantees: CWD = workspace root, per-tab HISTFILE, no env bleed.
# The host (scriptschnellng) should delegate PTY creation to the workspace agent
# (llmleaf-web /api/terminal/*). This script is the fallback for the host's
# own shell() when delegation is not yet configured.
set -e
WORKSPACE="${WORKSPACE_FOLDER:-/workspace}"
# Fallback if /workspace not present (local dev without bwrap)
if [ ! -d "$WORKSPACE" ]; then
  WORKSPACE="$(pwd)"
fi
TAB_ID="${TAB_ID:-$$}"
HISTFILE="/tmp/agent-session/history.${TAB_ID}"
mkdir -p "$(dirname "$HISTFILE")"
touch "$HISTFILE"
cd "$WORKSPACE"
export PWD="$WORKSPACE"
export HISTFILE
export TERM="${TERM:-xterm-256color}"
export PS1="[tab:$TAB_ID] \w\$ "
echo "isolated terminal — pid=$TAB_ID cwd=$(pwd) hist=$HISTFILE (workspace-agent owned)"
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  return 0 2>/dev/null || true
else
  exec bash --init-file <(echo "history -r \"$HISTFILE\" 2>/dev/null; trap 'history -w \"$HISTFILE\"' EXIT")
fi
