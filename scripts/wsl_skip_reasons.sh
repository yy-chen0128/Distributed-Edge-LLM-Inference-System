#!/usr/bin/env bash
# List skip reasons (a skip is not a pass -- see docs/plan).
set -u
PY="$HOME/venvs/pair/bin/python"
PROJ="$(find /mnt/d/Newproject -maxdepth 2 -type d -name project 2>/dev/null | head -1)"
cd "$PROJ" || exit 1
export PYTHONPATH=.
"$PY" -m pytest edge_llm_scheduler/tests/ -q -rs -p no:cacheprovider 2>&1 | grep -A1 -i "SKIPPED" | head -20
