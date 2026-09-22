#!/usr/bin/env bash
# Run the full test suite (ASCII only).
set -u
PY="$HOME/venvs/pair/bin/python"
cd "$HOME/pair" || exit 1
export PYTHONPATH=.
"$PY" -m pytest edge_llm_scheduler/tests -q -p no:cacheprovider 2>&1 | tail -25
