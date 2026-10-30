#!/usr/bin/env bash
# Run the seven pattern scripts in order, keeping one log per script.
# 08_event_log.py is an exercise (several LLM calls on a 600-line log), so it
# stays out of the default run: .venv/bin/python 08_event_log.py
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p logs
for script in 0[1-7]*.py; do
  echo "=== $script ==="
  .venv/bin/python "$script" 2>&1 | tee "logs/${script%.py}.log" | tail -n 5
done
