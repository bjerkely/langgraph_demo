#!/usr/bin/env bash
# Run all seven patterns in order, keeping one log per script.
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p logs
for script in 0*.py; do
  echo "=== $script ==="
  .venv/bin/python "$script" 2>&1 | tee "logs/${script%.py}.log" | tail -n 5
done
