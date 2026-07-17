#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="${1:-pruner_8000}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PID_FILE="$PROJECT_DIR/evaluation/runtime/$SERVICE_NAME/service.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No PID file for service $SERVICE_NAME"
  exit 0
fi

PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  for _ in $(seq 1 30); do
    kill -0 "$PID" 2>/dev/null || break
    sleep 1
  done
fi
rm -f "$PID_FILE"
echo "Stopped service $SERVICE_NAME (PID=$PID)"
