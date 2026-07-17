#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 GPU_ID MODEL_DIR [PORT] [SERVICE_NAME]" >&2
  exit 2
fi

GPU_ID="$1"
MODEL_DIR="$(cd "$2" && pwd)"
PORT="${3:-8000}"
SERVICE_NAME="${4:-pruner_$PORT}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_DIR="$PROJECT_DIR/evaluation/runtime/$SERVICE_NAME"
PID_FILE="$RUNTIME_DIR/service.pid"
LOG_FILE="$RUNTIME_DIR/service.log"

for required in config.json model.safetensors tokenizer_config.json; do
  if [[ ! -f "$MODEL_DIR/$required" ]]; then
    echo "Missing $MODEL_DIR/$required" >&2
    exit 1
  fi
done
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Service already running with PID $(cat "$PID_FILE")" >&2
  exit 1
fi

mkdir -p "$RUNTIME_DIR"
CUDA_VISIBLE_DEVICES="$GPU_ID" nohup swe-pruner \
  --model-path "$MODEL_DIR" \
  --port "$PORT" \
  > "$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"

DEADLINE=$((SECONDS + ${SERVICE_START_TIMEOUT:-300}))
until curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "SWE-Pruner service exited during startup" >&2
    tail -n 80 "$LOG_FILE" >&2 || true
    exit 1
  fi
  if (( SECONDS >= DEADLINE )); then
    echo "Timed out waiting for SWE-Pruner on port $PORT" >&2
    tail -n 80 "$LOG_FILE" >&2 || true
    exit 1
  fi
  sleep 2
done

echo "SWE-Pruner ready: PID=$PID GPU=$GPU_ID PORT=$PORT MODEL=$MODEL_DIR"
echo "Log: $LOG_FILE"
