#!/usr/bin/env bash

_swepruner_find_python() {
  local candidates=()
  local candidate
  local resolved

  if [[ -n "${PYTHON_BIN:-}" ]]; then
    candidates+=("$PYTHON_BIN")
  fi
  candidates+=(python3.14 python3.13 python3.12 python3.11 python3)
  candidates+=("$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3")

  for candidate in "${candidates[@]}"; do
    resolved="$(command -v "$candidate" 2>/dev/null || true)"
    if [[ -n "$resolved" ]] && "$resolved" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
      PYTHON_BIN="$resolved"
      export PYTHON_BIN
      return 0
    fi
  done

  echo "Python 3.11+ was not found. Set PYTHON_BIN to an offline Python 3.11+ executable." >&2
  return 2
}

_swepruner_find_python
unset -f _swepruner_find_python

