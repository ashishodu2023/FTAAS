#!/usr/bin/env bash
# Start FTAAS as a single unified process (UI + all APIs on one port).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="${ROOT}:${ROOT}/packages/ftaas_sdk:${ROOT}/services:${PYTHONPATH:-}"
export FTAAS_CONFIG="${FTAAS_CONFIG:-$ROOT/configs/settings.yaml}"
export FTAAS_DATA_DIR="${FTAAS_DATA_DIR:-$ROOT/data}"
# All logical services share the unified gateway
export FTAAS_JOBS_URL="${FTAAS_JOBS_URL:-http://127.0.0.1:8080}"
export FTAAS_DATASETS_URL="${FTAAS_DATASETS_URL:-http://127.0.0.1:8080}"
export FTAAS_PIPELINES_URL="${FTAAS_PIPELINES_URL:-http://127.0.0.1:8080}"
export FTAAS_SERVING_URL="${FTAAS_SERVING_URL:-http://127.0.0.1:8080}"

mkdir -p "$ROOT/data" "$ROOT/logs"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

# Stop any previous unified or split processes
"$ROOT/scripts/stop_all.sh" >/dev/null 2>&1 || true
for p in 8000 8001 8002 8003 8080; do
  if command -v lsof >/dev/null 2>&1; then
    lsof -tiTCP:"$p" -sTCP:LISTEN 2>/dev/null | xargs kill -9 2>/dev/null || true
  fi
done

PORT="${FTAAS_PORT:-8080}"
echo "==> Starting FTAAS on :${PORT}"
python -m ftaas_app.main >"$ROOT/logs/ftaas.log" 2>&1 &
echo $! >"$ROOT/logs/ftaas.pid"

sleep 2
if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "  ✓ FTAAS  http://127.0.0.1:${PORT}"
  echo "  UI + API → http://127.0.0.1:${PORT}"
else
  echo "  … starting (see logs/ftaas.log)"
  tail -n 40 "$ROOT/logs/ftaas.log" || true
fi
echo "Stop with  ./scripts/stop_all.sh"
