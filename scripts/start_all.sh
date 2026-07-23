#!/usr/bin/env bash
# Start all FTAAS services locally (no Docker required).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="${ROOT}:${ROOT}/packages/mdlc_sdk:${ROOT}/services:${PYTHONPATH:-}"
export FTAAS_CONFIG="${FTAAS_CONFIG:-$ROOT/configs/settings.yaml}"
export FTAAS_DATA_DIR="${FTAAS_DATA_DIR:-$ROOT/data}"

mkdir -p "$ROOT/data" "$ROOT/logs"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

echo "==> Starting MDS      :8001"
python -m mds.main >"$ROOT/logs/mds.log" 2>&1 &
echo $! >"$ROOT/logs/mds.pid"

echo "==> Starting Pipelineserv :8002"
python -m pipelineserv.main >"$ROOT/logs/pipelineserv.log" 2>&1 &
echo $! >"$ROOT/logs/pipelineserv.pid"

echo "==> Starting Aimlopsserv :8003"
python -m aimlopsserv.main >"$ROOT/logs/aimlopsserv.log" 2>&1 &
echo $! >"$ROOT/logs/aimlopsserv.pid"

echo "==> Starting MDLC Serv :8000"
python -m mdlc_server.main >"$ROOT/logs/mdlc.log" 2>&1 &
echo $! >"$ROOT/logs/mdlc.pid"

echo "==> Starting Cosmos UI :8080"
python -c "from ui.cosmos_ui.app import main; main()" >"$ROOT/logs/cosmos_ui.log" 2>&1 &
echo $! >"$ROOT/logs/cosmos_ui.pid"

sleep 2
echo
echo "Services:"
for s in mds:8001 pipelineserv:8002 aimlopsserv:8003 mdlc:8000 cosmos_ui:8080; do
  name="${s%%:*}"; port="${s##*:}"
  if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
    echo "  ✓ ${name}  http://127.0.0.1:${port}"
  else
    echo "  … ${name}  starting (see logs/${name}.log)"
  fi
done
echo
echo "Cosmos UI → http://127.0.0.1:8080"
echo "Stop with  ./scripts/stop_all.sh"
