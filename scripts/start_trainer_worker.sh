#!/usr/bin/env bash
# Start FTAAS GPU trainer worker (control/UI stays elsewhere).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${ROOT}/packages/ftaas_sdk:${ROOT}/services${PYTHONPATH:+:$PYTHONPATH}"
export FTAAS_TRAIN_MODE="${FTAAS_TRAIN_MODE:-local}"
export FTAAS_TRAIN_DEVICE="${FTAAS_TRAIN_DEVICE:-auto}"
export FTAAS_TRAINER_PORT="${FTAAS_TRAINER_PORT:-${PORT:-8090}}"
exec python -m trainer_worker.main
