#!/usr/bin/env bash
# End-to-end smoke: register dataset → finetune → wait → deploy → prompt
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${ROOT}/packages/mdlc_sdk:${ROOT}/services:${PYTHONPATH:-}"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

python "$ROOT/examples/e2e_finetune.py"
