#!/usr/bin/env bash
# Run FTAAS unit + integration tests.
# Usage:
#   ./scripts/run_tests.sh           # unit + fast integration
#   ./scripts/run_tests.sh --slow    # include real tiny-gpt2 fine-tune
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export PYTHONPATH="${ROOT}:${ROOT}/packages/ftaas_sdk:${ROOT}/services${PYTHONPATH:+:$PYTHONPATH}"

python -m pip install -q pytest httpx

SLOW=0
if [[ "${1:-}" == "--slow" ]]; then
  SLOW=1
fi

echo "==> Unit tests"
python -m pytest -m "unit" -q

echo "==> Integration tests (API)"
python -m pytest -m "integration and not slow" -q

if [[ "$SLOW" == "1" ]]; then
  echo "==> Slow system tests (real tiny-gpt2 LoRA)"
  export FTAAS_FORCE_REAL_TRAIN=1
  python -m pytest -m "slow" -q
else
  echo "(skip slow — pass --slow to run real fine-tune)"
fi

echo "OK"
