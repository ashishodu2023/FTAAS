#!/usr/bin/env bash
# Deploy FTAAS to Railway and prepare ftaas.org DNS.
# Run one command per line in your terminal (no trailing comments).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v railway >/dev/null 2>&1; then
  echo "Install Railway CLI: brew install railway" >&2
  exit 1
fi

if ! railway whoami >/dev/null 2>&1; then
  echo "Not logged in. Run alone:"
  echo "  railway login --browserless"
  exit 1
fi

if ! railway status >/dev/null 2>&1; then
  echo "==> Creating Railway project ftaas"
  railway init -n ftaas
fi

# Empty project from `init` has no service — create + link one.
if ! railway service status >/dev/null 2>&1; then
  echo "==> Creating service ftaas"
  railway add --service ftaas
  railway service link ftaas
fi

echo "==> Setting production variables"
railway variable set \
  FTAAS_PUBLIC_URL=https://ftaas.org \
  FTAAS_DATA_DIR=/app/data \
  PYTHONPATH=/app:/app/packages/ftaas_sdk:/app/services

echo "==> Deploying"
railway up --detach --ci -m "Deploy FTAAS console to ftaas.org"

echo "==> Domains"
railway domain ftaas.org || true
railway domain www.ftaas.org || true
railway domain || true

echo
echo "Done. Add the DNS records Railway printed at your registrar."
echo "Verify later with:  curl -sf https://ftaas.org/health"
