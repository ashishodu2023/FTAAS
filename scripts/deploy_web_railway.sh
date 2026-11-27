#!/usr/bin/env bash
# Deploy the Node product site and point ftaas.org at it.
# Prerequisites: railway login, existing Python gateway service already live.
# Run one command per line — do not paste comments on the same line.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v railway >/dev/null 2>&1; then
  echo "Install Railway CLI first (https://docs.railway.com/guides/cli)" >&2
  exit 1
fi

if ! railway whoami >/dev/null 2>&1; then
  echo "Not logged in. Run:"
  echo "  railway login --browserless"
  exit 1
fi

GATEWAY_URL="${FTAAS_GATEWAY_URL:-https://ftaas-production.up.railway.app}"

echo "==> Ensure project is linked"
railway status >/dev/null

echo "==> Create or link ftaas-web service"
railway add --service ftaas-web 2>/dev/null || true
railway service link ftaas-web

echo "==> Configure product-site variables"
# FTAAS_API_TOKEN must already exist as a Railway shared/variable secret — do not pass secrets on argv.
railway variable set \
  FTAAS_GATEWAY_URL="$GATEWAY_URL" \
  FTAAS_REQUIRE_AUTH=1

echo "==> Deploy Node site (Dockerfile web/Dockerfile from repo root)"
railway up --detach --ci -m "Deploy ftaas.org product site"

echo "==> Attach public domains to the product site"
railway domain ftaas.org || true
railway domain www.ftaas.org || true
railway domain || true

echo
echo "Done."
echo "1. Set FTAAS_API_TOKEN on BOTH ftaas-web and the Python gateway in the Railway dashboard."
echo "2. Move ftaas.org / www off the gateway service if they still point there; keep the gateway on *.up.railway.app or api.ftaas.org."
echo "3. Verify: curl -sf https://ftaas.org/health"
