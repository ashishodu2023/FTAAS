#!/usr/bin/env bash
# Deploy the Node product site and point ftaas.org at it.
# Prerequisites: railway login; Python gateway already live; FTAAS_API_TOKEN set on both services in the dashboard.
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
railway variable set \
  FTAAS_GATEWAY_URL="$GATEWAY_URL" \
  FTAAS_REQUIRE_AUTH=1

echo "==> Deploy Node site (web/ context + web/Dockerfile — not the gateway image)"
railway up web --path-as-root --service ftaas-web --detach --ci -m "Deploy ftaas.org product site"

echo
echo "==> Domains"
echo "Before attaching ftaas.org / www.ftaas.org here, remove them from the Python gateway service"
echo "in the Railway dashboard (Settings → Networking). Then run:"
echo "  railway service link ftaas-web"
echo "  railway domain ftaas.org"
echo "  railway domain www.ftaas.org"
echo
echo "Set FTAAS_API_TOKEN on BOTH ftaas-web and the Python gateway (same value)."
echo "Keep the gateway off public custom domains — use its *.up.railway.app URL or a private api host."
echo "Verify: curl -sf https://ftaas.org/health   # expect service: ftaas-web"
