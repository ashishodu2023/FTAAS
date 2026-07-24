# Deploy FTAAS to Railway and attach ftaas.org
#
# 1) Login (one-time):
#      railway login --browserless
# 2) From repo root:
#      ./scripts/deploy_railway.sh
# 3) Point DNS at Railway (see printed instructions).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v railway >/dev/null 2>&1; then
  echo "Install Railway CLI: brew install railway" >&2
  exit 1
fi

if ! railway whoami >/dev/null 2>&1; then
  echo "Not logged in. Run: railway login --browserless" >&2
  exit 1
fi

# Create/link project if needed
if ! railway status >/dev/null 2>&1; then
  echo "==> Creating Railway project ftaas"
  railway init --name ftaas || railway link
fi

echo "==> Setting production variables"
railway variables set \
  FTAAS_PUBLIC_URL=https://ftaas.org \
  FTAAS_DATA_DIR=/data \
  PYTHONPATH=/app:/app/packages/ftaas_sdk:/app/services

echo "==> Deploying"
railway up --detach --ci --message "Deploy FTAAS console to ftaas.org"

echo "==> Generating Railway domain (if none)"
railway domain 2>/dev/null || true

echo
echo "Next — custom domain:"
echo "  railway domain add ftaas.org"
echo "  railway domain add www.ftaas.org"
echo
echo "Then at your registrar DNS for ftaas.org set the CNAME/A records Railway prints."
echo "Also set: railway variables set FTAAS_PUBLIC_URL=https://ftaas.org"
