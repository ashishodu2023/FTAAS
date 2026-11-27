# Deploying FTAAS to ftaas.org (Railway)

Run **one command per line** — do not paste `# comments` on the same line (Railway treats `#` as an argument).

## 1. Login

```bash
cd /Users/ashishverma/Downloads/FTAAS
railway login --browserless
```

Complete the pairing code in the browser, then confirm:

```bash
railway whoami
```

## 2. Deploy

```bash
./scripts/deploy_railway.sh
```

Or manually:

```bash
railway init -n ftaas
railway add --service ftaas
railway service link ftaas
railway variable set FTAAS_PUBLIC_URL=https://ftaas.org FTAAS_DATA_DIR=/data PYTHONPATH=/app:/app/packages/ftaas_sdk:/app/services
railway up --detach --ci -m "Deploy FTAAS"
railway domain ftaas.org
railway domain www.ftaas.org
```

> Note: CLI is `railway domain ftaas.org` — **not** `railway domain add …`  
> Variables: `railway variable set` — **not** `railway variables set`

## 3. Custom domain (ftaas.org)

If `railway domain ftaas.org` returns **Unauthorized** even after a successful login, add the domain in the dashboard instead (CLI bug / permission quirk):

1. Open [Railway project → ftaas service](https://railway.com/project/9e91ef5e-c67a-447f-a4dc-9f49dea6f5ef)
2. **Settings → Networking → Custom Domain**
3. Add `ftaas.org` and `www.ftaas.org`
4. Copy the CNAME + TXT records Railway shows

At your registrar, add those records exactly. Typical pattern:

| Type | Name | Value |
|------|------|--------|
| CNAME / ALIAS | `@` | *(from Railway)* |
| CNAME | `www` | *(from Railway)* |
| TXT | *(as shown)* | *(ownership verification)* |

Until DNS is set, `ftaas.org` will not resolve. Meanwhile use the Railway URL: `https://ftaas-production.up.railway.app`

## 4. Verify

```bash
curl -sf https://ftaas.org/health
open https://ftaas.org/
```

## 5. Two parts on ftaas.org

This file deploys the **Python gateway** (console + `/v1`). The Node product site in `web/` is the public front door for the same domain. It does not train.

### Cut over the public domain to the Node site

1. Keep the Python gateway on its Railway URL (`https://ftaas-production.up.railway.app`) or attach `api.ftaas.org`.
2. Deploy the product site:

```bash
./scripts/deploy_web_railway.sh
```

Or manually (from the repo root, after `railway login`):

```bash
railway add --service ftaas-web
railway service link ftaas-web
railway variable set FTAAS_GATEWAY_URL=https://ftaas-production.up.railway.app FTAAS_REQUIRE_AUTH=1
# Set FTAAS_API_TOKEN in the Railway dashboard (same value on gateway + ftaas-web)
railway up --detach --ci -m "Deploy ftaas.org product site"
railway domain ftaas.org
railway domain www.ftaas.org
```

3. In the Railway dashboard, remove `ftaas.org` / `www.ftaas.org` from the **gateway** service so only `ftaas-web` owns them.
4. Verify:

```bash
curl -sf https://ftaas.org/health
# expect service: ftaas-web
```

Set `FTAAS_API_TOKEN` on both services. The public Node proxy requires `Authorization: Bearer …` for every `/gateway` call. The workbench has a token field (session-only). The GPU worker stays private (`FTAAS_TRAINER_URL`, port 8090). Do not put Node into the gateway `Dockerfile`. Host `PORT` wins over any local `FTAAS_WEB_PORT`.
