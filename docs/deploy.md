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

1. Keep the Python gateway on its Railway URL (`https://ftaas-production.up.railway.app`). Prefer **no** public custom domain on the gateway after cutover; if you need one, use a separate host such as `api.ftaas.org`.
2. Set the **same** `FTAAS_API_TOKEN` on the gateway and on `ftaas-web` in the Railway dashboard. When that variable is set, every gateway `/v1` route requires `Authorization: Bearer …`.
3. Deploy the product site (uses the `web/` Docker context — not the gateway image):

```bash
./scripts/deploy_web_railway.sh
```

Or manually (from the repo root, after `railway login`):

```bash
railway add --service ftaas-web
railway service link ftaas-web
railway variable set FTAAS_GATEWAY_URL=https://ftaas-production.up.railway.app FTAAS_REQUIRE_AUTH=1
railway up web --path-as-root --service ftaas-web --detach --ci -m "Deploy ftaas.org product site"
```

4. In the Railway dashboard, remove `ftaas.org` / `www.ftaas.org` from the **gateway** service, then attach them to `ftaas-web`:

```bash
railway service link ftaas-web
railway domain ftaas.org
railway domain www.ftaas.org
```

5. Verify:

```bash
curl -sf https://ftaas.org/health
# expect service: ftaas-web
```

The public Node proxy requires `Authorization: Bearer …` for every `/gateway` call. The workbench has a token field (session-only). The GPU worker stays private (`FTAAS_TRAINER_URL`, port 8090). Host `PORT` wins over any local `FTAAS_WEB_PORT`.
