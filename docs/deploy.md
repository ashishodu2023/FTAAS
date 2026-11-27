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

This file deploys the **Python gateway** (console + `/v1`) and attaches `ftaas.org` to that service. The Node product site in `web/` is the other part: the pages for the same domain. It does not train.

Leave this Railway service where it is until a second host is running `web/`. Build that host from the **repo root**:

```bash
docker build -f web/Dockerfile -t ftaas-web .
```

Point it at `ftaas.org` / `www.ftaas.org`, and set `FTAAS_GATEWAY_URL` to this service (the `*.up.railway.app` URL, or `https://api.ftaas.org`). Set `FTAAS_API_TOKEN` on both the site and the gateway when the site is public — the Node proxy requires `Authorization: Bearer …` on write routes when that token is set. The GPU worker stays private (`FTAAS_TRAINER_URL`, port 8090). Do not put Node into the gateway `Dockerfile`; `railway.toml` builds that image for the API and console. Host `PORT` wins over any local `FTAAS_WEB_PORT`.
