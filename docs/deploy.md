# Deploying FTAAS to ftaas.org (Railway)

## Status
- Domain purchased: `ftaas.org`
- DNS may still be propagating (NXDOMAIN until nameservers are set)

## One-time setup

```bash
cd /Users/ashishverma/Downloads/FTAAS
railway login --browserless   # open the printed URL / enter the code
chmod +x scripts/deploy_railway.sh
./scripts/deploy_railway.sh
```

## Attach the domain

```bash
railway domain add ftaas.org
railway domain add www.ftaas.org
```

Railway will print DNS records. At your registrar (wherever you bought `ftaas.org`):

| Type | Name | Value |
|------|------|--------|
| CNAME or ALIAS | `@` / `ftaas.org` | *(value from Railway)* |
| CNAME | `www` | *(value from Railway)* |

Or if Railway asks you to use their nameservers, point the domain NS records there instead.

## Env

```bash
railway variables set FTAAS_PUBLIC_URL=https://ftaas.org
```

## Verify

```bash
curl -sf https://ftaas.org/health
open https://ftaas.org/
```
