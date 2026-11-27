/**
 * FTAAS product site.
 * Proxies /gateway/* to the Python gateway (FTAAS_GATEWAY_URL, default http://127.0.0.1:8080).
 * Does not train. When FTAAS_API_TOKEN is set, mutating /gateway calls require Bearer auth.
 */
import fs from "node:fs";
import http from "node:http";
import https from "node:https";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { mergeCatalog, STATIC_CATALOG } from "./catalog-data.js";
import { catalogPage, homePage, notFoundPage, pipelinePage, scopePage, workbenchPage } from "./pages.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PUBLIC_DIR = path.join(__dirname, "public");
// Host PORT wins (Railway/Render). FTAAS_WEB_PORT is a local override only when PORT is unset.
const PORT = Number(process.env.PORT || process.env.FTAAS_WEB_PORT || 3000);
const GATEWAY = (process.env.FTAAS_GATEWAY_URL || "http://127.0.0.1:8080").replace(/\/$/, "");
const API_TOKEN = (process.env.FTAAS_API_TOKEN || "").trim();
const WRITE_METHODS = new Set(["POST", "PUT", "PATCH", "DELETE"]);

const MIME = {
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".html": "text/html; charset=utf-8",
  ".json": "application/json; charset=utf-8",
};

function send(res, status, body, type = "text/html; charset=utf-8") {
  if (res.headersSent) return;
  res.writeHead(status, { "content-type": type, "cache-control": "no-store" });
  res.end(body);
}

function sendJson(res, status, obj) {
  send(res, status, JSON.stringify(obj), "application/json; charset=utf-8");
}

async function fetchJson(pathname, timeoutMs = 1500) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const headers = {};
    if (API_TOKEN) headers.authorization = `Bearer ${API_TOKEN}`;
    const res = await fetch(`${GATEWAY}${pathname}`, { signal: ctrl.signal, headers });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

function serveStatic(res, pathname) {
  const rel = path.normalize(pathname.slice("/static/".length));
  if (rel.startsWith("..") || path.isAbsolute(rel)) {
    send(res, 400, "Bad path", "text/plain; charset=utf-8");
    return;
  }
  const file = path.join(PUBLIC_DIR, rel);
  if (!file.startsWith(PUBLIC_DIR + path.sep) && file !== PUBLIC_DIR) {
    send(res, 400, "Bad path", "text/plain; charset=utf-8");
    return;
  }
  fs.readFile(file, (err, buf) => {
    if (err) {
      send(res, 404, "Not found", "text/plain; charset=utf-8");
      return;
    }
    send(res, 200, buf, MIME[path.extname(file)] || "application/octet-stream");
  });
}

function authorized(req) {
  if (!API_TOKEN) return true;
  if (!WRITE_METHODS.has((req.method || "GET").toUpperCase())) return true;
  const header = req.headers.authorization || "";
  const match = /^Bearer\s+(.+)$/i.exec(header);
  return Boolean(match && match[1] === API_TOKEN);
}

function proxyTimeoutMs(pathname) {
  // Prompt / train-adjacent calls can wait on model load.
  if (pathname.includes("/prompt") || pathname.includes("/finetune") || pathname.includes("/train")) {
    return 600_000;
  }
  return 120_000;
}

function proxy(req, res) {
  if (!authorized(req)) {
    sendJson(res, 401, {
      detail: "Authorization required. Set Authorization: Bearer <FTAAS_API_TOKEN>.",
    });
    return;
  }

  let dest;
  let gatewayOrigin;
  try {
    gatewayOrigin = new URL(GATEWAY).origin;
    const destPath = req.url.slice("/gateway".length) || "/";
    dest = new URL(destPath, GATEWAY);
  } catch {
    sendJson(res, 400, { detail: "Bad gateway path" });
    return;
  }
  if (dest.origin !== gatewayOrigin) {
    sendJson(res, 400, { detail: "Gateway origin mismatch" });
    return;
  }

  const headers = { accept: req.headers.accept || "application/json" };
  if (req.headers["content-type"]) headers["content-type"] = req.headers["content-type"];
  if (req.headers["content-length"]) headers["content-length"] = req.headers["content-length"];
  if (API_TOKEN) headers.authorization = `Bearer ${API_TOKEN}`;
  else if (req.headers.authorization) headers.authorization = req.headers.authorization;

  const transport = dest.protocol === "https:" ? https : http;
  const timeout = proxyTimeoutMs(dest.pathname);
  let settled = false;

  const preq = transport.request(
    dest,
    { method: req.method, headers, timeout },
    (pres) => {
      const out = { ...pres.headers, "cache-control": "no-store" };
      res.writeHead(pres.statusCode || 502, out);
      pres.pipe(res);
    },
  );
  preq.on("timeout", () => {
    preq.destroy(new Error("Gateway request timed out"));
  });
  preq.on("error", (err) => {
    if (settled || res.headersSent) return;
    settled = true;
    sendJson(res, 502, {
      detail: `FTAAS gateway unreachable at ${GATEWAY}${err?.message ? ` (${err.message})` : ""}`,
    });
  });
  req.on("aborted", () => preq.destroy());
  req.pipe(preq);
}

const server = http.createServer(async (req, res) => {
  let url;
  try {
    const host = req.headers.host || "127.0.0.1";
    url = new URL(req.url || "/", `http://${host}`);
  } catch {
    send(res, 400, "Bad request", "text/plain; charset=utf-8");
    return;
  }

  try {
    if (url.pathname.startsWith("/gateway")) {
      proxy(req, res);
      return;
    }
    if (url.pathname.startsWith("/static/")) {
      if (req.method !== "GET" && req.method !== "HEAD") {
        send(res, 405, "Method not allowed", "text/plain; charset=utf-8");
        return;
      }
      serveStatic(res, url.pathname);
      return;
    }
    if (url.pathname === "/health") {
      const gateway = await fetchJson("/health", 1200);
      sendJson(res, 200, {
        status: "ok",
        service: "ftaas-web",
        gateway: GATEWAY,
        gateway_status: gateway?.status === "ok" ? "ok" : "unreachable",
        console_url: `${GATEWAY}/`,
        auth_required: Boolean(API_TOKEN),
      });
      return;
    }
    if (req.method !== "GET" && req.method !== "HEAD") {
      send(res, 405, "Method not allowed", "text/plain; charset=utf-8");
      return;
    }

    if (url.pathname === "/") {
      send(res, 200, homePage());
      return;
    }
    if (url.pathname === "/pipeline") {
      send(res, 200, pipelinePage());
      return;
    }
    if (url.pathname === "/catalog") {
      const [live, evalStack] = await Promise.all([fetchJson("/v1/catalog"), fetchJson("/v1/eval/stack")]);
      const catalog = mergeCatalog(live);
      send(res, 200, catalogPage({ catalog, evalStack, gatewayUp: Boolean(live) }));
      return;
    }
    if (url.pathname === "/workbench") {
      const live = await fetchJson("/v1/catalog");
      const catalog = live ? mergeCatalog(live) : STATIC_CATALOG;
      send(
        res,
        200,
        workbenchPage({
          catalog,
          gateway: GATEWAY,
          catalogSource: live ? "gateway" : "static",
        }),
      );
      return;
    }
    if (url.pathname === "/scope") {
      send(res, 200, scopePage());
      return;
    }

    send(res, 404, notFoundPage());
  } catch (err) {
    sendJson(res, 500, { detail: err?.message || "Internal error" });
  }
});

server.listen(PORT, "0.0.0.0", () => {
  console.log(`FTAAS product site http://127.0.0.1:${PORT}`);
  console.log(`Proxying /gateway → ${GATEWAY}`);
  if (API_TOKEN) console.log("Write proxy auth: FTAAS_API_TOKEN is set");
});
