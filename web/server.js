/**
 * FTAAS product site.
 * Proxies /gateway/* to the Python gateway (FTAAS_GATEWAY_URL, default http://127.0.0.1:8080).
 * Does not train, and does not read gateway credentials.
 */
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { mergeCatalog, STATIC_CATALOG } from "./catalog-data.js";
import { catalogPage, homePage, notFoundPage, pipelinePage, scopePage, workbenchPage } from "./pages.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PUBLIC_DIR = path.join(__dirname, "public");
const PORT = Number(process.env.FTAAS_WEB_PORT || process.env.PORT || 3000);
const GATEWAY = (process.env.FTAAS_GATEWAY_URL || "http://127.0.0.1:8080").replace(/\/$/, "");

const MIME = {
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".html": "text/html; charset=utf-8",
  ".json": "application/json; charset=utf-8",
};

function send(res, status, body, type = "text/html; charset=utf-8") {
  res.writeHead(status, { "content-type": type, "cache-control": "no-store" });
  res.end(body);
}

async function fetchJson(pathname, timeoutMs = 1500) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${GATEWAY}${pathname}`, { signal: ctrl.signal });
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

function proxy(req, res) {
  let dest;
  try {
    const destPath = req.url.slice("/gateway".length) || "/";
    dest = new URL(destPath, GATEWAY);
  } catch {
    send(res, 400, JSON.stringify({ detail: "Bad gateway path" }), "application/json; charset=utf-8");
    return;
  }
  if (dest.origin !== new URL(GATEWAY).origin) {
    send(res, 400, JSON.stringify({ detail: "Gateway origin mismatch" }), "application/json; charset=utf-8");
    return;
  }

  const headers = { accept: req.headers.accept || "application/json" };
  if (req.headers["content-type"]) headers["content-type"] = req.headers["content-type"];
  if (req.headers["content-length"]) headers["content-length"] = req.headers["content-length"];

  const preq = http.request(
    dest,
    { method: req.method, headers, timeout: 120_000 },
    (pres) => {
      const out = { ...pres.headers, "cache-control": "no-store" };
      res.writeHead(pres.statusCode || 502, out);
      pres.pipe(res);
    },
  );
  preq.on("timeout", () => preq.destroy());
  preq.on("error", () => {
    if (res.headersSent) return;
    send(
      res,
      502,
      JSON.stringify({ detail: `FTAAS gateway unreachable at ${GATEWAY}` }),
      "application/json; charset=utf-8",
    );
  });
  req.pipe(preq);
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url || "/", `http://${req.headers.host || "127.0.0.1"}`);

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
    send(
      res,
      200,
      JSON.stringify({
        status: "ok",
        service: "ftaas-web",
        gateway: GATEWAY,
        gateway_status: gateway?.status === "ok" ? "ok" : "unreachable",
        console_url: `${GATEWAY}/`,
      }),
      "application/json; charset=utf-8",
    );
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
});

server.listen(PORT, "0.0.0.0", () => {
  console.log(`FTAAS product site http://127.0.0.1:${PORT}`);
  console.log(`Proxying /gateway → ${GATEWAY}`);
});
