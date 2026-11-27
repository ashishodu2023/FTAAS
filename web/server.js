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
const gatewayHost = (() => {
  try {
    return new URL(GATEWAY).hostname;
  } catch {
    return "";
  }
})();
const gatewayIsLocal = gatewayHost === "127.0.0.1" || gatewayHost === "localhost";
// Public deploys must set FTAAS_API_TOKEN. Localhost stays open for demos unless FTAAS_REQUIRE_AUTH=1.
const AUTH_REQUIRED =
  process.env.FTAAS_REQUIRE_AUTH === "1" ||
  (Boolean(API_TOKEN) && process.env.FTAAS_ALLOW_OPEN_PROXY !== "1") ||
  (!gatewayIsLocal && process.env.FTAAS_ALLOW_OPEN_PROXY !== "1");

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

function bearerToken(req) {
  const header = req.headers.authorization || "";
  const match = /^Bearer\s+(.+)$/i.exec(header);
  return match ? match[1].trim() : "";
}

function authorized(req) {
  if (!AUTH_REQUIRED) return true;
  if (!API_TOKEN) return false;
  const presented = bearerToken(req);
  return Boolean(presented && presented === API_TOKEN);
}

function isUnsafeDatasetPath(value) {
  if (typeof value !== "string" || !value.trim()) return true;
  const p = value.trim();
  if (p.includes("\0")) return true;
  // Allow relative paths and gs:// labels; block file/http and absolute FS paths.
  if (/^(file|https?|ftp):/i.test(p)) return true;
  if (/^[a-zA-Z]:[\\/]/.test(p) || p.startsWith("\\\\") || p.startsWith("/")) return true;
  if (p.includes("..")) return true;
  return false;
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
      detail:
        AUTH_REQUIRED && !API_TOKEN
          ? "Server misconfigured: FTAAS_API_TOKEN is required for a public gateway proxy."
          : "Authorization required. Set Authorization: Bearer <FTAAS_API_TOKEN>.",
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

  const method = (req.method || "GET").toUpperCase();
  const maxBody = Number(process.env.FTAAS_PROXY_MAX_BODY || 1_048_576); // 1 MiB
  const declared = Number(req.headers["content-length"] || 0);
  if (declared > maxBody) {
    sendJson(res, 413, { detail: `Request body too large (max ${maxBody} bytes).` });
    return;
  }

  const chunks = [];
  let size = 0;
  let tooLarge = false;
  req.on("data", (chunk) => {
    if (tooLarge) return;
    size += chunk.length;
    if (size > maxBody) {
      tooLarge = true;
      chunks.length = 0;
      return;
    }
    chunks.push(chunk);
  });
  req.on("end", () => {
    if (tooLarge) {
      sendJson(res, 413, { detail: `Request body too large (max ${maxBody} bytes).` });
      return;
    }
    const bodyBuf = Buffer.concat(chunks);
    if (WRITE_METHODS.has(method) && bodyBuf.length) {
      try {
        const payload = JSON.parse(bodyBuf.toString("utf8"));
        const pathValue =
          payload?.gcs_path ?? payload?.path ?? payload?.dataset_path ?? payload?.uri;
        if (pathValue != null && isUnsafeDatasetPath(String(pathValue))) {
          sendJson(res, 400, {
            detail:
              "Dataset path must be a relative path under the gateway data root (no absolute, file://, or .. paths).",
          });
          return;
        }
      } catch {
        // Non-JSON bodies pass through unchanged.
      }
    }

    const headers = { accept: req.headers.accept || "application/json" };
    if (req.headers["content-type"]) headers["content-type"] = req.headers["content-type"];
    if (bodyBuf.length) headers["content-length"] = String(bodyBuf.length);
    if (API_TOKEN) headers.authorization = `Bearer ${API_TOKEN}`;
    else if (req.headers.authorization) headers.authorization = req.headers.authorization;

    const transport = dest.protocol === "https:" ? https : http;
    const timeout = proxyTimeoutMs(dest.pathname);
    let settled = false;

    const preq = transport.request(
      dest,
      { method, headers, timeout },
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
    if (bodyBuf.length) preq.end(bodyBuf);
    else preq.end();
  });
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
        auth_required: AUTH_REQUIRED,
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
  if (AUTH_REQUIRED) {
    console.log(
      API_TOKEN
        ? "Gateway proxy auth: required (FTAAS_API_TOKEN)"
        : "Gateway proxy auth: REQUIRED but FTAAS_API_TOKEN is missing",
    );
  } else {
    console.log("Gateway proxy auth: open (local demo)");
  }
});
