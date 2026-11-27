const NAV = [
  ["/", "Overview"],
  ["/pipeline", "How it works"],
  ["/catalog", "Catalog"],
  ["/workbench", "Workbench"],
  ["/scope", "Audience"],
];

export function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export function layout({ title, current, main, pageClass = "" }) {
  const links = NAV.map(([href, label]) => {
    const active = href === current ? ' aria-current="page"' : "";
    return `<a href="${href}"${active}>${label}</a>`;
  }).join("");

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>${escapeHtml(title)} — FTAAS</title>
  <meta name="description" content="Fine Tuning as a Service — preview, register, LoRA/QLoRA train, and prompt on one shared gateway." />
  <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate" />
  <meta http-equiv="Pragma" content="no-cache" />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Nunito:wght@500;600;700;800&display=swap" rel="stylesheet" />
  <link rel="stylesheet" href="/static/styles.css?v=build27" />
</head>
<body class="${escapeHtml(pageClass)}">
  <a class="skip" href="#content">Skip to content</a>
  <header class="site-header">
    <div class="shell header-inner">
      <a class="brand" href="/">
        <span class="logo">FTAAS</span>
        <span class="brand-mark" aria-hidden="true"></span>
      </a>
      <nav class="nav" aria-label="Primary">${links}</nav>
      <div class="header-meta">
        <span class="build-stamp" title="Product site build marker">UI build 27</span>
        <p class="status" id="gateway-pill" role="status">Checking gateway…</p>
        <a class="btn compact secondary" href="/workbench">Start</a>
      </div>
    </div>
  </header>
  <main id="content" class="shell">${main}</main>
  <footer class="site-footer">
    <div class="shell footer-inner">
      <div>
        <p class="footer-brand">FTAAS</p>
        <p class="footer-copy">Fine Tuning as a Service · <a href="https://ftaas.org">ftaas.org</a></p>
      </div>
      <div class="footer-links">
        <a href="/workbench">Workbench</a>
        <a href="/catalog">Catalog</a>
        <span id="console-link"></span>
      </div>
    </div>
  </footer>
  <script src="/static/site.js?v=build27" defer></script>
</body>
</html>`;
}
