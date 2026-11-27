const NAV = [
  ["/", "Product"],
  ["/pipeline", "Pipeline"],
  ["/catalog", "Catalog"],
  ["/workbench", "Workbench"],
  ["/scope", "Scope"],
];

export function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export function layout({ title, current, main }) {
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
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Nunito:wght@500;700;800&display=swap" rel="stylesheet" />
  <link rel="stylesheet" href="/static/styles.css" />
</head>
<body>
  <header class="top">
    <div class="top-row">
      <a class="brand" href="/">
        <span class="logo">FTAAS</span>
        <span class="tag">Product site</span>
      </a>
      <p class="gateway-pill" id="gateway-pill">Checking gateway…</p>
    </div>
    <nav class="nav">${links}</nav>
  </header>
  <main>${main}</main>
  <footer>
    <span>Fine Tuning as a Service · <a href="https://ftaas.org">ftaas.org</a></span>
    <span id="console-link"></span>
  </footer>
  <script src="/static/site.js" defer></script>
</body>
</html>`;
}
