/** Shared chrome: gateway status + console link. */
(() => {
  const pill = document.getElementById("gateway-pill");
  const consoleLink = document.getElementById("console-link");

  fetch("/health")
    .then((res) => res.json())
    .then((data) => {
      const up = data.gateway_status === "ok";
      if (pill) {
        pill.textContent = up ? "Gateway connected" : "Gateway offline";
        pill.classList.toggle("ok", up);
        pill.classList.toggle("down", !up);
      }
      if (consoleLink && data.console_url) {
        consoleLink.innerHTML = "";
        const a = document.createElement("a");
        a.href = data.console_url;
        a.textContent = "Operator console";
        a.rel = "noreferrer";
        consoleLink.appendChild(a);
      }
    })
    .catch(() => {
      if (pill) {
        pill.textContent = "Status unknown";
        pill.classList.add("down");
      }
    });
})();
