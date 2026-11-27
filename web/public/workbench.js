/**
 * Workbench — same /v1 calls as ui/console/static/console.js, via /gateway.
 */
(() => {
  const TERMINAL = new Set(["succeeded", "failed", "cancelled"]);
  const boot = JSON.parse(document.getElementById("boot").textContent || "{}");
  const catalog = boot.catalog || {};

  const flashEl = document.getElementById("flash");
  const tokenInp = document.getElementById("inp-api-token");
  const TOKEN_KEY = "ftaas_api_token";
  let memoryToken = "";
  function readStoredToken() {
    try {
      return sessionStorage.getItem(TOKEN_KEY) || "";
    } catch {
      return memoryToken;
    }
  }
  function writeStoredToken(value) {
    memoryToken = value;
    try {
      sessionStorage.setItem(TOKEN_KEY, value);
    } catch {
      // Browser blocked storage — keep the in-memory token only.
    }
  }
  if (tokenInp) {
    tokenInp.value = readStoredToken();
    const onTokenUpdate = () => {
      writeStoredToken(tokenInp.value.trim());
      refresh().catch((err) => flash(err.message, true));
    };
    tokenInp.addEventListener("change", onTokenUpdate);
    tokenInp.addEventListener("blur", onTokenUpdate);
  }
  const frameworkSel = document.getElementById("sel-framework");
  const frameworkNote = document.getElementById("framework-note");
  const techSel = document.getElementById("sel-technique");
  const trainHint = document.getElementById("train-plane-hint");
  const datasetSel = document.getElementById("sel-dataset");
  const versionInp = document.getElementById("inp-dataset-version");
  const modelSel = document.getElementById("sel-model");
  const endpointSel = document.getElementById("sel-endpoint");
  const listDatasets = document.getElementById("list-datasets");
  const listModels = document.getElementById("list-models");
  const listEndpoints = document.getElementById("list-endpoints");
  const jobsBody = document.getElementById("tbody-jobs");
  const previewBox = document.getElementById("dataset-preview");
  const btnPreview = document.getElementById("btn-preview");
  const btnRegister = document.getElementById("btn-register");
  const promptResult = document.getElementById("prompt-result");
  const promptOut = document.getElementById("prompt-out");
  const logTitle = document.getElementById("job-detail-title");
  const logView = document.getElementById("job-log-view");
  const progressWrap = document.getElementById("job-progress-wrap");
  const progressBar = document.getElementById("job-progress-bar");

  const state = {
    datasets: [],
    jobs: [],
    models: [],
    endpoints: [],
    selectedJobId: null,
    previewKey: null,
    previewOk: false,
  };
  let pollTimer = null;
  let refreshSeq = 0;

  function el(tag, attrs = {}, children = []) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) {
      if (value == null || value === false) continue;
      if (key === "className") node.className = value;
      else if (key === "text") node.textContent = value;
      else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2).toLowerCase(), value);
      else if (key === "dataset") Object.assign(node.dataset, value);
      else node.setAttribute(key, value === true ? "" : String(value));
    }
    for (const child of [].concat(children)) {
      if (child == null || child === false) continue;
      node.append(child.nodeType ? child : document.createTextNode(String(child)));
    }
    return node;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function setMutedList(list, message) {
    clear(list);
    list.append(el("li", { className: "muted", text: message }));
  }

  function flash(message, isErr = false) {
    if (!flashEl) return;
    flashEl.hidden = !message;
    flashEl.textContent = message || "";
    flashEl.classList.toggle("err", Boolean(isErr));
  }

  function apiHeaders(extra = {}) {
    const headers = { Accept: "application/json", ...extra };
    const token = (tokenInp?.value || readStoredToken() || "").trim();
    if (token) headers.Authorization = `Bearer ${token}`;
    return headers;
  }

  async function api(path, opts = {}) {
    const res = await fetch(`/gateway${path}`, {
      ...opts,
      headers: apiHeaders({
        ...(opts.body ? { "Content-Type": "application/json" } : {}),
        ...(opts.headers || {}),
      }),
    });
    const text = await res.text();
    let data = null;
    try {
      data = text ? JSON.parse(text) : null;
    } catch {
      data = { detail: text };
    }
    if (!res.ok) {
      if (res.status === 401) {
        throw new Error("Authorization required — enter the API token above.");
      }
      const detail = data?.detail;
      const msg = typeof detail === "string" ? detail : JSON.stringify(detail || data || res.statusText);
      throw new Error(msg);
    }
    return data;
  }

  function fillFrameworks() {
    const backends = catalog.backends || {};
    const selected = catalog.defaults?.framework || "transformers";
    clear(frameworkSel);
    for (const name of catalog.frameworks || ["transformers"]) {
      const note = backends[name] ? ` — ${backends[name]}` : "";
      const option = el("option", { value: name, text: `${name}${note}` });
      if (name === selected) option.selected = true;
      frameworkSel.append(option);
    }
    showFrameworkNote();
  }

  function showFrameworkNote() {
    frameworkNote.textContent = (catalog.backends || {})[frameworkSel.value] || "";
  }

  function fillTechniques() {
    const groups = catalog.techniques || {};
    const order = [
      ["PEFT", groups.peft || ["lora", "qlora"]],
      ["Full", groups.full_finetune || []],
      ["Instruction", groups.instruction_tuning || []],
      ["Alignment", groups.alignment || []],
    ];
    const preferred = catalog.defaults?.technique || "lora";
    clear(techSel);
    for (const [label, items] of order) {
      const group = el("optgroup", { label });
      for (const name of items || []) {
        const option = el("option", { value: name, text: name });
        if (name === preferred) option.selected = true;
        group.append(option);
      }
      techSel.append(group);
    }
    showTrainHint();
  }

  function qloraBlocked() {
    if (techSel.value !== "qlora") return false;
    const train = catalog.training || {};
    if (String(train.mode || "").toLowerCase() === "remote") return false;
    const q = train.qlora;
    if (!q) return boot.catalogSource !== "gateway";
    return !q.available && !q.cpu_fallback;
  }

  function showTrainHint() {
    const train = catalog.training || {};
    const q = train.qlora || {};
    const device = train.device || {};
    if (boot.catalogSource !== "gateway") {
      trainHint.textContent = "Catalog is the offline copy. Connect the gateway before creating a job.";
      return;
    }
    const cuda = device.cuda_available ? "CUDA" : "no CUDA";
    const mode = train.mode || "local";
    const qlora =
      mode === "remote"
        ? "QLoRA allowed via remote trainer"
        : q.available
          ? "QLoRA ready"
          : "QLoRA needs GPU+bnb";
    trainHint.textContent = `Training plane: ${mode} · ${cuda} · ${qlora}`;
  }

  function renderDatasets() {
    const current = datasetSel.value;
    if (!state.datasets.length) {
      setMutedList(listDatasets, "No datasets yet");
      clear(datasetSel);
      datasetSel.append(el("option", { value: "", text: "Register a dataset first" }));
      return;
    }
    clear(listDatasets);
    clear(datasetSel);
    for (const d of state.datasets) {
      listDatasets.append(
        el("li", {}, [
          el("code", { text: `${d.dataset_id}:${d.version}` }),
          document.createTextNode(` — ${d.name || "dataset"} (${d.num_rows ?? "?"} rows)`),
        ]),
      );
      const option = el("option", {
        value: d.dataset_id,
        text: `${d.name || d.dataset_id} (${d.dataset_id}:${d.version})`,
        dataset: { version: String(d.version) },
      });
      datasetSel.append(option);
    }
    if (current) datasetSel.value = current;
    syncVersion();
  }

  function syncVersion() {
    const opt = datasetSel.selectedOptions[0];
    if (opt?.dataset.version) versionInp.value = opt.dataset.version;
  }

  function renderModels() {
    const current = modelSel.value;
    if (!state.models.length) {
      setMutedList(listModels, "No models registered");
      clear(modelSel);
      modelSel.append(el("option", { value: "", text: "No model yet" }));
      return;
    }
    clear(listModels);
    clear(modelSel);
    for (const m of state.models) {
      listModels.append(
        el("li", {}, [
          el("code", { text: m.model_name }),
          document.createTextNode(` v${m.version}`),
        ]),
      );
      const value = `${m.model_name}::${m.version}`;
      modelSel.append(
        el("option", {
          value,
          text: `${m.model_name} v${m.version}`,
          dataset: { modelName: m.model_name, version: String(m.version) },
        }),
      );
    }
    if (current && [...modelSel.options].some((o) => o.value === current)) modelSel.value = current;
  }

  function renderEndpoints() {
    const current = endpointSel.value;
    if (!state.endpoints.length) {
      setMutedList(listEndpoints, "No endpoints");
      clear(endpointSel);
      endpointSel.append(el("option", { value: "", text: "Deploy a model first" }));
      return;
    }
    clear(listEndpoints);
    clear(endpointSel);
    for (const e of state.endpoints) {
      listEndpoints.append(
        el("li", {}, [
          el("code", { text: e.endpoint_id }),
          document.createTextNode(` — ${e.model_name} @ ${e.inference_framework}`),
        ]),
      );
      endpointSel.append(
        el("option", {
          value: e.endpoint_id,
          text: `${e.endpoint_id} — ${e.model_name}`,
        }),
      );
    }
    if (current && [...endpointSel.options].some((o) => o.value === current)) endpointSel.value = current;
  }

  function renderJobs() {
    clear(jobsBody);
    if (!state.jobs.length) {
      jobsBody.append(
        el("tr", {}, [el("td", { colSpan: "6", className: "muted", text: "No jobs yet" })]),
      );
      return;
    }
    for (const job of state.jobs) {
      const progress = job.progress || {};
      const pct = Math.max(0, Math.min(100, Number(progress.percent) || 0));
      const actions = el("td", { className: "actions" }, [
        el("button", {
          type: "button",
          className: "btn small",
          text: "View log",
          dataset: { logs: job.job_id },
        }),
      ]);
      if (!TERMINAL.has(job.status)) {
        actions.append(
          el("button", {
            type: "button",
            className: "btn small danger",
            text: "Stop",
            dataset: { stop: job.job_id },
          }),
        );
      }
      const bar = el("div", { className: "bar" });
      bar.style.width = `${pct}%`;
      jobsBody.append(
        el("tr", { dataset: { job: job.job_id } }, [
          el("td", {}, [el("code", { text: job.job_id })]),
          el("td", { text: job.model_name }),
          el("td", { text: `${job.framework} / ${job.technique}` }),
          el("td", {}, [el("span", { className: `pill ${job.status}`, text: job.status })]),
          el("td", {}, [
            el("div", { className: "progress" }, [bar]),
            el("small", { text: progress.message || job.error || "—" }),
          ]),
          actions,
        ]),
      );
    }
  }

  function previewFingerprint(pathValue, formatValue) {
    const path = pathValue ?? document.getElementById("inp-gcs-path").value.trim();
    const format = formatValue ?? document.getElementById("sel-ds-format").value;
    return `${path}::${format}`;
  }

  function renderPreview(data, requestedKey) {
    if (requestedKey !== previewFingerprint()) {
      state.previewOk = false;
      btnRegister.disabled = true;
      return;
    }
    clear(previewBox);
    previewBox.hidden = false;
    previewBox.append(
      el("p", {
        className: "hint",
        text: `${data.num_rows ?? "?"} rows · ${data.format || ""}`,
      }),
    );
    for (const warning of data.warnings || []) {
      previewBox.append(el("p", { className: "hint warn-text", text: warning }));
    }
    const rows = data.samples || data.rows || [];
    const cols = data.columns || (rows[0] ? Object.keys(rows[0]) : []);
    const thead = el("thead", {}, [
      el(
        "tr",
        {},
        cols.map((col) => el("th", { text: col })),
      ),
    ]);
    const tbody = el("tbody");
    if (!rows.length) {
      tbody.append(el("tr", {}, [el("td", { className: "muted", text: "No sample rows" })]));
    } else {
      for (const row of rows.slice(0, 5)) {
        tbody.append(
          el(
            "tr",
            {},
            cols.map((col) =>
              el("td", {
                text: typeof row[col] === "object" ? JSON.stringify(row[col]) : String(row[col] ?? ""),
              }),
            ),
          ),
        );
      }
    }
    previewBox.append(el("div", { className: "table-scroll" }, [el("table", {}, [thead, tbody])]));
    state.previewOk = true;
    state.previewKey = requestedKey;
    btnRegister.disabled = false;
  }

  async function showLogs(jobId) {
    state.selectedJobId = jobId;
    const data = await api(`/v1/jobs/${encodeURIComponent(jobId)}/logs`);
    if (state.selectedJobId !== jobId) return;
    const progress = data.progress || {};
    const pct = Math.max(0, Math.min(100, Number(progress.percent) || 0));
    logTitle.textContent = `${jobId} · ${data.status || ""}`;
    progressWrap.hidden = false;
    progressBar.style.width = `${pct}%`;
    const lines = (data.logs || []).map((line) => {
      if (typeof line === "string") return line;
      return `${line.ts || ""} ${line.message || JSON.stringify(line)}`.trim();
    });
    if (data.error) lines.push(data.error);
    logView.textContent = lines.join("\n") || progress.message || "No log lines yet.";
  }

  async function refresh() {
    const seq = ++refreshSeq;
    const [datasets, jobs, models, endpoints] = await Promise.all([
      api("/v1/datasets"),
      api("/v1/jobs"),
      api("/v1/models"),
      api("/v1/endpoints"),
    ]);
    if (seq !== refreshSeq) return;
    state.datasets = datasets;
    state.jobs = jobs;
    state.models = models;
    state.endpoints = endpoints;
    renderDatasets();
    renderJobs();
    renderModels();
    renderEndpoints();
    const active = state.jobs.some((job) => !TERMINAL.has(job.status));
    if (active && !pollTimer) {
      pollTimer = setInterval(() => refresh().catch((err) => flash(err.message, true)), 4000);
    }
    if (!active && pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
    if (state.selectedJobId) {
      showLogs(state.selectedJobId).catch(() => {});
    }
  }

  fillFrameworks();
  fillTechniques();
  frameworkSel.addEventListener("change", showFrameworkNote);
  techSel.addEventListener("change", showTrainHint);
  datasetSel.addEventListener("change", syncVersion);

  document.getElementById("inp-gcs-path").addEventListener("input", () => {
    state.previewOk = false;
    btnRegister.disabled = true;
  });
  document.getElementById("sel-ds-format").addEventListener("change", () => {
    state.previewOk = false;
    btnRegister.disabled = true;
  });

  btnPreview.addEventListener("click", async () => {
    const path = document.getElementById("inp-gcs-path").value.trim();
    const format = document.getElementById("sel-ds-format").value;
    if (!path) {
      flash("Enter a dataset path to preview", true);
      return;
    }
    if (path.startsWith("/") || path.startsWith("file://") || /^[A-Za-z]:[\\/]/.test(path)) {
      flash("Use a path relative to the gateway working directory (for example examples/data/alpaca_sample.jsonl).", true);
      return;
    }
    const requestedKey = previewFingerprint(path, format);
    btnPreview.disabled = true;
    try {
      const data = await api("/v1/datasets/preview", {
        method: "POST",
        body: JSON.stringify({ gcs_path: path, format, limit: 5 }),
      });
      renderPreview(data, requestedKey);
      if (state.previewOk) flash(`Preview ready — ${data.num_rows ?? "?"} rows.`);
    } catch (err) {
      state.previewOk = false;
      btnRegister.disabled = true;
      previewBox.hidden = true;
      flash(err.message, true);
    } finally {
      btnPreview.disabled = false;
    }
  });

  document.getElementById("form-register").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const fd = new FormData(ev.target);
    const path = String(fd.get("gcs_path") || "").trim();
    if (path.startsWith("/") || path.startsWith("file://") || /^[A-Za-z]:[\\/]/.test(path)) {
      flash("Refuse absolute paths. Use a relative gateway path.", true);
      return;
    }
    if (!state.previewOk || state.previewKey !== previewFingerprint()) {
      flash("Preview the dataset first, then register", true);
      return;
    }
    btnRegister.disabled = true;
    try {
      const ds = await api("/v1/datasets/register", {
        method: "POST",
        body: JSON.stringify({
          gcs_path: path,
          name: fd.get("name") || null,
          format: fd.get("format") || "jsonl",
        }),
      });
      flash(`Registered ${ds.dataset_id}:${ds.version}`);
      state.previewOk = false;
      previewBox.hidden = true;
      await refresh();
      datasetSel.value = ds.dataset_id;
      syncVersion();
    } catch (err) {
      flash(err.message, true);
      btnRegister.disabled = false;
    }
  });

  document.getElementById("form-job").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const fd = new FormData(ev.target);
    if (!fd.get("dataset_id")) {
      flash("Register a dataset first", true);
      return;
    }
    if (qloraBlocked()) {
      flash(
        "QLoRA needs CUDA and bitsandbytes on this gateway. Use technique lora for a CPU demo, or set FTAAS_TRAIN_MODE=remote with a GPU worker.",
        true,
      );
      return;
    }
    const button = ev.target.querySelector('button[type="submit"]');
    button.disabled = true;
    try {
      const loraR = Number(fd.get("lora_r") || 8);
      const job = await api("/v1/jobs/finetune", {
        method: "POST",
        body: JSON.stringify({
          model_name: fd.get("model_name"),
          framework: fd.get("framework"),
          technique: fd.get("technique"),
          dataset: {
            dataset_id: fd.get("dataset_id"),
            version: String(fd.get("dataset_version") || "1"),
          },
          parameters: {
            max_steps: Math.max(12, Number(fd.get("max_steps") || 24)),
            learning_rate: Number(fd.get("learning_rate") || 0.001),
            lora_r: loraR,
            lora_alpha: loraR * 2,
            per_device_train_batch_size: 1,
            max_seq_length: 64,
          },
        }),
      });
      flash(`Job ${job.job_id} queued`);
      state.selectedJobId = job.job_id;
      await refresh();
      await showLogs(job.job_id);
    } catch (err) {
      flash(err.message, true);
    } finally {
      button.disabled = false;
    }
  });

  jobsBody.addEventListener("click", async (ev) => {
    const logs = ev.target.closest("[data-logs]");
    const stop = ev.target.closest("[data-stop]");
    try {
      if (stop) {
        await api(`/v1/jobs/${encodeURIComponent(stop.dataset.stop)}/cancel`, { method: "POST" });
        flash(`Stop requested for ${stop.dataset.stop}`);
        await refresh();
        return;
      }
      if (logs) await showLogs(logs.dataset.logs);
    } catch (err) {
      flash(err.message, true);
    }
  });

  document.getElementById("form-deploy").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const selected = modelSel.selectedOptions[0];
    if (!selected || !selected.value) {
      flash("No model to deploy yet", true);
      return;
    }
    const modelName = selected.dataset.modelName || selected.value.split("::")[0];
    const modelVersion = selected.dataset.version || selected.value.split("::")[1] || undefined;
    try {
      const ep = await api("/v1/endpoints", {
        method: "POST",
        body: JSON.stringify({
          model_name: modelName,
          model_version: modelVersion,
          inference_framework: "transformers",
          use_adapters: true,
        }),
      });
      flash(`Endpoint ${ep.endpoint_id} ready`);
      await refresh();
      endpointSel.value = ep.endpoint_id;
    } catch (err) {
      flash(err.message, true);
    }
  });

  document.getElementById("form-prompt").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const fd = new FormData(ev.target);
    const endpointId = fd.get("endpoint_id");
    if (!endpointId) {
      flash("Deploy an endpoint first", true);
      return;
    }
    try {
      const resp = await api(`/v1/endpoints/${encodeURIComponent(endpointId)}/prompt`, {
        method: "POST",
        body: JSON.stringify({
          prompt: fd.get("prompt"),
          max_tokens: 48,
          temperature: 0,
        }),
      });
      promptResult.hidden = false;
      promptOut.textContent = resp.completion || JSON.stringify(resp, null, 2);
      flash("Prompt completed");
    } catch (err) {
      flash(err.message, true);
    }
  });

  document.getElementById("btn-refresh").addEventListener("click", () => {
    refresh().then(() => flash("Refreshed")).catch((err) => flash(err.message, true));
  });

  refresh().catch((err) => {
    flash(err.message, true);
    setMutedList(listDatasets, "Gateway offline — start it with ./scripts/start_all.sh");
    clear(jobsBody);
    jobsBody.append(el("tr", {}, [el("td", { colSpan: "6", className: "muted", text: "Gateway offline" })]));
  });
})();
