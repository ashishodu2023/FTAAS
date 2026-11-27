/**
 * Workbench — same /v1 calls as ui/console/static/console.js, via /gateway.
 */
(() => {
  const TERMINAL = new Set(["succeeded", "failed", "cancelled"]);
  const boot = JSON.parse(document.getElementById("boot").textContent || "{}");
  const catalog = boot.catalog || {};

  const flashEl = document.getElementById("flash");
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

  function esc(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function flash(message, isErr = false) {
    if (!flashEl) return;
    flashEl.hidden = !message;
    flashEl.textContent = message || "";
    flashEl.classList.toggle("err", Boolean(isErr));
  }

  async function api(path, opts = {}) {
    const res = await fetch(`/gateway${path}`, {
      headers: {
        Accept: "application/json",
        ...(opts.body ? { "Content-Type": "application/json" } : {}),
      },
      ...opts,
    });
    const text = await res.text();
    let data = null;
    try {
      data = text ? JSON.parse(text) : null;
    } catch {
      data = { detail: text };
    }
    if (!res.ok) {
      const detail = data?.detail;
      const msg = typeof detail === "string" ? detail : JSON.stringify(detail || data || res.statusText);
      throw new Error(msg);
    }
    return data;
  }

  function fillFrameworks() {
    const backends = catalog.backends || {};
    const selected = catalog.defaults?.framework || "transformers";
    frameworkSel.innerHTML = (catalog.frameworks || ["transformers"]).map((name) => {
      const note = backends[name] ? ` — ${backends[name]}` : "";
      return `<option value="${esc(name)}"${name === selected ? " selected" : ""}>${esc(name)}${esc(note)}</option>`;
    }).join("");
    showFrameworkNote();
  }

  function showFrameworkNote() {
    const note = (catalog.backends || {})[frameworkSel.value] || "";
    frameworkNote.textContent = note;
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
    techSel.innerHTML = order
      .map(([label, items]) => {
        const opts = items
          .map((name) => `<option value="${esc(name)}"${name === preferred ? " selected" : ""}>${esc(name)}</option>`)
          .join("");
        return `<optgroup label="${esc(label)}">${opts}</optgroup>`;
      })
      .join("");
    showTrainHint();
  }

  function qloraBlocked() {
    if (techSel.value !== "qlora") return false;
    const q = catalog.training?.qlora;
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
    const qlora = q.available ? "QLoRA ready" : "QLoRA needs GPU+bnb";
    trainHint.textContent = `Training plane: ${train.mode || "local"} · ${cuda} · ${qlora}`;
  }

  function renderDatasets() {
    if (!state.datasets.length) {
      listDatasets.innerHTML = `<li class="muted">No datasets yet</li>`;
      datasetSel.innerHTML = `<option value="">Register a dataset first</option>`;
      return;
    }
    listDatasets.innerHTML = state.datasets
      .map(
        (d) =>
          `<li><code>${esc(d.dataset_id)}:${esc(d.version)}</code> — ${esc(d.name || "dataset")} (${esc(d.num_rows ?? "?")} rows)</li>`,
      )
      .join("");
    const current = datasetSel.value;
    datasetSel.innerHTML = state.datasets
      .map(
        (d) =>
          `<option value="${esc(d.dataset_id)}" data-version="${esc(d.version)}">${esc(d.name || d.dataset_id)} (${esc(d.dataset_id)}:${esc(d.version)})</option>`,
      )
      .join("");
    if (current) datasetSel.value = current;
    syncVersion();
  }

  function syncVersion() {
    const opt = datasetSel.selectedOptions[0];
    if (opt?.dataset.version) versionInp.value = opt.dataset.version;
  }

  function renderModels() {
    if (!state.models.length) {
      listModels.innerHTML = `<li class="muted">No models registered</li>`;
      modelSel.innerHTML = `<option value="">No model yet</option>`;
      return;
    }
    listModels.innerHTML = state.models
      .map((m) => `<li><code>${esc(m.model_name)}</code> v${esc(m.version)}</li>`)
      .join("");
    modelSel.innerHTML = state.models
      .map((m) => `<option value="${esc(m.model_name)}">${esc(m.model_name)} v${esc(m.version)}</option>`)
      .join("");
  }

  function renderEndpoints() {
    if (!state.endpoints.length) {
      listEndpoints.innerHTML = `<li class="muted">No endpoints</li>`;
      endpointSel.innerHTML = `<option value="">Deploy a model first</option>`;
      return;
    }
    listEndpoints.innerHTML = state.endpoints
      .map(
        (e) =>
          `<li><code>${esc(e.endpoint_id)}</code> — ${esc(e.model_name)} @ ${esc(e.inference_framework)}</li>`,
      )
      .join("");
    endpointSel.innerHTML = state.endpoints
      .map(
        (e) =>
          `<option value="${esc(e.endpoint_id)}">${esc(e.endpoint_id)} — ${esc(e.model_name)}</option>`,
      )
      .join("");
  }

  function renderJobs() {
    if (!state.jobs.length) {
      jobsBody.innerHTML = `<tr><td colspan="6" class="muted">No jobs yet</td></tr>`;
      return;
    }
    jobsBody.innerHTML = state.jobs
      .map((job) => {
        const progress = job.progress || {};
        const pct = Math.max(0, Math.min(100, Number(progress.percent) || 0));
        const stop = TERMINAL.has(job.status)
          ? ""
          : `<button type="button" class="btn small danger" data-stop="${esc(job.job_id)}">Stop</button>`;
        return `<tr data-job="${esc(job.job_id)}">
          <td><code>${esc(job.job_id)}</code></td>
          <td>${esc(job.model_name)}</td>
          <td>${esc(job.framework)} / ${esc(job.technique)}</td>
          <td><span class="pill ${esc(job.status)}">${esc(job.status)}</span></td>
          <td>
            <div class="progress"><div class="bar" style="width:${pct}%"></div></div>
            <small>${esc(progress.message || job.error || "—")}</small>
          </td>
          <td class="actions">
            <button type="button" class="btn small" data-logs="${esc(job.job_id)}">View log</button>
            ${stop}
          </td>
        </tr>`;
      })
      .join("");
  }

  function previewFingerprint() {
    const path = document.getElementById("inp-gcs-path").value.trim();
    const format = document.getElementById("sel-ds-format").value;
    return `${path}::${format}`;
  }

  function renderPreview(data) {
    const rows = data.samples || data.rows || [];
    const cols = data.columns || (rows[0] ? Object.keys(rows[0]) : []);
    const head = cols.map((col) => `<th>${esc(col)}</th>`).join("");
    const body = (Array.isArray(rows) ? rows : [])
      .slice(0, 5)
      .map((row) => `<tr>${cols.map((col) => `<td>${esc(typeof row[col] === "object" ? JSON.stringify(row[col]) : row[col])}</td>`).join("")}</tr>`)
      .join("");
    const warnings = (data.warnings || []).map((warning) => `<p class="hint warn-text">${esc(warning)}</p>`).join("");
    previewBox.hidden = false;
    previewBox.innerHTML = `<p class="hint">${esc(data.num_rows ?? "?")} rows · ${esc(data.format || "")}</p>
      ${warnings}
      <div class="table-scroll"><table><thead><tr>${head}</tr></thead><tbody>${body || `<tr><td class="muted">No sample rows</td></tr>`}</tbody></table></div>`;
    state.previewOk = true;
    state.previewKey = previewFingerprint();
    btnRegister.disabled = false;
  }

  async function showLogs(jobId) {
    state.selectedJobId = jobId;
    const data = await api(`/v1/jobs/${encodeURIComponent(jobId)}/logs`);
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
    const [datasets, jobs, models, endpoints] = await Promise.all([
      api("/v1/datasets"),
      api("/v1/jobs"),
      api("/v1/models"),
      api("/v1/endpoints"),
    ]);
    state.datasets = datasets;
    state.jobs = jobs;
    state.models = models;
    state.endpoints = endpoints;
    renderDatasets();
    renderJobs();
    renderModels();
    renderEndpoints();
    const active = state.jobs.some((job) => !TERMINAL.has(job.status));
    if (active && !pollTimer) pollTimer = setInterval(() => refresh().catch((err) => flash(err.message, true)), 4000);
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
    if (!path) {
      flash("Enter a dataset path to preview", true);
      return;
    }
    btnPreview.disabled = true;
    try {
      const data = await api("/v1/datasets/preview", {
        method: "POST",
        body: JSON.stringify({
          gcs_path: path,
          format: document.getElementById("sel-ds-format").value,
          limit: 5,
        }),
      });
      renderPreview(data);
      flash(`Preview ready — ${data.num_rows ?? "?"} rows.`);
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
    if (!state.previewOk || state.previewKey !== previewFingerprint()) {
      flash("Preview the dataset first, then register", true);
      return;
    }
    btnRegister.disabled = true;
    try {
      const ds = await api("/v1/datasets/register", {
        method: "POST",
        body: JSON.stringify({
          gcs_path: fd.get("gcs_path"),
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
      flash("QLoRA needs CUDA and bitsandbytes on this gateway. Use technique lora for a CPU demo, or point FTAAS_TRAIN_MODE=remote at a GPU worker.", true);
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
      if (logs) {
        await showLogs(logs.dataset.logs);
      }
    } catch (err) {
      flash(err.message, true);
    }
  });

  document.getElementById("form-deploy").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const fd = new FormData(ev.target);
    if (!fd.get("model_name")) {
      flash("No model to deploy yet", true);
      return;
    }
    try {
      const ep = await api("/v1/endpoints", {
        method: "POST",
        body: JSON.stringify({
          model_name: fd.get("model_name"),
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
    listDatasets.innerHTML = `<li class="muted">Gateway offline — start it with ./scripts/start_all.sh</li>`;
    jobsBody.innerHTML = `<tr><td colspan="6" class="muted">Gateway offline</td></tr>`;
  });
})();
