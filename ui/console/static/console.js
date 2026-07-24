/**
 * FTAAS Console — wires UI forms to gateway /v1 APIs.
 * Flow: preview → register → create job → poll → deploy → prompt
 */
(() => {
  const TERMINAL = new Set(["succeeded", "failed", "cancelled"]);
  const boot = JSON.parse(document.getElementById("boot").textContent || "{}");

  const flashEl = document.getElementById("flash");
  const techSel = document.getElementById("sel-technique");
  const datasetSel = document.getElementById("sel-dataset");
  const versionInp = document.getElementById("inp-dataset-version");
  const modelSel = document.getElementById("sel-model");
  const endpointSel = document.getElementById("sel-endpoint");
  const jobsBody = document.getElementById("tbody-jobs");
  const listDatasets = document.getElementById("list-datasets");
  const listModels = document.getElementById("list-models");
  const listEndpoints = document.getElementById("list-endpoints");
  const promptResult = document.getElementById("prompt-result");
  const promptOut = document.getElementById("prompt-out");
  const btnPreviewDs = document.getElementById("btn-preview-ds");
  const btnRegisterDs = document.getElementById("btn-register-ds");
  const previewBox = document.getElementById("dataset-preview");
  const previewMeta = document.getElementById("dataset-preview-meta");
  const previewWarn = document.getElementById("dataset-preview-warn");
  const previewTable = document.getElementById("dataset-preview-table");
  const inpGcsPath = document.getElementById("inp-gcs-path");
  const selDsFormat = document.getElementById("sel-ds-format");

  const jobDetailTitle = document.getElementById("job-detail-title");
  const jobDetailMeta = document.getElementById("job-detail-meta");
  const jobProgressWrap = document.getElementById("job-progress-wrap");
  const jobProgressBar = document.getElementById("job-progress-bar");
  const jobLogView = document.getElementById("job-log-view");
  const btnStopJob = document.getElementById("btn-stop-job");
  const logModal = document.getElementById("log-modal");
  const logModalTitle = document.getElementById("log-modal-title");
  const logModalMeta = document.getElementById("log-modal-meta");
  const logModalBody = document.getElementById("log-modal-body");
  const logModalProgressWrap = document.getElementById("log-modal-progress-wrap");
  const logModalProgressBar = document.getElementById("log-modal-progress-bar");

  let pollTimer = null;
  let state = {
    catalog: boot.catalog || {},
    datasets: boot.datasets || [],
    jobs: boot.jobs || [],
    models: boot.models || [],
    endpoints: boot.endpoints || [],
    selectedJobId: null,
    previewKey: null,
    previewOk: false,
  };

  function esc(s) {
    return String(s ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function progressOf(j) {
    const p = j.progress || {};
    const percent = Math.max(0, Math.min(100, Number(p.percent) || 0));
    const message = p.message || j.error || (TERMINAL.has(j.status) ? j.status : "…");
    return { percent, message, phase: p.phase || "", step: p.step, max_steps: p.max_steps };
  }

  function flash(msg, isErr = false) {
    if (!flashEl) return;
    flashEl.hidden = !msg;
    flashEl.textContent = msg || "";
    flashEl.classList.toggle("err", !!isErr);
    if (msg && !isErr) {
      clearTimeout(flash._t);
      flash._t = setTimeout(() => {
        flashEl.hidden = true;
      }, 4000);
    }
  }

  async function api(path, opts = {}) {
    const res = await fetch(path, {
      headers: { Accept: "application/json", ...(opts.body ? { "Content-Type": "application/json" } : {}) },
      ...opts,
    });
    let data = null;
    const text = await res.text();
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

  function techniqueOptions(catalog) {
    const techs = catalog?.techniques || {};
    const groups = [
      ["PEFT", techs.peft || ["lora", "qlora", "dora"]],
      ["Full", techs.full_finetune || ["16bit_full", "frozen"]],
      ["Instruction", techs.instruction_tuning || ["sft"]],
      ["Alignment", techs.alignment || ["dpo", "ppo", "orpo"]],
    ];
    const preferred = boot.defaults?.technique || "lora";
    techSel.innerHTML = groups
      .map(
        ([label, items]) =>
          `<optgroup label="${label}">${items
            .map((t) => `<option value="${t}" ${t === preferred ? "selected" : ""}>${t}</option>`)
            .join("")}</optgroup>`
      )
      .join("");
  }

  function renderDatasets(datasets) {
    state.datasets = datasets;
    if (!datasets.length) {
      listDatasets.innerHTML = `<li class="muted">No datasets yet</li>`;
      datasetSel.innerHTML = `<option value="">Register a dataset first</option>`;
      versionInp.value = "1";
      return;
    }
    listDatasets.innerHTML = datasets
      .map(
        (d) =>
          `<li data-id="${d.dataset_id}" data-version="${d.version}"><code>${d.dataset_id}:${d.version}</code> — ${d.name || ""} (${d.num_rows ?? "?"} rows)</li>`
      )
      .join("");
    const prev = datasetSel.value;
    datasetSel.innerHTML = datasets
      .map(
        (d) =>
          `<option value="${d.dataset_id}" data-version="${d.version}" ${d.dataset_id === prev ? "selected" : ""}>${d.name || d.dataset_id} (${d.dataset_id}:${d.version})</option>`
      )
      .join("");
    syncDatasetVersion();
  }

  function syncDatasetVersion() {
    const opt = datasetSel.selectedOptions[0];
    if (opt) versionInp.value = opt.dataset.version || "1";
  }

  function renderJobs(jobs) {
    state.jobs = jobs;
    if (!jobs.length) {
      jobsBody.innerHTML = `<tr class="empty"><td colspan="7" class="muted">No jobs yet</td></tr>`;
      return;
    }
    if (state.selectedJobId && !jobs.some((j) => j.job_id === state.selectedJobId)) {
      state.selectedJobId = jobs[0].job_id;
    }
    jobsBody.innerHTML = jobs
      .map((j) => {
        const deployBtn = j.registered_model_name
          ? `<button type="button" class="btn-small" data-deploy="${esc(j.registered_model_name)}">Deploy</button>`
          : "";
        const stopBtn = !TERMINAL.has(j.status)
          ? `<button type="button" class="btn-danger btn-stop-inline" data-stop="${esc(j.job_id)}">Stop</button>`
          : "";
        const logsBtn = `<button type="button" class="btn-small" data-logs="${esc(j.job_id)}">View log</button>`;
        const prog = progressOf(j);
        const active = j.job_id === state.selectedJobId ? " active" : "";
        return `<tr data-job="${esc(j.job_id)}" class="job-row${active}">
          <td><code>${esc(j.job_id)}</code></td>
          <td>${esc(j.model_name)}</td>
          <td>${esc(j.framework)} / ${esc(j.technique)}</td>
          <td><span class="pill ${esc(j.status)}">${esc(j.status)}</span></td>
          <td class="progress-cell">
            <div class="progress"><div class="bar" style="width:${prog.percent}%"></div></div>
            <small title="${esc(prog.message)}">${esc(prog.message)}</small>
          </td>
          <td><code>${esc(j.pipeline_id || "—")}</code></td>
          <td class="actions-cell">${logsBtn}${stopBtn}${deployBtn}</td>
        </tr>`;
      })
      .join("");
    if (state.selectedJobId) {
      refreshSelectedLogs({ openModal: false }).catch(() => {});
    }
  }

  function formatLogs(logs) {
    if (!logs || !logs.length) return "(no log lines yet)";
    return logs
      .map((line) => {
        const ts = line.ts ? String(line.ts).replace("T", " ").slice(0, 19) : "";
        return `${ts}  ${line.message || JSON.stringify(line)}`;
      })
      .join("\n");
  }

  function paintJobDetail(payload, { openModal = false } = {}) {
    if (!jobLogView) return;
    const prog = payload.progress || {};
    const percent = Math.max(0, Math.min(100, Number(prog.percent) || 0));
    const title = payload.job_id || "Job";
    const bits = [payload.status || ""];
    if (prog.phase) bits.push(prog.phase);
    if (prog.step != null && prog.max_steps != null) bits.push(`step ${prog.step}/${prog.max_steps}`);
    if (prog.loss != null) bits.push(`loss ${Number(prog.loss).toFixed(4)}`);
    const meta = bits.filter(Boolean).join(" · ");

    jobDetailTitle.textContent = title;
    jobDetailMeta.textContent = meta;
    if (btnStopJob) {
      const busy = payload.status && !TERMINAL.has(payload.status);
      btnStopJob.hidden = !busy;
      btnStopJob.disabled = false;
      btnStopJob.textContent = "Stop training";
    }
    jobProgressWrap.hidden = false;
    jobProgressBar.style.width = `${percent}%`;
    let text = formatLogs(payload.logs);
    if (payload.error) text += `\n\nERROR: ${payload.error}`;
    if (payload.metrics && Object.keys(payload.metrics).length) {
      text += `\n\nmetrics: ${JSON.stringify(payload.metrics)}`;
    }
    const atBottom =
      jobLogView.scrollHeight - jobLogView.scrollTop - jobLogView.clientHeight < 40;
    jobLogView.textContent = text;
    if (atBottom) jobLogView.scrollTop = jobLogView.scrollHeight;

    document.getElementById("job-detail")?.classList.add("job-detail-active");

    if (logModalTitle) logModalTitle.textContent = `Logs · ${title}`;
    if (logModalMeta) logModalMeta.textContent = meta || "—";
    if (logModalBody) {
      logModalBody.textContent = text;
      logModalBody.scrollTop = logModalBody.scrollHeight;
    }
    if (logModalProgressWrap && logModalProgressBar) {
      logModalProgressWrap.hidden = false;
      logModalProgressBar.style.width = `${percent}%`;
    }
    if (openModal) openLogModal();
  }

  function openLogModal() {
    if (!logModal) return;
    logModal.hidden = false;
    document.body.classList.add("modal-open");
  }

  function closeLogModal() {
    if (!logModal) return;
    logModal.hidden = true;
    document.body.classList.remove("modal-open");
  }

  async function refreshSelectedLogs(opts = {}) {
    if (!state.selectedJobId) return;
    const data = await api(`/v1/jobs/${encodeURIComponent(state.selectedJobId)}/logs`);
    paintJobDetail(data, opts);
  }

  function selectJob(jobId, { scroll = false, openModal = false } = {}) {
    state.selectedJobId = jobId;
    jobsBody.querySelectorAll("tr.job-row").forEach((tr) => {
      tr.classList.toggle("active", tr.dataset.job === jobId);
    });
    refreshSelectedLogs({ openModal })
      .then(() => {
        if (scroll && !openModal) {
          document.getElementById("job-detail")?.scrollIntoView({ behavior: "smooth", block: "center" });
        }
        if (openModal) flash(`Logs for ${jobId}`);
      })
      .catch((e) => flash(e.message, true));
  }

  async function stopJob(jobId) {
    if (!jobId) return;
    if (!window.confirm(`Stop training for ${jobId}? Current step will finish, then the job cancels.`)) {
      return;
    }
    if (btnStopJob) {
      btnStopJob.disabled = true;
      btnStopJob.textContent = "Stopping…";
    }
    try {
      await api(`/v1/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST" });
      flash(`Stop requested for ${jobId}`);
      await refreshJobs();
      if (state.selectedJobId === jobId) await refreshSelectedLogs();
    } catch (e) {
      flash(e.message, true);
      if (btnStopJob) {
        btnStopJob.disabled = false;
        btnStopJob.textContent = "Stop training";
      }
    }
  }

  function renderModels(models) {
    state.models = models;
    if (!models.length) {
      listModels.innerHTML = `<li class="muted">No models registered</li>`;
      modelSel.innerHTML = `<option value="">Train a job first</option>`;
      return;
    }
    listModels.innerHTML = models
      .map((m) => `<li><code>${m.model_name}</code> v${m.version} — run ${m.run_id}</li>`)
      .join("");
    const prev = modelSel.value;
    modelSel.innerHTML = models
      .map(
        (m) =>
          `<option value="${m.model_name}" ${m.model_name === prev ? "selected" : ""}>${m.model_name} v${m.version}</option>`
      )
      .join("");
  }

  function renderEndpoints(endpoints) {
    state.endpoints = endpoints;
    if (!endpoints.length) {
      listEndpoints.innerHTML = `<li class="muted">No endpoints</li>`;
      endpointSel.innerHTML = `<option value="">Deploy a model first</option>`;
      return;
    }
    listEndpoints.innerHTML = endpoints
      .map(
        (e) =>
          `<li><code>${e.endpoint_id}</code> — ${e.model_name}@${e.inference_framework}</li>`
      )
      .join("");
    const prev = endpointSel.value;
    endpointSel.innerHTML = endpoints
      .map(
        (e) =>
          `<option value="${e.endpoint_id}" ${e.endpoint_id === prev ? "selected" : ""}>${e.endpoint_id} — ${e.model_name}</option>`
      )
      .join("");
  }

  async function refreshAll() {
    const [catalog, datasets, jobs, models, endpoints] = await Promise.all([
      api("/v1/catalog"),
      api("/v1/datasets"),
      api("/v1/jobs"),
      api("/v1/models"),
      api("/v1/endpoints"),
    ]);
    state.catalog = catalog;
    techniqueOptions(catalog);
    renderDatasets(datasets);
    renderJobs(jobs);
    renderModels(models);
    renderEndpoints(endpoints);
    maybePoll();
  }

  async function refreshJobs() {
    const jobs = await api("/v1/jobs");
    renderJobs(jobs);
    const models = await api("/v1/models");
    renderModels(models);
    maybePoll();
  }

  function maybePoll() {
    const busy = (state.jobs || []).some((j) => !TERMINAL.has(j.status));
    const selectedBusy =
      state.selectedJobId &&
      (state.jobs || []).some((j) => j.job_id === state.selectedJobId && !TERMINAL.has(j.status));
    if ((busy || selectedBusy) && !pollTimer) {
      pollTimer = setInterval(() => {
        refreshJobs().catch((e) => flash(e.message, true));
      }, 1500);
    }
    if (!busy && !selectedBusy && pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
      refreshAll().catch(() => {});
    }
  }

  function previewFingerprint() {
    return `${(inpGcsPath?.value || "").trim()}|${selDsFormat?.value || "jsonl"}`;
  }

  function invalidatePreview() {
    state.previewOk = false;
    state.previewKey = null;
    if (btnRegisterDs) {
      btnRegisterDs.disabled = true;
      btnRegisterDs.title = "Preview the file first";
    }
  }

  function cellText(v) {
    if (v == null) return "";
    if (typeof v === "object") return JSON.stringify(v);
    return String(v);
  }

  function renderDatasetPreview(data) {
    if (!previewBox) return;
    previewBox.hidden = false;
    const rows = data.num_rows ?? "?";
    const cols = (data.columns || []).join(", ") || "—";
    previewMeta.innerHTML =
      `<strong>${esc(rows)}</strong> rows · columns: <code>${esc(cols)}</code>` +
      `<br/><span class="muted">resolved: <code>${esc(data.resolved_path || "")}</code></span>`;
    if (data.warnings && data.warnings.length) {
      previewWarn.hidden = false;
      previewWarn.textContent = data.warnings.join(" ");
    } else {
      previewWarn.hidden = true;
      previewWarn.textContent = "";
    }
    const columns = data.columns?.length
      ? data.columns
      : Object.keys((data.samples && data.samples[0]) || {});
    const thead = previewTable.querySelector("thead");
    const tbody = previewTable.querySelector("tbody");
    thead.innerHTML = `<tr>${columns.map((c) => `<th>${esc(c)}</th>`).join("")}</tr>`;
    const samples = data.samples || [];
    tbody.innerHTML = samples.length
      ? samples
          .map(
            (row) =>
              `<tr>${columns
                .map((c) => {
                  const t = cellText(row[c]);
                  return `<td title="${esc(t)}">${esc(t)}</td>`;
                })
                .join("")}</tr>`
          )
          .join("")
      : `<tr><td colspan="${Math.max(columns.length, 1)}" class="muted">No sample rows</td></tr>`;
    state.previewOk = true;
    state.previewKey = previewFingerprint();
    if (btnRegisterDs) {
      btnRegisterDs.disabled = false;
      btnRegisterDs.title = "Register this dataset";
    }
  }

  async function runDatasetPreview() {
    const path = (inpGcsPath?.value || "").trim();
    if (!path) {
      flash("Enter a dataset path to preview", true);
      return;
    }
    if (btnPreviewDs) {
      btnPreviewDs.disabled = true;
      btnPreviewDs.textContent = "Previewing…";
    }
    try {
      const data = await api("/v1/datasets/preview", {
        method: "POST",
        body: JSON.stringify({
          gcs_path: path,
          format: selDsFormat?.value || "jsonl",
          limit: 5,
        }),
      });
      renderDatasetPreview(data);
      flash(`Preview ready — ${data.num_rows ?? "?"} rows. Review then register.`);
    } catch (e) {
      invalidatePreview();
      if (previewBox) previewBox.hidden = true;
      flash(e.message, true);
    } finally {
      if (btnPreviewDs) {
        btnPreviewDs.disabled = false;
        btnPreviewDs.textContent = "Preview data";
      }
    }
  }

  function setBusy(form, busy) {
    const btn = form.querySelector('button[type="submit"]');
    if (!btn) return;
    btn.disabled = busy || (btn === btnRegisterDs && !state.previewOk);
    btn.dataset.label = btn.dataset.label || btn.textContent;
    btn.textContent = busy ? "Working…" : btn.dataset.label;
  }

  document.getElementById("form-register").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const form = ev.target;
    const fd = new FormData(form);
    if (!state.previewOk || state.previewKey !== previewFingerprint()) {
      flash("Preview the dataset first, then register", true);
      invalidatePreview();
      return;
    }
    setBusy(form, true);
    try {
      const ds = await api("/v1/datasets/register", {
        method: "POST",
        body: JSON.stringify({
          gcs_path: fd.get("gcs_path"),
          name: fd.get("name") || null,
          format: fd.get("format") || "jsonl",
        }),
      });
      flash(`Registered ${ds.dataset_id}:${ds.version} (${ds.num_rows ?? "?"} rows)`);
      invalidatePreview();
      if (previewBox) previewBox.hidden = true;
      await refreshAll();
      datasetSel.value = ds.dataset_id;
      syncDatasetVersion();
    } catch (e) {
      flash(e.message, true);
    } finally {
      setBusy(form, false);
    }
  });

  btnPreviewDs?.addEventListener("click", () => {
    runDatasetPreview().catch((e) => flash(e.message, true));
  });
  inpGcsPath?.addEventListener("input", invalidatePreview);
  selDsFormat?.addEventListener("change", invalidatePreview);
  document.getElementById("form-job").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const form = ev.target;
    const fd = new FormData(form);
    if (!fd.get("dataset_id")) {
      flash("Register a dataset first", true);
      return;
    }
    setBusy(form, true);
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
            max_steps: Number(fd.get("max_steps") || 12),
            learning_rate: Number(fd.get("learning_rate") || 1e-3),
            lora_r: loraR,
            lora_alpha: loraR * 2,
            per_device_train_batch_size: 1,
            max_seq_length: 64,
          },
        }),
      });
      flash(`Job ${job.job_id} queued — watch progress below`);
      state.selectedJobId = job.job_id;
      await refreshJobs();
      selectJob(job.job_id);
      maybePoll();
      document.getElementById("job-detail")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    } catch (e) {
      flash(e.message, true);
    } finally {
      setBusy(form, false);
    }
  });

  document.getElementById("form-deploy").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const form = ev.target;
    const fd = new FormData(form);
    if (!fd.get("model_name")) {
      flash("No model to deploy yet — wait for a succeeded job", true);
      return;
    }
    setBusy(form, true);
    try {
      const ep = await api("/v1/endpoints", {
        method: "POST",
        body: JSON.stringify({
          model_name: fd.get("model_name"),
          inference_framework: fd.get("inference_framework"),
          use_adapters: true,
        }),
      });
      flash(`Endpoint ${ep.endpoint_id} ready`);
      const endpoints = await api("/v1/endpoints");
      renderEndpoints(endpoints);
      endpointSel.value = ep.endpoint_id;
    } catch (e) {
      flash(e.message, true);
    } finally {
      setBusy(form, false);
    }
  });

  document.getElementById("form-prompt").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const form = ev.target;
    const fd = new FormData(form);
    const endpointId = fd.get("endpoint_id");
    if (!endpointId) {
      flash("Deploy an endpoint first", true);
      return;
    }
    setBusy(form, true);
    try {
      const resp = await api(`/v1/endpoints/${endpointId}/prompt`, {
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
    } catch (e) {
      flash(e.message, true);
    } finally {
      setBusy(form, false);
    }
  });

  datasetSel.addEventListener("change", syncDatasetVersion);

  document.getElementById("btn-refresh-jobs").addEventListener("click", () => {
    refreshAll()
      .then(() => flash("Refreshed"))
      .catch((e) => flash(e.message, true));
  });

  jobsBody.addEventListener("click", (ev) => {
    const stop = ev.target.closest("[data-stop]");
    if (stop) {
      ev.preventDefault();
      ev.stopPropagation();
      stopJob(stop.dataset.stop).catch((e) => flash(e.message, true));
      return;
    }
    const logsBtn = ev.target.closest("[data-logs]");
    if (logsBtn) {
      ev.preventDefault();
      ev.stopPropagation();
      selectJob(logsBtn.dataset.logs, { openModal: true });
      return;
    }
    const btn = ev.target.closest("[data-deploy]");
    if (btn) {
      modelSel.value = btn.dataset.deploy;
      document.getElementById("form-deploy").scrollIntoView({ behavior: "smooth", block: "center" });
      flash(`Selected model ${btn.dataset.deploy} for deploy`);
      return;
    }
    const row = ev.target.closest("tr.job-row[data-job]");
    if (row) selectJob(row.dataset.job, { scroll: true });
  });

  btnStopJob?.addEventListener("click", () => {
    if (state.selectedJobId) stopJob(state.selectedJobId).catch((e) => flash(e.message, true));
  });

  document.querySelectorAll("[data-close-log-modal]").forEach((el) => {
    el.addEventListener("click", () => closeLogModal());
  });
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && logModal && !logModal.hidden) closeLogModal();
  });

  // init
  techniqueOptions(state.catalog);
  syncDatasetVersion();
  if (state.jobs.length) {
    const busy = state.jobs.find((j) => !TERMINAL.has(j.status));
    selectJob((busy || state.jobs[0]).job_id);
  }
  maybePoll();
  refreshAll().catch((e) => flash(e.message, true));
})();
