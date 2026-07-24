/**
 * FTAAS Console — wires UI forms to gateway /v1 APIs.
 * Flow: register → create job → poll → deploy → prompt
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

  let pollTimer = null;
  let state = {
    catalog: boot.catalog || {},
    datasets: boot.datasets || [],
    jobs: boot.jobs || [],
    models: boot.models || [],
    endpoints: boot.endpoints || [],
  };

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
    jobsBody.innerHTML = jobs
      .map((j) => {
        const deployBtn = j.registered_model_name
          ? `<button type="button" class="btn-small" data-deploy="${j.registered_model_name}">Deploy</button>`
          : "";
        return `<tr data-job="${j.job_id}">
          <td><code>${j.job_id}</code></td>
          <td>${j.model_name}</td>
          <td>${j.framework} / ${j.technique}</td>
          <td><span class="pill ${j.status}">${j.status}</span></td>
          <td><code>${JSON.stringify(j.metrics || {})}</code></td>
          <td><code>${j.pipeline_id || "—"}</code></td>
          <td>${deployBtn}</td>
        </tr>`;
      })
      .join("");
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
    if (busy && !pollTimer) {
      pollTimer = setInterval(() => {
        refreshJobs().catch((e) => flash(e.message, true));
      }, 2000);
    }
    if (!busy && pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
      // one more pull for models after success
      refreshAll().catch(() => {});
    }
  }

  function setBusy(form, busy) {
    const btn = form.querySelector('button[type="submit"]');
    if (!btn) return;
    btn.disabled = busy;
    btn.dataset.label = btn.dataset.label || btn.textContent;
    btn.textContent = busy ? "Working…" : btn.dataset.label;
  }

  document.getElementById("form-register").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const form = ev.target;
    const fd = new FormData(form);
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
      flash(`Registered ${ds.dataset_id}:${ds.version}`);
      await refreshAll();
      datasetSel.value = ds.dataset_id;
      syncDatasetVersion();
    } catch (e) {
      flash(e.message, true);
    } finally {
      setBusy(form, false);
    }
  });

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
            max_steps: Number(fd.get("max_steps") || 10),
            learning_rate: Number(fd.get("learning_rate") || 2e-4),
            lora_r: loraR,
            lora_alpha: loraR * 2,
          },
        }),
      });
      flash(`Job ${job.job_id} queued — training…`);
      await refreshJobs();
      maybePoll();
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
          max_tokens: 128,
          temperature: 0.7,
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
    const btn = ev.target.closest("[data-deploy]");
    if (!btn) return;
    modelSel.value = btn.dataset.deploy;
    document.getElementById("form-deploy").scrollIntoView({ behavior: "smooth", block: "center" });
    flash(`Selected model ${btn.dataset.deploy} for deploy`);
  });

  // init
  techniqueOptions(state.catalog);
  syncDatasetVersion();
  maybePoll();
  // soft refresh to stay in sync
  refreshAll().catch((e) => flash(e.message, true));
})();
