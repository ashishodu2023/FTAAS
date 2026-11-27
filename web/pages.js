import { escapeHtml, layout } from "./layout.js";

function page(current, title, main, pageClass = "") {
  return layout({ title, current, main, pageClass });
}

const HERO_VISUAL = `
<div class="hero-visual" role="img" aria-label="Diagram of the FTAAS loop from dataset to prompt">
  <div class="flow-board">
    <div class="flow-line" aria-hidden="true"></div>
    <div class="flow-nodes">
      <div class="flow-node tone-a"><span>Data</span></div>
      <div class="flow-node tone-b"><span>Train</span></div>
      <div class="flow-node tone-c"><span>Track</span></div>
      <div class="flow-node tone-d"><span>Prompt</span></div>
    </div>
    <p class="flow-caption">One shared gateway · real LoRA / QLoRA</p>
  </div>
</div>`;

export function homePage() {
  return page(
    "/",
    "Fine Tuning as a Service",
    `
    <section class="hero-stage">
      <div class="hero-copy">
        <p class="brand-lockup">FTAAS</p>
        <h1>Fine-tune open models on one clear loop</h1>
        <p class="lede">Preview a dataset, run real LoRA or QLoRA, then prompt the adapter — for a solo operator today, or a small team sharing the same gateway.</p>
        <div class="actions">
          <a class="btn" href="/workbench">Open workbench</a>
          <a class="btn secondary" href="/pipeline">See how it works</a>
        </div>
      </div>
      <div class="hero-media" aria-hidden="false">
        ${HERO_VISUAL}
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <h2>The loop</h2>
        <p>Four steps. Same APIs the Python SDK and operator console already use.</p>
      </div>
      <ol class="loop">
        <li>
          <span class="step-num">01</span>
          <div>
            <h3>Preview &amp; register</h3>
            <p>Inspect a jsonl, json, or csv path, then register it. Sample: <code>examples/data/alpaca_sample.jsonl</code>.</p>
          </div>
        </li>
        <li>
          <span class="step-num">02</span>
          <div>
            <h3>Train</h3>
            <p>Create a job. Default <code>distilgpt2</code> + LoRA for CPU demos. QLoRA when CUDA and bitsandbytes are ready.</p>
          </div>
        </li>
        <li>
          <span class="step-num">03</span>
          <div>
            <h3>Track</h3>
            <p>Watch status, progress, and logs. Metrics land on the job and in MLflow.</p>
          </div>
        </li>
        <li>
          <span class="step-num">04</span>
          <div>
            <h3>Prompt</h3>
            <p>Deploy an endpoint and ask. Serving today is local Hugging Face / PEFT generate.</p>
          </div>
        </li>
      </ol>
    </section>

    <section class="section">
      <div class="section-head">
        <h2>What is live</h2>
        <p>Grounded in the gateway you already run — not a mock.</p>
      </div>
      <div class="status-list">
        <div class="status-row"><span>Dataset preview &amp; register</span><span class="badge live">Live</span></div>
        <div class="status-row"><span>Job logs, progress, cancel</span><span class="badge live">Live</span></div>
        <div class="status-row"><span>transformers + PEFT LoRA / DoRA · TRL SFT / DPO</span><span class="badge live">Live</span></div>
        <div class="status-row"><span>QLoRA on CUDA · remote trainer</span><span class="badge live">Live</span></div>
        <div class="status-row"><span>Prompt via local HF / PEFT</span><span class="badge live">Live</span></div>
        <div class="status-row"><span>verl · llama-factory · unsloth · axolotl</span><span class="badge soft">Alias → PEFT</span></div>
        <div class="status-row"><span>vLLM · Ray Serve · real gs:// fetch</span><span class="badge next">Planned</span></div>
      </div>
    </section>
    `,
    "page-home",
  );
}

export function pipelinePage() {
  return page(
    "/pipeline",
    "How it works",
    `
    <section class="page-intro">
      <p class="eyebrow">Architecture</p>
      <h1>One gateway, five modules</h1>
      <p class="lede">By default every step below is the same FastAPI process on port 8080. The browser workbench and <code>ftaas.Client</code> both call it.</p>
    </section>

    <ol class="pipeline">
      <li>
        <h2>Preview / register</h2>
        <p><code>services/registry</code> · <code>POST /v1/datasets/preview</code>, <code>POST /v1/datasets/register</code>. A <code>gs://</code> path is only a label until the file is on disk under the local mirror.</p>
      </li>
      <li>
        <h2>Create job</h2>
        <p><code>services/control</code> · <code>POST /v1/jobs/finetune</code>. Catalog, logs, cancel, and the model list live here too.</p>
      </li>
      <li>
        <h2>Orchestrate</h2>
        <p><code>services/workflow</code> plus <code>runner/local</code>. Airflow is optional and off in settings.</p>
      </li>
      <li>
        <h2>Train</h2>
        <p><code>training/frameworks</code> — real torch updates. Or send the job to <code>trainer_worker</code> on port 8090 with <code>FTAAS_TRAIN_MODE=remote</code>.</p>
      </li>
      <li>
        <h2>Deploy &amp; prompt</h2>
        <p><code>services/deploy</code> · local HF / PEFT generate. vLLM and Ray Serve are listed and not wired up.</p>
      </li>
    </ol>

    <section class="section code-section">
      <div class="section-head">
        <h2>Same calls from Python</h2>
        <p>Install with <code>pip install ftaas</code> once 0.2.0 is on PyPI, or from this repo.</p>
      </div>
      <pre class="code">from ftaas import Client, Framework, Technique, HyperParameters

with Client() as c:
    ds = c.register_dataset("examples/data/alpaca_sample.jsonl", name="alpaca")
    job = c.create_finetune_job(
        model_name="distilgpt2",
        dataset=ds,
        framework=Framework.TRANSFORMERS,
        technique=Technique.LORA,
        parameters=HyperParameters(max_steps=12, max_seq_length=64),
    )
    job = c.wait_for_job(job.job_id)
    ep = c.create_endpoint(job.registered_model_name, inference_framework="transformers")
    print(c.prompt(ep.endpoint_id, "What is LoRA?").completion)</pre>
    </section>
    `,
  );
}

function techniqueGroups(catalog) {
  const groups = catalog.techniques || {};
  return Object.entries(groups)
    .map(([name, items]) => {
      const chips = (items || []).map((item) => `<span class="chip">${escapeHtml(item)}</span>`).join("");
      return `<div class="tech-group"><h3>${escapeHtml(name.replaceAll("_", " "))}</h3><div class="chips">${chips}</div></div>`;
    })
    .join("");
}

export function catalogPage({ catalog, evalStack, gatewayUp }) {
  const backends = catalog.backends || {};
  const frameworkRows = (catalog.frameworks || [])
    .map((name) => {
      const note = backends[name] || "listed in configs/settings.yaml";
      const live = String(note).startsWith("live");
      return `<tr>
        <td><code>${escapeHtml(name)}</code></td>
        <td>${escapeHtml(note)}</td>
        <td><span class="badge ${live ? "live" : "soft"}">${live ? "Live" : "Alias"}</span></td>
      </tr>`;
    })
    .join("");

  const inferenceRows = Object.entries(catalog.inference || {})
    .map(([name, note]) => {
      const live = String(note).startsWith("live");
      return `<tr>
        <td><code>${escapeHtml(name)}</code></td>
        <td>${escapeHtml(note)}</td>
        <td><span class="badge ${live ? "live" : "next"}">${live ? "Live" : "Planned"}</span></td>
      </tr>`;
    })
    .join("");

  const phases = (catalog.phases || [])
    .map((phase) => {
      const available = phase.status === "available";
      return `<li>
        <span class="phase-num">Phase ${escapeHtml(phase.phase)}</span>
        <span class="phase-name">${escapeHtml(phase.name)}</span>
        <span class="badge ${available ? "live" : "next"}">${escapeHtml(phase.status)}</span>
      </li>`;
    })
    .join("");

  const training = catalog.training || {};
  const qlora = training.qlora || {};
  const device = training.device || {};
  const source =
    catalog.source === "gateway"
      ? "Loaded live from <code>GET /v1/catalog</code>."
      : "Gateway offline — showing the static lists from settings, plus the backend labels the catalog route returns.";

  const deviceBlock =
    catalog.source === "gateway"
      ? `<p class="callout">Training plane: <strong>${escapeHtml(training.mode || "local")}</strong> · CUDA ${device.cuda_available ? "yes" : "no"} · QLoRA ${qlora.available ? "ready" : "needs GPU + bitsandbytes"}.</p>`
      : `<p class="callout">Start the gateway with <code>./scripts/start_all.sh</code> to refresh this page with live device info.</p>`;

  const evalNote = evalStack
    ? `<p class="hint">Serving stack: live ${(evalStack.live || []).map(escapeHtml).join(", ") || "—"} · planned ${(evalStack.planned || []).map(escapeHtml).join(", ") || "—"}.</p>`
    : "";

  return page(
    "/catalog",
    "Catalog",
    `
    <section class="page-intro">
      <p class="eyebrow">Capabilities</p>
      <h1>Catalog</h1>
      <p class="lede">${source}</p>
    </section>
    ${deviceBlock}
    <section class="section">
      <div class="section-head">
        <h2>Frameworks</h2>
        <p>What the name in the dropdown actually runs.</p>
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Name</th><th>Backend</th><th></th></tr></thead>
          <tbody>${frameworkRows}</tbody>
        </table>
      </div>
    </section>
    <section class="section">
      <div class="section-head">
        <h2>Techniques</h2>
        <p>From <code>configs/settings.yaml</code> and the live catalog.</p>
      </div>
      <div class="tech-grid">${techniqueGroups(catalog)}</div>
    </section>
    <div class="two-col">
      <section class="section">
        <div class="section-head">
          <h2>Inference</h2>
        </div>
        <div class="table-wrap">
          <table>
            <thead><tr><th>Name</th><th>Status</th><th></th></tr></thead>
            <tbody>${inferenceRows}</tbody>
          </table>
        </div>
        ${evalNote}
      </section>
      <section class="section">
        <div class="section-head">
          <h2>Phases</h2>
        </div>
        <ul class="phase-list">${phases}</ul>
      </section>
    </div>
    `,
  );
}

export function workbenchPage(boot) {
  const bootJson = JSON.stringify(boot).replace(/</g, "\\u003c");
  return page(
    "/workbench",
    "Workbench",
    `
    <section class="page-intro">
      <p class="eyebrow">Run</p>
      <h1>Workbench</h1>
      <p class="lede">Preview → register → train → deploy → prompt. Requests go through this site to the Python gateway. Creating a job starts a real training run.</p>
    </section>
    <div id="flash" class="banner" hidden></div>

    <div class="workbench">
      <aside class="workbench-nav" aria-label="Workbench steps">
        <a href="#step-data">1 · Dataset</a>
        <a href="#step-job">2 · Job</a>
        <a href="#step-jobs">3 · Monitor</a>
        <a href="#step-deploy">4 · Deploy</a>
        <a href="#step-prompt">5 · Prompt</a>
      </aside>

      <div class="workbench-main">
        <section class="panel" id="step-data">
          <div class="panel-head">
            <h2>1. Dataset</h2>
            <p class="hint">Path must be readable by the gateway process.</p>
          </div>
          <form id="form-register" class="form">
            <label>Path on the gateway
              <input name="gcs_path" id="inp-gcs-path" required value="examples/data/alpaca_sample.jsonl" autocomplete="off" />
            </label>
            <div class="form-row">
              <label>Name <input name="name" value="alpaca-sample" /></label>
              <label>Format
                <select name="format" id="sel-ds-format">
                  <option value="jsonl" selected>jsonl</option>
                  <option value="json">json</option>
                  <option value="csv">csv</option>
                </select>
              </label>
            </div>
            <div class="btn-row">
              <button type="button" class="btn secondary" id="btn-preview">Preview</button>
              <button type="submit" class="btn" id="btn-register" disabled>Register</button>
            </div>
          </form>
          <div id="dataset-preview" class="preview-box" hidden></div>
          <h3 class="subhead">Registered datasets</h3>
          <ul class="list" id="list-datasets"><li class="muted">Waiting for the gateway…</li></ul>
        </section>

        <section class="panel" id="step-job">
          <div class="panel-head">
            <h2>2. Fine-tune job</h2>
            <p class="hint">QLoRA is blocked when the catalog says this machine cannot run it.</p>
          </div>
          <form id="form-job" class="form">
            <label>Model <input name="model_name" id="inp-model" required value="distilgpt2" /></label>
            <label>Framework <select name="framework" id="sel-framework"></select></label>
            <p class="hint" id="framework-note"></p>
            <label>Technique <select name="technique" id="sel-technique"></select></label>
            <p class="hint" id="train-plane-hint"></p>
            <div class="form-row">
              <label>Dataset <select name="dataset_id" id="sel-dataset" required></select></label>
              <label>Version <input name="dataset_version" id="inp-dataset-version" value="1" /></label>
            </div>
            <div class="form-row three">
              <label>max_steps <input type="number" name="max_steps" value="24" min="12" /></label>
              <label>learning rate <input name="learning_rate" value="0.001" /></label>
              <label>lora_r <input type="number" name="lora_r" value="8" min="1" /></label>
            </div>
            <button type="submit" class="btn">Create fine-tune job</button>
          </form>
        </section>

        <section class="panel wide" id="step-jobs">
          <div class="panel-head">
            <h2>3. Jobs</h2>
            <button type="button" class="btn ghost compact" id="btn-refresh">Refresh</button>
          </div>
          <div class="table-wrap">
            <table>
              <thead>
                <tr><th>Job</th><th>Model</th><th>Framework</th><th>Status</th><th>Progress</th><th></th></tr>
              </thead>
              <tbody id="tbody-jobs"><tr><td colspan="6" class="muted">Waiting for the gateway…</td></tr></tbody>
            </table>
          </div>
          <div class="job-detail">
            <h3 id="job-detail-title">Select a job to view logs</h3>
            <div class="progress big" id="job-progress-wrap" hidden><div class="bar" id="job-progress-bar"></div></div>
            <pre class="log-view" id="job-log-view">Logs from GET /v1/jobs/{id}/logs appear here.</pre>
          </div>
        </section>

        <div class="two-col">
          <section class="panel" id="step-deploy">
            <div class="panel-head">
              <h2>4. Deploy</h2>
            </div>
            <ul class="list" id="list-models"><li class="muted">No models yet</li></ul>
            <form id="form-deploy" class="form">
              <label>Model <select name="model_name" id="sel-model" required></select></label>
              <label>Inference
                <select name="inference_framework">
                  <option value="transformers" selected>transformers (local PEFT / HF)</option>
                </select>
              </label>
              <button type="submit" class="btn">Create endpoint</button>
            </form>
          </section>

          <section class="panel" id="step-prompt">
            <div class="panel-head">
              <h2>5. Prompt</h2>
            </div>
            <ul class="list" id="list-endpoints"><li class="muted">No endpoints yet</li></ul>
            <form id="form-prompt" class="form">
              <label>Endpoint <select name="endpoint_id" id="sel-endpoint" required></select></label>
              <label>Prompt <textarea name="prompt" rows="3" required>What is LoRA?</textarea></label>
              <button type="submit" class="btn">Send prompt</button>
            </form>
            <div id="prompt-result" class="result" hidden>
              <h3>Completion</h3>
              <pre id="prompt-out"></pre>
            </div>
          </section>
        </div>
      </div>
    </div>
    <script id="boot" type="application/json">${bootJson}</script>
    <script src="/static/workbench.js?v=pro3" defer></script>
    `,
    "page-workbench",
  );
}

export function scopePage() {
  return page(
    "/scope",
    "Audience",
    `
    <section class="page-intro">
      <p class="eyebrow">Audience</p>
      <h1>Who ftaas.org serves next</h1>
      <p class="lede">Same product, on the domain you already have. A few more people on the gateway you already run.</p>
    </section>
    <div class="audience">
      <article>
        <p class="eyebrow">Today</p>
        <h2>One operator</h2>
        <p>A developer on a laptop, or a CPU control plane plus one GPU worker. They use the Jinja console or <code>ftaas.Client</code>. There is no login.</p>
      </article>
      <article>
        <p class="eyebrow">Next</p>
        <h2>One shared gateway</h2>
        <p>A small applied-ML group: one person prepares the jsonl, one launches LoRA or QLoRA, one prompts the adapter. Still one registry and one trainer.</p>
      </article>
    </div>
    <section class="section">
      <div class="section-head">
        <h2>What grows inside this loop</h2>
        <p>Enhancements, not a different product.</p>
      </div>
      <ul class="plain-list">
        <li>Upload a dataset and fetch a real <code>gs://</code> object into the mirror path the registry already computes.</li>
        <li>A job list the group can read — those routes already exist.</li>
        <li>A catalog that says transformers and TRL train; the other four names are aliases.</li>
        <li>Prompt checks plus <code>statistics.json</code> as the eval view phase 2 claims.</li>
        <li>The remote trainer switch that already exists.</li>
        <li>One shared bearer token on write routes when the host is not localhost.</li>
      </ul>
      <p class="hint">Not a marketplace, not a labeling tool, and not a second MLflow.</p>
    </section>
    `,
  );
}

export function notFoundPage() {
  return page(
    "",
    "Not found",
    `
    <section class="page-intro">
      <h1>Page not found</h1>
      <p class="lede"><a href="/">Back to overview</a> · <a href="/workbench">Workbench</a></p>
    </section>
    `,
  );
}
