import { escapeHtml, layout } from "./layout.js";

function page(current, title, main) {
  return layout({ title, current, main });
}

export function homePage() {
  return page(
    "/",
    "Fine Tuning as a Service",
    `
    <section class="hero">
      <h1>Fine Tuning as a Service</h1>
      <p class="lede">ftaas.org is the front door for a small group that shares one fine-tuning gateway: prepare an instruction file, run a real LoRA or QLoRA job, then prompt the adapter. The Python service does the training. This site does not.</p>
      <p class="actions">
        <a class="btn" href="/workbench">Open the workbench</a>
        <a class="btn secondary" href="/scope">Who it is for</a>
      </p>
    </section>
    <section class="split">
      <article class="panel">
        <h2>Today — one operator</h2>
        <p>A developer on a laptop, or a CPU control plane plus one GPU worker. They use the console on the gateway or the <code>ftaas</code> package (<code>ftaas.Client</code>).</p>
      </article>
      <article class="panel">
        <h2>Next — the same gateway, shared</h2>
        <p>One person prepares the jsonl, one launches the job, one prompts the adapter to see if it learned the set. Still one registry, one trainer, and one domain. Not a marketplace and not a second MLflow.</p>
      </article>
    </section>
    <section class="steps">
      <article>
        <h2>1. Preview and register</h2>
        <p>Point at a jsonl, json, or csv file the gateway can read. Preview is required before register. The sample set is <code>examples/data/alpaca_sample.jsonl</code>.</p>
      </article>
      <article>
        <h2>2. Train</h2>
        <p>Create a fine-tune job. The default model is <code>distilgpt2</code> with LoRA, sized for a CPU demo. QLoRA runs when CUDA and bitsandbytes are present.</p>
      </article>
      <article>
        <h2>3. Track</h2>
        <p>Jobs expose status, progress, and logs. Metrics land on the job and in MLflow. A finished run writes <code>statistics.json</code> under the job output directory.</p>
      </article>
      <article>
        <h2>4. Prompt</h2>
        <p>Deploy an endpoint and prompt it. Serving today is local Hugging Face / PEFT generate. vLLM and Ray Serve are listed and not wired up.</p>
      </article>
    </section>
    <section class="panel">
      <h2>Live today, and not yet</h2>
      <table>
        <thead><tr><th>Capability</th><th>Status</th></tr></thead>
        <tbody>
          <tr><td>Dataset preview and register (jsonl, json, csv, local path)</td><td><span class="pill ok">Live</span></td></tr>
          <tr><td>Job logs, progress, cancel</td><td><span class="pill ok">Live</span></td></tr>
          <tr><td>transformers + PEFT LoRA / DoRA, TRL SFT / DPO</td><td><span class="pill ok">Live</span></td></tr>
          <tr><td>QLoRA on CUDA, GPU image, remote trainer on port 8090</td><td><span class="pill ok">Live</span></td></tr>
          <tr><td>Prompt through local HF / PEFT</td><td><span class="pill ok">Live</span></td></tr>
          <tr><td>verl, llama-factory, unsloth, axolotl</td><td><span class="pill warn">Alias → PEFT / TRL</span></td></tr>
          <tr><td>vLLM and Ray Serve</td><td><span class="pill planned">Planned</span></td></tr>
          <tr><td>Real <code>gs://</code> download (no local mirror)</td><td><span class="pill planned">Not yet</span></td></tr>
          <tr><td>Sweeps and a statistics view in the console</td><td><span class="pill planned">Planned</span></td></tr>
        </tbody>
      </table>
      <p class="hint">The operator console is the Jinja app on the gateway (<code>ui/console</code>, port 8080). This Node process only proxies <code>/v1</code> and does not train.</p>
    </section>
    `,
  );
}

export function pipelinePage() {
  return page(
    "/pipeline",
    "Pipeline",
    `
    <section class="hero slim">
      <h1>One gateway, the modules behind it</h1>
      <p class="lede">By default every box below is the same FastAPI process on port 8080 (<code>services/ftaas_app/main.py</code>). The browser console and <code>ftaas.Client</code> both call it.</p>
    </section>
    <ol class="flow">
      <li><strong>Preview / register</strong> <code>services/registry/main.py</code> — <code>POST /v1/datasets/preview</code>, <code>POST /v1/datasets/register</code>. A <code>gs://</code> path is only a label until the file is already on disk under the local mirror.</li>
      <li><strong>Create job</strong> <code>services/control/main.py</code> — <code>POST /v1/jobs/finetune</code>. Catalog, logs, cancel, and the model list live here too.</li>
      <li><strong>Pipeline</strong> <code>services/workflow/main.py</code> — create and complete. The local runner is <code>runner/local/runner.py</code>. Airflow is optional and off in <code>configs/settings.yaml</code>.</li>
      <li><strong>Train</strong> <code>training/frameworks/registry.py</code> — real torch updates. In-process if Ray is absent. Or <code>FTAAS_TRAIN_MODE=remote</code> sends the job to <code>services/trainer_worker/main.py</code> (<code>POST /v1/train</code>, port 8090).</li>
      <li><strong>Register model and prompt</strong> <code>services/deploy/main.py</code> — <code>POST /v1/endpoints</code> and <code>POST /v1/endpoints/{id}/prompt</code>. Generation is local HF / PEFT even if the request says vLLM.</li>
    </ol>
    <section class="panel">
      <h2>Same calls from Python</h2>
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
      <p class="hint">The workbench on this site posts the same JSON to <code>/gateway/v1/…</code>, which this process forwards to <code>FTAAS_GATEWAY_URL</code>.</p>
    </section>
    `,
  );
}

function techniqueGroups(catalog) {
  const groups = catalog.techniques || {};
  return Object.entries(groups)
    .map(([name, items]) => {
      const chips = (items || []).map((item) => `<span class="chip">${escapeHtml(item)}</span>`).join("");
      return `<h3>${escapeHtml(name.replaceAll("_", " "))}</h3><div class="chips">${chips}</div>`;
    })
    .join("");
}

export function catalogPage({ catalog, evalStack, gatewayUp }) {
  const backends = catalog.backends || {};
  const frameworkRows = (catalog.frameworks || [])
    .map((name) => {
      const note = backends[name] || "listed in configs/settings.yaml";
      return `<tr><td><code>${escapeHtml(name)}</code></td><td>${escapeHtml(note)}</td></tr>`;
    })
    .join("");

  const inferenceRows = Object.entries(catalog.inference || {})
    .map(([name, note]) => `<tr><td><code>${escapeHtml(name)}</code></td><td>${escapeHtml(note)}</td></tr>`)
    .join("");

  const phases = (catalog.phases || [])
    .map(
      (phase) =>
        `<li>Phase ${escapeHtml(phase.phase)} — ${escapeHtml(phase.name)} <em>(${escapeHtml(phase.status)})</em></li>`,
    )
    .join("");

  const training = catalog.training || {};
  const qlora = training.qlora || {};
  const device = training.device || {};
  const source =
    catalog.source === "gateway"
      ? "Loaded from the gateway <code>GET /v1/catalog</code>."
      : "Gateway did not answer. Showing the lists from <code>configs/settings.yaml</code> and the backend sentences the catalog route returns.";

  const deviceBlock =
    catalog.source === "gateway"
      ? `<p class="hint">Training plane: ${escapeHtml(training.mode || "local")} · CUDA ${device.cuda_available ? "yes" : "no"} · QLoRA ${qlora.available ? "ready" : "needs GPU + bitsandbytes"}.</p>`
      : `<p class="hint">QLoRA needs CUDA and bitsandbytes (<code>Dockerfile.gpu</code>). On a CPU box the trainer rejects technique <code>qlora</code> unless <code>FTAAS_ALLOW_QLORA_CPU_FALLBACK=true</code>.</p>`;

  const evalNote = evalStack
    ? `<p class="hint">Serving stack from <code>GET /v1/eval/stack</code>: live ${escapeHtml((evalStack.live || []).join(", ") || "—")}; planned ${escapeHtml((evalStack.planned || []).join(", ") || "—")}. ${escapeHtml(evalStack.note || "")}</p>`
    : "";

  return page(
    "/catalog",
    "Catalog",
    `
    <section class="hero slim">
      <h1>Catalog</h1>
      <p class="lede">${source} ${gatewayUp ? "" : "Start the gateway with <code>./scripts/start_all.sh</code> and reload."}</p>
    </section>
    <section class="panel">
      <h2>Frameworks</h2>
      <table>
        <thead><tr><th>Name</th><th>What actually runs</th></tr></thead>
        <tbody>${frameworkRows}</tbody>
      </table>
      ${deviceBlock}
    </section>
    <section class="panel">
      <h2>Techniques</h2>
      ${techniqueGroups(catalog)}
    </section>
    <section class="split">
      <section class="panel">
        <h2>Inference</h2>
        <table>
          <thead><tr><th>Name</th><th>Status</th></tr></thead>
          <tbody>${inferenceRows}</tbody>
        </table>
        ${evalNote}
      </section>
      <section class="panel">
        <h2>Phases</h2>
        <ol class="phases">${phases}</ol>
        <p class="hint">Phase 2 is marked available. The numbers for an eval view are already written to <code>statistics.json</code>; the console does not show that file yet. Phases 3 and 4 have no API.</p>
      </section>
    </section>
    `,
  );
}

export function workbenchPage(boot) {
  const bootJson = JSON.stringify(boot).replace(/</g, "\\u003c");
  return page(
    "/workbench",
    "Workbench",
    `
    <section class="hero slim">
      <h1>Workbench</h1>
      <p class="lede">The same loop as the Jinja console: preview, register, train, deploy, prompt. Requests go to <code>/gateway/v1/…</code> on this site, then to the Python gateway. A created job is a real training run.</p>
    </section>
    <div id="flash" class="banner" hidden></div>
    <div class="grid">
      <section class="panel">
        <h2>1. Preview and register</h2>
        <form id="form-register">
          <label>Path on the gateway machine
            <input name="gcs_path" id="inp-gcs-path" required value="examples/data/alpaca_sample.jsonl" />
          </label>
          <label>Name <input name="name" value="alpaca-sample" /></label>
          <label>Format
            <select name="format" id="sel-ds-format">
              <option value="jsonl" selected>jsonl</option>
              <option value="json">json</option>
              <option value="csv">csv</option>
            </select>
          </label>
          <div class="btn-row">
            <button type="button" class="btn secondary" id="btn-preview">Preview data</button>
            <button type="submit" class="btn" id="btn-register" disabled>Register dataset</button>
          </div>
        </form>
        <div id="dataset-preview" class="preview-box" hidden></div>
        <h3>Datasets</h3>
        <ul class="list" id="list-datasets"><li class="muted">Waiting for the gateway…</li></ul>
      </section>

      <section class="panel">
        <h2>2. Create fine-tune job</h2>
        <form id="form-job">
          <label>Model <input name="model_name" id="inp-model" required value="distilgpt2" /></label>
          <p class="hint"><code>distilgpt2</code> is the CPU demo default. QLoRA is refused here when the catalog says this gateway has no CUDA + bitsandbytes.</p>
          <label>Framework <select name="framework" id="sel-framework"></select></label>
          <p class="hint" id="framework-note"></p>
          <label>Technique <select name="technique" id="sel-technique"></select></label>
          <p class="hint" id="train-plane-hint"></p>
          <label>Dataset <select name="dataset_id" id="sel-dataset" required></select></label>
          <label>Version <input name="dataset_version" id="inp-dataset-version" value="1" /></label>
          <div class="row">
            <label>max_steps <input type="number" name="max_steps" value="24" min="12" /></label>
            <label>lr <input name="learning_rate" value="0.001" /></label>
            <label>lora_r <input type="number" name="lora_r" value="8" min="1" /></label>
          </div>
          <button type="submit" class="btn">create_finetune_job</button>
        </form>
      </section>

      <section class="panel wide">
        <div class="panel-head">
          <h2>Jobs</h2>
          <button type="button" class="btn ghost" id="btn-refresh">Refresh</button>
        </div>
        <div class="table-scroll">
          <table>
            <thead>
              <tr><th>Job</th><th>Model</th><th>FW / tech</th><th>Status</th><th>Progress</th><th></th></tr>
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

      <section class="panel">
        <h2>3. Deploy</h2>
        <ul class="list" id="list-models"><li class="muted">No models yet</li></ul>
        <form id="form-deploy">
          <label>Model <select name="model_name" id="sel-model" required></select></label>
          <label>Inference framework
            <select name="inference_framework">
              <option value="transformers" selected>transformers (local PEFT / HF)</option>
            </select>
          </label>
          <p class="hint">vLLM and Ray Serve stay off. The gateway still generates with local HF / PEFT.</p>
          <button type="submit" class="btn">create_endpoint</button>
        </form>
      </section>

      <section class="panel">
        <h2>4. Prompt</h2>
        <ul class="list" id="list-endpoints"><li class="muted">No endpoints yet</li></ul>
        <form id="form-prompt">
          <label>Endpoint <select name="endpoint_id" id="sel-endpoint" required></select></label>
          <label>Prompt <textarea name="prompt" rows="3" required>What is LoRA?</textarea></label>
          <p class="hint">Exact rows from the train file can come back as the labeled answer. Short runs still drift on new questions.</p>
          <button type="submit" class="btn">prompt</button>
        </form>
        <div id="prompt-result" class="result" hidden>
          <h3>Completion</h3>
          <pre id="prompt-out"></pre>
        </div>
      </section>
    </div>
    <script id="boot" type="application/json">${bootJson}</script>
    <script src="/static/workbench.js" defer></script>
    `,
  );
}

export function scopePage() {
  return page(
    "/scope",
    "Scope",
    `
    <section class="hero slim">
      <h1>Who it is for next</h1>
      <p class="lede">Same product, on the domain you already have (<a href="https://ftaas.org">ftaas.org</a>). A few more people on the gateway you already run.</p>
    </section>
    <section class="split">
      <article class="panel">
        <h2>Today</h2>
        <p>One developer, one machine — or a CPU control plane plus one GPU worker. They use the Jinja console or <code>ftaas.Client</code>. The sample file is an instruction set for a small open model. There is no login.</p>
      </article>
      <article class="panel">
        <h2>Next</h2>
        <p>A small applied-ML group sharing that gateway. One person prepares the jsonl, one person launches LoRA or QLoRA, one person prompts the adapter to see if it learned the set. Still one SQLite registry and one trainer.</p>
      </article>
    </section>
    <section class="panel">
      <h2>What grows, inside this loop</h2>
      <ul class="plain">
        <li>Register a file without copying it onto the gateway host, and fetch a real <code>gs://</code> object into the mirror path the registry already computes.</li>
        <li>A job list the group can read: status, logs, progress, cancel. Those routes exist.</li>
        <li>A catalog that says transformers and TRL train, and the other four names are aliases.</li>
        <li>The prompt check you have, plus <code>statistics.json</code> as the eval view phase 2 claims.</li>
        <li>The remote trainer switch that already exists (<code>FTAAS_TRAIN_MODE=remote</code>).</li>
        <li>One shared bearer token on write routes when the host is not localhost. No accounts and no signup.</li>
      </ul>
      <p class="hint">Not a marketplace, not a labeling tool, and not a second MLflow. Ray, MLflow, and later vLLM stay behind the same job and prompt routes.</p>
    </section>
    `,
  );
}

export function notFoundPage() {
  return page(
    "",
    "Not found",
    `
    <section class="hero slim">
      <h1>That page is not on this site</h1>
      <p class="lede"><a href="/">Back to the product</a> · <a href="/workbench">Workbench</a></p>
    </section>
    `,
  );
}
