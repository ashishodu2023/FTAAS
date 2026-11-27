# Fine Tuning as a Service (FTAAS)

The gateway behind [ftaas.org](https://ftaas.org). One operator can run it today. A small applied-ML group can share that same gateway next: register an instruction file, run a real LoRA or QLoRA job, read logs, and prompt the adapter. The PyPI package `ftaas` is the SDK for that gateway.

**preview → register → orchestrate → train (transformers / TRL · LoRA) → MLflow → prompt (local HF/PEFT) → Console / API**

```
ftaas.Client / notebook           Console
              \                    /
               ▼                  ▼
              FTAAS gateway  (:8080)
         ┌─────────┼──────────┐
         ▼         ▼          ▼
      registry   control    deploy
                   │
                workflow
                   │
            runner · MLflow · HF generate
```

## What’s live vs planned

| Capability | Status |
|------------|--------|
| Dataset preview + register (jsonl/json/csv, local paths) | **Live** |
| Job logs + progress in Console / `GET /v1/jobs/{id}/logs` | **Live** |
| Real fine-tune (`transformers` + PEFT LoRA/DoRA; `trl` SFT/DPO) | **Live** |
| **QLoRA (4-bit bitsandbytes)** on CUDA | **Live** — see [docs/gpu-training.md](docs/gpu-training.md) |
| GPU Docker (`Dockerfile.gpu`) / RunPod / GCE | **Live** |
| Remote trainer worker (control/UI offload) | **Live** — `FTAAS_TRAIN_MODE=remote` |
| Framework aliases (`verl`, `llama-factory`, `unsloth`, `axolotl`) | Compat → PEFT/TRL backend |
| Prompt via local Transformers/PEFT | **Live** |
| vLLM / Ray Serve serving | **Planned** (UI options disabled) |
| Real `gs://` GCS (no local mirror) | **Not yet** — use local/file paths |

## Components

| Module | Role |
|--------|------|
| **gateway** (`ftaas_app`) | Single process on `:8080` — Console + APIs |
| **console** | Web UI: preview → register → train → logs → deploy → prompt |
| **registry** | Dataset preview + registration → `id:version` |
| **control** | Fine-tune jobs, logs/progress, model registry |
| **workflow** | Pipeline create / complete |
| **deploy** | Endpoints + prompt API (HF/PEFT generate) |
| **runner** | Local train pipeline (Airflow optional); can offload to GPU worker |
| **trainer_worker** | Optional GPU process — QLoRA/LoRA, uploads artifacts to control |
| **ftaas** SDK | `Client` for notebooks / automation |

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./scripts/start_all.sh          # → http://127.0.0.1:8080
./scripts/e2e_smoke.sh
./scripts/stop_all.sh
```

### GPU / QLoRA / remote trainer

```bash
# All-in-one GPU box
docker build -f Dockerfile.gpu -t ftaas-gpu .
docker run --gpus all -p 8080:8080 -e FTAAS_TRAIN_DEVICE=cuda ftaas-gpu

# Or: control on CPU, train on GPU worker — see docs/gpu-training.md
./scripts/start_trainer_worker.sh   # :8090
```

Full guide: [docs/gpu-training.md](docs/gpu-training.md).

### Product site

Two parts. The Python gateway (`http://127.0.0.1:8080`, and the Railway service behind [ftaas.org](https://ftaas.org)) trains and serves `/v1`. The Node site on port 3000 is the public front door and proxies those same routes (preview, register, jobs, deploy, prompt):

```bash
cd web && npm start
# FTAAS_GATEWAY_URL=http://127.0.0.1:8080  FTAAS_WEB_PORT=3000
# If 3000 is busy: npm run start:alt  →  http://127.0.0.1:3001
```

Open http://127.0.0.1:3000 (or `:3001` with `npm run start:alt`). Pages: `/`, `/pipeline`, `/catalog`, `/workbench`, `/scope`. The site does not train; creating a job from the workbench starts a real run on the gateway.

## SDK

```python
from ftaas import Client, Framework, Technique, HyperParameters

with Client() as c:
    preview = c.preview_dataset("examples/data/alpaca_sample.jsonl")
    print(preview["num_rows"], preview["columns"])
    ds = c.register_dataset("examples/data/alpaca_sample.jsonl", name="alpaca")
    job = c.create_finetune_job(
        model_name="distilgpt2",
        dataset=ds,
        framework=Framework.TRANSFORMERS,
        technique=Technique.LORA,
        parameters=HyperParameters(max_steps=12, max_seq_length=64),
    )
    job = c.wait_for_job(job.job_id)
    print(c.get_job_logs(job.job_id)["progress"])
    model = c.get_model(job.registered_model_name)
    ep = c.create_endpoint(model.model_name, inference_framework="transformers")
    print(c.prompt(ep.endpoint_id, "What is LoRA?").completion)
```

## Layout

```
FTAAS/
├── packages/ftaas_sdk/ftaas/   # Client SDK
├── services/
│   ├── ftaas_app/              # gateway
│   ├── registry/
│   ├── control/
│   ├── workflow/
│   ├── deploy/
│   └── trainer_worker/         # optional GPU offload
├── runner/                     # local + Airflow DAG
├── training/
├── ui/console/                 # operator console on the gateway
├── web/                        # Node product site (proxies /v1)
├── Dockerfile                  # CPU (Railway)
├── Dockerfile.gpu              # CUDA + bitsandbytes
├── docs/gpu-training.md
├── tests/                      # unit + system integration
├── examples/
└── scripts/
```

## Test results

```bash
./scripts/run_tests.sh          # unit + API
./scripts/run_tests.sh --slow   # + real tiny-gpt2 fine-tune
```

See [docs/testing.md](docs/testing.md) for markers and statistics fields.

## License

Apache-2.0
