# Fine Tuning as a Service (FTAAS)

End-to-end LLM fine-tuning platform:

**dataset registration → job orchestration → training (HF / TRL / LLaMA-Factory / Unsloth / Axolotl with LoRA / QLoRA / DoRA) → MLflow tracking → vLLM / Ray Serve deployment → API / UI**

```
FTAAS SDK / Jupyter              FTAAS UI
              \                    /
               \                  /
                ▼                ▼
              FTAAS gateway  (:8080)
         ┌─────────┼─────────┐
         ▼         ▼         ▼
     Datasets    Jobs     Serving
         │         │         │
         │    Pipelines      │
         │         │         │
         └────► Runner / Ray / MLflow / vLLM
```

## What you get

| Layer | Component | Role |
|-------|-----------|------|
| Gateway | **FTAAS** `:8080` | Single process: UI + all APIs |
| UI / SDK | Web UI + `ftaas` SDK | Job setup, datasets, tracking, prompt |
| Jobs | Job lifecycle + model registry | `create_finetune_job`, status, `register_model` |
| Datasets | Dataset registry | `register_dataset` → `id:version` |
| Pipelines | Pipeline tracking | `create_pipeline` / complete |
| Serving | Deploy + prompt | Endpoints (vLLM / Ray Serve path) |
| Infra | Local runner · Ray · MLflow | Train → track → register |

### Supported frameworks
Hugging Face Transformers · TRL · Verl · LLaMA-Factory · Unsloth · Axolotl

### Supported techniques
- **Full FT:** 16-bit full, frozen
- **PEFT:** LoRA, QLoRA, DoRA, LoRA+, LongLoRA, LoftQ, PiSSA, Prefix Tuning, Adapter Tuning, BitFit, IA3
- **Instruction:** SFT, multimodal instruction, Self-Instruct
- **Alignment:** Reward modeling, PPO, DPO, ORPO

> Real GPU training uses Transformers + PEFT (LoRA/DoRA/SFT). Other frameworks can run in mock mode so the control plane works without a GPU cluster.

## Quick start

```bash
cd FTAAS
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # includes greenlet for async SQLAlchemy

chmod +x scripts/*.sh
./scripts/start_all.sh
```

Open **http://127.0.0.1:8080**.

```bash
./scripts/e2e_smoke.sh
./scripts/stop_all.sh
```

## Flow

1. SDK/UI → Datasets `register_dataset(path)` → `dataset_id`, `version`
2. → Jobs `create_finetune_job(model, framework, dataset, parameters)` → `job_id`
3. Jobs → Pipelines `create_pipeline()` → `pipeline_id`
4. Local runner (Airflow-equivalent): download dataset → Ray cluster → train → poll
5. Log metrics / params / model → MLflow
6. `register_model` → Jobs registry
7. Pipeline complete → job `succeeded`
8. Serving `create_endpoint` → prompt on UI/API

## SDK

```python
from ftaas import FTAASClient, Framework, Technique, HyperParameters

with FTAASClient() as c:
    ds = c.register_dataset("examples/data/alpaca_sample.jsonl", name="alpaca")
    job = c.create_finetune_job(
        model_name="sshleifer/tiny-gpt2",
        dataset=ds,
        framework=Framework.TRANSFORMERS,
        technique=Technique.LORA,
        parameters=HyperParameters(max_steps=10),
    )
    job = c.wait_for_job(job.job_id)
    model = c.get_model(job.registered_model_name)
    ep = c.create_endpoint(model.model_name, inference_framework="vllm")
    print(c.prompt(ep.endpoint_id, "What is LoRA?").completion)
```

```bash
export PYTHONPATH="$PWD:$PWD/packages/ftaas_sdk:$PWD/services"
python -m ftaas.cli catalog
python -m ftaas.cli register-dataset examples/data/alpaca_sample.jsonl --name alpaca
```

## Layout

```
FTAAS/
├── configs/settings.yaml
├── packages/ftaas_sdk/ftaas/   # Python SDK + shared models
├── services/
│   ├── ftaas_app/              # Unified gateway (one process)
│   ├── jobs/                   # Job lifecycle + model registry
│   ├── datasets/               # Dataset registry
│   ├── pipelines/
│   └── serving/                # Deploy / prompt
├── orchestrator/
│   ├── local_runner/
│   └── airflow/dags/
├── training/
│   ├── frameworks/
│   ├── ray_jobs/
│   └── templates/
├── ui/web/
├── examples/
└── scripts/
```

## Configuration

Env vars (`FTAAS_` prefix), all defaulting to the unified gateway:

- `FTAAS_JOBS_URL`, `FTAAS_DATASETS_URL`, `FTAAS_PIPELINES_URL`, `FTAAS_SERVING_URL`
- `FTAAS_DATA_DIR`, `FTAAS_PORT` (default `8080`)
- Ray: `integrations.ray.mock`
- Airflow: `integrations.airflow.enabled`

## License

Apache-2.0 (see `LICENSE`).
