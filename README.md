# Fine Tuning as a Service (FTAAS)

Open-source reference implementation of an end-to-end **Fine Tuning as a Service** platform, mirroring the Cosmos.AI / AIML MDLC architecture:

```
MDLC Python SDK / Jupyter          Cosmos UI
              \                      /
               \                    /
                ▼                  ▼
              AIML MDLC Serv  (job lifecycle + model registry)
               /        |        \
              ▼         ▼         ▼
        Pipelineserv   MDS    Aimlopsserv
              |         |         |
         Apache Airflow |    vLLM / Adapters / Ray Serve
              |         |         |
         Ray cluster ←──┘      MLflow
         LLM Repo
```

## What you get

| Layer | Component | Port | Role |
|-------|-----------|------|------|
| UI / SDK | Cosmos UI | 8080 | Job setup, datasets, tracking, prompt |
| UI / SDK | `mdlc` Python SDK | — | Notebook / CLI client |
| Orchestration | AIML MDLC Serv | 8000 | `create_finetune_job`, status, `register_model` |
| Services | MDS | 8001 | `register_dataset(gcs_path)` → `id:version` |
| Services | Pipelineserv | 8002 | `create_pipeline` / complete |
| Services | Aimlopsserv | 8003 | Create endpoint → deploy → prompt API |
| Infra | Local runner (Airflow DAG equivalent) | — | Sequence from the flow diagram |
| Infra | Ray (mock or real) | — | `create_cluster` / `submit_training_job` |
| Infra | MLflow | 5000 (optional) | metrics / params / model artifacts |

### Supported frameworks
Hugging Face Transformers · TRL · Verl · LLaMA-Factory · Unsloth · Axolotl

### Supported techniques
- **Full FT:** 16-bit full, frozen
- **PEFT:** LoRA, QLoRA, DoRA, LoRA+, LongLoRA, LoftQ, PiSSA, Prefix Tuning, Adapter Tuning, BitFit, IA3
- **Instruction:** SFT, multimodal instruction, Self-Instruct
- **Alignment:** Reward modeling, PPO, DPO, ORPO

> Real GPU training path uses Transformers + PEFT (LoRA/DoRA/SFT). Other frameworks run in **mock mode** (artifacts + MLflow-compatible metrics) so the full control plane works without a GPU cluster.

## Quick start (local)

```bash
cd /Users/ashishverma/Downloads/FTAAS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # or: pip install fastapi uvicorn ... (see Dockerfile for minimal)

chmod +x scripts/*.sh
./scripts/start_all.sh
```

Open **http://127.0.0.1:8080** (Cosmos UI).

Run the SDK smoke test:

```bash
./scripts/e2e_smoke.sh
```

Stop:

```bash
./scripts/stop_all.sh
```

## Sequence (matches the architecture diagram)

1. `CosmosUI/SDK` → MDS `register_dataset(gcs_path)` → `dataset_id`, `version`
2. → MDLC `create_finetune_job(model, framework, dataset, parameters)` → persist → `job_id`
3. MDLC → Pipelineserv `create_pipeline()` → `pipeline_id`
4. Schedule Airflow / local runner
5. Runner: `download_dataset` → `load_parameters` → Ray `create_cluster` → `submit_training_job` → `poll_job_status`
6. `log_metrics` / `log_parameters` / `log_model` → MLflow
7. `register_model` → MDLC DB
8. Pipeline complete → job `succeeded`
9. UI `get_job_status` / `get_model` → Aimlopsserv deploy → prompt on UI/API

## SDK example

```python
from mdlc import MDLCClient, Framework, Technique, HyperParameters

with MDLCClient() as c:
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

CLI:

```bash
export PYTHONPATH="$PWD:$PWD/packages/mdlc_sdk:$PWD/services"
python -m mdlc.cli catalog
python -m mdlc.cli register-dataset examples/data/alpaca_sample.jsonl --name alpaca
```

## Phase roadmap (from product milestones)

| Phase | Milestone | Status in this repo |
|-------|-----------|---------------------|
| 0 | Fine-Tuning & RL Templates | `training/templates/` |
| 1 | Fine-Tuning UI | Cosmos UI on :8080 |
| 2 | Fine-Tune & Evaluate | Jobs + Aimlopsserv endpoints + eval stack API |
| 3 | Resource Optimization | Planned (catalog stub) |
| 4 | Sweeps & Optimization | Planned (catalog stub) |

## Docker Compose

```bash
docker compose up --build
```

## Layout

```
FTAAS/
├── configs/settings.yaml
├── packages/mdlc_sdk/mdlc/     # Python SDK + shared models
├── services/
│   ├── mdlc_server/            # AIML MDLC Serv
│   ├── mds/                    # Dataset registry
│   ├── pipelineserv/
│   └── aimlopsserv/            # Deploy / prompt
├── orchestrator/
│   ├── local_runner/           # Airflow-equivalent pipeline
│   └── airflow/dags/           # Optional real Airflow DAG
├── training/
│   ├── frameworks/             # Framework adapters
│   ├── ray_jobs/               # Cluster + submit/poll
│   └── templates/              # Phase 0 YAML templates
├── ui/cosmos_ui/               # Cosmos UI
├── examples/                   # Sample data + e2e script
└── scripts/                    # start / stop / smoke
```

## Configuration

Edit `configs/settings.yaml` or set env vars (`FTAAS_` prefix):

- `FTAAS_MDLC_URL`, `FTAAS_MDS_URL`, `FTAAS_PIPELINESERV_URL`, `FTAAS_AIMLOPSSERV_URL`
- `FTAAS_DATA_DIR` — SQLite DBs + datasets + outputs
- Ray: `integrations.ray.mock: true|false`
- Airflow: `integrations.airflow.enabled: true` to use the DAG instead of the local runner

## License

Apache-2.0 (see `LICENSE`).
