# Fine Tuning as a Service (FTAAS)

**register → orchestrate → train (HF / TRL / LLaMA-Factory / Unsloth / Axolotl · LoRA / QLoRA / DoRA) → MLflow → vLLM / Ray Serve → API / Console**

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
            runner · Ray · MLflow · vLLM
```

## Components

| Module | Role |
|--------|------|
| **gateway** (`ftaas_app`) | Single process on `:8080` — Console + APIs |
| **console** | Web UI for setup, tracking, prompt |
| **registry** | Dataset registration → `id:version` |
| **control** | Fine-tune jobs + model registry |
| **workflow** | Pipeline create / complete |
| **deploy** | Endpoints (vLLM / Ray Serve path) + prompt API |
| **runner** | Local (or Airflow) train pipeline |
| **ftaas** SDK | `Client` for notebooks / automation |

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./scripts/start_all.sh          # → http://127.0.0.1:8080
./scripts/e2e_smoke.sh
./scripts/stop_all.sh
```

## SDK

```python
from ftaas import Client, Framework, Technique, HyperParameters

with Client() as c:
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

## Layout

```
FTAAS/
├── packages/ftaas_sdk/ftaas/   # Client SDK
├── services/
│   ├── ftaas_app/              # gateway
│   ├── registry/
│   ├── control/
│   ├── workflow/
│   └── deploy/
├── runner/                     # local + Airflow DAG
├── training/
├── ui/console/
├── examples/
└── scripts/
```

## License

Apache-2.0
