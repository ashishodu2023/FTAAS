# Fine Tuning as a Service (FTAAS)

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
| **runner** | Local train pipeline (Airflow optional) |
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
    preview = c.preview_dataset("examples/data/alpaca_sample.jsonl")
    print(preview["num_rows"], preview["columns"])
    ds = c.register_dataset("examples/data/alpaca_sample.jsonl", name="alpaca")
    job = c.create_finetune_job(
        model_name="sshleifer/tiny-gpt2",
        dataset=ds,
        framework=Framework.TRANSFORMERS,
        technique=Technique.LORA,
        parameters=HyperParameters(max_steps=10),
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
│   └── deploy/
├── runner/                     # local + Airflow DAG
├── training/
├── ui/console/
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
