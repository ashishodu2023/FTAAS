# Testing FTAAS

## Quick start

```bash
cd /Users/ashishverma/Downloads/FTAAS
source .venv/bin/activate
pip install -r requirements.txt   # includes pytest + train stack
./scripts/run_tests.sh            # unit + API integration
./scripts/run_tests.sh --slow     # real sshleifer/tiny-gpt2 LoRA + statistics
```

## What is covered

| Suite | Marker | Covers |
|-------|--------|--------|
| Unit | `unit` | config/data dirs, real trainer stats, registry helpers, capability matrix |
| Integration | `integration` | `/health`, catalog, register dataset, deploy (TestClient) |
| System / real train | `slow` | live uvicorn → finetune tiny-gpt2 LoRA → job metrics + `statistics.json` |

## No mock mode

FTAAS always runs **real** gradient updates (torch / transformers / peft / TRL). Missing datasets or missing train deps raise errors instead of writing fake adapters.

Statistics land in `data/outputs/<job_id>/statistics.json` and on the job:

- `train_loss` / `final_train_loss`, `loss_first`, `loss_last`, `loss_curve_len`
- `trainable_params`, `total_params`, `trainable_pct`
- `duration_seconds`, `num_examples`, `real=1`
