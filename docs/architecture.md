# FTAAS Architecture

## System diagram

```
┌────────────────────┐   ┌────────────────────┐
│   FTAAS Python SDK │   │      FTAAS UI       │
│     (Jupyter)      │   │                    │
└─────────┬──────────┘   └─────────┬──────────┘
          └────────────┬───────────┘
                       ▼
              ┌─────────────────┐
              │  FTAAS gateway  │
              └────────┬────────┘
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │ Pipelines│  │ Datasets │  │ Serving  │
  └────┬─────┘  └────┬─────┘  └────┬─────┘
       ▼             │             ▼
 Local runner /      │      Create Endpoint
 Apache Airflow      │      Inference FW
       │             │      Model Deploy
       ▼             │      prompt UI / API
     ┌───┐           │
     │RAY│◄──────────┘
     └───┘
       │
  Model repo · MLflow · vLLM / Ray Serve
```

## Training flow

1. Select open-source LLM
2. Enter GCS/local path → register → Datasets **or** pick `dataset_id:version`
3. Select training framework + hyperparameters (Jobs API)
4. Training (Ray) + Logging & Tracking (MLflow)
5. **Training Completes** → branch:
   - **deployment:** Create Endpoint → Select Inference FW → Model Deploy → UI/API (Serving)
   - **evaluation:** vLLM → Adapters → Ray Serve → UI/API (MLflow + Ray)

## Pipeline steps

Implemented in `orchestrator/local_runner/runner.py` and service APIs:

| Step | Call | Component |
|------|------|-----------|
| 1–2 | register_dataset | Datasets |
| 3–5 | create_finetune_job + persist | Jobs |
| 6–8 | create_pipeline + persist | Pipelines / Jobs |
| 9–10 | schedule | Local runner / Airflow |
| 11 | download_dataset | Datasets |
| 12–17 | load_parameters, create_cluster, submit, poll | Ray helpers |
| 18–20 | log metrics/params/model | MLflow |
| 21–23 | register_model | Jobs |
| 24–26 | complete job | Pipelines → Jobs |
| 27–29 | get_job_status / get_model | Jobs / UI |
