# FTAAS Architecture

## System diagram

```
┌────────────────────┐   ┌────────────────────┐
│  MDLC Python SDK   │   │     Cosmos UI      │
│     (Jupyter)      │   │    (Cosmos.AI)     │
└─────────┬──────────┘   └─────────┬──────────┘
          └────────────┬───────────┘
                       ▼
              ┌─────────────────┐
              │  AIML MDLC Serv │
              └────────┬────────┘
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
  ┌────────────┐ ┌─────────┐ ┌──────────────┐
  │Pipelineserv│ │   MDS   │ │ Aimlopsserv  │
  └─────┬──────┘ └────┬────┘ └──────┬───────┘
        ▼             │             ▼
  Apache Airflow      │      Create Endpoint
        │             │      Inference FW
        ▼             │      Model Deploy
      ┌───┐           │      prompt UI / API
      │RAY│◄──────────┘
      └───┘
        │
   LLM Repo · MLflow
```

## Training flow

1. Select Open Source LLM (LLM Repo)
2. Enter GCS path → register → MDS **or** pick `dataset_id:version`
3. Select Training Framework + Input Hyperparameters (MDLC)
4. Training (Ray) + Logging & Tracking (MLflow)
5. **Training Completes** → branch:
   - **deployment:** Create Endpoint → Select Inference FW → Model Deploy → UI/API (Aimlopsserv)
   - **evaluation:** vLLM → Adapters → Ray Serve → UI/API (MLflow + Ray)

## Sequence diagram steps (1–29)

Implemented in `orchestrator/local_runner/runner.py` and service APIs:

| Step | Call | Service |
|------|------|---------|
| 1–2 | register_dataset | MDS |
| 3–5 | create_finetune_job + persist | MDLC |
| 6–8 | create_pipeline + persist | Pipelineserv / MDLC |
| 9–10 | schedule | Local runner / Airflow |
| 11 | download_dataset | MDS |
| 12–17 | load_parameters, create_cluster, submit, poll | Ray helpers |
| 18–20 | log metrics/params/model | MLflow |
| 21–23 | register_model | MDLC |
| 24–26 | complete job | Pipelineserv → MDLC |
| 27–29 | get_job_status / get_model | MDLC / UI |
