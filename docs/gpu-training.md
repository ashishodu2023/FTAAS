# GPU training & remote trainer

FTAAS can run three ways:

1. **CPU control plane** (Railway / laptop) — LoRA on small models (`distilgpt2` / `gpt2`)
2. **GPU all-in-one** — same process with CUDA Torch + bitsandbytes (real QLoRA, ~7B)
3. **Split plane** — control/UI on CPU; training on a remote GPU worker

## 1. Local GPU (Docker)

```bash
docker build -f Dockerfile.gpu -t ftaas-gpu .
docker run --gpus all -p 8080:8080 \
  -e FTAAS_TRAIN_DEVICE=cuda \
  -v "$PWD/data:/app/data" \
  ftaas-gpu
```

Open http://127.0.0.1:8080 — pick technique `qlora` and a Hugging Face model that fits VRAM
(e.g. `TinyLlama/TinyLlama-1.1B-Chat-v1.0` or `mistralai/Mistral-7B-Instruct-v0.2` on ≥16–24GB).

Without Docker:

```bash
pip install -r requirements.txt
pip install -r requirements-gpu.txt  # after CUDA torch
export FTAAS_TRAIN_DEVICE=cuda
./scripts/start_all.sh
```

## 2. RunPod / GCE

| Step | Action |
|------|--------|
| Image | Build/push `Dockerfile.gpu`, or start a CUDA 12.1 PyTorch pod and `git clone` + `pip install` |
| Env | `FTAAS_TRAIN_DEVICE=cuda`, `FTAAS_DATA_DIR=/workspace/data`, optional `HF_TOKEN` |
| Port | Expose `8080` (control+UI) or only `8090` if this box is a **trainer worker** |
| Volume | Persist `/app/data` (or `/workspace/data`) so adapters survive restarts |

**GCE example** (L4 / T4 VM):

```bash
# on the VM
git clone https://github.com/ashishodu2023/FTAAS.git && cd FTAAS
docker build -f Dockerfile.gpu -t ftaas-gpu .
docker run --gpus all -d -p 8080:8080 --name ftaas \
  -e FTAAS_TRAIN_DEVICE=cuda \
  -e FTAAS_PUBLIC_URL=http://YOUR_VM_IP:8080 \
  -v /mnt/ftaas-data:/app/data \
  ftaas-gpu
```

## 3. Remote trainer (control stays on Railway)

Keep **ftaas.org** (or any CPU deploy) as the UI/API. Point it at a GPU worker:

**On Railway (control):**

```
FTAAS_TRAIN_MODE=remote
FTAAS_TRAINER_URL=https://your-runpod-worker:8090
FTAAS_PUBLIC_URL=https://ftaas.org
```

**On the GPU box:**

```bash
docker run --gpus all -p 8090:8090 \
  -e FTAAS_TRAIN_DEVICE=cuda \
  -e FTAAS_TRAIN_MODE=local \
  -e PORT=8090 \
  ftaas-gpu python -m trainer_worker.main
```

Or:

```bash
./scripts/start_trainer_worker.sh
```

Flow:

1. Console creates a job on control
2. Runner POSTs to `{TRAINER_URL}/v1/train` with dataset download URL
3. Worker trains (QLoRA/LoRA), zips adapters, `POST /v1/jobs/{id}/artifacts`
4. Control registers the model and Console can deploy/prompt as usual

Worker health: `GET http://gpu:8090/health` → includes `device.cuda_available` and `bitsandbytes`.

## Env reference

| Variable | Default | Meaning |
|----------|---------|---------|
| `FTAAS_TRAIN_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `FTAAS_TRAIN_MODE` | `local` | `local` in-process, or `remote` offload |
| `FTAAS_TRAINER_URL` | _(empty)_ | GPU worker base URL when mode=remote |
| `FTAAS_PUBLIC_URL` | `http://127.0.0.1:8080` | URL workers use to fetch datasets / (control) |
| `FTAAS_ALLOW_QLORA_CPU_FALLBACK` | `false` | If true, `qlora` on CPU silently uses LoRA |

## QLoRA notes

- Requires **CUDA** + **bitsandbytes** (`requirements-gpu.txt` / `Dockerfile.gpu`)
- On CPU (e.g. Railway), selecting `qlora` fails with a clear error unless fallback is enabled
- Catalog `GET /v1/catalog` → `training.qlora.available` reflects current host capability
