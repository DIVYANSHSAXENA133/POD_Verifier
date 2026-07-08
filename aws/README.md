# POD scoring pipeline — AWS

Single container Lambda that scores Proof-of-Delivery photos for quality and flags
bad PODs for the operations team. **Single invocation per day** covers the whole
dataset: read POD rows from Postgres → expand links (bound to AWB/trip) → concurrent
in-memory image download → ImageNet-normalized EfficientNet inference → idempotent
write to PostgreSQL. Triggered once daily by EventBridge Scheduler.

**No SAM.** Infra is plain **CloudFormation** ([`infra/stack.yaml`](infra/stack.yaml))
applied with [`provision-stack.sh`](provision-stack.sh); image delivery is
[`deploy.sh`](deploy.sh). Full runbook: [`DEPLOYMENT.md`](DEPLOYMENT.md).

## Architecture (single invocation, resilient)

One EventBridge trigger (empty event `{}`) runs one invocation that processes the
entire day in bounded-memory windows and is safe against the 15-minute wall:

- **Concurrent downloads** — `ThreadPoolExecutor`; the download, not the model, was
  the original bottleneck.
- **Bounded memory** — one `WINDOW_SIZE` of images in memory at a time; flat regardless
  of dataset size.
- **Resume-from-checkpoint** — scored rows in Postgres are the checkpoint; a retry
  processes only the remainder.
- **Idempotent upsert** — unique `(awb, pod_link, run_date)`; retries fill gaps, never dup.
- **Every input gets an outcome** — download failures are recorded (`status='download_failed'`),
  never silently dropped and never penalised.
- **Clock-aware continuation** — near the wall it flushes and queues exactly one
  checkpointed continuation (bounded by `MAX_CONTINUATIONS`); async retries + SQS DLQ
  back it up.

## Configuration

Runtime env (set by CloudFormation, not baked into code): **source**
(`SOURCE_QUERY` — the SQL that returns the day's POD rows), **pipeline tuning**
(`MAX_DOWNLOAD_WORKERS`, `WINDOW_SIZE`, `INFERENCE_BATCH_SIZE`), **scoring**
(`FLAG_THRESHOLD`, `IMAGENET_NORMALIZE`), **resilience** (`CONTINUATION_SAFETY_MS`,
`MAX_CONTINUATIONS`), and **Postgres** (`PG_*`). See [`config.env.example`](config.env.example);
use Secrets Manager / CI secrets for real passwords.

> **Do not disable `IMAGENET_NORMALIZE`.** The model was trained with ImageNet
> normalization; scoring without it collapses recall.

## Layout

- `lambda_scorer/` — Docker image: `Dockerfile`, `handler.py`, `model/best.pt`, `src/`, `tests/`.
- `infra/` — `stack.yaml` (CloudFormation) and `schema.sql` (with the idempotency key).
- `eval/` — `evaluate_model.py` + `run_gold_eval.sh`: measure the model against a human gold set.
- `deploy.sh` — build CPU image, push to ECR, roll the Lambda (`DRY_RUN`/`SKIP_LAMBDA_UPDATE` supported).
- `provision-stack.sh` — `aws cloudformation deploy` (VPC, Lambda, Scheduler, DLQ, IAM, env).
- `DEPLOYMENT.md` — step-by-step deploy runbook.

## Model weights (`lambda_scorer/model/best.pt`)

Trained `MultiHeadEfficientNet` (EfficientNet-B0 backbone, 4 attribute heads) checkpoint,
tracked in Git and copied into the image at `/opt/model/best.pt`. The Docker build
verifies it loads into the architecture with 0 key mismatches.

## Tests

```bash
cd aws/lambda_scorer && python3 -m venv .venv && source .venv/bin/activate \
  && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
  && pip install -r requirements.txt -r requirements-dev.txt \
  && pytest tests/ -v
```

## Build smoke-test (no AWS)

```bash
cd aws && DRY_RUN=true ./deploy.sh
```

**Schedule:** cron `25 18` UTC daily (23:55 IST); tune in `stack.yaml`.
