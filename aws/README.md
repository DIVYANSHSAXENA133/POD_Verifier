# POD scoring pipeline — AWS

Single container Lambda: Metabase ingest, `/tmp` staging, EfficientNet inference, PostgreSQL. Daily trigger via EventBridge Scheduler.

**No SAM.** Infra is plain **CloudFormation** ([`infra/stack.yaml`](infra/stack.yaml)) applied with [`provision-stack.sh`](provision-stack.sh); image delivery is [`deploy.sh`](deploy.sh).

## Configuration

At runtime the Lambda reads **Metabase** (`METABASE_URL`, `METABASE_API_KEY`, `METABASE_CARD_ID`), **batching** (`FETCH_BATCH_SIZE`, `INFERENCE_BATCH_SIZE`), **threshold** (`FLAG_THRESHOLD`), and **Postgres** (`PG_*`) from **environment variables** set in the stack template (not baked into code).

See [`config.env.example`](config.env.example) for naming; use Secrets Manager / CI secrets for real passwords and API keys.

## Layout

- `lambda_scorer/` — Docker image: `Dockerfile`, `handler.py`, `model/best.pt`, `src/`.
- `infra/` — **`stack.yaml`** (CloudFormation), `schema.sql`, `ephemeral_peak_mb.py`.
- `deploy.sh` — build CPU image, push to ECR; updates Lambda unless `SKIP_LAMBDA_UPDATE=true`.
- `provision-stack.sh` — `aws cloudformation deploy` stack (VPC, Lambda, Scheduler, IAM, env vars).
- `config.env.example` — reference only.

## Model weights (`lambda_scorer/model/best.pt`)

Trained checkpoint at `model/best.pt` (tracked in Git). Docker copies it to `/opt/model/best.pt`.

Dummy init checkpoint: `python3 aws/scripts/create_init_checkpoint.py`

## `/tmp` sizing

Rough peak vs `FETCH_BATCH_SIZE`: `python3 infra/ephemeral_peak_mb.py --fetch-batch-size 500 ...`  
Tune **`TmpEphemeralMB`** (`TMP_EPHEMERAL_MB` export for `provision-stack.sh`).

## Unit tests

```bash
cd aws/lambda_scorer && python3 -m venv .venv && source .venv/bin/activate \
  && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
  && pip install -r requirements.txt -r requirements-dev.txt \
  && pytest tests/ -v
```

## Smoke-test Docker build (no AWS)

Same `docker buildx build --platform linux/amd64 --provenance=false` path as production deploy; skips ECR/Lambda:

```bash
cd aws && chmod +x deploy.sh && DRY_RUN=true ./deploy.sh
```

Requires Docker Desktop/daemon. Loads a local image **`pod-pipeline-local:${IMAGE_TAG:-latest}`**.

## First-time deploy

1. **RDS schema:** `psql -f infra/schema.sql` …
2. **ECR:** create repo (Console/CLI); set `ECR_REPOSITORY`/`IMAGE_TAG` to match.
3. **Push image only** (no Lambda yet — skips `update-function-code`):

   ```bash
   cd aws && chmod +x deploy.sh provision-stack.sh
   export AWS_REGION=ap-south-1
   export ECR_REPOSITORY=pod-pipeline
   export IMAGE_TAG=latest
   export STAGE=prod
   SKIP_LAMBDA_UPDATE=true ./deploy.sh
   ```

   Use the printed URI for **`SCORER_IMAGE_URI`** in step 4 (e.g.  
   `account.dkr.ecr.ap-south-1.amazonaws.com/pod-pipeline:latest`).

4. **Create stack** (private subnets comma-separated; secrets from env/CI):

   ```bash
   export STACK_NAME=pod-scoring-prod
   export VPC_ID=vpc-xxx
   export SUBNET_IDS=subnet-a,subnet-b
   export METABASE_URL=https://your-metabase
   export METABASE_API_KEY=***
   export PG_HOST=your-db.region.rds.amazonaws.com
   export PG_PASSWORD=***
   export SCORER_IMAGE_URI="$(aws sts get-caller-identity --query Account --output text).dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}"
   ./provision-stack.sh
   ```

5. **Later image updates:** build, push, roll Lambda to the new digest:

   ```bash
   ./deploy.sh
   ```

   Default **`LAMBDA_FUNCTION=pod-pipeline-${STAGE}`** (override if you renamed the function).

**Schedule:** cron **25 18 UTC** daily (tune in `stack.yaml` if you need a different wall time).

## GitHub Projects

Optional tracking in GitHub Projects for ops tasks.
