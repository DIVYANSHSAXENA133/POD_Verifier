# POD Verifier — Lambda Deployment (no SAM)

Deployment-ready module for the **normalization-fixed** single-invocation pipeline.
Infra is plain **CloudFormation** + Docker + AWS CLI — **no SAM, no CDK**.

## What ships

- **Container Lambda** (`lambda_scorer/`): `handler.py` (single-invocation, resilient,
  **ImageNet-normalized inference by default**), `src/model.py`, `model/best.pt`.
- **`Dockerfile`**: CPU-only torch 2.2.2 / torchvision 0.17.2 (pinned), `numpy<2`;
  build-time check that deps import and the checkpoint loads with 0 key mismatches.
- **Infra** (`infra/stack.yaml`): VPC Lambda (10 GB / 900 s), EventBridge daily trigger
  (empty event `{}`), async retries + SQS DLQ, S3 state bucket, IAM.
- **Scripts**: `deploy.sh` (build → ECR → update function), `provision-stack.sh`
  (`aws cloudformation deploy`), `infra/schema.sql` (v2 with the idempotency key).

## Prerequisites

- Docker (with buildx) running.
- AWS CLI v2 authenticated (`AWS_PROFILE` or keys) with ECR/Lambda/CFN/IAM permissions.
- `lambda_scorer/model/best.pt` present (baked into the image).
- Reachability: the Lambda's VPC subnets need egress to the Metabase host, the POD
  image host, and the RDS Postgres endpoint (NAT or VPC endpoints).

## One-time setup

1. **Database schema** (idempotent; adds `status`/`failure_reason` + the unique
   `(awb, pod_link, run_date)` key used for upsert/resume):

   ```bash
   psql "host=<PG_HOST> dbname=pod_classifier user=postgres" -f infra/schema.sql
   ```

2. **ECR repo**:

   ```bash
   export AWS_REGION=ap-south-1
   aws ecr create-repository --repository-name pod-pipeline --region "$AWS_REGION"
   ```

3. **Build & push image only** (no Lambda yet):

   ```bash
   cd aws && chmod +x deploy.sh provision-stack.sh
   export ECR_REPOSITORY=pod-pipeline IMAGE_TAG=latest STAGE=prod
   SKIP_LAMBDA_UPDATE=true ./deploy.sh
   ```

   Note the printed image URI for the next step.

4. **Create the stack** (private subnets comma-separated; secrets from env/CI —
   never commit them):

   ```bash
   export STACK_NAME=pod-scoring-prod
   export VPC_ID=vpc-xxx
   export SUBNET_IDS=subnet-a,subnet-b
   export METABASE_URL=https://metabase.blitznow.in
   export METABASE_API_KEY=***
   export METABASE_CARD_ID=10989          # scoring card
   export PG_HOST=<db>.rds.amazonaws.com
   export PG_PASSWORD=***
   export SCORER_IMAGE_URI="$(aws sts get-caller-identity --query Account --output text).dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}"
   # Optional operating knobs (defaults shown):
   #   FLAG_THRESHOLD=0.7  MAX_DOWNLOAD_WORKERS=64  WINDOW_SIZE=800  IMAGENET_NORMALIZE=true
   ./provision-stack.sh
   ```

## Routine updates

Build, push, and roll the Lambda to the new image digest:

```bash
cd aws && ./deploy.sh
```

`DRY_RUN=true ./deploy.sh` builds the image locally only (no AWS) — a good pre-push smoke test.

## Verify after deploy

```bash
# Manual run (same payload the scheduler sends)
aws lambda invoke --function-name pod-pipeline-prod \
  --payload '{}' --cli-binary-format raw-in-base64-out /dev/stdout

# Rows land in Postgres
psql ... -c "SELECT status, count(*) FROM pod_scores WHERE run_date=CURRENT_DATE GROUP BY status;"

# Coverage metrics (CloudWatch namespace 'PODPipeline'): ImagesScored/Failed/Uncovered
```

Expect the response body to report `status=complete` with `scored + failed == total`.
If it reports `continuing`, a bounded continuation was queued (large day) — it resumes automatically.

## Operating notes (from the gold-set evaluation)

- **ImageNet normalization is on by default** (`IMAGENET_NORMALIZE=true`) — this is the
  correctness fix; do not set it to `false` in production.
- **FLAG threshold**: the eval favours ~0.55–0.60 (precision-first for penalties) over the
  0.7 default. Tune `FLAG_THRESHOLD` and the `0.7` in the `pod_scores_flagged` view together.
- **Coverage**: download failures are recorded (`status='download_failed'`) and excluded from
  PASS/FLAG, so an AWB is never penalised on a failed fetch.

## Rollback

Re-point the function at the previous image tag:

```bash
aws lambda update-function-code --function-name pod-pipeline-prod \
  --image-uri <account>.dkr.ecr.<region>.amazonaws.com/pod-pipeline:<previous-tag> --region "$AWS_REGION"
```

CloudFormation changes roll back via `aws cloudformation deploy` of the prior template or the console.
