#!/usr/bin/env bash
# Create or update the POD pipeline stack (plain CloudFormation — no SAM).
#
# Prereqs: AWS CLI v2, credentials, image already in ECR (see deploy.sh).
#
# Required env (example):
#   export AWS_REGION=ap-south-1
#   export STACK_NAME=pod-scoring-prod
#   export STAGE=prod
#   export VPC_ID=vpc-xxx
#   export SUBNET_IDS=subnet-a,subnet-b
#   export METABASE_URL=https://metabase.example
#   export METABASE_API_KEY=secret
#   export PG_HOST=db.xxx.rds.amazonaws.com
#   export PG_PASSWORD=secret
#   export SCORER_IMAGE_URI=123456789012.dkr.ecr.ap-south-1.amazonaws.com/pod-pipeline:latest
#
# Optional: METABASE_CARD_ID FETCH_BATCH_SIZE FLAG_THRESHOLD INFERENCE_BATCH_SIZE
#           PG_PORT PG_DATABASE PG_USER TMP_EPHEMERAL_MB
#
set -euo pipefail

AWS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${AWS_DIR}/infra/stack.yaml"

AWS_REGION="${AWS_REGION:-ap-south-1}"
STACK_NAME="${STACK_NAME:-pod-scoring-prod}"
STAGE="${STAGE:-prod}"
VPC_ID="${VPC_ID:-}"
SUBNET_IDS="${SUBNET_IDS:-}"
METABASE_URL="${METABASE_URL:-}"
METABASE_API_KEY="${METABASE_API_KEY:-}"
METABASE_CARD_ID="${METABASE_CARD_ID:-10989}"
FETCH_BATCH_SIZE="${FETCH_BATCH_SIZE:-500}"
FLAG_THRESHOLD="${FLAG_THRESHOLD:-0.7}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-64}"
PG_HOST="${PG_HOST:-}"
PG_PASSWORD="${PG_PASSWORD:-}"
PG_PORT="${PG_PORT:-5432}"
PG_DATABASE="${PG_DATABASE:-pod_classifier}"
PG_USER="${PG_USER:-postgres}"
TMP_EPHEMERAL_MB="${TMP_EPHEMERAL_MB:-5120}"
SCORER_IMAGE_URI="${SCORER_IMAGE_URI:-}"

if [[ -z "$VPC_ID" || -z "$SUBNET_IDS" ]]; then
  echo "Set VPC_ID and SUBNET_IDS (comma-separated private subnets)." >&2
  exit 1
fi
if [[ -z "$METABASE_URL" || -z "$METABASE_API_KEY" ]]; then
  echo "Set METABASE_URL and METABASE_API_KEY." >&2
  exit 1
fi
if [[ -z "$PG_HOST" || -z "$PG_PASSWORD" ]]; then
  echo "Set PG_HOST and PG_PASSWORD." >&2
  exit 1
fi
if [[ -z "$SCORER_IMAGE_URI" ]]; then
  echo "Set SCORER_IMAGE_URI (same ECR tag you pushed with aws/deploy.sh)." >&2
  exit 1
fi

OVERRIDES=(
  "Stage=${STAGE}"
  "MetabaseUrl=${METABASE_URL}"
  "MetabaseApiKey=${METABASE_API_KEY}"
  "MetabaseCardId=${METABASE_CARD_ID}"
  "FetchBatchSize=${FETCH_BATCH_SIZE}"
  "FlagThreshold=${FLAG_THRESHOLD}"
  "InferenceBatchSize=${INFERENCE_BATCH_SIZE}"
  "PgHost=${PG_HOST}"
  "PgPassword=${PG_PASSWORD}"
  "PgPort=${PG_PORT}"
  "PgDatabase=${PG_DATABASE}"
  "PgUser=${PG_USER}"
  "VpcId=${VPC_ID}"
  "SubnetIds=${SUBNET_IDS}"
  "ScorerImageUri=${SCORER_IMAGE_URI}"
  "TmpEphemeralMB=${TMP_EPHEMERAL_MB}"
)

echo "==> cloudformation deploy stack=${STACK_NAME} region=${AWS_REGION}"
aws cloudformation deploy \
  --stack-name "${STACK_NAME}" \
  --template-file "${TEMPLATE}" \
  --capabilities CAPABILITY_NAMED_IAM CAPABILITY_IAM \
  --region "${AWS_REGION}" \
  --parameter-overrides "${OVERRIDES[@]}"

echo "==> outputs"
aws cloudformation describe-stacks \
  --stack-name "${STACK_NAME}" \
  --region "${AWS_REGION}" \
  --query "Stacks[0].Outputs" \
  --output table

echo "Done."
