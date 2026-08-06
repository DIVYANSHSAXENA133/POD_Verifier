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
#   export SOURCE_QUERY="SELECT awb, trip_id, pod FROM pod_manual_verification WHERE created_date = CURRENT_DATE"
#   export PG_HOST=db.xxx.rds.amazonaws.com
#   export PG_PASSWORD=secret
#   export SCORER_IMAGE_URI=123456789012.dkr.ecr.ap-south-1.amazonaws.com/pod-pipeline:latest
#
# Optional: FLAG_THRESHOLD INFERENCE_BATCH_SIZE MAX_DOWNLOAD_WORKERS WINDOW_SIZE
#           IMAGENET_NORMALIZE PG_PORT PG_DATABASE PG_USER TMP_EPHEMERAL_MB
#
set -euo pipefail

AWS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${AWS_DIR}/infra/stack.yaml"

AWS_REGION="${AWS_REGION:-us-east-2}"
STACK_NAME="${STACK_NAME:-pod-scoring-prod}"
STAGE="${STAGE:-prod}"
VPC_ID="${VPC_ID:-}"
SUBNET_IDS="${SUBNET_IDS:-}"
SOURCE_QUERY="${SOURCE_QUERY:-}"
FLAG_THRESHOLD="${FLAG_THRESHOLD:-0.7}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-64}"
MAX_DOWNLOAD_WORKERS="${MAX_DOWNLOAD_WORKERS:-64}"
WINDOW_SIZE="${WINDOW_SIZE:-800}"
IMAGENET_NORMALIZE="${IMAGENET_NORMALIZE:-true}"
PG_HOST="${PG_HOST:-}"
PG_PASSWORD="${PG_PASSWORD:-}"
PG_PORT="${PG_PORT:-5432}"
PG_DATABASE="${PG_DATABASE:-pod_classifier}"
PG_USER="${PG_USER:-postgres}"
TMP_EPHEMERAL_MB="${TMP_EPHEMERAL_MB:-512}"   # in-memory design uses no /tmp
SCORER_IMAGE_URI="${SCORER_IMAGE_URI:-}"

if [[ -z "$VPC_ID" || -z "$SUBNET_IDS" ]]; then
  echo "Set VPC_ID and SUBNET_IDS (comma-separated private subnets)." >&2
  exit 1
fi
if [[ -z "$SOURCE_QUERY" ]]; then
  echo "Set SOURCE_QUERY (SQL returning awb, trip_id, and POD links)." >&2
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
  "SourceQuery=${SOURCE_QUERY}"
  "FlagThreshold=${FLAG_THRESHOLD}"
  "InferenceBatchSize=${INFERENCE_BATCH_SIZE}"
  "MaxDownloadWorkers=${MAX_DOWNLOAD_WORKERS}"
  "WindowSize=${WINDOW_SIZE}"
  "ImagenetNormalize=${IMAGENET_NORMALIZE}"
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
