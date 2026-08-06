#!/usr/bin/env bash
# Build POD Lambda CPU image → ECR login → push → lambda update-function-code.
# NO SAM — plain Docker + AWS CLI. Infra is CloudFormation (aws/infra/stack.yaml)
# applied by aws/provision-stack.sh.
#
# Handler: single-invocation, resilient scorer — query Postgres for POD rows, expand links,
# concurrently download images to memory, ImageNet-normalized EfficientNet scoring,
# idempotent upsert to Postgres, resume + bounded self-continuation. Scheduler
# sends an empty event {} once per day.
#
# Local build smoke-test without AWS credentials:
#   DRY_RUN=true ./deploy.sh
#
# First CFN bootstrap (push image only, no Lambda yet):
#   SKIP_LAMBDA_UPDATE=true ./deploy.sh
#
# Routine update (build, push, roll Lambda to the new image):
#   ./deploy.sh
#
set -euo pipefail

AWS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CTX="${AWS_DIR}/lambda_scorer"

# The trained checkpoint is baked into the image — fail fast if it's missing.
if [[ ! -f "${CTX}/model/best.pt" ]]; then
  echo "ERROR: ${CTX}/model/best.pt not found. Commit/copy the trained checkpoint first." >&2
  exit 1
fi

AWS_REGION="${AWS_REGION:-us-east-2}"
STAGE="${STAGE:-stg}"
ECR_REPOSITORY="${ECR_REPOSITORY:-pod-pipeline}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
LOCAL_NAME="${LOCAL_NAME:-pod-pipeline-local}"
SKIP_LAMBDA_UPDATE="${SKIP_LAMBDA_UPDATE:-false}"
DRY_RUN="${DRY_RUN:-false}"

LAMBDA_FUNCTION="${LAMBDA_FUNCTION:-pod-pipeline-${STAGE}}"

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "==> DRY_RUN: docker build only (${CTX}) linux/amd64"
  docker buildx build \
    --platform linux/amd64 \
    --provenance=false \
    --load \
    -t "${LOCAL_NAME}:${IMAGE_TAG}" \
    "${CTX}"
  echo "DRY_RUN OK: local image ${LOCAL_NAME}:${IMAGE_TAG}"
  exit 0
fi

if [[ -n "${AWS_ACCOUNT_ID:-}" ]]; then
  :
else
  if ! AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"; then
    echo "aws sts get-caller-identity failed; set AWS_ACCOUNT_ID or fix credentials / AWS_PROFILE." >&2
    exit 1
  fi
fi

REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
REMOTE_URI="${REGISTRY}/${ECR_REPOSITORY}:${IMAGE_TAG}"

echo "==> ECR login ${REGISTRY}"
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${REGISTRY}"

echo "==> docker buildx (${CTX}) linux/amd64 (same as reference deploy.sh)"
docker buildx build \
  --platform linux/amd64 \
  --provenance=false \
  -t "${LOCAL_NAME}:${IMAGE_TAG}" \
  "${CTX}"

docker tag "${LOCAL_NAME}:${IMAGE_TAG}" "${REMOTE_URI}"

echo "==> docker push ${REMOTE_URI}"
docker push "${REMOTE_URI}"

echo "Image pushed to ECR!"
echo "Image URI: ${REMOTE_URI}"

if [[ "${SKIP_LAMBDA_UPDATE}" == "true" ]]; then
  echo "SKIP_LAMBDA_UPDATE=true → skipping lambda update-function-code."
else
  echo "==> aws lambda update-function-code ${LAMBDA_FUNCTION}"
  aws lambda update-function-code \
    --function-name "${LAMBDA_FUNCTION}" \
    --image-uri "${REMOTE_URI}" \
    --region "${AWS_REGION}"
fi

echo "Done."
