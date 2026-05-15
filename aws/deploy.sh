#!/usr/bin/env bash
# Build POD Lambda CPU image → ECR login → push → lambda update-function-code.
# Mirrors the Downloads reference flow; values are parameterized (account from STS unless overridden).
#
# Local test without AWS credentials:
#   DRY_RUN=true ./deploy.sh
#
# First CFN bootstrap (no Lambda yet):
#   SKIP_LAMBDA_UPDATE=true ./deploy.sh
#
set -euo pipefail

AWS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CTX="${AWS_DIR}/lambda_scorer"

AWS_REGION="${AWS_REGION:-ap-south-1}"
STAGE="${STAGE:-prod}"
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
