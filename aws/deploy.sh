#!/usr/bin/env bash
# POD pipeline — SAM build + deploy.
#
# Prerequisites: AWS CLI v2, SAM CLI, Docker (running), logged-in AWS credentials.
#
# Usage:
#   cd aws && chmod +x deploy.sh
#   SAM_PARAMETER_OVERRIDES='Key=Val ...' ./deploy.sh           # samconfig default env
#   ./deploy.sh prod                                             # short-hand for samconfig prod
#
# Export SAM_PARAMETER_OVERRIDES unless parameters are already stored by `sam deploy --guided`.
# Keys must satisfy CloudFormation template.yaml Parameters:
#   Stage MetabaseUrl MetabaseApiKey MetabaseCardId FetchBatchSize FlagThreshold InferenceBatchSize
#   PgHost PgPort PgDatabase PgUser PgPassword VpcId SubnetIds ScorerImageUri [TmpEphemeralMB]
#
# Optional env:
#   SAM_CONFIG_ENV      Override samconfig env (default: default). Ignored if first arg is prod|dev|default.
#   SAM_NO_CONFIRM=true Skip interactive changeset confirmation.
#   SAM_RESOLVE_IMAGE_REPOS=true   Let SAM create/use ECR repos (first-time image flows).
#
# Example:
#   export SAM_PARAMETER_OVERRIDES='Stage=prod MetabaseUrl=https://meta.example MetabaseApiKey=secret MetabaseCardId=10989 FetchBatchSize=500 FlagThreshold=0.7 InferenceBatchSize=64 PgHost=....rds.amazonaws.com PgPort=5432 PgDatabase=pod_classifier PgUser=postgres PgPassword=secret VpcId=vpc-xxx SubnetIds=subnet-a,subnet-b ScorerImageUri=account.dkr.ecr.region.amazonaws.com/pod:latest TmpEphemeralMB=5120'
#   ./deploy.sh prod
#
set -euo pipefail

AWS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${AWS_ROOT}"

SAM_TEMPLATE="${SAM_TEMPLATE:-infra/template.yaml}"
CONFIG_ENV="${SAM_CONFIG_ENV:-default}"

usage() {
  cat <<'EOF'
Usage:
  SAM_PARAMETER_OVERRIDES='Key=Val ...' ./deploy.sh [prod|dev|default] [-- extra sam deploy args]

See header comments in deploy.sh for required parameter keys.
EOF
  exit 0
}

EXTRA_SAM_DEPLOY_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage ;;
    --)
      shift
      EXTRA_SAM_DEPLOY_ARGS+=("$@")
      break
      ;;
    prod|dev|default)
      CONFIG_ENV="$1"
      shift
      ;;
    *)
      EXTRA_SAM_DEPLOY_ARGS+=("$1")
      shift
      ;;
  esac
done

command -v sam >/dev/null 2>&1 || { echo "sam CLI not found"; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "docker not found"; exit 1; }

DEPLOY_ARGS=(
  --config-file infra/samconfig.toml
  --config-env "${CONFIG_ENV}"
)

if [[ "${SAM_NO_CONFIRM:-false}" == "true" ]]; then
  DEPLOY_ARGS+=(--no-confirm-changeset)
fi

if [[ "${SAM_RESOLVE_IMAGE_REPOS:-false}" == "true" ]]; then
  DEPLOY_ARGS+=(--resolve-image-repos)
fi

if [[ -n "${SAM_PARAMETER_OVERRIDES:-}" ]]; then
  DEPLOY_ARGS+=(--parameter-overrides "${SAM_PARAMETER_OVERRIDES}")
fi

echo "==> SAM validate (${SAM_TEMPLATE})"
sam validate --template-file "${SAM_TEMPLATE}"

echo "==> SAM build"
sam build --template-file "${SAM_TEMPLATE}"

echo "==> SAM deploy (config-env=${CONFIG_ENV})"
sam deploy "${DEPLOY_ARGS[@]}" "${EXTRA_SAM_DEPLOY_ARGS[@]}"

echo "Done."
