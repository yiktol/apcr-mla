#!/usr/bin/env bash
#
# ChurnGuard frontend deploy (Option 2, same-origin). Builds the Vite SPA with a
# RELATIVE API base so the browser calls /api on the CloudFront domain, deploys
# the 08-frontend.yaml stack (private S3 + CloudFront OAC + /api/* -> backend),
# syncs the built dist/ to the asset bucket, and invalidates CloudFront.
#
# This script is intentionally NOT wired into deploy.sh: it is an explicit,
# standalone invocation. The backend origin domain is a required input and is
# never hardcoded.
#
# Usage:
#   ./deploy-frontend.sh <backend-origin-domain>
#   BACKEND_ORIGIN_DOMAIN=<backend-origin-domain> ./deploy-frontend.sh
#
# Every aws call is pinned to us-east-1 (the machine default region differs).

set -euo pipefail

REGION="us-east-1"
ACCOUNT="875692608981"
export AWS_DEFAULT_REGION="$REGION"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
INFRA_DIR="$HERE"
FRONTEND_DIR="$REPO_ROOT/frontend"

STACK_NAME="churnguard-frontend"
TEMPLATE="$INFRA_DIR/08-frontend.yaml"

log() { printf '\n=== %s ===\n' "$*"; }

# --- explicit-invocation guard: backend origin domain is required ----------
BACKEND_ORIGIN_DOMAIN="${1:-${BACKEND_ORIGIN_DOMAIN:-}}"
if [[ -z "$BACKEND_ORIGIN_DOMAIN" ]]; then
  echo "ERROR: backend origin domain is required." >&2
  echo "Pass it as the first argument or set BACKEND_ORIGIN_DOMAIN." >&2
  echo "Usage: $0 <backend-origin-domain>" >&2
  exit 1
fi

dump_stack_events() {
  local stack="$1"
  echo "---- last events for stack $stack ----" >&2
  aws cloudformation describe-stack-events \
    --region "$REGION" \
    --stack-name "$stack" \
    --max-items 25 \
    --query 'StackEvents[].[Timestamp,ResourceStatus,ResourceType,LogicalResourceId,ResourceStatusReason]' \
    --output table >&2 || true
}

deploy_stack() {
  local stack="$1"; shift
  local template="$1"; shift
  log "deploy stack $stack"
  if ! aws cloudformation deploy \
      --region "$REGION" \
      --stack-name "$stack" \
      --template-file "$template" \
      --capabilities CAPABILITY_NAMED_IAM \
      --no-fail-on-empty-changeset \
      "$@"; then
    echo "ERROR: deploy of $stack failed" >&2
    dump_stack_events "$stack"
    exit 1
  fi
}

stack_output() {
  local stack="$1"; shift
  local key="$1"; shift
  aws cloudformation describe-stacks \
    --region "$REGION" \
    --stack-name "$stack" \
    --query "Stacks[0].Outputs[?OutputKey=='${key}'].OutputValue" \
    --output text
}

# --- 1. build the SPA (same-origin: relative /api, read-only online) -------
log "build frontend (VITE_API_BASE=/api VITE_READ_ONLY=true)"
(
  cd "$FRONTEND_DIR"
  VITE_API_BASE=/api VITE_READ_ONLY=true npm run build
)

# --- 2. deploy the hosting stack ------------------------------------------
deploy_stack "$STACK_NAME" "$TEMPLATE" \
  --parameter-overrides "BackendOriginDomain=${BACKEND_ORIGIN_DOMAIN}"

# --- 3. resolve stack outputs ---------------------------------------------
ASSET_BUCKET="$(stack_output "$STACK_NAME" AssetBucketName)"
DISTRIBUTION_ID="$(stack_output "$STACK_NAME" DistributionId)"
if [[ -z "$ASSET_BUCKET" || "$ASSET_BUCKET" == "None" ]]; then
  echo "ERROR: could not resolve AssetBucketName from $STACK_NAME outputs" >&2
  exit 1
fi
if [[ -z "$DISTRIBUTION_ID" || "$DISTRIBUTION_ID" == "None" ]]; then
  echo "ERROR: could not resolve DistributionId from $STACK_NAME outputs" >&2
  exit 1
fi
echo "asset bucket:    $ASSET_BUCKET"
echo "distribution id: $DISTRIBUTION_ID"

# --- 4. sync built assets to the private bucket ---------------------------
log "sync dist/ to s3://${ASSET_BUCKET}/"
aws s3 sync "$FRONTEND_DIR/dist/" "s3://${ASSET_BUCKET}/" --delete

# --- 5. invalidate CloudFront so clients fetch the new build --------------
log "invalidate CloudFront distribution $DISTRIBUTION_ID"
aws cloudfront create-invalidation \
  --distribution-id "$DISTRIBUTION_ID" \
  --paths '/*'

log "frontend deploy complete"
DISTRIBUTION_DOMAIN="$(stack_output "$STACK_NAME" DistributionDomainName)"
echo "SPA is served at: https://${DISTRIBUTION_DOMAIN}"
echo "API routed same-origin: https://${DISTRIBUTION_DOMAIN}/api/* -> ${BACKEND_ORIGIN_DOMAIN}"
