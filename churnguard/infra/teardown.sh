#!/usr/bin/env bash
#
# ChurnGuard teardown. Reverses deploy.sh:
#
#   1. Sweep the churnguard-bg- runtime resources FIRST (blue/green models +
#      endpoint configs the backend created outside CloudFormation). ONLY the
#      churnguard-bg- sub-prefix is swept so a stack-owned resource is never
#      touched (the two prefixes never overlap).
#   2. Delete endpoint + events stacks (events, mme, async, serverless, realtime),
#      then registry-pipeline.
#   3. Empty all four S3 buckets, then delete foundation.
#   4. Wait on stack-delete-complete and report leftovers.
#
# Every aws call is pinned to us-east-1. Errors on individual best-effort sweeps
# are tolerated; the final leftovers report is authoritative.

set -uo pipefail

REGION="us-east-1"
ACCOUNT="875692608981"
export AWS_DEFAULT_REGION="$REGION"

BG_PREFIX="churnguard-bg-"
DATA_BUCKET="churnguard-data-${ACCOUNT}-${REGION}"
MODELS_BUCKET="churnguard-models-${ACCOUNT}-${REGION}"
ASYNC_BUCKET="churnguard-async-${ACCOUNT}-${REGION}"
BATCH_BUCKET="churnguard-batch-${ACCOUNT}-${REGION}"

log() { printf '\n=== %s ===\n' "$*"; }

# --- 1. sweep churnguard-bg- runtime resources ----------------------------
log "sweep ${BG_PREFIX} runtime resources (blue/green leftovers)"

sweep_endpoint_configs() {
  local names
  names="$(aws sagemaker list-endpoint-configs \
    --region "$REGION" \
    --name-contains "$BG_PREFIX" \
    --query 'EndpointConfigs[].EndpointConfigName' \
    --output text 2>/dev/null || true)"
  for name in $names; do
    case "$name" in
      ${BG_PREFIX}*)
        echo "delete endpoint-config $name"
        aws sagemaker delete-endpoint-config \
          --region "$REGION" --endpoint-config-name "$name" || true
        ;;
    esac
  done
}

sweep_models() {
  local names
  names="$(aws sagemaker list-models \
    --region "$REGION" \
    --name-contains "$BG_PREFIX" \
    --query 'Models[].ModelName' \
    --output text 2>/dev/null || true)"
  for name in $names; do
    case "$name" in
      ${BG_PREFIX}*)
        echo "delete model $name"
        aws sagemaker delete-model \
          --region "$REGION" --model-name "$name" || true
        ;;
    esac
  done
}

sweep_endpoint_configs
sweep_models

# --- 2 & 3. delete stacks in reverse, emptying buckets before foundation --
delete_stack() {
  local stack="$1"
  log "delete stack $stack"
  if ! aws cloudformation describe-stacks \
      --region "$REGION" --stack-name "$stack" >/dev/null 2>&1; then
    echo "stack $stack not present, skipping"
    return 0
  fi
  aws cloudformation delete-stack --region "$REGION" --stack-name "$stack" || true
  echo "waiting for $stack delete to complete..."
  if ! aws cloudformation wait stack-delete-complete \
      --region "$REGION" --stack-name "$stack"; then
    echo "WARNING: wait for $stack delete did not complete cleanly" >&2
  fi
}

empty_bucket() {
  local bucket="$1"
  if aws s3api head-bucket --region "$REGION" --bucket "$bucket" >/dev/null 2>&1; then
    echo "emptying s3://$bucket"
    aws s3 rm "s3://$bucket" --region "$REGION" --recursive || true
  else
    echo "bucket $bucket not present, skipping empty"
  fi
}

# Endpoint + events stacks (reverse of deploy order), then registry-pipeline.
delete_stack churnguard-events
delete_stack churnguard-mme
delete_stack churnguard-async
delete_stack churnguard-serverless
delete_stack churnguard-realtime
delete_stack churnguard-registry-pipeline

# Empty all four buckets before deleting foundation (which owns them).
log "empty S3 buckets before deleting foundation"
empty_bucket "$DATA_BUCKET"
empty_bucket "$MODELS_BUCKET"
empty_bucket "$ASYNC_BUCKET"
empty_bucket "$BATCH_BUCKET"

delete_stack churnguard-foundation

# --- 4. report leftovers --------------------------------------------------
log "leftover report"
echo "-- churnguard-* endpoints --"
aws sagemaker list-endpoints \
  --region "$REGION" --name-contains churnguard \
  --query 'Endpoints[].EndpointName' --output text || true
echo "-- churnguard-* models --"
aws sagemaker list-models \
  --region "$REGION" --name-contains churnguard \
  --query 'Models[].ModelName' --output text || true
echo "-- churnguard-* endpoint-configs --"
aws sagemaker list-endpoint-configs \
  --region "$REGION" --name-contains churnguard \
  --query 'EndpointConfigs[].EndpointConfigName' --output text || true
echo "-- churnguard-* stacks (should be empty) --"
aws cloudformation list-stacks \
  --region "$REGION" \
  --stack-status-filter CREATE_COMPLETE UPDATE_COMPLETE ROLLBACK_COMPLETE DELETE_FAILED \
  --query "StackSummaries[?starts_with(StackName, 'churnguard')].StackName" \
  --output text || true

log "teardown complete"
echo "If any churnguard-* resources are listed above, inspect and remove them manually."
