#!/usr/bin/env bash
#
# ChurnGuard end-to-end deploy. Provisions the seven CloudFormation stacks and
# runs the ml/ provisioning scripts in the order the design requires:
#
#   1. foundation            (buckets + SNS + SSM)
#   2. ml/upload_dataset.py  (raw Telco CSV -> data bucket)
#   3. ml/build_pipeline.py  (author + upload pipeline definition + code/)
#   4. registry-pipeline     (ModelPackageGroup + Pipeline via S3 location)
#   5. ml/verify_mme_capability.py  (real MME probe -> resolves the MME image)
#   6. ml/run_training.py    (5a preprocess -> 5b train, writes SSM artifact uri)
#   7. ml/build_mme_models.py (mme/churn-v1.tar.gz + churn-v2.tar.gz)
#   8. realtime, serverless, async, mme, events (pass ModelArtifactUri)
#
# Every aws call is pinned to us-east-1 (the machine default region differs).
# Stops on the first non-zero step; dumps describe-stack-events on a failed
# CloudFormation deploy. No simulation: every step hits real AWS.

set -euo pipefail

REGION="us-east-1"
ACCOUNT="875692608981"
export AWS_DEFAULT_REGION="$REGION"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
INFRA_DIR="$HERE"
ML_DIR="$REPO_ROOT/ml"
VENV_PY="$ML_DIR/.venv/bin/python"

DATA_BUCKET="churnguard-data-${ACCOUNT}-${REGION}"
MODELS_BUCKET="churnguard-models-${ACCOUNT}-${REGION}"
ASYNC_BUCKET="churnguard-async-${ACCOUNT}-${REGION}"
EXECUTION_ROLE_ARN="arn:aws:iam::${ACCOUNT}:role/AmazonSageMaker-ExecutionRole"
ASYNC_TOPIC_ARN="arn:aws:sns:${REGION}:${ACCOUNT}:churnguard-async-notifications"
PIPELINE_DEF_KEY="pipeline/pipeline-definition.json"
SSM_ARTIFACT_PARAM="/churnguard/model-artifact-uri"

NOTIFICATION_EMAIL="${NOTIFICATION_EMAIL:-}"

log() { printf '\n=== %s ===\n' "$*"; }

require_venv() {
  if [[ ! -x "$VENV_PY" ]]; then
    echo "ERROR: ml virtualenv python not found at $VENV_PY" >&2
    echo "Run: bash $ML_DIR/setup-venv.sh" >&2
    exit 1
  fi
}

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

require_venv

# --- 1. foundation --------------------------------------------------------
FOUNDATION_PARAMS=( "ExecutionRoleArn=${EXECUTION_ROLE_ARN}" )
if [[ -n "$NOTIFICATION_EMAIL" ]]; then
  FOUNDATION_PARAMS+=( "NotificationEmail=${NOTIFICATION_EMAIL}" )
fi
deploy_stack churnguard-foundation "$INFRA_DIR/01-foundation.yaml" \
  --parameter-overrides "${FOUNDATION_PARAMS[@]}"

# --- 2. dataset -----------------------------------------------------------
log "upload dataset (ml/upload_dataset.py)"
"$VENV_PY" "$ML_DIR/upload_dataset.py"

# --- 3. pipeline definition (author + upload def + code/ scripts) ---------
log "build + upload pipeline definition (ml/build_pipeline.py)"
"$VENV_PY" "$ML_DIR/build_pipeline.py"

# --- 4. registry + pipeline ----------------------------------------------
deploy_stack churnguard-registry-pipeline "$INFRA_DIR/02-registry-pipeline.yaml" \
  --parameter-overrides \
    "ExecutionRoleArn=${EXECUTION_ROLE_ARN}" \
    "PipelineDefinitionBucket=${DATA_BUCKET}" \
    "PipelineDefinitionKey=${PIPELINE_DEF_KEY}"

# --- 5. MME capability probe (resolves the MME serving image) -------------
log "probe MME capability (ml/verify_mme_capability.py)"
MME_PROBE_OUT="$("$VENV_PY" "$ML_DIR/verify_mme_capability.py")"
echo "$MME_PROBE_OUT"
MME_IMAGE_URI="$(printf '%s\n' "$MME_PROBE_OUT" | sed -n 's/^MME_IMAGE_URI=//p' | tail -n1)"
if [[ -z "$MME_IMAGE_URI" ]]; then
  echo "ERROR: MME probe did not emit MME_IMAGE_URI" >&2
  exit 1
fi
echo "resolved MME image: $MME_IMAGE_URI"

# --- 6. baseline training (5a preprocess -> 5b train) ---------------------
log "baseline preprocess + train (ml/run_training.py)"
"$VENV_PY" "$ML_DIR/run_training.py"

MODEL_ARTIFACT_URI="$(aws ssm get-parameter \
  --region "$REGION" \
  --name "$SSM_ARTIFACT_PARAM" \
  --query 'Parameter.Value' \
  --output text)"
if [[ -z "$MODEL_ARTIFACT_URI" || "$MODEL_ARTIFACT_URI" == "None" ]]; then
  echo "ERROR: ${SSM_ARTIFACT_PARAM} not set after training" >&2
  exit 1
fi
echo "model artifact: $MODEL_ARTIFACT_URI"

# --- 7. MME artifacts (v1 copy + v2 retrain) ------------------------------
log "build MME artifacts (ml/build_mme_models.py)"
"$VENV_PY" "$ML_DIR/build_mme_models.py"

# --- 8. endpoint + events stacks ------------------------------------------
deploy_stack churnguard-realtime "$INFRA_DIR/03-realtime.yaml" \
  --parameter-overrides \
    "ExecutionRoleArn=${EXECUTION_ROLE_ARN}" \
    "ModelArtifactUri=${MODEL_ARTIFACT_URI}"

deploy_stack churnguard-serverless "$INFRA_DIR/04-serverless.yaml" \
  --parameter-overrides \
    "ExecutionRoleArn=${EXECUTION_ROLE_ARN}" \
    "ModelArtifactUri=${MODEL_ARTIFACT_URI}"

deploy_stack churnguard-async "$INFRA_DIR/05-async.yaml" \
  --parameter-overrides \
    "ExecutionRoleArn=${EXECUTION_ROLE_ARN}" \
    "ModelArtifactUri=${MODEL_ARTIFACT_URI}" \
    "AsyncBucket=${ASYNC_BUCKET}" \
    "AsyncTopicArn=${ASYNC_TOPIC_ARN}"

deploy_stack churnguard-mme "$INFRA_DIR/06-mme.yaml" \
  --parameter-overrides \
    "ExecutionRoleArn=${EXECUTION_ROLE_ARN}" \
    "ModelsBucket=${MODELS_BUCKET}" \
    "ImageUri=${MME_IMAGE_URI}"

deploy_stack churnguard-events "$INFRA_DIR/07-events.yaml" \
  --parameter-overrides \
    "AsyncTopicArn=${ASYNC_TOPIC_ARN}"

log "deploy complete"
echo "All seven stacks deployed. Endpoints: churnguard-realtime, churnguard-serverless, churnguard-async, churnguard-mme."
echo "Model artifact: $MODEL_ARTIFACT_URI"
echo "MME image: $MME_IMAGE_URI"
