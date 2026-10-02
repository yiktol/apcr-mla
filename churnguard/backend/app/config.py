"""Runtime configuration resolution.

Resolution order for every value (first hit wins):
  1. environment variable override (deploy-time escape hatch)
  2. SSM Parameter Store (published by the foundation stack)
  3. the fixed design default (so the backend boots even before SSM is seeded)

SSM is read once at startup; failures to read SSM are non-fatal — the design
defaults are authoritative resource names, so a missing/unreachable parameter
simply falls through to the default. Nothing here fabricates a prediction; it
only resolves which real resources the routes address.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict

from botocore.exceptions import BotoCoreError, ClientError

from .aws import AWS_REGION, boto_client

log = logging.getLogger("churnguard.config")

ACCOUNT_ID = "875692608981"

# Fixed design defaults (see docs/design.md "Naming conventions").
DEFAULT_DATA_BUCKET = f"churnguard-data-{ACCOUNT_ID}-{AWS_REGION}"
DEFAULT_MODELS_BUCKET = f"churnguard-models-{ACCOUNT_ID}-{AWS_REGION}"
DEFAULT_ASYNC_BUCKET = f"churnguard-async-{ACCOUNT_ID}-{AWS_REGION}"
DEFAULT_BATCH_BUCKET = f"churnguard-batch-{ACCOUNT_ID}-{AWS_REGION}"

DEFAULT_REALTIME_ENDPOINT = "churnguard-realtime"
DEFAULT_SERVERLESS_ENDPOINT = "churnguard-serverless"
DEFAULT_ASYNC_ENDPOINT = "churnguard-async"
DEFAULT_MME_ENDPOINT = "churnguard-mme"

DEFAULT_SNS_TOPIC_ARN = (
    f"arn:aws:sns:{AWS_REGION}:{ACCOUNT_ID}:churnguard-async-notifications"
)
DEFAULT_BATCH_MODEL_NAME = "churnguard-realtime-model"

DEFAULT_MODEL_PACKAGE_GROUP = "churnguard-churn"
DEFAULT_PIPELINE_NAME = "churnguard-pipeline"
DEFAULT_EVENTS_LOG_GROUP = "/churnguard/events"
DEFAULT_ROLLBACK_ALARM = "churnguard-realtime-ModelError"

# (env var override, SSM parameter name, default value)
_SPEC = {
    "data_bucket": ("CHURNGUARD_DATA_BUCKET", "/churnguard/data-bucket", DEFAULT_DATA_BUCKET),
    "models_bucket": ("CHURNGUARD_MODELS_BUCKET", "/churnguard/models-bucket", DEFAULT_MODELS_BUCKET),
    "async_bucket": ("CHURNGUARD_ASYNC_BUCKET", "/churnguard/async-bucket", DEFAULT_ASYNC_BUCKET),
    "batch_bucket": ("CHURNGUARD_BATCH_BUCKET", "/churnguard/batch-bucket", DEFAULT_BATCH_BUCKET),
    "realtime_endpoint": ("CHURNGUARD_REALTIME_ENDPOINT", "/churnguard/realtime-endpoint", DEFAULT_REALTIME_ENDPOINT),
    "serverless_endpoint": ("CHURNGUARD_SERVERLESS_ENDPOINT", "/churnguard/serverless-endpoint", DEFAULT_SERVERLESS_ENDPOINT),
    "async_endpoint": ("CHURNGUARD_ASYNC_ENDPOINT", "/churnguard/async-endpoint", DEFAULT_ASYNC_ENDPOINT),
    "mme_endpoint": ("CHURNGUARD_MME_ENDPOINT", "/churnguard/mme-endpoint", DEFAULT_MME_ENDPOINT),
    "sns_topic_arn": ("CHURNGUARD_SNS_TOPIC_ARN", "/churnguard/async-topic-arn", DEFAULT_SNS_TOPIC_ARN),
    "batch_model_name": ("CHURNGUARD_BATCH_MODEL_NAME", "/churnguard/batch-model-name", DEFAULT_BATCH_MODEL_NAME),
    "model_package_group": ("CHURNGUARD_MODEL_PACKAGE_GROUP", "/churnguard/model-package-group", DEFAULT_MODEL_PACKAGE_GROUP),
    "pipeline_name": ("CHURNGUARD_PIPELINE_NAME", "/churnguard/pipeline-name", DEFAULT_PIPELINE_NAME),
    "events_log_group": ("CHURNGUARD_EVENTS_LOG_GROUP", "/churnguard/events-log-group", DEFAULT_EVENTS_LOG_GROUP),
    "rollback_alarm": ("CHURNGUARD_ROLLBACK_ALARM", "/churnguard/rollback-alarm", DEFAULT_ROLLBACK_ALARM),
}


@dataclass
class Config:
    region: str = AWS_REGION
    data_bucket: str = DEFAULT_DATA_BUCKET
    models_bucket: str = DEFAULT_MODELS_BUCKET
    async_bucket: str = DEFAULT_ASYNC_BUCKET
    batch_bucket: str = DEFAULT_BATCH_BUCKET
    realtime_endpoint: str = DEFAULT_REALTIME_ENDPOINT
    serverless_endpoint: str = DEFAULT_SERVERLESS_ENDPOINT
    async_endpoint: str = DEFAULT_ASYNC_ENDPOINT
    mme_endpoint: str = DEFAULT_MME_ENDPOINT
    sns_topic_arn: str = DEFAULT_SNS_TOPIC_ARN
    batch_model_name: str = DEFAULT_BATCH_MODEL_NAME
    model_package_group: str = DEFAULT_MODEL_PACKAGE_GROUP
    pipeline_name: str = DEFAULT_PIPELINE_NAME
    events_log_group: str = DEFAULT_EVENTS_LOG_GROUP
    rollback_alarm: str = DEFAULT_ROLLBACK_ALARM
    feature_columns_key: str = "processed/feature_columns.json"

    def endpoint_names(self) -> Dict[str, str]:
        return {
            "realtime": self.realtime_endpoint,
            "serverless": self.serverless_endpoint,
            "async": self.async_endpoint,
            "mme": self.mme_endpoint,
        }

    def bucket_names(self) -> Dict[str, str]:
        return {
            "data": self.data_bucket,
            "models": self.models_bucket,
            "async": self.async_bucket,
            "batch": self.batch_bucket,
        }


def _resolve_from_ssm() -> Dict[str, str]:
    """Best-effort read of every SSM parameter. Missing params are skipped."""
    resolved: Dict[str, str] = {}
    try:
        ssm = boto_client("ssm")
    except Exception as exc:  # pragma: no cover - client construction rarely fails
        log.warning("could not create SSM client, using defaults: %s", exc)
        return resolved

    names = [ssm_name for (_env, ssm_name, _default) in _SPEC.values()]
    try:
        page = ssm.get_parameters(Names=names)
    except (ClientError, BotoCoreError) as exc:
        log.warning("SSM get_parameters failed, using defaults/env only: %s", exc)
        return resolved

    by_name = {p["Name"]: p["Value"] for p in page.get("Parameters", [])}
    for field_name, (_env, ssm_name, _default) in _SPEC.items():
        if ssm_name in by_name:
            resolved[field_name] = by_name[ssm_name]
    return resolved


def load_config() -> Config:
    """Resolve configuration: env override > SSM > fixed default."""
    ssm_values = _resolve_from_ssm()
    kwargs: Dict[str, str] = {}
    for field_name, (env_var, _ssm_name, default) in _SPEC.items():
        env_val = os.environ.get(env_var)
        if env_val:
            kwargs[field_name] = env_val
        elif field_name in ssm_values:
            kwargs[field_name] = ssm_values[field_name]
        else:
            kwargs[field_name] = default
    return Config(**kwargs)
