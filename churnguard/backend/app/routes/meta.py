"""Health and architecture metadata routes."""
from __future__ import annotations

import logging

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import APIRouter, Depends, Request

from ..aws import AWS_REGION, boto_client
from ..config import Config
from ..deps import get_config, get_schema
from ..inference import FeatureSchema

log = logging.getLogger("churnguard.meta")

router = APIRouter(prefix="/api", tags=["meta"])


@router.get("/health")
def health(
    request: Request,
    cfg: Config = Depends(get_config),
    schema: FeatureSchema = Depends(get_schema),
):
    sm = boto_client("sagemaker")
    endpoint_status = {}
    for label, name in cfg.endpoint_names().items():
        endpoint_status[label] = _describe_status(sm, name)

    return {
        "region": AWS_REGION,
        "endpoints": cfg.endpoint_names(),
        "buckets": cfg.bucket_names(),
        "snsTopicArn": cfg.sns_topic_arn,
        "featureSchemaLoaded": schema.loaded,
        "endpointStatus": endpoint_status,
    }


def _describe_status(sm, name):
    try:
        desc = sm.describe_endpoint(EndpointName=name)
        return desc.get("EndpointStatus", "Unknown")
    except (ClientError, BotoCoreError) as exc:
        code = getattr(exc, "response", {}).get("Error", {}).get("Code", "") if hasattr(exc, "response") else ""
        if code in ("ValidationException", "ResourceNotFound", "ResourceNotFoundException"):
            return "NotFound"
        log.warning("describe_endpoint failed for %s: %s", name, exc)
        return "Error"


@router.get("/architecture")
def architecture():
    """Static descriptive metadata for the landing-page diagram (not simulated)."""
    nodes = [
        {"id": "dataset", "label": "Telco Churn Dataset", "type": "data"},
        {"id": "pipeline", "label": "SageMaker Pipeline", "type": "pipeline"},
        {"id": "registry", "label": "Model Registry (churnguard-churn)", "type": "registry"},
        {"id": "events", "label": "EventBridge + Lambda", "type": "events"},
        {"id": "realtime", "label": "Real-time Endpoint", "type": "endpoint"},
        {"id": "serverless", "label": "Serverless Endpoint", "type": "endpoint"},
        {"id": "async", "label": "Async Endpoint + SNS", "type": "endpoint"},
        {"id": "batch", "label": "Batch Transform", "type": "batch"},
        {"id": "mme", "label": "Multi-Model Endpoint", "type": "endpoint"},
        {"id": "bluegreen", "label": "Blue/Green Deploy", "type": "deploy"},
    ]
    edges = [
        {"from": "dataset", "to": "pipeline"},
        {"from": "pipeline", "to": "registry"},
        {"from": "registry", "to": "events"},
        {"from": "registry", "to": "bluegreen"},
        {"from": "bluegreen", "to": "realtime"},
        {"from": "dataset", "to": "realtime"},
        {"from": "dataset", "to": "serverless"},
        {"from": "dataset", "to": "async"},
        {"from": "dataset", "to": "batch"},
        {"from": "dataset", "to": "mme"},
    ]
    return {"nodes": nodes, "edges": edges}
