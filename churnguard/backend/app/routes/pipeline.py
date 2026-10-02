"""SageMaker Pipeline, Model Registry, and EventBridge-log routes."""
from __future__ import annotations

import logging

from botocore.exceptions import ClientError
from fastapi import APIRouter, Depends

from ..aws import boto_client
from ..config import Config
from ..deps import get_config
from ..errors import ApiError
from ..models import PipelineRunRequest, RegistryApproveRequest

log = logging.getLogger("churnguard.pipeline")

router = APIRouter(prefix="/api", tags=["pipeline"])


@router.post("/pipeline/run")
def pipeline_run(_req: PipelineRunRequest | None = None, cfg: Config = Depends(get_config)):
    sm = boto_client("sagemaker")
    resp = sm.start_pipeline_execution(PipelineName=cfg.pipeline_name)
    arn = resp["PipelineExecutionArn"]
    log.info("pipeline execution started %s", arn)
    return {"pipelineExecutionArn": arn, "status": "Executing"}


@router.get("/pipeline/status")
def pipeline_status(arn: str, cfg: Config = Depends(get_config)):
    sm = boto_client("sagemaker")
    desc = sm.describe_pipeline_execution(PipelineExecutionArn=arn)
    steps_resp = sm.list_pipeline_execution_steps(PipelineExecutionArn=arn)
    steps = []
    for step in steps_resp.get("PipelineExecutionSteps", []):
        steps.append({
            "name": step.get("StepName"),
            "status": step.get("StepStatus"),
            "startTime": _iso(step.get("StartTime")),
            "endTime": _iso(step.get("EndTime")),
            "failureReason": step.get("FailureReason"),
        })
    return {
        "arn": arn,
        "status": desc.get("PipelineExecutionStatus"),
        "steps": steps,
    }


@router.get("/registry")
def registry(cfg: Config = Depends(get_config)):
    sm = boto_client("sagemaker")
    versions = []
    try:
        packages = sm.list_model_packages(ModelPackageGroupName=cfg.model_package_group)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in (
            "ValidationException", "ResourceNotFound", "ResourceNotFoundException",
        ):
            # Group not created yet -> legitimately empty, HTTP 200.
            return {"group": cfg.model_package_group, "versions": []}
        raise

    for pkg in packages.get("ModelPackageSummaryList", []):
        arn = pkg.get("ModelPackageArn")
        metrics = _metrics_for(sm, arn)
        versions.append({
            "arn": arn,
            "version": pkg.get("ModelPackageVersion"),
            "status": pkg.get("ModelPackageStatus"),
            "approvalStatus": pkg.get("ModelApprovalStatus"),
            "createdAt": _iso(pkg.get("CreationTime")),
            "metrics": metrics,
        })
    return {"group": cfg.model_package_group, "versions": versions}


def _metrics_for(sm, arn):
    try:
        desc = sm.describe_model_package(ModelPackageName=arn)
    except ClientError:
        return {}
    stats = (
        desc.get("ModelMetrics", {}).get("ModelQuality", {})
        .get("Statistics", {})
    )
    # Metrics attached as a PropertyFile are not inline; return what's present.
    out = {}
    for key in ("accuracy", "auc"):
        if key in stats:
            out[key] = stats[key]
    return out


@router.post("/registry/approve")
def registry_approve(req: RegistryApproveRequest, cfg: Config = Depends(get_config)):
    sm = boto_client("sagemaker")
    # Validate the ARN belongs to the churnguard-churn group.
    try:
        desc = sm.describe_model_package(ModelPackageName=req.modelPackageArn)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in ("ValidationException", "ResourceNotFound", "ResourceNotFoundException"):
            raise ApiError(400, "INVALID_PACKAGE",
                           f"model package not found: {req.modelPackageArn}")
        raise

    group = desc.get("ModelPackageGroupName")
    if group != cfg.model_package_group:
        raise ApiError(
            400, "INVALID_PACKAGE",
            f"model package {req.modelPackageArn} belongs to {group!r}, "
            f"not {cfg.model_package_group!r}",
        )

    if desc.get("ModelApprovalStatus") == "Approved":
        # Idempotent: already approved -> return current state.
        log.warning("model package already approved: %s", req.modelPackageArn)
        return {"modelPackageArn": req.modelPackageArn, "approvalStatus": "Approved"}

    sm.update_model_package(
        ModelPackageArn=req.modelPackageArn,
        ModelApprovalStatus="Approved",
    )
    log.info("approved model package %s", req.modelPackageArn)
    return {"modelPackageArn": req.modelPackageArn, "approvalStatus": "Approved"}


@router.get("/events/recent")
def events_recent(cfg: Config = Depends(get_config), limit: int = 20):
    logs = boto_client("logs")
    try:
        resp = logs.filter_log_events(
            logGroupName=cfg.events_log_group,
            limit=limit,
        )
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in (
            "ResourceNotFoundException", "ResourceNotFound",
        ):
            # Log group not created yet -> no events, HTTP 200.
            return {"events": []}
        raise

    events = []
    for ev in resp.get("events", []):
        message = ev.get("message", "")
        events.append({
            "timestamp": ev.get("timestamp"),
            "modelPackageArn": _extract_arn(message),
            "message": message,
        })
    return {"events": events}


def _extract_arn(message: str):
    for token in message.replace('"', " ").replace(",", " ").split():
        if token.startswith("arn:aws:sagemaker:") and ":model-package/" in token:
            return token
    return None


def _iso(value):
    if value is None:
        return None
    try:
        return value.isoformat()
    except AttributeError:
        return str(value)
