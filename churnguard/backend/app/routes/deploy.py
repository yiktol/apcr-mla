"""Blue/green deployment routes (slide-36 canary/linear guardrails).

Runtime resources use the distinct ``churnguard-bg-`` sub-prefix with
``<rev> = int(time.time())`` so repeated runs never collide and teardown can
sweep them without touching CloudFormation-owned resources.
"""
from __future__ import annotations

import logging
import time

from botocore.exceptions import ClientError
from fastapi import APIRouter, Depends

from ..aws import boto_client
from ..config import Config
from ..deps import get_config
from ..errors import ApiError
from ..models import BlueGreenRequest

log = logging.getLogger("churnguard.deploy")

router = APIRouter(prefix="/api", tags=["deploy"])


def _approved_package(sm, arn: str, cfg: Config):
    try:
        desc = sm.describe_model_package(ModelPackageName=arn)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in ("ValidationException", "ResourceNotFound", "ResourceNotFoundException"):
            raise ApiError(400, "INVALID_PACKAGE", f"model package not found: {arn}")
        raise
    if desc.get("ModelPackageGroupName") != cfg.model_package_group:
        raise ApiError(400, "INVALID_PACKAGE",
                       f"model package {arn} is not in {cfg.model_package_group}")
    if desc.get("ModelApprovalStatus") != "Approved":
        raise ApiError(409, "NOT_APPROVED",
                       f"model package {arn} is not Approved")
    return desc


@router.post("/deploy/bluegreen")
def deploy_bluegreen(req: BlueGreenRequest, cfg: Config = Depends(get_config)):
    sm = boto_client("sagemaker")
    _approved_package(sm, req.modelPackageArn, cfg)

    rev = int(time.time())
    model_name = f"churnguard-bg-model-{rev}"
    config_name = f"churnguard-bg-config-{rev}"

    # Create a model from the approved package (SageMaker resolves the artifact
    # + image from the package's inference spec).
    sm.create_model(
        ModelName=model_name,
        Containers=[{"ModelPackageName": req.modelPackageArn}],
        ExecutionRoleArn=_execution_role(cfg),
    )
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            "VariantName": "AllTraffic",
            "ModelName": model_name,
            "InitialInstanceCount": 3,
            "InstanceType": "ml.m5.large",
        }],
    )

    if req.mode == "canary":
        routing = {
            "Type": "CANARY",
            "CanarySize": {"Type": "CAPACITY_PERCENT", "Value": req.canaryPercent},
            "WaitIntervalInSeconds": req.bakeTimeSeconds,
        }
    else:
        routing = {
            "Type": "LINEAR",
            "LinearStepSize": {"Type": "CAPACITY_PERCENT", "Value": req.linearStepPercent},
            "WaitIntervalInSeconds": req.bakeTimeSeconds,
        }

    sm.update_endpoint(
        EndpointName=cfg.realtime_endpoint,
        EndpointConfigName=config_name,
        DeploymentConfig={
            "BlueGreenUpdatePolicy": {
                "TrafficRoutingConfiguration": routing,
                "TerminationWaitInSeconds": 300,
                "MaximumExecutionTimeoutInSeconds": 1800,
            },
            "AutoRollbackConfiguration": {
                "Alarms": [{"AlarmName": cfg.rollback_alarm}],
            },
        },
    )
    log.info("blue/green update started rev=%d mode=%s", rev, req.mode)
    return {
        "endpoint": cfg.realtime_endpoint,
        "newConfig": config_name,
        "newModel": model_name,
        "mode": req.mode,
        "rev": rev,
        "status": "Updating",
    }


def _execution_role(cfg: Config) -> str:
    # The SageMaker execution role is fixed for this account (see design).
    import os
    return os.environ.get(
        "CHURNGUARD_EXECUTION_ROLE_ARN",
        "arn:aws:iam::875692608981:role/AmazonSageMaker-ExecutionRole",
    )


@router.get("/deploy/bluegreen/status")
def deploy_bluegreen_status(cfg: Config = Depends(get_config)):
    sm = boto_client("sagemaker")
    desc = sm.describe_endpoint(EndpointName=cfg.realtime_endpoint)
    return {
        "status": desc.get("EndpointStatus"),
        "endpointConfig": desc.get("EndpointConfigName"),
        "lastDeploymentStatus": desc.get("LastDeploymentConfig") and "present" or None,
        "pendingDeploymentSummary": _summarize_pending(desc.get("PendingDeploymentSummary")),
    }


def _summarize_pending(pending):
    if not pending:
        return None
    variants = []
    for v in pending.get("ProductionVariants", []):
        variants.append({
            "variantName": v.get("VariantName"),
            "currentWeight": v.get("CurrentWeight"),
            "desiredWeight": v.get("DesiredWeight"),
            "currentInstanceCount": v.get("CurrentInstanceCount"),
            "desiredInstanceCount": v.get("DesiredInstanceCount"),
            "variantStatus": [s.get("Status") for s in v.get("VariantStatus", [])],
        })
    return {"endpointConfigName": pending.get("EndpointConfigName"), "variants": variants}
