"""Shared route helpers: app state accessors and endpoint-readiness checks."""
from __future__ import annotations

from typing import Dict

from fastapi import Request

from .config import Config
from .errors import ApiError
from .inference import FeatureSchema


def get_config(request: Request) -> Config:
    return request.app.state.config


def get_schema(request: Request) -> FeatureSchema:
    return request.app.state.feature_schema


def get_rowcounts(request: Request) -> Dict[str, int]:
    """In-memory map {inferenceId|jobName -> input row count}.

    Backed by the input CSV in S3 (the backend can re-derive the count by
    reading the object's line count if this entry is missing after a restart).
    """
    return request.app.state.rowcounts


def ensure_endpoint_in_service(sm_client, endpoint_name: str) -> str:
    """Return the endpoint status, raising 409 ENDPOINT_NOT_READY if not InService.

    ``describe_endpoint`` is a real AWS call; a missing endpoint surfaces as the
    underlying ClientError (mapped to 404 by the error envelope).
    """
    resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
    status = resp.get("EndpointStatus", "Unknown")
    if status != "InService":
        raise ApiError(
            status_code=409,
            code="ENDPOINT_NOT_READY",
            message=f"endpoint {endpoint_name} is {status}, not InService",
            detail={"endpoint": endpoint_name, "status": status},
        )
    return status
