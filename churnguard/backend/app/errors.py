"""Shared error envelope and FastAPI exception handlers.

Every error leaves the backend in the shape::

    {"error": {"code": str, "message": str, "aws_request_id": str|null, "detail": any}}

A ``botocore.exceptions.ClientError`` is mapped to an HTTP status by its AWS
error code; the real AWS request id is surfaced so a failure can be traced in
CloudTrail. Input-validation failures are logged WARNING; everything else that
reaches the AWS layer is logged ERROR with the request id.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from botocore.exceptions import ClientError
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .inference import ModelNotReadyError

log = logging.getLogger("churnguard.errors")


class ApiError(Exception):
    """Explicit, route-raised error with a pinned code + HTTP status."""

    def __init__(self, status_code: int, code: str, message: str, detail: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.detail = detail


def envelope(code: str, message: str, aws_request_id: Optional[str] = None, detail: Any = None) -> dict:
    return {
        "error": {
            "code": code,
            "message": message,
            "aws_request_id": aws_request_id,
            "detail": detail,
        }
    }


def _request_id(exc: ClientError) -> Optional[str]:
    meta = exc.response.get("ResponseMetadata", {}) if exc.response else {}
    return meta.get("RequestId")


def _map_client_error(exc: ClientError) -> tuple[int, str]:
    """Map an AWS error code to (http_status, envelope_code)."""
    code = exc.response.get("Error", {}).get("Code", "") if exc.response else ""
    if code in ("ValidationException", "ModelError"):
        return 400, code
    if code in ("ResourceNotFound", "ResourceNotFoundException", "ValidationError"):
        return 404, code
    if code in ("ThrottlingException", "Throttling", "TooManyRequestsException"):
        return 429, code
    return 502, code or "AWSError"


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _handle_api_error(_request: Request, exc: ApiError):
        # Route-raised business errors (409 ENDPOINT_NOT_READY, 409 NOT_APPROVED,
        # 400 INVALID_PACKAGE, ...). These are recoverable/expected -> WARNING.
        log.warning("api error %s: %s", exc.code, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content=envelope(exc.code, exc.message, detail=exc.detail),
        )

    @app.exception_handler(ModelNotReadyError)
    async def _handle_model_not_ready(_request: Request, exc: ModelNotReadyError):
        log.warning("model not ready: %s", exc)
        return JSONResponse(
            status_code=409,
            content=envelope("MODEL_NOT_READY", str(exc)),
        )

    @app.exception_handler(ClientError)
    async def _handle_client_error(_request: Request, exc: ClientError):
        status, code = _map_client_error(exc)
        request_id = _request_id(exc)
        message = exc.response.get("Error", {}).get("Message", str(exc)) if exc.response else str(exc)
        log.error("aws ClientError code=%s status=%s request_id=%s: %s",
                  code, status, request_id, message)
        return JSONResponse(
            status_code=status,
            content=envelope(code, message, aws_request_id=request_id),
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation(_request: Request, exc: RequestValidationError):
        # Input validation never makes an AWS call -> WARNING, HTTP 422.
        detail = _jsonable_errors(exc.errors())
        log.warning("input validation failed: %s", detail)
        return JSONResponse(
            status_code=422,
            content=envelope("VALIDATION_ERROR", "input validation failed", detail=detail),
        )


def _jsonable_errors(errors: Any) -> Any:
    """Strip non-JSON-serializable objects (e.g. ctx exceptions) from pydantic errors."""
    safe = []
    for err in errors:
        item = {k: v for k, v in err.items() if k != "ctx"}
        ctx = err.get("ctx")
        if ctx:
            item["ctx"] = {k: str(v) for k, v in ctx.items()}
        safe.append(item)
    return safe
