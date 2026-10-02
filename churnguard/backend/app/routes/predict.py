"""Synchronous + asynchronous scoring routes.

realtime / serverless invoke the endpoint directly; async writes the input CSV
to S3 and launches ``invoke_endpoint_async``. Every parser is the single pinned
``inference`` parser; no route fabricates a probability.
"""
from __future__ import annotations

import logging
import time
import uuid
from urllib.parse import urlparse

from botocore.exceptions import ClientError
from fastapi import APIRouter, Depends, Request

from ..aws import boto_client
from ..config import Config
from ..deps import ensure_endpoint_in_service, get_config, get_rowcounts, get_schema
from ..errors import ApiError
from ..inference import (
    FeatureSchema,
    parse_csv_predictions,
    predictions_from_body,
    records_to_csv,
    to_prediction,
)
from ..models import AsyncRequest, PredictRequest

log = logging.getLogger("churnguard.predict")

router = APIRouter(prefix="/api", tags=["predict"])

COLD_START_THRESHOLD_MS = 1500


def _records_or_400(req: PredictRequest):
    records = req.resolved_records()
    if not records:
        raise ApiError(400, "EMPTY_REQUEST", "provide 'features' or non-empty 'records'")
    return records


def _invoke(endpoint: str, csv_body: str):
    runtime = boto_client("sagemaker-runtime")
    sm = boto_client("sagemaker")
    ensure_endpoint_in_service(sm, endpoint)
    start = time.perf_counter()
    resp = runtime.invoke_endpoint(
        EndpointName=endpoint,
        ContentType="text/csv",
        Accept="text/csv",
        Body=csv_body.encode("utf-8"),
    )
    latency_ms = int((time.perf_counter() - start) * 1000)
    body = resp["Body"].read().decode("utf-8")
    return body, latency_ms


@router.post("/predict/realtime")
def predict_realtime(
    req: PredictRequest,
    cfg: Config = Depends(get_config),
    schema: FeatureSchema = Depends(get_schema),
):
    records = _records_or_400(req)
    csv_body = records_to_csv([r.model_dump() for r in records], schema.columns())
    body, latency_ms = _invoke(cfg.realtime_endpoint, csv_body)
    predictions = predictions_from_body(body)
    _assert_count(predictions, records, cfg.realtime_endpoint)
    log.info("realtime scored %d records in %dms", len(records), latency_ms)
    return {
        "endpoint": cfg.realtime_endpoint,
        "predictions": predictions,
        "latencyMs": latency_ms,
    }


@router.post("/predict/serverless")
def predict_serverless(
    req: PredictRequest,
    cfg: Config = Depends(get_config),
    schema: FeatureSchema = Depends(get_schema),
):
    records = _records_or_400(req)
    csv_body = records_to_csv([r.model_dump() for r in records], schema.columns())
    body, latency_ms = _invoke(cfg.serverless_endpoint, csv_body)
    predictions = predictions_from_body(body)
    _assert_count(predictions, records, cfg.serverless_endpoint)
    log.info("serverless scored %d records in %dms", len(records), latency_ms)
    # SageMaker emits NO cold-start signal for serverless. We report measured
    # latency plus a clearly-derived heuristic, never a measured coldStart bool.
    return {
        "endpoint": cfg.serverless_endpoint,
        "predictions": predictions,
        "latencyMs": latency_ms,
        "coldStartLikely": latency_ms > COLD_START_THRESHOLD_MS,
        "coldStartThresholdMs": COLD_START_THRESHOLD_MS,
    }


def _assert_count(predictions, records, endpoint):
    if len(predictions) != len(records):
        # The container returned a count that doesn't match input rows: surface
        # the real mismatch rather than guessing which row maps to which score.
        raise ApiError(
            502,
            "PREDICTION_COUNT_MISMATCH",
            f"endpoint {endpoint} returned {len(predictions)} predictions for "
            f"{len(records)} input rows",
        )


@router.post("/predict/async")
def predict_async(
    req: AsyncRequest,
    request: Request,
    cfg: Config = Depends(get_config),
    schema: FeatureSchema = Depends(get_schema),
):
    records = req.records
    csv_body = records_to_csv([r.model_dump() for r in records], schema.columns())
    inference_id = str(uuid.uuid4())
    input_key = f"input/{inference_id}.csv"

    s3 = boto_client("s3")
    s3.put_object(Bucket=cfg.async_bucket, Key=input_key, Body=csv_body.encode("utf-8"))
    input_location = f"s3://{cfg.async_bucket}/{input_key}"

    runtime = boto_client("sagemaker-runtime")
    sm = boto_client("sagemaker")
    ensure_endpoint_in_service(sm, cfg.async_endpoint)
    # Pass our InferenceId so SageMaker names the output deterministically under
    # the endpoint's configured S3OutputPath: <S3OutputPath>/<InferenceId>.out.
    # The response's OutputLocation is authoritative, so we use that for the
    # result key rather than reconstructing it.
    invoke_resp = runtime.invoke_endpoint_async(
        EndpointName=cfg.async_endpoint,
        InputLocation=input_location,
        ContentType="text/csv",
        Accept="text/csv",
        InferenceId=inference_id,
    )
    output_location = invoke_resp["OutputLocation"]

    # Record the row count keyed by BOTH our inference id and the real output
    # object name so the result route (which keys off the output key) can gate
    # Completed on the row count.
    rowcounts = get_rowcounts(request)
    rowcounts[inference_id] = len(records)
    out_name = output_location.rstrip("/").split("/")[-1]
    if out_name.endswith(".out"):
        out_name = out_name[: -len(".out")]
    rowcounts[out_name] = len(records)
    log.info("async submitted %d rows id=%s out=%s", len(records), inference_id, output_location)
    return {
        "endpoint": cfg.async_endpoint,
        "inferenceId": inference_id,
        "outputLocation": output_location,
        "inputLocation": input_location,
        "snsTopicArn": cfg.sns_topic_arn,
        "rowCount": len(records),
        "status": "InProgress",
    }


def _parse_s3_uri(uri: str):
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ApiError(400, "INVALID_S3_URI", f"not an s3:// uri: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def _recorded_rowcount(request: Request, cfg: Config, output_location: str, bucket: str, key: str):
    """Resolve the recorded input row count, re-deriving from the input CSV if
    the in-memory entry was lost (e.g. after a backend restart)."""
    inference_id = key.split("/")[-1]
    if inference_id.endswith(".out"):
        inference_id = inference_id[: -len(".out")]
    rowcounts = get_rowcounts(request)
    if inference_id in rowcounts:
        return rowcounts[inference_id]
    # Re-derive from the input object we wrote.
    s3 = boto_client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=f"input/{inference_id}.csv")
    except ClientError:
        return None
    body = obj["Body"].read().decode("utf-8").strip()
    count = len([ln for ln in body.splitlines() if ln.strip()])
    rowcounts[inference_id] = count
    return count


@router.get("/predict/async/result")
def predict_async_result(
    outputLocation: str,
    request: Request,
    cfg: Config = Depends(get_config),
):
    bucket, key = _parse_s3_uri(outputLocation)
    s3 = boto_client("s3")

    # 1. Check <outputLocation>.out.failure FIRST so a failed job that also left
    #    a partial/empty .out is reported Failed, never Completed.
    failure_key = f"{key}.failure"
    try:
        failure = s3.get_object(Bucket=bucket, Key=failure_key)
        reason = failure["Body"].read().decode("utf-8")
        log.error("async job failed id=%s: %s", key, reason)
        return {"status": "Failed", "reason": reason}
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") not in ("NoSuchKey", "404"):
            raise

    # 2. Else check <outputLocation>.out. Absent -> InProgress.
    try:
        out = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return {"status": "InProgress"}
        raise

    body = out["Body"].read().decode("utf-8")
    tokens = parse_csv_predictions(body)

    # 3. Gate Completed on row-count match. A short/empty parse stays InProgress.
    rows = _recorded_rowcount(request, cfg, outputLocation, bucket, key)
    if rows is not None and len(tokens) != rows:
        log.info("async out present but %d tokens != %s rows -> InProgress", len(tokens), rows)
        return {"status": "InProgress"}
    if rows is None and not tokens:
        return {"status": "InProgress"}

    predictions = [to_prediction(p) for p in tokens]
    return {"status": "Completed", "predictions": predictions}
