"""Multi-model endpoint routes."""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends

from ..aws import boto_client
from ..config import Config
from ..deps import ensure_endpoint_in_service, get_config, get_schema
from ..errors import ApiError
from ..inference import FeatureSchema, predictions_from_body, records_to_csv
from ..models import MultiModelRequest

log = logging.getLogger("churnguard.hosting")

router = APIRouter(prefix="/api", tags=["hosting"])


@router.post("/hosting/multimodel")
def multimodel_invoke(
    req: MultiModelRequest,
    cfg: Config = Depends(get_config),
    schema: FeatureSchema = Depends(get_schema),
):
    csv_body = records_to_csv([req.features.model_dump()], schema.columns())
    runtime = boto_client("sagemaker-runtime")
    sm = boto_client("sagemaker")
    ensure_endpoint_in_service(sm, cfg.mme_endpoint)

    start = time.perf_counter()
    resp = runtime.invoke_endpoint(
        EndpointName=cfg.mme_endpoint,
        ContentType="text/csv",
        Accept="text/csv",
        TargetModel=req.targetModel,
        Body=csv_body.encode("utf-8"),
    )
    latency_ms = int((time.perf_counter() - start) * 1000)
    predictions = predictions_from_body(resp["Body"].read().decode("utf-8"))
    if len(predictions) != 1:
        raise ApiError(
            502, "PREDICTION_COUNT_MISMATCH",
            f"mme endpoint returned {len(predictions)} predictions for 1 input row",
        )
    log.info("mme scored target=%s in %dms", req.targetModel, latency_ms)
    return {
        "endpoint": cfg.mme_endpoint,
        "targetModel": req.targetModel,
        "prediction": predictions[0],
        "latencyMs": latency_ms,
    }


@router.get("/hosting/multimodel/models")
def multimodel_models(cfg: Config = Depends(get_config)):
    s3 = boto_client("s3")
    resp = s3.list_objects_v2(Bucket=cfg.models_bucket, Prefix="mme/")
    models = []
    for obj in resp.get("Contents", []):
        key = obj["Key"]
        if key.endswith(".tar.gz"):
            models.append({
                "targetModel": key.split("/")[-1],
                "key": key,
                "size": obj.get("Size"),
            })
    return {"endpoint": cfg.mme_endpoint, "models": models}
