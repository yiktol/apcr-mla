"""Batch transform routes.

The default path strips the label column 0 from the headerless, label-first
test split before writing ``batch.csv`` so the transform job receives
label-free, feature-ordered rows that match the pinned inference contract.
"""
from __future__ import annotations

import logging
import time
from urllib.parse import urlparse

from botocore.exceptions import ClientError
from fastapi import APIRouter, Body, Depends, Request

from ..aws import boto_client
from ..config import Config
from ..deps import get_config, get_rowcounts
from ..errors import ApiError
from ..inference import parse_csv_predictions, to_prediction
from ..models import BatchRequest

log = logging.getLogger("churnguard.batch")

router = APIRouter(prefix="/api", tags=["batch"])

TEST_SPLIT_KEY = "processed/test/test.csv"


def _strip_label_column(csv_text: str) -> tuple[str, int]:
    """Drop column 0 (the label) from each row. Return (stripped_csv, row_count)."""
    out_lines = []
    for line in csv_text.splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        out_lines.append(",".join(parts[1:]))
    return "\n".join(out_lines), len(out_lines)


def _count_rows(csv_text: str) -> int:
    return len([ln for ln in csv_text.splitlines() if ln.strip()])


@router.post("/batch")
def start_batch(
    request: Request,
    req: BatchRequest | None = Body(default=None),
    cfg: Config = Depends(get_config),
):
    s3 = boto_client("s3")
    input_uri = req.inputS3Uri if req else None

    if input_uri:
        # Caller-supplied input must ALREADY be label-free, feature-ordered,
        # headerless CSV. We do not strip or reorder it; we only read its row
        # count to assert len(predictions) == rows on fetch.
        in_bucket, in_key = _parse_s3_uri(input_uri)
        obj = s3.get_object(Bucket=in_bucket, Key=in_key)
        rows = _count_rows(obj["Body"].read().decode("utf-8"))
        batch_input_uri = input_uri
    else:
        # Default path: read the label-first test split and strip column 0.
        try:
            test_obj = s3.get_object(Bucket=cfg.data_bucket, Key=TEST_SPLIT_KEY)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                raise ApiError(
                    409, "MODEL_NOT_READY",
                    f"test split not present at s3://{cfg.data_bucket}/{TEST_SPLIT_KEY}",
                )
            raise
        stripped, rows = _strip_label_column(test_obj["Body"].read().decode("utf-8"))
        batch_input_key = "input/batch.csv"
        s3.put_object(Bucket=cfg.batch_bucket, Key=batch_input_key,
                      Body=stripped.encode("utf-8"))
        batch_input_uri = f"s3://{cfg.batch_bucket}/{batch_input_key}"

    job_name = f"churnguard-batch-{int(time.time())}"
    output_uri = f"s3://{cfg.batch_bucket}/output/"

    sm = boto_client("sagemaker")
    sm.create_transform_job(
        TransformJobName=job_name,
        ModelName=cfg.batch_model_name,
        TransformInput={
            "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": batch_input_uri}},
            "ContentType": "text/csv",
            "SplitType": "Line",
        },
        TransformOutput={"S3OutputPath": output_uri, "Accept": "text/csv", "AssembleWith": "Line"},
        TransformResources={"InstanceType": "ml.m5.large", "InstanceCount": 1},
    )

    get_rowcounts(request)[job_name] = rows
    log.info("batch job %s started rows=%d", job_name, rows)
    return {"jobName": job_name, "status": "InProgress", "outputLocation": output_uri}


@router.get("/batch/{jobName}")
def batch_status(
    jobName: str,
    request: Request,
    cfg: Config = Depends(get_config),
):
    sm = boto_client("sagemaker")
    desc = sm.describe_transform_job(TransformJobName=jobName)
    status = desc.get("TransformJobStatus", "Unknown")
    output_uri = desc.get("TransformOutput", {}).get("S3OutputPath")
    result = {"jobName": jobName, "status": status, "outputLocation": output_uri}

    if status == "Failed":
        result["failureReason"] = desc.get("FailureReason", "unknown")
        return result

    rows = get_rowcounts(request).get(jobName)
    if rows is not None:
        result["rowCount"] = rows

    if status == "Completed":
        predictions = _fetch_batch_predictions(cfg, desc, rows, jobName)
        result["predictions"] = predictions
    return result


def _fetch_batch_predictions(cfg: Config, desc, rows, job_name):
    """Fetch and parse the .out; assert len == recorded rows else real error."""
    input_uri = (
        desc.get("TransformInput", {}).get("DataSource", {})
        .get("S3DataSource", {}).get("S3Uri", "")
    )
    output_path = desc.get("TransformOutput", {}).get("S3OutputPath", "")
    out_bucket, out_prefix = _parse_s3_uri(output_path)
    # Batch Transform writes <input-filename>.out under the output prefix.
    input_name = input_uri.rstrip("/").split("/")[-1]
    out_key = f"{out_prefix.rstrip('/')}/{input_name}.out"

    s3 = boto_client("s3")
    obj = s3.get_object(Bucket=out_bucket, Key=out_key)
    tokens = parse_csv_predictions(obj["Body"].read().decode("utf-8"))

    if rows is not None and len(tokens) != rows:
        raise ApiError(
            502, "PREDICTION_COUNT_MISMATCH",
            f"batch job {job_name} completed but produced {len(tokens)} "
            f"predictions for {rows} input rows",
        )
    return [to_prediction(p) for p in tokens]


def _parse_s3_uri(uri: str):
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ApiError(400, "INVALID_S3_URI", f"not an s3:// uri: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")
