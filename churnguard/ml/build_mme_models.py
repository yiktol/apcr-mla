#!/usr/bin/env python3
"""Build the two multi-model-endpoint artifacts under the models bucket mme/ prefix.

churn-v1.tar.gz: a copy of the baseline training artifact (from SSM
                 /churnguard/model-artifact-uri).
churn-v2.tar.gz: a SECOND variant REALLY retrained with different
                 hyperparameters (max_depth=7, eta=0.1) on the same processed
                 splits, then uploaded.

No fake data: v2 is a real SageMaker XGBoost training job. All region-pinned to
us-east-1.
"""
import argparse
import sys
from urllib.parse import urlparse

import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput

REGION = "us-east-1"
ACCOUNT = "875692608981"
EXECUTION_ROLE = f"arn:aws:iam::{ACCOUNT}:role/AmazonSageMaker-ExecutionRole"
DATA_BUCKET = f"churnguard-data-{ACCOUNT}-{REGION}"
MODELS_BUCKET = f"churnguard-models-{ACCOUNT}-{REGION}"
INSTANCE_TYPE = "ml.m5.large"
SSM_ARTIFACT_PARAM = "/churnguard/model-artifact-uri"

XGBOOST_IMAGE = (
    f"683313688378.dkr.ecr.{REGION}.amazonaws.com/sagemaker-xgboost:1.7-1"
)

# v2 is deliberately different from the baseline (max_depth 5 / eta 0.2).
V2_HYPERPARAMETERS = {
    "objective": "binary:logistic",
    "num_round": "150",
    "max_depth": "7",
    "eta": "0.1",
    "subsample": "0.8",
    "eval_metric": "auc",
}

MME_V1_KEY = "mme/churn-v1.tar.gz"
MME_V2_KEY = "mme/churn-v2.tar.gz"


def _parse_s3(uri):
    parsed = urlparse(uri)
    return parsed.netloc, parsed.path.lstrip("/")


def read_baseline_artifact(ssm):
    resp = ssm.get_parameter(Name=SSM_ARTIFACT_PARAM)
    return resp["Parameter"]["Value"]


def copy_v1(s3, baseline_uri):
    """Copy the baseline artifact to mme/churn-v1.tar.gz."""
    src_bucket, src_key = _parse_s3(baseline_uri)
    s3.copy_object(
        Bucket=MODELS_BUCKET,
        Key=MME_V1_KEY,
        CopySource={"Bucket": src_bucket, "Key": src_key},
    )
    uri = f"s3://{MODELS_BUCKET}/{MME_V1_KEY}"
    print(f"copied baseline -> {uri}")
    return uri


def train_v2(session, processed_base):
    """Really retrain a second variant and return its model.tar.gz S3 URI."""
    estimator = Estimator(
        image_uri=XGBOOST_IMAGE,
        role=EXECUTION_ROLE,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        output_path=f"s3://{MODELS_BUCKET}/training-v2",
        sagemaker_session=session,
        base_job_name="churnguard-mme-v2-training",
    )
    estimator.set_hyperparameters(**V2_HYPERPARAMETERS)
    estimator.fit(
        inputs={
            "train": TrainingInput(
                s3_data=f"{processed_base}/train", content_type="text/csv"
            ),
            "validation": TrainingInput(
                s3_data=f"{processed_base}/validation", content_type="text/csv"
            ),
        },
        wait=True,
        logs=True,
    )
    print(f"v2 training complete -> {estimator.model_data}")
    return estimator.model_data


def copy_v2(s3, v2_artifact_uri):
    src_bucket, src_key = _parse_s3(v2_artifact_uri)
    s3.copy_object(
        Bucket=MODELS_BUCKET,
        Key=MME_V2_KEY,
        CopySource={"Bucket": src_bucket, "Key": src_key},
    )
    uri = f"s3://{MODELS_BUCKET}/{MME_V2_KEY}"
    print(f"uploaded v2 -> {uri}")
    return uri


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build MME v1/v2 artifacts")
    parser.add_argument(
        "--processed-base",
        default=f"s3://{DATA_BUCKET}/processed",
    )
    args = parser.parse_args(argv)

    boto_session = boto3.Session(region_name=REGION)
    session = sagemaker.Session(boto_session=boto_session, default_bucket=DATA_BUCKET)
    s3 = boto_session.client("s3", region_name=REGION)
    ssm = boto_session.client("ssm", region_name=REGION)

    baseline_uri = read_baseline_artifact(ssm)
    copy_v1(s3, baseline_uri)

    v2_artifact_uri = train_v2(session, args.processed_base)
    copy_v2(s3, v2_artifact_uri)

    print("MME artifacts ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
