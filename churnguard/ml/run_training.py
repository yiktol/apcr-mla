#!/usr/bin/env python3
"""Baseline provisioning path: preprocess (5a) then train (5b), both REAL jobs.

5a. Run ml/scripts/preprocessing.py as a REAL SageMaker Processing job using the
    SDK-resolved SKLearn processing image. Writes processed/{train,validation,
    test}/ splits + processed/feature_columns.json to the data bucket. This is
    byte-for-byte the same preprocessing the pipeline's DataProcessing step runs.

5b. Run a standalone built-in XGBoost (1.7-1 literal) training job on the
    processed train + validation splits with the same hyperparameters, writing
    model.tar.gz.

5a MUST complete and be awaited before 5b starts. The resulting artifact S3 URI
is written to SSM /churnguard/model-artifact-uri.

All region-pinned to us-east-1. No fake data: both are real SageMaker jobs.
"""
import argparse
import os
import sys

import boto3
import sagemaker
from sagemaker import image_uris
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor

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

HYPERPARAMETERS = {
    "objective": "binary:logistic",
    "num_round": "150",
    "max_depth": "5",
    "eta": "0.2",
    "subsample": "0.8",
    "eval_metric": "auc",
}

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(HERE, "scripts")


def _make_session():
    boto_session = boto3.Session(region_name=REGION)
    return sagemaker.Session(boto_session=boto_session, default_bucket=DATA_BUCKET)


def run_preprocessing(session, raw_uri, processed_base):
    """5a: real SageMaker Processing job. Blocks until the job completes."""
    # SDK-resolve the SKLearn image (NOT a literal 1.2-1 tag). The sklearn
    # framework has no separate `processing` scope in the image_uris registry;
    # SageMaker Processing runs the same `sagemaker-scikit-learn` image resolved
    # via the training scope (platform/python-qualified, e.g. 1.2-1-cpu-py3).
    sklearn_image = image_uris.retrieve(
        framework="sklearn",
        region=REGION,
        version="1.2-1",
        image_scope="training",
        instance_type=INSTANCE_TYPE,
    )
    processor = ScriptProcessor(
        image_uri=sklearn_image,
        command=["python3"],
        role=EXECUTION_ROLE,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        sagemaker_session=session,
        base_job_name="churnguard-baseline-processing",
    )

    processor.run(
        code=os.path.join(SCRIPTS_DIR, "preprocessing.py"),
        inputs=[
            ProcessingInput(source=raw_uri, destination="/opt/ml/processing/input"),
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/output/train",
                destination=f"{processed_base}/train",
            ),
            ProcessingOutput(
                output_name="validation",
                source="/opt/ml/processing/output/validation",
                destination=f"{processed_base}/validation",
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/output/test",
                destination=f"{processed_base}/test",
            ),
            ProcessingOutput(
                output_name="processed",
                source="/opt/ml/processing/output/processed",
                destination=f"{processed_base}/processed",
            ),
        ],
        arguments=[
            "--input-dir", "/opt/ml/processing/input",
            "--output-dir", "/opt/ml/processing/output",
        ],
        wait=True,  # 5a MUST complete before 5b.
        logs=True,
    )
    print("5a preprocessing complete")


def run_training(session, processed_base):
    """5b: real built-in XGBoost training job. Returns the model.tar.gz S3 URI."""
    estimator = Estimator(
        image_uri=XGBOOST_IMAGE,
        role=EXECUTION_ROLE,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        output_path=f"s3://{MODELS_BUCKET}/training",
        sagemaker_session=session,
        base_job_name="churnguard-baseline-training",
    )
    estimator.set_hyperparameters(**HYPERPARAMETERS)

    estimator.fit(
        inputs={
            "train": TrainingInput(
                s3_data=f"{processed_base}/train",
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=f"{processed_base}/validation",
                content_type="text/csv",
            ),
        },
        wait=True,
        logs=True,
    )
    artifact_uri = estimator.model_data
    print(f"5b training complete -> {artifact_uri}")
    return artifact_uri


def write_ssm(artifact_uri):
    ssm = boto3.client("ssm", region_name=REGION)
    ssm.put_parameter(
        Name=SSM_ARTIFACT_PARAM,
        Value=artifact_uri,
        Type="String",
        Overwrite=True,
    )
    print(f"wrote {SSM_ARTIFACT_PARAM} = {artifact_uri}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Baseline preprocess + train path")
    parser.add_argument(
        "--raw-uri",
        default=f"s3://{DATA_BUCKET}/raw/Telco-Customer-Churn.csv",
    )
    parser.add_argument(
        "--processed-base",
        default=f"s3://{DATA_BUCKET}/processed",
    )
    args = parser.parse_args(argv)

    session = _make_session()

    # 5a MUST complete before 5b (wait=True above enforces the ordering).
    run_preprocessing(session, args.raw_uri, args.processed_base)
    artifact_uri = run_training(session, args.processed_base)
    write_ssm(artifact_uri)

    print("baseline path complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
