#!/usr/bin/env python3
"""Author the slide-33 SageMaker Pipeline DAG and emit its definition JSON.

DAG (per the approved design):
  ProcessingStep "DataProcessing"  (SKLearn processor, SDK-resolved image)
    -> TrainingStep "ModelTraining" (built-in XGBoost 1.7-1 literal)
    -> ProcessingStep "ModelEvaluation" (PropertyFile "EvaluationReport")
    -> ConditionStep "CheckAccuracy" (JsonGet metrics.accuracy.value >= 0.78)
    -> RegisterModel "RegisterChurnModel" into group churnguard-churn
       with ModelApprovalStatus=PendingManualApproval

The SKLearn processing image is SDK-resolved via image_uris.retrieve (the tag is
NOT a bare 1.2-1 -- it is platform/python-qualified). The XGBoost 1.7-1 training
image tag is literal.

Writes the definition locally to infra/pipeline-definition.json for inspection.
With --dry-run it ONLY writes locally (no boto3 / no upload). Without --dry-run
it also uploads to s3://<data-bucket>/pipeline/pipeline-definition.json.

All region-pinned to us-east-1.
"""
import argparse
import json
import os
import sys

import boto3
import sagemaker
from sagemaker import image_uris
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.estimator import Estimator
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

REGION = "us-east-1"
ACCOUNT = "875692608981"
EXECUTION_ROLE = f"arn:aws:iam::{ACCOUNT}:role/AmazonSageMaker-ExecutionRole"
DATA_BUCKET = f"churnguard-data-{ACCOUNT}-{REGION}"
MODEL_PACKAGE_GROUP = "churnguard-churn"
PIPELINE_NAME = "churnguard-pipeline"

# XGBoost built-in image tag is LITERAL and used verbatim.
XGBOOST_IMAGE = (
    f"683313688378.dkr.ecr.{REGION}.amazonaws.com/sagemaker-xgboost:1.7-1"
)

ACCURACY_THRESHOLD = 0.78
INSTANCE_TYPE = "ml.m5.large"

HYPERPARAMETERS = {
    "objective": "binary:logistic",
    "num_round": "150",
    "max_depth": "5",
    "eta": "0.2",
    "subsample": "0.8",
    "eval_metric": "auc",
}

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
SCRIPTS_DIR = os.path.join(HERE, "scripts")
OUTPUT_JSON = os.path.join(REPO_ROOT, "infra", "pipeline-definition.json")

# The ProcessingSteps reference their entry-point scripts from S3 (not a local
# path) so authoring the definition is fully offline: a local `code=` path would
# make the SDK upload to the default bucket at definition time, which requires
# live credentials. deploy.sh / the non-dry-run path uploads these scripts to
# these exact keys before running the pipeline.
CODE_PREFIX = "code"
PREPROCESSING_CODE_URI = f"s3://{DATA_BUCKET}/{CODE_PREFIX}/preprocessing.py"
EVALUATION_CODE_URI = f"s3://{DATA_BUCKET}/{CODE_PREFIX}/evaluation.py"


def _resolve_sklearn_image(session):
    """SDK-resolve the SKLearn processing image URI (NOT a literal tag).

    SageMaker Processing with the SKLearn container uses the same ECR image as
    the SKLearn training/inference scope -- a single `sagemaker-scikit-learn`
    image runs the processing script. The `image_uris` registry does not expose
    a separate `processing` scope for the sklearn framework, so we resolve via
    the `training` scope. The point of the design (resolve, do NOT hard-code a
    bare `1.2-1` tag) is honored: this returns the platform/python-qualified tag
    (e.g. `1.2-1-cpu-py3`)."""
    return image_uris.retrieve(
        framework="sklearn",
        region=REGION,
        version="1.2-1",
        image_scope="training",
        instance_type=INSTANCE_TYPE,
    )


def build_pipeline(session):
    """Build and return the sagemaker.workflow Pipeline object."""
    raw_data_uri = ParameterString(
        name="RawDataUri",
        default_value=f"s3://{DATA_BUCKET}/raw/Telco-Customer-Churn.csv",
    )
    processed_uri = ParameterString(
        name="ProcessedUri",
        default_value=f"s3://{DATA_BUCKET}/processed",
    )

    sklearn_image = _resolve_sklearn_image(session)

    # --- Step 1: DataProcessing -------------------------------------------
    sklearn_processor = ScriptProcessor(
        image_uri=sklearn_image,
        command=["python3"],
        role=EXECUTION_ROLE,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        sagemaker_session=session,
        base_job_name="churnguard-processing",
    )

    processing_step = ProcessingStep(
        name="DataProcessing",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(
                source=raw_data_uri,
                destination="/opt/ml/processing/input",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/output/train",
                destination=f"s3://{DATA_BUCKET}/processed/train",
            ),
            ProcessingOutput(
                output_name="validation",
                source="/opt/ml/processing/output/validation",
                destination=f"s3://{DATA_BUCKET}/processed/validation",
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/output/test",
                destination=f"s3://{DATA_BUCKET}/processed/test",
            ),
            ProcessingOutput(
                output_name="processed",
                source="/opt/ml/processing/output/processed",
                destination=f"s3://{DATA_BUCKET}/processed/processed",
            ),
        ],
        code=PREPROCESSING_CODE_URI,
        job_arguments=[
            "--input-dir", "/opt/ml/processing/input",
            "--output-dir", "/opt/ml/processing/output",
        ],
    )

    # --- Step 2: ModelTraining --------------------------------------------
    estimator = Estimator(
        image_uri=XGBOOST_IMAGE,
        role=EXECUTION_ROLE,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        output_path=f"s3://{DATA_BUCKET}/pipeline/training-output",
        sagemaker_session=session,
        base_job_name="churnguard-training",
    )
    estimator.set_hyperparameters(**HYPERPARAMETERS)

    train_channel = TrainingInput(
        s3_data=processing_step.properties.ProcessingOutputConfig.Outputs[
            "train"
        ].S3Output.S3Uri,
        content_type="text/csv",
    )
    validation_channel = TrainingInput(
        s3_data=processing_step.properties.ProcessingOutputConfig.Outputs[
            "validation"
        ].S3Output.S3Uri,
        content_type="text/csv",
    )

    training_step = TrainingStep(
        name="ModelTraining",
        estimator=estimator,
        inputs={"train": train_channel, "validation": validation_channel},
    )

    # --- Step 3: ModelEvaluation ------------------------------------------
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    eval_processor = ScriptProcessor(
        image_uri=XGBOOST_IMAGE,
        command=["python3"],
        role=EXECUTION_ROLE,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        sagemaker_session=session,
        base_job_name="churnguard-evaluation",
    )

    evaluation_step = ProcessingStep(
        name="ModelEvaluation",
        processor=eval_processor,
        inputs=[
            ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=processing_step.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=f"s3://{DATA_BUCKET}/pipeline/evaluation",
            ),
        ],
        code=EVALUATION_CODE_URI,
        job_arguments=[
            "--model-dir", "/opt/ml/processing/model",
            "--test-dir", "/opt/ml/processing/test",
            "--output-dir", "/opt/ml/processing/evaluation",
        ],
        property_files=[evaluation_report],
    )

    # --- Step 5 (collection): RegisterModel -------------------------------
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                evaluation_step.arguments["ProcessingOutputConfig"]["Outputs"][0][
                    "S3Output"
                ]["S3Uri"]
            ),
            content_type="application/json",
        )
    )

    register_step = RegisterModel(
        name="RegisterChurnModel",
        estimator=estimator,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=[INSTANCE_TYPE],
        transform_instances=[INSTANCE_TYPE],
        model_package_group_name=MODEL_PACKAGE_GROUP,
        approval_status="PendingManualApproval",
        model_metrics=model_metrics,
    )

    # --- Step 4: CheckAccuracy condition ----------------------------------
    accuracy_condition = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=evaluation_step.name,
            property_file=evaluation_report,
            json_path="metrics.accuracy.value",
        ),
        right=ACCURACY_THRESHOLD,
    )

    condition_step = ConditionStep(
        name="CheckAccuracy",
        conditions=[accuracy_condition],
        if_steps=[register_step],
        else_steps=[],
    )

    pipeline = Pipeline(
        name=PIPELINE_NAME,
        parameters=[raw_data_uri, processed_uri],
        steps=[processing_step, training_step, evaluation_step, condition_step],
        sagemaker_session=session,
    )
    return pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(description="Author the ChurnGuard pipeline DAG")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write infra/pipeline-definition.json locally only (no S3 upload)",
    )
    parser.add_argument("--bucket", default=DATA_BUCKET)
    parser.add_argument("--key", default="pipeline/pipeline-definition.json")
    args = parser.parse_args(argv)

    boto_session = boto3.Session(region_name=REGION)
    session = PipelineSession(boto_session=boto_session, default_bucket=DATA_BUCKET)
    # Seed the default bucket so authoring never calls S3 list/create_bucket.
    session._default_bucket = DATA_BUCKET

    pipeline = build_pipeline(session)
    definition = pipeline.definition()  # JSON string

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as fh:
        # pipeline.definition() returns a JSON string; normalize to pretty JSON.
        fh.write(json.dumps(json.loads(definition), indent=2))
    print(f"wrote pipeline definition -> {OUTPUT_JSON}")

    if args.dry_run:
        print("dry-run: skipping S3 upload")
        return 0

    s3 = boto_session.client("s3", region_name=REGION)

    # Upload the ProcessingStep entry-point scripts to the S3 code URIs the
    # definition references, so a real pipeline run finds them.
    for local_name, code_uri in (
        ("preprocessing.py", PREPROCESSING_CODE_URI),
        ("evaluation.py", EVALUATION_CODE_URI),
    ):
        key = code_uri.split(f"s3://{DATA_BUCKET}/", 1)[1]
        with open(os.path.join(SCRIPTS_DIR, local_name), "rb") as fh:
            s3.put_object(
                Bucket=DATA_BUCKET, Key=key, Body=fh.read(),
                ContentType="text/x-python",
            )
        print(f"uploaded {local_name} -> {code_uri}")

    s3.put_object(
        Bucket=args.bucket,
        Key=args.key,
        Body=definition.encode("utf-8"),
        ContentType="application/json",
    )
    print(f"uploaded pipeline definition -> s3://{args.bucket}/{args.key}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
