#!/usr/bin/env python3
"""HIGH-2: verify (for real) whether the built-in XGBoost 1.7-1 image is
multi-model (Mode: MultiModel) capable in us-east-1.

NO fake result. The probe:
  1. Creates a throwaway Model with Mode=MultiModel on the built-in
     sagemaker-xgboost:1.7-1 image pointing at the real mme/ prefix.
  2. Creates a throwaway EndpointConfig + Endpoint, waits for InService.
  3. Calls invoke_endpoint(TargetModel="churn-v1.tar.gz") with a real feature
     row and checks a real probability comes back.
  4. Tears down the throwaway resources (always, even on failure).

On success prints:  MME_BUILTIN_SUPPORTED=true  + the pinned built-in URI.
On failure resolves the open-source XGBoost *framework* inference URI via
image_uris.retrieve(framework="xgboost", version="1.7-1",
image_scope="inference", instance_type="ml.m5.large") and prints
MME_BUILTIN_SUPPORTED=false + that URI so infra/06-mme.yaml can be pinned.

Exits non-zero ONLY if capability cannot be determined (e.g. teardown aside,
an inconclusive error that is neither a clear success nor a clear
"not supported"). A clear false result is a successful determination (exit 0).

All region-pinned to us-east-1.
"""
import argparse
import sys
import time

import boto3
from botocore.exceptions import ClientError
from sagemaker import image_uris

REGION = "us-east-1"
ACCOUNT = "875692608981"
EXECUTION_ROLE = f"arn:aws:iam::{ACCOUNT}:role/AmazonSageMaker-ExecutionRole"
MODELS_BUCKET = f"churnguard-models-{ACCOUNT}-{REGION}"
INSTANCE_TYPE = "ml.m5.large"

BUILTIN_IMAGE = (
    f"683313688378.dkr.ecr.{REGION}.amazonaws.com/sagemaker-xgboost:1.7-1"
)
MME_PREFIX_URI = f"s3://{MODELS_BUCKET}/mme/"
TARGET_MODEL = "churn-v1.tar.gz"

# Endpoint readiness bound: well under a demo's patience, long enough to spin up.
WAIT_TIMEOUT_SECONDS = 1200
POLL_SECONDS = 20


def _names():
    rev = int(time.time())
    return (
        f"churnguard-mmeprobe-model-{rev}",
        f"churnguard-mmeprobe-config-{rev}",
        f"churnguard-mmeprobe-endpoint-{rev}",
    )


def _resolve_framework_inference_uri():
    """Fallback image: open-source XGBoost framework inference container."""
    return image_uris.retrieve(
        framework="xgboost",
        region=REGION,
        version="1.7-1",
        image_scope="inference",
        instance_type=INSTANCE_TYPE,
    )


def _build_probe_csv(num_features):
    """A single all-zero feature row of the given width (shape only matters for
    the capability probe; it just needs the container to return a probability)."""
    return ",".join(["0"] * num_features)


def _feature_width():
    """Read feature_columns.json width from the data bucket if present; else a
    conservative default. The probe only needs the row to be accepted."""
    data_bucket = f"churnguard-data-{ACCOUNT}-{REGION}"
    s3 = boto3.client("s3", region_name=REGION)
    try:
        import json
        obj = s3.get_object(
            Bucket=data_bucket, Key="processed/processed/feature_columns.json"
        )
        cols = json.loads(obj["Body"].read())
        return len(cols)
    except ClientError:
        # Fall back to a plausible width; the probe tolerates a reasonable row.
        return 45


def teardown(sm, model_name, config_name, endpoint_name):
    for fn, kwargs in (
        (sm.delete_endpoint, {"EndpointName": endpoint_name}),
        (sm.delete_endpoint_config, {"EndpointConfigName": config_name}),
        (sm.delete_model, {"ModelName": model_name}),
    ):
        try:
            fn(**kwargs)
        except ClientError as exc:
            print(f"teardown warning: {exc}")


def wait_in_service(sm, endpoint_name):
    deadline = time.time() + WAIT_TIMEOUT_SECONDS
    while time.time() < deadline:
        desc = sm.describe_endpoint(EndpointName=endpoint_name)
        status = desc["EndpointStatus"]
        if status == "InService":
            return True
        if status == "Failed":
            raise RuntimeError(
                f"probe endpoint failed: {desc.get('FailureReason', 'unknown')}"
            )
        time.sleep(POLL_SECONDS)
    raise TimeoutError(f"probe endpoint not InService within {WAIT_TIMEOUT_SECONDS}s")


def probe(sm, runtime):
    model_name, config_name, endpoint_name = _names()
    created = False
    try:
        sm.create_model(
            ModelName=model_name,
            ExecutionRoleArn=EXECUTION_ROLE,
            PrimaryContainer={
                "Image": BUILTIN_IMAGE,
                "Mode": "MultiModel",
                "ModelDataUrl": MME_PREFIX_URI,
            },
        )
        created = True
        sm.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": model_name,
                    "InstanceType": INSTANCE_TYPE,
                    "InitialInstanceCount": 1,
                }
            ],
        )
        sm.create_endpoint(
            EndpointName=endpoint_name, EndpointConfigName=config_name
        )
        wait_in_service(sm, endpoint_name)

        body = _build_probe_csv(_feature_width())
        resp = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="text/csv",
            Accept="text/csv",
            TargetModel=TARGET_MODEL,
            Body=body.encode("utf-8"),
        )
        payload = resp["Body"].read().decode("utf-8").strip()
        # A real probability (parseable float) confirms MME routing works.
        float(payload.replace("\n", " ").split()[0].split(",")[0])
        return True, None
    except ClientError as exc:
        # A ValidationException on create_model / create_endpoint that names the
        # multi-model mode is a clear "not supported" signal.
        msg = str(exc)
        lowered = msg.lower()
        if ("multimodel" in lowered or "multi-model" in lowered
                or "multi model" in lowered):
            return False, msg
        # Model was created but endpoint invoke rejected MMS contract: not MME.
        if created and "targetmodel" in lowered:
            return False, msg
        # Anything else is inconclusive.
        raise
    finally:
        teardown(sm, model_name, config_name, endpoint_name)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Probe built-in XGBoost MME capability")
    args = parser.parse_args(argv)

    boto_session = boto3.Session(region_name=REGION)
    sm = boto_session.client("sagemaker", region_name=REGION)
    runtime = boto_session.client("sagemaker-runtime", region_name=REGION)

    try:
        supported, detail = probe(sm, runtime)
    except Exception as exc:  # inconclusive -> cannot determine capability
        print(f"MME capability could not be determined: {exc}", file=sys.stderr)
        return 2

    if supported:
        print("MME_BUILTIN_SUPPORTED=true")
        print(f"MME_IMAGE_URI={BUILTIN_IMAGE}")
        return 0

    fallback_uri = _resolve_framework_inference_uri()
    print("MME_BUILTIN_SUPPORTED=false")
    print(f"MME_FALLBACK_NOTE={detail}")
    print(f"MME_IMAGE_URI={fallback_uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
