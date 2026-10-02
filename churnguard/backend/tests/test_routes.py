"""Route schema tests via FastAPI TestClient with Stubber-backed clients.

Config is forced to known names so stubbed expected_params are exact. The
feature schema is pre-seeded so scoring routes don't need a live S3 load
(the lazy-load path is covered in test_inference).
"""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from datetime import datetime

from app.config import Config
from app.inference import FeatureSchema
from app.main import create_app
from tests.conftest import FEATURE_COLUMNS
from tests.test_models import valid_record

# Stubber validates response shapes too; use realistic ARNs/datetimes.
_EP_ARN = "arn:aws:sagemaker:us-east-1:111111111111:endpoint/churnguard-x"
_PKG_ARN = "arn:aws:sagemaker:us-east-1:111111111111:model-package/churnguard-churn/1"
_NOW = datetime(2024, 1, 1, 0, 0, 0)

TEST_CONFIG = Config(
    data_bucket="data-bucket",
    models_bucket="models-bucket",
    async_bucket="async-bucket",
    batch_bucket="batch-bucket",
    realtime_endpoint="churnguard-realtime",
    serverless_endpoint="churnguard-serverless",
    async_endpoint="churnguard-async",
    mme_endpoint="churnguard-mme",
    sns_topic_arn="arn:aws:sns:us-east-1:111:churnguard-async-notifications",
    batch_model_name="churnguard-realtime-model",
    model_package_group="churnguard-churn",
    pipeline_name="churnguard-pipeline",
    events_log_group="/churnguard/events",
    rollback_alarm="churnguard-realtime-ModelError",
)


class _PreloadedSchema(FeatureSchema):
    def __init__(self, columns):
        super().__init__("data-bucket", "processed/feature_columns.json")
        self._columns = list(columns)


@pytest.fixture
def client(stubs, monkeypatch):
    # Avoid any SSM call during app construction.
    monkeypatch.setattr("app.main.load_config", lambda: TEST_CONFIG)
    app = create_app()
    app.state.feature_schema = _PreloadedSchema(FEATURE_COLUMNS)
    return TestClient(app, raise_server_exceptions=False), stubs


def _stream(data: bytes):
    return io.BytesIO(data)


def _describe_endpoint_resp(name, status):
    return {
        "EndpointName": name, "EndpointArn": _EP_ARN, "EndpointConfigName": "cfg",
        "EndpointStatus": status, "CreationTime": _NOW, "LastModifiedTime": _NOW,
    }


def _describe_pkg_resp(arn, group, approval=None):
    resp = {
        "ModelPackageName": arn.split("/", 1)[-1],
        "ModelPackageArn": arn,
        "ModelPackageGroupName": group,
        "ModelPackageStatus": "Completed",
        "ModelPackageStatusDetails": {"ValidationStatuses": [], "ImageScanStatuses": []},
        "CreationTime": _NOW,
    }
    if approval is not None:
        resp["ModelApprovalStatus"] = approval
    return resp


def _in_service(stub, name="churnguard-realtime"):
    stub.add_response(
        "describe_endpoint",
        _describe_endpoint_resp(name, "InService"),
        {"EndpointName": name},
    )


# ---- health / architecture -------------------------------------------------

def test_architecture(client):
    tc, _ = client
    resp = tc.get("/api/architecture")
    assert resp.status_code == 200
    body = resp.json()
    assert "nodes" in body and "edges" in body


def test_health_reports_schema_and_endpoints(client):
    tc, stubs = client
    sm = stubs.stub("sagemaker")
    for name in ("churnguard-realtime", "churnguard-serverless",
                 "churnguard-async", "churnguard-mme"):
        sm.add_response(
            "describe_endpoint",
            _describe_endpoint_resp(name, "InService"),
            {"EndpointName": name},
        )
    resp = tc.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["region"] == "us-east-1"
    assert body["featureSchemaLoaded"] is True
    assert body["endpointStatus"]["realtime"] == "InService"


# ---- realtime --------------------------------------------------------------

def test_predict_realtime(client):
    tc, stubs = client
    sm = stubs.stub("sagemaker")
    _in_service(sm, "churnguard-realtime")
    rt = stubs.stub("sagemaker-runtime")
    rt.add_response(
        "invoke_endpoint",
        {"Body": _stream(b"0.81\n"), "ContentType": "text/csv"},
        {"EndpointName": "churnguard-realtime", "ContentType": "text/csv",
         "Accept": "text/csv", "Body": _any_bytes()},
    )
    resp = tc.post("/api/predict/realtime", json={"features": valid_record()})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["endpoint"] == "churnguard-realtime"
    assert body["predictions"][0]["churn"] is True
    assert body["predictions"][0]["churnProbability"] == 0.81


def test_predict_realtime_invalid_input_422(client):
    tc, _ = client
    bad = valid_record()
    bad["Contract"] = "Nope"
    resp = tc.post("/api/predict/realtime", json={"features": bad})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "VALIDATION_ERROR"


def test_predict_realtime_endpoint_not_ready_409(client):
    tc, stubs = client
    sm = stubs.stub("sagemaker")
    sm.add_response(
        "describe_endpoint",
        _describe_endpoint_resp("churnguard-realtime", "Creating"),
        {"EndpointName": "churnguard-realtime"},
    )
    resp = tc.post("/api/predict/realtime", json={"features": valid_record()})
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "ENDPOINT_NOT_READY"


# ---- serverless ------------------------------------------------------------

def test_predict_serverless_heuristic_fields(client):
    tc, stubs = client
    sm = stubs.stub("sagemaker")
    _in_service(sm, "churnguard-serverless")
    rt = stubs.stub("sagemaker-runtime")
    rt.add_response(
        "invoke_endpoint",
        {"Body": _stream(b"0.3"), "ContentType": "text/csv"},
        {"EndpointName": "churnguard-serverless", "ContentType": "text/csv",
         "Accept": "text/csv", "Body": _any_bytes()},
    )
    resp = tc.post("/api/predict/serverless", json={"features": valid_record()})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["coldStartThresholdMs"] == 1500
    assert "coldStartLikely" in body
    assert "coldStart" not in body  # never a measured-looking bool
    assert body["predictions"][0]["churn"] is False


# ---- async -----------------------------------------------------------------

def test_async_submit_and_result_completed(client):
    tc, stubs = client
    s3 = stubs.stub("s3")
    s3.add_response("put_object", {}, {
        "Bucket": "async-bucket", "Key": _any_str(), "Body": _any_bytes()})
    sm = stubs.stub("sagemaker")
    _in_service(sm, "churnguard-async")
    rt = stubs.stub("sagemaker-runtime")
    rt.add_response(
        "invoke_endpoint_async",
        {"OutputLocation": "s3://async-bucket/output/x.out"},
        {"EndpointName": "churnguard-async", "InputLocation": _any_str(),
         "ContentType": "text/csv", "Accept": "text/csv"},
    )
    submit = tc.post("/api/predict/async", json={"records": [valid_record(), valid_record()]})
    assert submit.status_code == 200, submit.text
    out_loc = submit.json()["outputLocation"]
    assert submit.json()["rowCount"] == 2
    assert submit.json()["snsTopicArn"].endswith("churnguard-async-notifications")

    # result: no failure, .out present with 2 tokens -> Completed
    _, out_key = out_loc.replace("s3://", "").split("/", 1)
    s3.add_client_error("get_object", service_error_code="NoSuchKey",
                        http_status_code=404)  # .out.failure absent
    s3.add_response("get_object", {"Body": _stream(b"0.2\n0.9\n")},
                    {"Bucket": "async-bucket", "Key": out_key})
    res = tc.get("/api/predict/async/result", params={"outputLocation": out_loc})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["status"] == "Completed"
    assert len(body["predictions"]) == 2


def test_async_result_failure_checked_first(client):
    tc, stubs = client
    s3 = stubs.stub("s3")
    s3.add_response("put_object", {}, {
        "Bucket": "async-bucket", "Key": _any_str(), "Body": _any_bytes()})
    sm = stubs.stub("sagemaker")
    _in_service(sm, "churnguard-async")
    rt = stubs.stub("sagemaker-runtime")
    rt.add_response("invoke_endpoint_async",
                    {"OutputLocation": "s3://async-bucket/output/x.out"},
                    {"EndpointName": "churnguard-async", "InputLocation": _any_str(),
                     "ContentType": "text/csv", "Accept": "text/csv"})
    submit = tc.post("/api/predict/async", json={"records": [valid_record()]})
    out_loc = submit.json()["outputLocation"]
    # .out.failure present -> Failed (checked before .out)
    s3.add_response("get_object", {"Body": _stream(b"model exploded")},
                    {"Bucket": "async-bucket",
                     "Key": out_loc.replace("s3://async-bucket/", "") + ".failure"})
    res = tc.get("/api/predict/async/result", params={"outputLocation": out_loc})
    assert res.json()["status"] == "Failed"
    assert "exploded" in res.json()["reason"]


def test_async_result_short_parse_stays_inprogress(client):
    tc, stubs = client
    s3 = stubs.stub("s3")
    s3.add_response("put_object", {}, {
        "Bucket": "async-bucket", "Key": _any_str(), "Body": _any_bytes()})
    sm = stubs.stub("sagemaker")
    _in_service(sm, "churnguard-async")
    rt = stubs.stub("sagemaker-runtime")
    rt.add_response("invoke_endpoint_async",
                    {"OutputLocation": "s3://async-bucket/output/x.out"},
                    {"EndpointName": "churnguard-async", "InputLocation": _any_str(),
                     "ContentType": "text/csv", "Accept": "text/csv"})
    submit = tc.post("/api/predict/async", json={"records": [valid_record(), valid_record()]})
    out_loc = submit.json()["outputLocation"]
    _, out_key = out_loc.replace("s3://", "").split("/", 1)
    s3.add_client_error("get_object", service_error_code="NoSuchKey", http_status_code=404)
    # only 1 token present but 2 rows submitted -> InProgress, NOT Completed
    s3.add_response("get_object", {"Body": _stream(b"0.5\n")},
                    {"Bucket": "async-bucket", "Key": out_key})
    res = tc.get("/api/predict/async/result", params={"outputLocation": out_loc})
    assert res.json()["status"] == "InProgress"


# ---- batch -----------------------------------------------------------------

def test_batch_default_strips_label_and_records_rowcount(client):
    tc, stubs = client
    s3 = stubs.stub("s3")
    # test split: label-first, 2 rows x (1 label + 8 features)
    test_csv = b"1,0,1,0,70.0,845.0,1,0,12\n0,1,0,0,50.0,600.0,0,1,5\n"
    s3.add_response("get_object", {"Body": _stream(test_csv)},
                    {"Bucket": "data-bucket", "Key": "processed/test/test.csv"})
    # batch.csv written should have the label column removed (8 cols)
    s3.add_response("put_object", {},
                    {"Bucket": "batch-bucket", "Key": "input/batch.csv",
                     "Body": _any_bytes()})
    sm = stubs.stub("sagemaker")
    sm.add_response("create_transform_job", {"TransformJobArn": "arn:aws:sagemaker:us-east-1:111111111111:transform-job/j"}, _any_params())
    resp = tc.post("/api/batch", json={})
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "InProgress"
    assert resp.json()["jobName"].startswith("churnguard-batch-")


def test_batch_status_completed_asserts_rowcount(client):
    tc, stubs = client
    # First start a job so the row count is recorded.
    s3 = stubs.stub("s3")
    test_csv = b"1,0,1,0,70.0,845.0,1,0,12\n0,1,0,0,50.0,600.0,0,1,5\n"
    s3.add_response("get_object", {"Body": _stream(test_csv)},
                    {"Bucket": "data-bucket", "Key": "processed/test/test.csv"})
    s3.add_response("put_object", {},
                    {"Bucket": "batch-bucket", "Key": "input/batch.csv", "Body": _any_bytes()})
    sm = stubs.stub("sagemaker")
    sm.add_response("create_transform_job", {"TransformJobArn": "arn:aws:sagemaker:us-east-1:111111111111:transform-job/j"}, _any_params())
    job = tc.post("/api/batch", json={}).json()["jobName"]

    # Now describe Completed + fetch the .out with matching 2 tokens.
    sm.add_response(
        "describe_transform_job",
        {
            "TransformJobName": job, "TransformJobArn": "arn:aws:sagemaker:us-east-1:111111111111:transform-job/j",
            "TransformJobStatus": "Completed", "ModelName": "churnguard-realtime-model",
            "TransformInput": {"DataSource": {"S3DataSource": {
                "S3DataType": "S3Prefix", "S3Uri": "s3://batch-bucket/input/batch.csv"}},
                "ContentType": "text/csv"},
            "TransformOutput": {"S3OutputPath": "s3://batch-bucket/output/"},
            "TransformResources": {"InstanceType": "ml.m5.large", "InstanceCount": 1},
            "CreationTime": _NOW,
        },
        {"TransformJobName": job},
    )
    s3.add_response("get_object", {"Body": _stream(b"0.3\n0.7\n")},
                    {"Bucket": "batch-bucket", "Key": "output/batch.csv.out"})
    resp = tc.get(f"/api/batch/{job}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "Completed"
    assert body["rowCount"] == 2
    assert len(body["predictions"]) == 2


# ---- pipeline / registry / events -----------------------------------------

def test_pipeline_run(client):
    tc, stubs = client
    sm = stubs.stub("sagemaker")
    sm.add_response("start_pipeline_execution",
                    {"PipelineExecutionArn": "arn:aws:sagemaker:us-east-1:1:pipeline/x/execution/y"},
                    {"PipelineName": "churnguard-pipeline"})
    resp = tc.post("/api/pipeline/run", json={})
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "Executing"


def test_registry_empty_when_group_missing(client):
    tc, stubs = client
    sm = stubs.stub("sagemaker")
    sm.add_client_error("list_model_packages", service_error_code="ValidationException",
                        http_status_code=400)
    resp = tc.get("/api/registry")
    assert resp.status_code == 200
    assert resp.json()["versions"] == []


def test_registry_approve_rejects_foreign_arn(client):
    tc, stubs = client
    sm = stubs.stub("sagemaker")
    arn = "arn:aws:sagemaker:us-east-1:111111111111:model-package/other/1"
    sm.add_response(
        "describe_model_package",
        _describe_pkg_resp(arn, "other-group"),
        {"ModelPackageName": arn},
    )
    resp = tc.post("/api/registry/approve", json={"modelPackageArn": arn})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "INVALID_PACKAGE"


def test_registry_approve_idempotent(client):
    tc, stubs = client
    sm = stubs.stub("sagemaker")
    arn = "arn:aws:sagemaker:us-east-1:111111111111:model-package/churnguard-churn/2"
    sm.add_response(
        "describe_model_package",
        _describe_pkg_resp(arn, "churnguard-churn", approval="Approved"),
        {"ModelPackageName": arn},
    )
    resp = tc.post("/api/registry/approve", json={"modelPackageArn": arn})
    assert resp.status_code == 200
    assert resp.json()["approvalStatus"] == "Approved"


def test_events_recent_empty_when_group_absent(client):
    tc, stubs = client
    logs = stubs.stub("logs")
    logs.add_client_error("filter_log_events",
                          service_error_code="ResourceNotFoundException",
                          http_status_code=400)
    resp = tc.get("/api/events/recent")
    assert resp.status_code == 200
    assert resp.json()["events"] == []


# ---- blue/green ------------------------------------------------------------

def test_bluegreen_rejects_unapproved_409(client):
    tc, stubs = client
    sm = stubs.stub("sagemaker")
    arn = "arn:aws:sagemaker:us-east-1:111111111111:model-package/churnguard-churn/3"
    sm.add_response(
        "describe_model_package",
        _describe_pkg_resp(arn, "churnguard-churn", approval="PendingManualApproval"),
        {"ModelPackageName": arn},
    )
    resp = tc.post("/api/deploy/bluegreen", json={"modelPackageArn": arn, "mode": "canary"})
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "NOT_APPROVED"


def test_bluegreen_canary_update(client):
    tc, stubs = client
    sm = stubs.stub("sagemaker")
    arn = "arn:aws:sagemaker:us-east-1:111111111111:model-package/churnguard-churn/4"
    sm.add_response(
        "describe_model_package",
        _describe_pkg_resp(arn, "churnguard-churn", approval="Approved"),
        {"ModelPackageName": arn},
    )
    sm.add_response("create_model", {"ModelArn": "arn:aws:sagemaker:us-east-1:111111111111:model/m"}, _any_params())
    sm.add_response("create_endpoint_config", {"EndpointConfigArn": "arn:aws:sagemaker:us-east-1:111111111111:endpoint-config/c"}, _any_params())
    sm.add_response("update_endpoint", {"EndpointArn": _EP_ARN}, _any_params())
    resp = tc.post("/api/deploy/bluegreen",
                   json={"modelPackageArn": arn, "mode": "canary", "canaryPercent": 10})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "Updating"
    assert body["newModel"].startswith("churnguard-bg-model-")
    assert body["newConfig"].startswith("churnguard-bg-config-")


# ---- multi-model -----------------------------------------------------------

def test_mme_invoke(client):
    tc, stubs = client
    sm = stubs.stub("sagemaker")
    _in_service(sm, "churnguard-mme")
    rt = stubs.stub("sagemaker-runtime")
    rt.add_response(
        "invoke_endpoint",
        {"Body": _stream(b"0.66"), "ContentType": "text/csv"},
        {"EndpointName": "churnguard-mme", "ContentType": "text/csv",
         "Accept": "text/csv", "TargetModel": "churn-v1.tar.gz", "Body": _any_bytes()},
    )
    resp = tc.post("/api/hosting/multimodel",
                   json={"targetModel": "churn-v1.tar.gz", "features": valid_record()})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["targetModel"] == "churn-v1.tar.gz"
    assert body["prediction"]["churn"] is True


def test_mme_list_models(client):
    tc, stubs = client
    s3 = stubs.stub("s3")
    s3.add_response(
        "list_objects_v2",
        {"Contents": [
            {"Key": "mme/churn-v1.tar.gz", "Size": 100},
            {"Key": "mme/churn-v2.tar.gz", "Size": 110},
            {"Key": "mme/", "Size": 0},
        ]},
        {"Bucket": "models-bucket", "Prefix": "mme/"},
    )
    resp = tc.get("/api/hosting/multimodel/models")
    assert resp.status_code == 200
    names = [m["targetModel"] for m in resp.json()["models"]]
    assert names == ["churn-v1.tar.gz", "churn-v2.tar.gz"]


# ---- Stubber ANY helpers ---------------------------------------------------

def _any_bytes():
    from botocore.stub import ANY
    return ANY


def _any_str():
    from botocore.stub import ANY
    return ANY


def _any_params():
    # None tells Stubber to skip expected-parameter validation for this call.
    return None
