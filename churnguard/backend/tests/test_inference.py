"""Unit tests for the shared inference primitives."""
from __future__ import annotations

import io

import pytest

from app.inference import (
    CHURN_DECISION_THRESHOLD,
    FeatureSchema,
    ModelNotReadyError,
    encode_record,
    parse_csv_predictions,
    predictions_from_body,
    records_to_csv,
    to_prediction,
)
from tests.conftest import FEATURE_COLUMNS


def _sample_record():
    return {
        "tenure": 12,
        "MonthlyCharges": 70.35,
        "TotalCharges": 845.5,
        "Contract": "One year",
        "gender": "Female",
    }


def test_threshold_is_half():
    assert CHURN_DECISION_THRESHOLD == 0.5


def test_encode_record_builds_vector_in_file_order():
    vector = encode_record(_sample_record(), FEATURE_COLUMNS)
    # order: Contract_Month-to-month, Contract_One year, Contract_Two year,
    #        MonthlyCharges, TotalCharges, gender_Female, gender_Male, tenure
    assert vector == [0.0, 1.0, 0.0, 70.35, 845.5, 1.0, 0.0, 12.0]
    assert len(vector) == len(FEATURE_COLUMNS)


def test_records_to_csv_no_label_no_header():
    csv = records_to_csv([_sample_record()], FEATURE_COLUMNS)
    assert "\n" not in csv  # single row
    assert csv.count(",") == len(FEATURE_COLUMNS) - 1


def test_parse_single_row():
    assert parse_csv_predictions("0.73") == [0.73]


def test_parse_single_row_trailing_newline():
    assert parse_csv_predictions("0.73\n") == [0.73]


def test_parse_multi_row_newlines_and_commas():
    assert parse_csv_predictions("0.1\n0.9\n0.4\n") == [0.1, 0.9, 0.4]
    assert parse_csv_predictions("0.1,0.9,0.4") == [0.1, 0.9, 0.4]


def test_parse_drops_empty_tokens():
    assert parse_csv_predictions("\n\n0.5\n\n") == [0.5]


def test_to_prediction_threshold_boundary():
    assert to_prediction(0.5)["churn"] is True
    assert to_prediction(0.49)["churn"] is False
    assert to_prediction(0.5)["churnProbability"] == 0.5


def test_predictions_from_body_maps_in_order():
    preds = predictions_from_body("0.2\n0.8")
    assert preds[0]["churn"] is False
    assert preds[1]["churn"] is True


def test_schema_loads_and_caches(stubs, feature_columns_body):
    s3 = stubs.stub("s3")
    s3.add_response(
        "get_object",
        {"Body": _stream(feature_columns_body)},
        {"Bucket": "data-bucket", "Key": "processed/feature_columns.json"},
    )
    schema = FeatureSchema("data-bucket", "processed/feature_columns.json")
    assert schema.loaded is False
    cols = schema.columns()
    assert cols == FEATURE_COLUMNS
    assert schema.loaded is True
    # Second call must NOT hit S3 again (no second stub queued).
    assert schema.columns() == FEATURE_COLUMNS


def test_schema_missing_raises_model_not_ready_and_retries(stubs):
    s3 = stubs.stub("s3")
    s3.add_client_error(
        "get_object", service_error_code="NoSuchKey", http_status_code=404,
        expected_params={"Bucket": "data-bucket", "Key": "k.json"},
    )
    schema = FeatureSchema("data-bucket", "k.json")
    with pytest.raises(ModelNotReadyError):
        schema.columns()
    # Cache stays empty so the next request retries.
    assert schema.loaded is False


def _stream(data: bytes):
    return io.BytesIO(data)
