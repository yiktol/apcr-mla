"""Unit tests for the Pydantic request models."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models import ChurnFeatures, PredictRequest


def valid_record():
    return {
        "tenure": 24,
        "MonthlyCharges": 89.1,
        "TotalCharges": 2100.5,
        "SeniorCitizen": 0,
        "gender": "Male",
        "Partner": "Yes",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
    }


def test_valid_record_parses():
    feats = ChurnFeatures(**valid_record())
    assert feats.tenure == 24
    assert feats.Contract == "Month-to-month"


def test_all_19_fields_required():
    rec = valid_record()
    del rec["Contract"]
    with pytest.raises(ValidationError) as exc:
        ChurnFeatures(**rec)
    assert any(e["loc"] == ("Contract",) for e in exc.value.errors())


def test_tenure_bounds():
    rec = valid_record()
    rec["tenure"] = 200
    with pytest.raises(ValidationError):
        ChurnFeatures(**rec)


def test_negative_monthly_charges_rejected():
    rec = valid_record()
    rec["MonthlyCharges"] = -5
    with pytest.raises(ValidationError):
        ChurnFeatures(**rec)


def test_invalid_contract_literal_rejected():
    rec = valid_record()
    rec["Contract"] = "Three year"
    with pytest.raises(ValidationError):
        ChurnFeatures(**rec)


def test_extra_field_forbidden():
    rec = valid_record()
    rec["unexpected"] = "x"
    with pytest.raises(ValidationError):
        ChurnFeatures(**rec)


def test_predict_request_single_and_batch():
    single = PredictRequest(features=ChurnFeatures(**valid_record()))
    assert len(single.resolved_records()) == 1
    batch = PredictRequest(records=[ChurnFeatures(**valid_record())] * 3)
    assert len(batch.resolved_records()) == 3


def test_predict_request_batch_max_100():
    with pytest.raises(ValidationError):
        PredictRequest(records=[ChurnFeatures(**valid_record())] * 101)
