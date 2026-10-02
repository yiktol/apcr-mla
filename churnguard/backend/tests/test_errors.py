"""Unit tests for the error-envelope mapping."""
from __future__ import annotations

from botocore.exceptions import ClientError

from app.errors import _map_client_error, _request_id, envelope


def _client_error(code, message="boom", request_id="req-123"):
    return ClientError(
        {
            "Error": {"Code": code, "Message": message},
            "ResponseMetadata": {"RequestId": request_id},
        },
        "SomeOperation",
    )


def test_validation_exception_maps_400():
    status, code = _map_client_error(_client_error("ValidationException"))
    assert status == 400
    assert code == "ValidationException"


def test_model_error_maps_400():
    status, _ = _map_client_error(_client_error("ModelError"))
    assert status == 400


def test_resource_not_found_maps_404():
    status, _ = _map_client_error(_client_error("ResourceNotFound"))
    assert status == 404


def test_throttling_maps_429():
    status, _ = _map_client_error(_client_error("ThrottlingException"))
    assert status == 429


def test_unknown_maps_502():
    status, code = _map_client_error(_client_error("SomethingElse"))
    assert status == 502
    assert code == "SomethingElse"


def test_request_id_extracted():
    assert _request_id(_client_error("X", request_id="abc")) == "abc"


def test_envelope_shape():
    env = envelope("CODE", "msg", aws_request_id="r1", detail={"a": 1})
    assert env == {
        "error": {
            "code": "CODE",
            "message": "msg",
            "aws_request_id": "r1",
            "detail": {"a": 1},
        }
    }
