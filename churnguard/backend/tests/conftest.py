"""Pytest fixtures: Stubber-backed boto3 clients.

The HARD CONSTRAINT: the running demo always hits real AWS. The ONLY mocks
anywhere are ``botocore.stub.Stubber`` instances, used here to exercise the
boto3 transport without a network. Each test registers the exact responses it
expects; an unmatched call raises, so these tests assert the real request
shapes the routes send.
"""
from __future__ import annotations

import json
from typing import Dict

import boto3
import pytest
from botocore.config import Config as BotoConfig
from botocore.stub import Stubber

from app.aws import AWS_REGION

# A fixed feature schema fixture (feature columns only, inference order,
# excluding the label). Mix of numeric + one-hot categorical columns.
FEATURE_COLUMNS = [
    "Contract_Month-to-month",
    "Contract_One year",
    "Contract_Two year",
    "MonthlyCharges",
    "TotalCharges",
    "gender_Female",
    "gender_Male",
    "tenure",
]


def make_client(service: str):
    """A real boto3 client (no network calls will be made; a Stubber intercepts)."""
    return boto3.client(
        service,
        region_name=AWS_REGION,
        aws_access_key_id="test",
        aws_secret_access_key="test",
        config=BotoConfig(retries={"max_attempts": 0}),
    )


class StubRegistry:
    """Holds one stubbed client per service and routes boto_client() to them."""

    def __init__(self):
        self.clients: Dict[str, object] = {}
        self.stubbers: Dict[str, Stubber] = {}

    def client(self, service: str):
        if service not in self.clients:
            c = make_client(service)
            s = Stubber(c)
            s.activate()
            self.clients[service] = c
            self.stubbers[service] = s
        return self.clients[service]

    def stub(self, service: str) -> Stubber:
        self.client(service)
        return self.stubbers[service]

    def assert_complete(self):
        for s in self.stubbers.values():
            s.assert_no_pending_responses()

    def deactivate(self):
        for s in self.stubbers.values():
            s.deactivate()


@pytest.fixture
def stubs(monkeypatch):
    """Patch every module's boto_client to return Stubber-backed clients."""
    registry = StubRegistry()

    def fake_boto_client(service: str):
        return registry.client(service)

    # Patch the factory where it is imported/used.
    import app.aws
    import app.config
    import app.inference
    import app.routes.predict
    import app.routes.batch
    import app.routes.pipeline
    import app.routes.deploy
    import app.routes.hosting
    import app.routes.meta

    for mod in (
        app.aws, app.config, app.inference,
        app.routes.predict, app.routes.batch, app.routes.pipeline,
        app.routes.deploy, app.routes.hosting, app.routes.meta,
    ):
        if hasattr(mod, "boto_client"):
            monkeypatch.setattr(mod, "boto_client", fake_boto_client)

    yield registry
    registry.deactivate()


@pytest.fixture
def feature_columns_body():
    return json.dumps(FEATURE_COLUMNS).encode("utf-8")
