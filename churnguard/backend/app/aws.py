"""Single boto3 client factory.

The machine's AWS CLI default region is ``ap-southeast-1``; the demo targets
``us-east-1`` exclusively. Every client in the backend MUST be created here so
``region_name`` is pinned in exactly one place and can never drift.
"""
from __future__ import annotations

import boto3

AWS_REGION = "us-east-1"


def boto_client(service: str):
    """Return a boto3 client for ``service`` pinned to ``us-east-1``.

    This is the only place in the backend that constructs an AWS client, so the
    region invariant holds for every runtime AWS call.
    """
    return boto3.client(service, region_name=AWS_REGION)
