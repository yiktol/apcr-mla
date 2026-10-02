"""ChurnGuard FastAPI backend package.

Every AWS call is real at runtime; the only mocks permitted anywhere are
``botocore.stub.Stubber`` instances inside the unit tests. All boto3 clients
are created via :func:`app.aws.boto_client`, which pins ``region_name`` to
``us-east-1``.
"""
