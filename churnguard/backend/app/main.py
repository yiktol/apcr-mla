"""FastAPI application wiring.

Builds the app, loads config from SSM at startup (with env/default fallbacks),
constructs the single lazy feature-schema cache, registers the error handlers,
and mounts every route module. CORS is restricted to localhost dev origins.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import load_config
from .errors import register_exception_handlers
from .inference import FeatureSchema
from .routes import batch, deploy, hosting, meta, pipeline, predict

logging.basicConfig(level=logging.INFO)

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]


def create_app() -> FastAPI:
    app = FastAPI(title="ChurnGuard API", version="1.0.0")

    config = load_config()
    app.state.config = config
    app.state.feature_schema = FeatureSchema(config.data_bucket, config.feature_columns_key)
    app.state.rowcounts = {}

    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)

    for module in (meta, predict, batch, pipeline, deploy, hosting):
        app.include_router(module.router)

    return app


app = create_app()
