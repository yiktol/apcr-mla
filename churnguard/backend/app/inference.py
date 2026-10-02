"""Shared inference primitives: the pinned decision threshold, the lazy
feature-schema loader, the one-hot encoder, and the single text/csv parser.

Every scoring route (realtime, serverless, mme, batch, async) uses the helpers
here so the probability->label decision and the response parsing are identical
everywhere. Nothing in this module invents a probability: a missing feature
schema raises ``ModelNotReadyError`` (409 MODEL_NOT_READY) and a short/empty
parse is reported honestly, never padded.
"""
from __future__ import annotations

import json
import logging
import threading
from typing import Dict, List

from botocore.exceptions import ClientError

from .aws import boto_client

log = logging.getLogger("churnguard.inference")

# --- Single source of truth for the probability -> label decision. ----------
# Lives here and nowhere else; every route imports it. 0.5 is the default
# binary cutoff, intentionally untuned for the mild (~27%) class imbalance to
# keep the demo output stable and explainable (documented in the README).
CHURN_DECISION_THRESHOLD = 0.5

NUMERIC_COLUMNS = ("tenure", "MonthlyCharges", "TotalCharges")


class ModelNotReadyError(Exception):
    """Raised when ``feature_columns.json`` is genuinely absent in S3.

    Maps to HTTP 409 MODEL_NOT_READY. It is recoverable: the loader retries on
    the next request, so the backend self-heals once training writes the file.
    """


class FeatureSchema:
    """Lazy-loaded, retry-on-miss cache for ``processed/feature_columns.json``.

    The schema is loaded from S3 on the first scoring request (not at startup)
    and cached on success. A genuine ``NoSuchKey`` raises ``ModelNotReadyError``
    and leaves the cache empty so the next request retries. Any other S3 error
    propagates (handled by the error envelope as a real AWS failure).
    """

    def __init__(self, bucket: str, key: str):
        self._bucket = bucket
        self._key = key
        self._columns: List[str] | None = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._columns is not None

    def columns(self) -> List[str]:
        """Return the cached feature columns, loading from S3 if needed."""
        if self._columns is not None:
            return self._columns
        with self._lock:
            if self._columns is not None:  # another thread won the race
                return self._columns
            self._columns = self._load()
            return self._columns

    def _load(self) -> List[str]:
        s3 = boto_client("s3")
        try:
            obj = s3.get_object(Bucket=self._bucket, Key=self._key)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in ("NoSuchKey", "404", "NoSuchBucket"):
                log.warning(
                    "feature schema not present yet s3://%s/%s (%s) -> MODEL_NOT_READY",
                    self._bucket, self._key, code,
                )
                raise ModelNotReadyError(
                    f"feature_columns.json not found at s3://{self._bucket}/{self._key}"
                ) from exc
            raise
        body = obj["Body"].read()
        columns = json.loads(body)
        if not isinstance(columns, list) or not all(isinstance(c, str) for c in columns):
            raise ValueError("feature_columns.json must be a JSON list of strings")
        log.info("feature schema loaded: %d columns", len(columns))
        return columns


def encode_record(record: Dict[str, object], feature_columns: List[str]) -> List[float]:
    """Build a feature vector of ``len(feature_columns)`` values in file order.

    Numeric columns map by name directly. Categorical values are expressed as
    one-hot column names of the form ``<field>_<value>`` (matching the
    ``pd.get_dummies`` naming the preprocessing script produces). There is NO
    label slot — ``feature_columns`` already excludes the label.
    """
    # Pre-compute the one-hot column names this record activates.
    active = set()
    for field, value in record.items():
        if field in NUMERIC_COLUMNS:
            continue
        active.add(f"{field}_{value}")

    vector: List[float] = []
    for col in feature_columns:
        if col in NUMERIC_COLUMNS:
            vector.append(float(record[col]))
        else:
            vector.append(1.0 if col in active else 0.0)
    return vector


def records_to_csv(records: List[Dict[str, object]], feature_columns: List[str]) -> str:
    """Serialize records to headerless, label-free CSV in feature order."""
    lines = []
    for rec in records:
        vector = encode_record(rec, feature_columns)
        lines.append(",".join(_fmt(v) for v in vector))
    return "\n".join(lines)


def _fmt(value: float) -> str:
    # Keep integers compact; the serving container accepts either form.
    if value == int(value):
        return str(int(value))
    return repr(value)


def parse_csv_predictions(body: str) -> List[float]:
    """The single pinned text/csv response parser.

    Split on any whitespace/newlines/commas, drop empty tokens, ``float()``-cast
    each remaining token. The i-th float corresponds to the i-th input row.
    """
    tokens: List[float] = []
    for raw in body.replace(",", " ").split():
        raw = raw.strip()
        if not raw:
            continue
        tokens.append(float(raw))
    return tokens


def to_prediction(probability: float) -> Dict[str, object]:
    """Map a probability to the pinned ``{churnProbability, churn}`` shape."""
    return {
        "churnProbability": probability,
        "churn": probability >= CHURN_DECISION_THRESHOLD,
    }


def predictions_from_body(body: str) -> List[Dict[str, object]]:
    """Parse a text/csv body into a list of prediction dicts (input order)."""
    return [to_prediction(p) for p in parse_csv_predictions(body)]
