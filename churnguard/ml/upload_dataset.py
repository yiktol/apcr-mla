#!/usr/bin/env python3
"""Download the real Telco Customer Churn CSV and upload it to the data bucket.

Real HTTP download (no fake data). Fails loudly (non-zero exit) if the dataset
does not have the expected shape (7043 rows x 21 columns) rather than proceeding
with partial/corrupt data.

Region is pinned to us-east-1 for every AWS call.
"""
import argparse
import io
import sys
import urllib.request

import boto3
import pandas as pd

REGION = "us-east-1"
DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/"
    "telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)
DEFAULT_BUCKET = "churnguard-data-875692608981-us-east-1"
DEFAULT_KEY = "raw/Telco-Customer-Churn.csv"

EXPECTED_ROWS = 7043
EXPECTED_COLS = 21


def download(url):
    """Return the raw CSV bytes from a real HTTP GET."""
    with urllib.request.urlopen(url, timeout=60) as resp:
        if resp.status != 200:
            raise RuntimeError(f"download failed: HTTP {resp.status} for {url}")
        return resp.read()


def validate(csv_bytes):
    """Assert the dataset shape; raise on mismatch."""
    df = pd.read_csv(io.BytesIO(csv_bytes))
    rows, cols = df.shape
    if rows != EXPECTED_ROWS or cols != EXPECTED_COLS:
        raise ValueError(
            f"unexpected dataset shape rows={rows} cols={cols}; "
            f"expected rows={EXPECTED_ROWS} cols={EXPECTED_COLS}"
        )
    return rows, cols


def upload(csv_bytes, bucket, key):
    s3 = boto3.client("s3", region_name=REGION)
    s3.put_object(Bucket=bucket, Key=key, Body=csv_bytes, ContentType="text/csv")
    return f"s3://{bucket}/{key}"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Upload Telco churn dataset to S3")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--key", default=DEFAULT_KEY)
    parser.add_argument("--url", default=DATASET_URL)
    args = parser.parse_args(argv)

    print(f"downloading {args.url}")
    csv_bytes = download(args.url)
    rows, cols = validate(csv_bytes)
    print(f"validated dataset rows={rows} cols={cols}")

    uri = upload(csv_bytes, args.bucket, args.key)
    print(f"uploaded -> {uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
