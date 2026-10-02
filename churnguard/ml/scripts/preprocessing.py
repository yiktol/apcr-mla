#!/usr/bin/env python3
"""SageMaker Processing entry point: deterministic Telco churn preprocessing.

This script owns ALL preprocessing so training and inference agree on the exact
feature vector. It is byte-for-byte the same script run by the pipeline's
`DataProcessing` ProcessingStep and by the baseline path in `run_training.py`.

Behavior (per the approved design):
  * read the raw Telco-Customer-Churn.csv
  * drop `customerID`
  * coerce `TotalCharges` to float, dropping the ~11 blank rows
  * map `Churn` Yes->1 / No->0 and emit it as column 0 (XGBoost label-first)
  * one-hot encode categoricals in a FIXED sorted column order
  * 70/15/15 train/validation/test split with random_state=42
  * write headerless, label-first CSV to train/, validation/, test/
  * write processed/feature_columns.json listing FEATURE columns only, in
    inference order, EXCLUDING the label `Churn`
  * assert the invariant: len(feature_columns) == training_csv_width - 1

When run as a SageMaker Processing job the SageMaker-managed paths are used
(/opt/ml/processing/...). For local unit runs pass --input-dir / --output-dir
(or --input-file) so the same code path can be exercised in a temp dir.
"""
import argparse
import json
import os
import sys

import pandas as pd

# SageMaker Processing default container mount points.
SM_INPUT_DIR = "/opt/ml/processing/input"
SM_OUTPUT_DIR = "/opt/ml/processing/output"

LABEL_COLUMN = "Churn"
DROP_COLUMN = "customerID"
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
RANDOM_STATE = 42


def _resolve_input_csv(input_dir, input_file):
    """Return the path to the raw CSV given a dir or an explicit file."""
    if input_file:
        return input_file
    # SageMaker Processing mounts the input channel as a directory; pick the
    # first .csv under it (there is exactly one raw file in this project).
    for name in sorted(os.listdir(input_dir)):
        if name.lower().endswith(".csv"):
            return os.path.join(input_dir, name)
    raise FileNotFoundError(f"No .csv found under input dir {input_dir!r}")


def preprocess(df):
    """Return (processed_df, feature_columns).

    processed_df has the integer label `Churn` as its FIRST column followed by
    the one-hot-encoded feature columns in a fixed sorted order. feature_columns
    is the ordered list of feature column names, EXCLUDING the label.
    """
    df = df.copy()

    if DROP_COLUMN in df.columns:
        df = df.drop(columns=[DROP_COLUMN])

    # TotalCharges arrives as strings with ~11 blank values; coerce + drop them.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)

    # Map the target to 1/0.
    df[LABEL_COLUMN] = df[LABEL_COLUMN].map({"Yes": 1, "No": 0}).astype(int)

    label = df[LABEL_COLUMN]
    features = df.drop(columns=[LABEL_COLUMN])

    # One-hot encode every non-numeric column; numeric columns pass through.
    categorical_cols = [c for c in features.columns if c not in NUMERIC_COLUMNS]
    encoded = pd.get_dummies(features, columns=categorical_cols, dtype=int)

    # FIXED, sorted column order -> deterministic across runs and machines.
    feature_columns = sorted(encoded.columns.tolist())
    encoded = encoded[feature_columns]

    # Label first, features follow (XGBoost built-in label-first, no header).
    processed = pd.concat([label.rename(LABEL_COLUMN), encoded], axis=1)
    return processed, feature_columns


def split_frame(processed):
    """70/15/15 train/validation/test split, random_state=42."""
    shuffled = processed.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    n = len(shuffled)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    train = shuffled.iloc[:n_train]
    validation = shuffled.iloc[n_train:n_train + n_val]
    test = shuffled.iloc[n_train + n_val:]
    return train, validation, test


def _write_split(frame, out_dir, name):
    """Write a headerless, label-first CSV split to <out_dir>/<name>/<name>.csv."""
    split_dir = os.path.join(out_dir, name)
    os.makedirs(split_dir, exist_ok=True)
    path = os.path.join(split_dir, f"{name}.csv")
    frame.to_csv(path, header=False, index=False)
    return path


def run(input_dir, output_dir, input_file=None):
    raw_path = _resolve_input_csv(input_dir, input_file)
    df = pd.read_csv(raw_path)

    processed, feature_columns = preprocess(df)
    train, validation, test = split_frame(processed)

    _write_split(train, output_dir, "train")
    _write_split(validation, output_dir, "validation")
    _write_split(test, output_dir, "test")

    # feature_columns.json lives under processed/ alongside the splits.
    processed_dir = os.path.join(output_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    fc_path = os.path.join(processed_dir, "feature_columns.json")
    with open(fc_path, "w") as fh:
        json.dump(feature_columns, fh, indent=2)

    # Invariant: the training CSV is exactly one column wider than the inference
    # vector, that extra column being the label at index 0.
    training_csv_width = train.shape[1]
    assert len(feature_columns) == training_csv_width - 1, (
        f"feature_columns width {len(feature_columns)} != "
        f"training_csv_width-1 {training_csv_width - 1}"
    )
    assert LABEL_COLUMN not in feature_columns, "label must not appear in feature_columns"

    print(
        f"preprocessing: rows={len(processed)} "
        f"train={len(train)} validation={len(validation)} test={len(test)} "
        f"features={len(feature_columns)} training_csv_width={training_csv_width}"
    )
    return feature_columns


def main(argv=None):
    parser = argparse.ArgumentParser(description="Telco churn preprocessing")
    parser.add_argument("--input-dir", default=SM_INPUT_DIR,
                        help="directory containing the raw CSV (SageMaker input channel)")
    parser.add_argument("--input-file", default=None,
                        help="explicit path to the raw CSV (overrides --input-dir discovery)")
    parser.add_argument("--output-dir", default=SM_OUTPUT_DIR,
                        help="root output directory for splits + processed/feature_columns.json")
    args = parser.parse_args(argv)

    run(args.input_dir, args.output_dir, input_file=args.input_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
