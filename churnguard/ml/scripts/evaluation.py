#!/usr/bin/env python3
"""SageMaker Processing entry point: model evaluation for the pipeline.

Loads the trained model artifact (model.tar.gz) and the headerless, label-first
test split, computes accuracy and AUC, and writes evaluation.json in the EXACT
nested shape the pipeline ConditionStep's JsonGet path `metrics.accuracy.value`
resolves against:

    {"metrics": {"accuracy": {"value": ...}, "auc": {"value": ...}}}

When run as a SageMaker Processing job the SageMaker-managed mount points are
used. For local unit runs pass --model-dir / --test-dir / --output-dir so the
same code path can run against a tiny locally-trained model in a temp dir.
"""
import argparse
import glob
import json
import os
import sys
import tarfile

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

# SageMaker Processing default container mount points.
SM_MODEL_DIR = "/opt/ml/processing/model"
SM_TEST_DIR = "/opt/ml/processing/test"
SM_OUTPUT_DIR = "/opt/ml/processing/evaluation"

DECISION_THRESHOLD = 0.5


def _extract_model_tar(model_dir):
    """If model.tar.gz is present, extract it in place so the booster is loadable."""
    tar_path = os.path.join(model_dir, "model.tar.gz")
    if os.path.exists(tar_path):
        with tarfile.open(tar_path) as tar:
            # filter="data" rejects unsafe members (path traversal) and is the
            # Python 3.14 default; set explicitly for forward-compatibility.
            try:
                tar.extractall(path=model_dir, filter="data")
            except TypeError:
                # Older Python without the filter kwarg.
                tar.extractall(path=model_dir)


def _load_booster(model_dir):
    """Load an XGBoost booster from the extracted artifact directory.

    The built-in XGBoost container writes the serialized booster as
    `xgboost-model` (pickle). We support both that and a plain .json/.model
    dump for local test artifacts.
    """
    _extract_model_tar(model_dir)

    # Built-in XGBoost artifact: a pickled booster named `xgboost-model`.
    pickled = os.path.join(model_dir, "xgboost-model")
    if os.path.exists(pickled):
        import pickle
        with open(pickled, "rb") as fh:
            return pickle.load(fh)

    # Fall back to any xgboost-native model file in the dir.
    for pattern in ("*.json", "*.model", "*.ubj", "*.bin"):
        matches = sorted(glob.glob(os.path.join(model_dir, pattern)))
        if matches:
            booster = xgb.Booster()
            booster.load_model(matches[0])
            return booster

    raise FileNotFoundError(f"No loadable XGBoost model found in {model_dir!r}")


def _resolve_test_csv(test_dir):
    for name in sorted(os.listdir(test_dir)):
        if name.lower().endswith(".csv"):
            return os.path.join(test_dir, name)
    raise FileNotFoundError(f"No .csv found under test dir {test_dir!r}")


def evaluate(booster, test_csv):
    """Return (accuracy, auc) for the booster against a label-first test CSV."""
    df = pd.read_csv(test_csv, header=None)
    y_true = df.iloc[:, 0].astype(int).to_numpy()
    x = df.iloc[:, 1:].to_numpy(dtype=np.float32)

    dmatrix = xgb.DMatrix(x)
    probs = booster.predict(dmatrix)
    probs = np.asarray(probs, dtype=np.float64).ravel()

    preds = (probs >= DECISION_THRESHOLD).astype(int)
    accuracy = float(accuracy_score(y_true, preds))
    # roc_auc_score needs both classes present; guard the degenerate case.
    if len(np.unique(y_true)) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_true, probs))
    return accuracy, auc


def run(model_dir, test_dir, output_dir):
    booster = _load_booster(model_dir)
    test_csv = _resolve_test_csv(test_dir)
    accuracy, auc = evaluate(booster, test_csv)

    report = {
        "metrics": {
            "accuracy": {"value": accuracy},
            "auc": {"value": auc},
        }
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "evaluation.json")
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"evaluation: accuracy={accuracy:.4f} auc={auc:.4f} -> {out_path}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Churn model evaluation")
    parser.add_argument("--model-dir", default=SM_MODEL_DIR,
                        help="directory holding model.tar.gz or an extracted booster")
    parser.add_argument("--test-dir", default=SM_TEST_DIR,
                        help="directory holding the headerless label-first test CSV")
    parser.add_argument("--output-dir", default=SM_OUTPUT_DIR,
                        help="directory to write evaluation.json into")
    args = parser.parse_args(argv)

    run(args.model_dir, args.test_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
