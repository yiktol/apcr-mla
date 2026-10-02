#!/usr/bin/env bash
# Build the dedicated SageMaker SDK virtualenv for ChurnGuard.
#
# The SYSTEM `sagemaker` python module is broken (no `image_uris`), so every
# SageMaker SDK call in this project MUST run inside this venv. This script
# creates ml/.venv, installs ml/requirements.txt, and pins the AWS region to
# us-east-1 (the machine default is ap-southeast-1).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

# Region is pinned everywhere; export it so any boto3/CLI invocation inheriting
# this environment defaults to us-east-1 rather than the machine's ap-southeast-1.
export AWS_DEFAULT_REGION=us-east-1

PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ ! -d "${VENV_DIR}" ]; then
  echo "Creating virtualenv at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "${SCRIPT_DIR}/requirements.txt"

echo "ml/.venv ready. AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION}"
python -c 'import sagemaker, boto3; from sagemaker import image_uris; print("sagemaker", sagemaker.__version__, "boto3", boto3.__version__)'
