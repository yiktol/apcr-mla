"""Pydantic request/response models.

``ChurnFeatures`` mirrors the 19 raw Telco feature columns (customerID and the
Churn label excluded). Numeric fields carry real bounds; categorical fields are
constrained to ``Literal`` enums of the actual dataset values so an invalid row
fails fast with a 422 and never reaches a real endpoint.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ChurnFeatures(BaseModel):
    """The 19 raw feature columns of the IBM Telco Customer Churn dataset."""

    model_config = {"extra": "forbid"}

    # Numeric (3).
    tenure: int = Field(..., ge=0, le=120, description="months with the company")
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)

    # SeniorCitizen is encoded 0/1 in the raw dataset.
    SeniorCitizen: Literal[0, 1]

    # Categorical (15).
    gender: Literal["Female", "Male"]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]


class PredictRequest(BaseModel):
    """Single record or a small batch (max 100) for the synchronous routes."""

    model_config = {"extra": "forbid"}

    features: Optional[ChurnFeatures] = None
    records: Optional[List[ChurnFeatures]] = Field(default=None, max_length=100)

    def resolved_records(self) -> List[ChurnFeatures]:
        if self.records is not None:
            return self.records
        if self.features is not None:
            return [self.features]
        return []


class AsyncRequest(BaseModel):
    model_config = {"extra": "forbid"}

    records: List[ChurnFeatures] = Field(..., min_length=1)


class BatchRequest(BaseModel):
    model_config = {"extra": "forbid"}

    inputS3Uri: Optional[str] = None


class PipelineRunRequest(BaseModel):
    model_config = {"extra": "forbid"}


class RegistryApproveRequest(BaseModel):
    model_config = {"extra": "forbid"}

    modelPackageArn: str


class BlueGreenRequest(BaseModel):
    model_config = {"extra": "forbid"}

    modelPackageArn: str
    mode: Literal["canary", "linear"] = "canary"
    canaryPercent: int = Field(default=10, ge=1, le=50)
    linearStepPercent: int = Field(default=20, ge=10, le=50)
    bakeTimeSeconds: int = Field(default=120, ge=30, le=1800)


class MultiModelRequest(BaseModel):
    model_config = {"extra": "forbid"}

    targetModel: Literal["churn-v1.tar.gz", "churn-v2.tar.gz"]
    features: ChurnFeatures
