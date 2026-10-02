// A real sample row from the IBM Telco Customer Churn dataset used as sensible
// form defaults. These are legitimate input values the user can edit; they are
// never presented as a prediction result.
import type { ChurnFeatures } from "./api/types";

export const DEFAULT_FEATURES: ChurnFeatures = {
  tenure: 12,
  MonthlyCharges: 70.35,
  TotalCharges: 845.5,
  SeniorCitizen: 0,
  gender: "Female",
  Partner: "Yes",
  Dependents: "No",
  PhoneService: "Yes",
  MultipleLines: "No",
  InternetService: "Fiber optic",
  OnlineSecurity: "No",
  OnlineBackup: "No",
  DeviceProtection: "No",
  TechSupport: "No",
  StreamingTV: "No",
  StreamingMovies: "No",
  Contract: "Month-to-month",
  PaperlessBilling: "Yes",
  PaymentMethod: "Electronic check",
};

export const NUMERIC_FIELDS = ["tenure", "MonthlyCharges", "TotalCharges"] as const;

// Select-field options mirror the Pydantic Literal enums in backend/app/models.py.
export const CATEGORICAL_OPTIONS: Record<string, string[]> = {
  SeniorCitizen: ["0", "1"],
  gender: ["Female", "Male"],
  Partner: ["Yes", "No"],
  Dependents: ["Yes", "No"],
  PhoneService: ["Yes", "No"],
  MultipleLines: ["Yes", "No", "No phone service"],
  InternetService: ["DSL", "Fiber optic", "No"],
  OnlineSecurity: ["Yes", "No", "No internet service"],
  OnlineBackup: ["Yes", "No", "No internet service"],
  DeviceProtection: ["Yes", "No", "No internet service"],
  TechSupport: ["Yes", "No", "No internet service"],
  StreamingTV: ["Yes", "No", "No internet service"],
  StreamingMovies: ["Yes", "No", "No internet service"],
  Contract: ["Month-to-month", "One year", "Two year"],
  PaperlessBilling: ["Yes", "No"],
  PaymentMethod: [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
  ],
};
