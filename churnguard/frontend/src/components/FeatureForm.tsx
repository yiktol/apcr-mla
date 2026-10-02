import { useState } from "react";
import ColumnLayout from "@cloudscape-design/components/column-layout";
import FormField from "@cloudscape-design/components/form-field";
import Input from "@cloudscape-design/components/input";
import Select from "@cloudscape-design/components/select";
import type { ChurnFeatures } from "../api/types";
import { CATEGORICAL_OPTIONS, NUMERIC_FIELDS } from "../features";

interface Props {
  value: ChurnFeatures;
  onChange: (next: ChurnFeatures) => void;
}

type NumericField = (typeof NUMERIC_FIELDS)[number];

// Client-side validation mirrors the backend numeric bounds so an obviously
// invalid value is caught before the request. The backend remains the final
// authority (returns 422 with field detail).
export function validateFeatures(value: ChurnFeatures): Record<string, string> {
  const errors: Record<string, string> = {};
  if (!Number.isInteger(value.tenure) || value.tenure < 0 || value.tenure > 120) {
    errors.tenure = "tenure must be an integer between 0 and 120";
  }
  if (!(value.MonthlyCharges >= 0)) {
    errors.MonthlyCharges = "MonthlyCharges must be >= 0";
  }
  if (!(value.TotalCharges >= 0)) {
    errors.TotalCharges = "TotalCharges must be >= 0";
  }
  return errors;
}

export default function FeatureForm({ value, onChange }: Props) {
  const [rawNumeric, setRawNumeric] = useState<Record<string, string>>({
    tenure: String(value.tenure),
    MonthlyCharges: String(value.MonthlyCharges),
    TotalCharges: String(value.TotalCharges),
  });

  const errors = validateFeatures(value);

  const updateNumeric = (field: NumericField, raw: string) => {
    setRawNumeric((prev) => ({ ...prev, [field]: raw }));
    const parsed = field === "tenure" ? parseInt(raw, 10) : parseFloat(raw);
    onChange({ ...value, [field]: Number.isNaN(parsed) ? -1 : parsed });
  };

  const categoricalFields = Object.keys(CATEGORICAL_OPTIONS);

  return (
    <ColumnLayout columns={3} borders="vertical">
      {NUMERIC_FIELDS.map((field) => (
        <FormField key={field} label={field} errorText={errors[field]}>
          <Input
            type="number"
            value={rawNumeric[field]}
            onChange={({ detail }) => updateNumeric(field, detail.value)}
            ariaLabel={field}
          />
        </FormField>
      ))}
      {categoricalFields.map((field) => {
        const options = CATEGORICAL_OPTIONS[field].map((o) => ({ label: o, value: o }));
        const current = String(value[field as keyof ChurnFeatures]);
        return (
          <FormField key={field} label={field}>
            <Select
              selectedOption={{ label: current, value: current }}
              options={options}
              ariaLabel={field}
              onChange={({ detail }) => {
                const v = detail.selectedOption.value as string;
                const coerced = field === "SeniorCitizen" ? (Number(v) as 0 | 1) : v;
                onChange({ ...value, [field]: coerced } as ChurnFeatures);
              }}
            />
          </FormField>
        );
      })}
    </ColumnLayout>
  );
}
