import Table from "@cloudscape-design/components/table";
import Badge from "@cloudscape-design/components/badge";
import type { Prediction } from "../api/types";

interface Props {
  predictions: Prediction[];
}

// Renders exactly the probabilities the backend returned, in input order. No
// value is derived or fabricated client-side.
export default function PredictionTable({ predictions }: Props) {
  return (
    <Table<Prediction & { index: number }>
      variant="embedded"
      items={predictions.map((p, i) => ({ ...p, index: i }))}
      columnDefinitions={[
        { id: "row", header: "Row", cell: (p) => p.index + 1 },
        {
          id: "prob",
          header: "Churn probability",
          cell: (p) => p.churnProbability.toFixed(6),
        },
        {
          id: "churn",
          header: "Churn",
          cell: (p) => (
            <Badge color={p.churn ? "red" : "green"}>{p.churn ? "Yes" : "No"}</Badge>
          ),
        },
      ]}
      empty="No predictions returned."
    />
  );
}
