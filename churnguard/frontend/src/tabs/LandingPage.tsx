import { useEffect, useState } from "react";
import Container from "@cloudscape-design/components/container";
import Header from "@cloudscape-design/components/header";
import SpaceBetween from "@cloudscape-design/components/space-between";
import Box from "@cloudscape-design/components/box";
import ColumnLayout from "@cloudscape-design/components/column-layout";
import StatusIndicator from "@cloudscape-design/components/status-indicator";
import Spinner from "@cloudscape-design/components/spinner";
import { api } from "../api/client";
import type { ArchitectureResponse, HealthResponse } from "../api/types";
import ArchitectureDiagram from "../components/ArchitectureDiagram";
import Identifier from "../components/Identifier";
import ErrorAlert from "../components/ErrorAlert";

export default function LandingPage() {
  const [architecture, setArchitecture] = useState<ArchitectureResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [arch, h] = await Promise.all([api.getArchitecture(), api.getHealth()]);
        if (!cancelled) {
          setArchitecture(arch);
          setHealth(h);
        }
      } catch (e) {
        if (!cancelled) setError(e);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const statusType = (s: string) => {
    if (s === "InService") return "success" as const;
    if (s === "NotFound") return "stopped" as const;
    if (s === "Error") return "error" as const;
    return "in-progress" as const;
  };

  return (
    <SpaceBetween size="l">
      <Container
        header={
          <Header
            variant="h1"
            description="One real customer-churn XGBoost model, served seven ways through SageMaker deployment and orchestration. Every tab calls a real backend route against real AWS resources in us-east-1."
          >
            ChurnGuard
          </Header>
        }
      >
        <Box variant="p">
          The scenario walks the continuous flow end to end: run the SageMaker Pipeline to
          register a model version, approve it to emit a real EventBridge event into the
          Lambda handoff, then promote that approved version onto the real-time endpoint with
          a blue/green traffic shift. The remaining tabs exercise real-time, serverless,
          async, batch, and multi-model inference against the same model.
        </Box>
      </Container>

      <Container header={<Header variant="h2">Architecture</Header>}>
        {loading && !architecture ? (
          <Spinner />
        ) : architecture ? (
          <ArchitectureDiagram architecture={architecture} />
        ) : (
          <ErrorAlert error={error} />
        )}
      </Container>

      <Container header={<Header variant="h2">Live backend health</Header>}>
        {loading && !health ? (
          <Spinner />
        ) : health ? (
          <SpaceBetween size="m">
            <ColumnLayout columns={3} variant="text-grid">
              <SpaceBetween size="xxs">
                <Box variant="awsui-key-label">Region</Box>
                <Box>{health.region}</Box>
              </SpaceBetween>
              <SpaceBetween size="xxs">
                <Box variant="awsui-key-label">Feature schema loaded</Box>
                <StatusIndicator type={health.featureSchemaLoaded ? "success" : "pending"}>
                  {health.featureSchemaLoaded ? "Loaded" : "Not loaded yet"}
                </StatusIndicator>
              </SpaceBetween>
              <Identifier label="SNS topic ARN" value={health.snsTopicArn} />
            </ColumnLayout>
            <Box variant="h3">Endpoint status</Box>
            <ColumnLayout columns={4} variant="text-grid">
              {Object.entries(health.endpointStatus).map(([label, status]) => (
                <SpaceBetween size="xxs" key={label}>
                  <Box variant="awsui-key-label">{label}</Box>
                  <StatusIndicator type={statusType(status)}>{status}</StatusIndicator>
                </SpaceBetween>
              ))}
            </ColumnLayout>
          </SpaceBetween>
        ) : (
          <ErrorAlert error={error} />
        )}
      </Container>
    </SpaceBetween>
  );
}
