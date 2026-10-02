import { useState } from "react";
import Button from "@cloudscape-design/components/button";
import Container from "@cloudscape-design/components/container";
import Header from "@cloudscape-design/components/header";
import SpaceBetween from "@cloudscape-design/components/space-between";
import Box from "@cloudscape-design/components/box";
import StatusIndicator from "@cloudscape-design/components/status-indicator";
import { api } from "../api/client";
import type { ChurnFeatures, ServerlessResponse } from "../api/types";
import { DEFAULT_FEATURES } from "../features";
import FeatureForm, { validateFeatures } from "../components/FeatureForm";
import PredictionTable from "../components/PredictionTable";
import Identifier from "../components/Identifier";
import ErrorAlert from "../components/ErrorAlert";

export default function ServerlessTab() {
  const [features, setFeatures] = useState<ChurnFeatures>(DEFAULT_FEATURES);
  const [result, setResult] = useState<ServerlessResponse | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);

  const invalid = Object.keys(validateFeatures(features)).length > 0;

  const score = async () => {
    setLoading(true);
    setError(null);
    try {
      setResult(await api.predictServerless(features));
    } catch (e) {
      setError(e);
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <SpaceBetween size="l">
      <Container
        header={
          <Header
            variant="h2"
            description="Serverless endpoint churnguard-serverless. Scales to zero when idle; the first call after idle pays a cold start."
          >
            Serverless scoring
          </Header>
        }
      >
        <SpaceBetween size="m">
          <FeatureForm value={features} onChange={setFeatures} />
          <Button variant="primary" onClick={score} loading={loading} disabled={invalid}>
            Score
          </Button>
        </SpaceBetween>
      </Container>

      <ErrorAlert error={error} />

      {result && (
        <Container header={<Header variant="h2">Result</Header>}>
          <SpaceBetween size="m">
            <PredictionTable predictions={result.predictions} />
            <Identifier label="Endpoint" value={result.endpoint} />
            <Box variant="awsui-key-label">Latency</Box>
            <Box>{result.latencyMs} ms</Box>
            <Box variant="awsui-key-label">Cold start (heuristic)</Box>
            <StatusIndicator type={result.coldStartLikely ? "warning" : "success"}>
              {result.coldStartLikely ? "Cold start likely" : "Warm likely"}
            </StatusIndicator>
            <Box variant="small" color="text-body-secondary">
              Heuristic only: inferred from measured latency ({result.latencyMs} ms) vs the{" "}
              {result.coldStartThresholdMs} ms threshold. This is an estimate, not an
              observed cold-start fact reported by SageMaker.
            </Box>
          </SpaceBetween>
        </Container>
      )}
    </SpaceBetween>
  );
}
