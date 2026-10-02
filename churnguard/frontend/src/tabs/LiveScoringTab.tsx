import { useState } from "react";
import Button from "@cloudscape-design/components/button";
import Container from "@cloudscape-design/components/container";
import Header from "@cloudscape-design/components/header";
import SpaceBetween from "@cloudscape-design/components/space-between";
import Box from "@cloudscape-design/components/box";
import { api } from "../api/client";
import type { ChurnFeatures, RealtimeResponse } from "../api/types";
import { DEFAULT_FEATURES } from "../features";
import FeatureForm, { validateFeatures } from "../components/FeatureForm";
import PredictionTable from "../components/PredictionTable";
import Identifier from "../components/Identifier";
import ErrorAlert from "../components/ErrorAlert";

export default function LiveScoringTab() {
  const [features, setFeatures] = useState<ChurnFeatures>(DEFAULT_FEATURES);
  const [result, setResult] = useState<RealtimeResponse | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);

  const invalid = Object.keys(validateFeatures(features)).length > 0;

  const score = async () => {
    setLoading(true);
    setError(null);
    try {
      setResult(await api.predictRealtime(features));
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
            description="Real-time endpoint churnguard-realtime. One synchronous invoke_endpoint per request."
          >
            Live scoring
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
          </SpaceBetween>
        </Container>
      )}
    </SpaceBetween>
  );
}
