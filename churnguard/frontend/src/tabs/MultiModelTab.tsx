import { useEffect, useState } from "react";
import Button from "@cloudscape-design/components/button";
import Container from "@cloudscape-design/components/container";
import Header from "@cloudscape-design/components/header";
import SpaceBetween from "@cloudscape-design/components/space-between";
import FormField from "@cloudscape-design/components/form-field";
import Select from "@cloudscape-design/components/select";
import Box from "@cloudscape-design/components/box";
import { api } from "../api/client";
import type {
  ChurnFeatures,
  MultiModelEntry,
  MultiModelPredictResponse,
} from "../api/types";
import { DEFAULT_FEATURES } from "../features";
import FeatureForm, { validateFeatures } from "../components/FeatureForm";
import PredictionTable from "../components/PredictionTable";
import Identifier from "../components/Identifier";
import ErrorAlert from "../components/ErrorAlert";

export default function MultiModelTab() {
  const [models, setModels] = useState<MultiModelEntry[]>([]);
  const [endpoint, setEndpoint] = useState<string | null>(null);
  const [target, setTarget] = useState<string | null>(null);
  const [features, setFeatures] = useState<ChurnFeatures>(DEFAULT_FEATURES);
  const [result, setResult] = useState<MultiModelPredictResponse | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);

  const invalid = Object.keys(validateFeatures(features)).length > 0;

  const loadModels = async () => {
    setError(null);
    try {
      const res = await api.getMultiModelModels();
      setModels(res.models);
      setEndpoint(res.endpoint);
      if (res.models.length > 0 && !target) {
        setTarget(res.models[0].targetModel);
      }
    } catch (e) {
      setError(e);
    }
  };

  useEffect(() => {
    loadModels();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const score = async () => {
    if (!target) return;
    setLoading(true);
    setError(null);
    try {
      setResult(await api.predictMultiModel(target, features));
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
            description="Multi-model endpoint churnguard-mme. One endpoint serves churn-v1 and churn-v2; the TargetModel header routes each request to the chosen artifact."
            actions={<Button onClick={loadModels}>Refresh models</Button>}
          >
            Hosting (Multi-Model)
          </Header>
        }
      >
        <SpaceBetween size="m">
          <Identifier label="Endpoint" value={endpoint} />
          <FormField label="Target model">
            <Select
              selectedOption={target ? { label: target, value: target } : null}
              placeholder={models.length ? "Select a model" : "No models listed"}
              options={models.map((m) => ({ label: m.targetModel, value: m.targetModel }))}
              onChange={({ detail }) => setTarget(detail.selectedOption.value ?? null)}
            />
          </FormField>
          <FeatureForm value={features} onChange={setFeatures} />
          <Button
            variant="primary"
            onClick={score}
            loading={loading}
            disabled={invalid || !target}
          >
            Score with selected model
          </Button>
        </SpaceBetween>
      </Container>

      <ErrorAlert error={error} />

      {result && (
        <Container header={<Header variant="h2">Result</Header>}>
          <SpaceBetween size="m">
            <Identifier label="Endpoint" value={result.endpoint} />
            <Identifier label="TargetModel routed to" value={result.targetModel} />
            <PredictionTable predictions={[result.prediction]} />
            <Box variant="awsui-key-label">Latency</Box>
            <Box>{result.latencyMs} ms</Box>
          </SpaceBetween>
        </Container>
      )}
    </SpaceBetween>
  );
}
