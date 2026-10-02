import { useEffect, useRef, useState } from "react";
import Button from "@cloudscape-design/components/button";
import Container from "@cloudscape-design/components/container";
import Header from "@cloudscape-design/components/header";
import SpaceBetween from "@cloudscape-design/components/space-between";
import StatusIndicator from "@cloudscape-design/components/status-indicator";
import Alert from "@cloudscape-design/components/alert";
import { api } from "../api/client";
import type { AsyncResultResponse, AsyncSubmitResponse, ChurnFeatures } from "../api/types";
import { DEFAULT_FEATURES } from "../features";
import FeatureForm, { validateFeatures } from "../components/FeatureForm";
import PredictionTable from "../components/PredictionTable";
import Identifier from "../components/Identifier";
import ErrorAlert from "../components/ErrorAlert";

export default function AsyncTab() {
  const [features, setFeatures] = useState<ChurnFeatures>(DEFAULT_FEATURES);
  const [submit, setSubmit] = useState<AsyncSubmitResponse | null>(null);
  const [result, setResult] = useState<AsyncResultResponse | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const invalid = Object.keys(validateFeatures(features)).length > 0;

  const clearTimer = () => {
    if (timer.current) {
      clearTimeout(timer.current);
      timer.current = null;
    }
  };
  useEffect(() => clearTimer, []);

  const poll = async (outputLocation: string) => {
    try {
      const r = await api.getAsyncResult(outputLocation);
      setResult(r);
      if (r.status === "InProgress") {
        timer.current = setTimeout(() => poll(outputLocation), 5000);
      }
    } catch (e) {
      setError(e);
    }
  };

  const start = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    clearTimer();
    try {
      const res = await api.predictAsync([features]);
      setSubmit(res);
      setResult({ status: "InProgress" });
      poll(res.outputLocation);
    } catch (e) {
      setError(e);
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
            description="Async endpoint churnguard-async. The backend writes the input to S3, calls invoke_endpoint_async, and polls the .out object. SNS notifies on success/error."
          >
            Async report
          </Header>
        }
      >
        <SpaceBetween size="m">
          <FeatureForm value={features} onChange={setFeatures} />
          <Button variant="primary" onClick={start} loading={loading} disabled={invalid}>
            Submit async job
          </Button>
        </SpaceBetween>
      </Container>

      <ErrorAlert error={error} />

      {submit && (
        <Container header={<Header variant="h2">Submission</Header>}>
          <SpaceBetween size="m">
            <Identifier label="Inference id" value={submit.inferenceId} />
            <Identifier label="Input S3 location" value={submit.inputLocation} />
            <Identifier label="Output S3 location" value={submit.outputLocation} />
            <Identifier label="SNS topic ARN" value={submit.snsTopicArn} />
            <SpaceBetween size="xxs">
              <strong>Status</strong>
              {result?.status === "Completed" ? (
                <StatusIndicator type="success">Completed</StatusIndicator>
              ) : result?.status === "Failed" ? (
                <StatusIndicator type="error">Failed</StatusIndicator>
              ) : (
                <StatusIndicator type="in-progress">InProgress</StatusIndicator>
              )}
            </SpaceBetween>
            {result?.status === "Failed" && (
              <Alert type="error" header="Async job failed">
                {result.reason}
              </Alert>
            )}
            {result?.status === "Completed" && (
              <PredictionTable predictions={result.predictions} />
            )}
          </SpaceBetween>
        </Container>
      )}
    </SpaceBetween>
  );
}
