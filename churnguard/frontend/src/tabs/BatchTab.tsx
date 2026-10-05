import { useEffect, useRef, useState } from "react";
import Button from "@cloudscape-design/components/button";
import Container from "@cloudscape-design/components/container";
import Header from "@cloudscape-design/components/header";
import SpaceBetween from "@cloudscape-design/components/space-between";
import StatusIndicator from "@cloudscape-design/components/status-indicator";
import Alert from "@cloudscape-design/components/alert";
import { api } from "../api/client";
import type { BatchStatusResponse } from "../api/types";
import Identifier from "../components/Identifier";
import PredictionTable from "../components/PredictionTable";
import ErrorAlert from "../components/ErrorAlert";
import { isReadOnly } from "../config/deployment";

const TERMINAL = ["Completed", "Failed", "Stopped"];

export default function BatchTab() {
  const [jobName, setJobName] = useState<string | null>(null);
  const [status, setStatus] = useState<BatchStatusResponse | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearTimer = () => {
    if (timer.current) {
      clearTimeout(timer.current);
      timer.current = null;
    }
  };

  useEffect(() => clearTimer, []);

  const poll = async (name: string) => {
    try {
      const s = await api.getBatchStatus(name);
      setStatus(s);
      if (!TERMINAL.includes(s.status)) {
        timer.current = setTimeout(() => poll(name), 5000);
      }
    } catch (e) {
      setError(e);
    }
  };

  const start = async () => {
    setLoading(true);
    setError(null);
    setStatus(null);
    clearTimer();
    try {
      const res = await api.runBatch();
      setJobName(res.jobName);
      setStatus({ jobName: res.jobName, status: res.status, outputLocation: res.outputLocation });
      poll(res.jobName);
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  };

  const indicator = (s: string) => {
    if (s === "Completed") return <StatusIndicator type="success">Completed</StatusIndicator>;
    if (s === "Failed" || s === "Stopped")
      return <StatusIndicator type="error">{s}</StatusIndicator>;
    return <StatusIndicator type="in-progress">{s}</StatusIndicator>;
  };

  return (
    <SpaceBetween size="l">
      {isReadOnly && (
        <Alert type="info" header="Read-only demo">
          Actions are disabled in the hosted deployment. Run locally to execute.
        </Alert>
      )}
      <Container
        header={
          <Header
            variant="h2"
            description="Batch Transform over the held-out test split. The backend strips the label column, writes batch.csv, and polls describe_transform_job."
          >
            Bulk / Batch
          </Header>
        }
      >
        <Button variant="primary" onClick={start} loading={loading} disabled={isReadOnly}>
          Start batch transform
        </Button>
      </Container>

      <ErrorAlert error={error} />

      {status && (
        <Container header={<Header variant="h2">Transform job</Header>}>
          <SpaceBetween size="m">
            <Identifier label="Job name" value={jobName} />
            <SpaceBetween size="xxs">
              <strong>Status</strong>
              {indicator(status.status)}
            </SpaceBetween>
            <Identifier label="Output S3 location" value={status.outputLocation} />
            {status.rowCount !== undefined && (
              <SpaceBetween size="xxs">
                <strong>Row count</strong>
                <span>{status.rowCount}</span>
              </SpaceBetween>
            )}
            {status.failureReason && (
              <Alert type="error" header="Job failed">
                {status.failureReason}
              </Alert>
            )}
            {status.predictions && <PredictionTable predictions={status.predictions} />}
          </SpaceBetween>
        </Container>
      )}
    </SpaceBetween>
  );
}
