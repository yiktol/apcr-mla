import { useEffect, useRef, useState } from "react";
import Button from "@cloudscape-design/components/button";
import Container from "@cloudscape-design/components/container";
import Header from "@cloudscape-design/components/header";
import SpaceBetween from "@cloudscape-design/components/space-between";
import StatusIndicator from "@cloudscape-design/components/status-indicator";
import Table from "@cloudscape-design/components/table";
import Box from "@cloudscape-design/components/box";
import Badge from "@cloudscape-design/components/badge";
import { api } from "../api/client";
import type {
  EventRecord,
  PipelineStatusResponse,
  PipelineStep,
  RegistryResponse,
  RegistryVersion,
} from "../api/types";
import Identifier from "../components/Identifier";
import ErrorAlert from "../components/ErrorAlert";

interface Props {
  registry: RegistryResponse | null;
  refreshRegistry: () => void;
}

const PIPELINE_TERMINAL = ["Succeeded", "Failed", "Stopped"];

function stepIndicator(status: string) {
  if (status === "Succeeded") return <StatusIndicator type="success">Succeeded</StatusIndicator>;
  if (status === "Executing") return <StatusIndicator type="in-progress">Executing</StatusIndicator>;
  if (status === "Failed") return <StatusIndicator type="error">Failed</StatusIndicator>;
  return <StatusIndicator type="pending">{status}</StatusIndicator>;
}

export default function PipelineRegistryTab({ registry, refreshRegistry }: Props) {
  const [pipelineArn, setPipelineArn] = useState<string | null>(null);
  const [pipeline, setPipeline] = useState<PipelineStatusResponse | null>(null);
  const [events, setEvents] = useState<EventRecord[]>([]);
  const [error, setError] = useState<unknown>(null);
  const [runLoading, setRunLoading] = useState(false);
  const [approveArn, setApproveArn] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearTimer = () => {
    if (timer.current) {
      clearTimeout(timer.current);
      timer.current = null;
    }
  };
  useEffect(() => clearTimer, []);

  const loadEvents = async () => {
    try {
      const res = await api.getRecentEvents();
      setEvents(res.events);
    } catch (e) {
      setError(e);
    }
  };

  useEffect(() => {
    loadEvents();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const pollPipeline = async (arn: string) => {
    try {
      const s = await api.getPipelineStatus(arn);
      setPipeline(s);
      if (!PIPELINE_TERMINAL.includes(s.status)) {
        timer.current = setTimeout(() => pollPipeline(arn), 5000);
      } else {
        refreshRegistry();
      }
    } catch (e) {
      setError(e);
    }
  };

  const runPipeline = async () => {
    setRunLoading(true);
    setError(null);
    clearTimer();
    try {
      const res = await api.runPipeline();
      setPipelineArn(res.pipelineExecutionArn);
      setPipeline({ arn: res.pipelineExecutionArn, status: res.status, steps: [] });
      pollPipeline(res.pipelineExecutionArn);
    } catch (e) {
      setError(e);
    } finally {
      setRunLoading(false);
    }
  };

  const approve = async (arn: string) => {
    setApproveArn(arn);
    setError(null);
    try {
      await api.approveModel(arn);
      refreshRegistry();
      // Approval emits a real EventBridge event -> Lambda handoff; refresh the panel.
      await loadEvents();
    } catch (e) {
      setError(e);
    } finally {
      setApproveArn(null);
    }
  };

  const versions = registry?.versions ?? [];

  return (
    <SpaceBetween size="l">
      <Container
        header={
          <Header
            variant="h2"
            description="Processing → Training → Evaluation → Condition → RegisterModel. Each step status comes from describe_pipeline_execution."
            actions={
              <Button variant="primary" onClick={runPipeline} loading={runLoading}>
                Run pipeline
              </Button>
            }
          >
            SageMaker Pipeline
          </Header>
        }
      >
        {pipeline ? (
          <SpaceBetween size="m">
            <Identifier label="Pipeline execution ARN" value={pipelineArn} />
            <SpaceBetween size="xxs">
              <strong>Execution status</strong>
              {stepIndicator(pipeline.status)}
            </SpaceBetween>
            <Table<PipelineStep>
              variant="embedded"
              items={pipeline.steps}
              columnDefinitions={[
                { id: "name", header: "Step", cell: (s) => s.name },
                { id: "status", header: "Status", cell: (s) => stepIndicator(s.status) },
                {
                  id: "failure",
                  header: "Failure reason",
                  cell: (s) => s.failureReason ?? "—",
                },
              ]}
              empty="No step status yet — the execution is starting."
            />
          </SpaceBetween>
        ) : (
          <Box color="text-body-secondary">
            No pipeline run yet. Click “Run pipeline” to start a real execution.
          </Box>
        )}
      </Container>

      <Container
        header={
          <Header
            variant="h2"
            description="Model package group churnguard-churn. Only the pipeline's RegisterModel step registers versions."
            actions={<Button onClick={refreshRegistry}>Refresh</Button>}
          >
            Model Registry
          </Header>
        }
      >
        {versions.length === 0 ? (
          <Box textAlign="center" color="text-body-secondary" padding="l">
            <SpaceBetween size="xs">
              <Box variant="strong">No versions yet — run the pipeline</Box>
              <Box variant="small">
                An empty registry on a fresh deploy is expected: the baseline artifact that
                stands up the endpoints is deliberately not registered. The first live
                pipeline run above registers a version here.
              </Box>
            </SpaceBetween>
          </Box>
        ) : (
          <Table<RegistryVersion>
            variant="embedded"
            items={versions}
            columnDefinitions={[
              { id: "version", header: "Version", cell: (v) => v.version },
              {
                id: "approval",
                header: "Approval",
                cell: (v) => (
                  <Badge color={v.approvalStatus === "Approved" ? "green" : "grey"}>
                    {v.approvalStatus}
                  </Badge>
                ),
              },
              {
                id: "accuracy",
                header: "Accuracy",
                cell: (v) => (v.metrics.accuracy !== undefined ? v.metrics.accuracy.toFixed(4) : "—"),
              },
              {
                id: "auc",
                header: "AUC",
                cell: (v) => (v.metrics.auc !== undefined ? v.metrics.auc.toFixed(4) : "—"),
              },
              { id: "arn", header: "ARN", cell: (v) => <Identifier label="" value={v.arn} /> },
              {
                id: "action",
                header: "Action",
                cell: (v) =>
                  v.approvalStatus === "Approved" ? (
                    <StatusIndicator type="success">Approved</StatusIndicator>
                  ) : (
                    <Button
                      onClick={() => approve(v.arn)}
                      loading={approveArn === v.arn}
                      disabled={approveArn !== null}
                    >
                      Approve
                    </Button>
                  ),
              },
            ]}
            empty="No versions."
          />
        )}
      </Container>

      <Container
        header={
          <Header
            variant="h2"
            description="Approving a version emits a real SageMaker Model Package State Change event; the EventBridge rule invokes the Lambda handler, which logs to /churnguard/events."
            actions={<Button onClick={loadEvents}>Refresh events</Button>}
          >
            EventBridge → Lambda handoff
          </Header>
        }
      >
        {events.length === 0 ? (
          <Box color="text-body-secondary">
            No events yet. Approve a model version above to trigger the EventBridge → Lambda
            handoff.
          </Box>
        ) : (
          <Table<EventRecord>
            variant="embedded"
            items={events}
            columnDefinitions={[
              { id: "ts", header: "Timestamp", cell: (e) => e.timestamp },
              {
                id: "arn",
                header: "Model package ARN",
                cell: (e) => <Identifier label="" value={e.modelPackageArn} />,
              },
              { id: "msg", header: "Message", cell: (e) => e.message },
            ]}
            empty="No events."
          />
        )}
      </Container>

      <ErrorAlert error={error} />
    </SpaceBetween>
  );
}
