import { useEffect, useRef, useState } from "react";
import Button from "@cloudscape-design/components/button";
import Container from "@cloudscape-design/components/container";
import Header from "@cloudscape-design/components/header";
import SpaceBetween from "@cloudscape-design/components/space-between";
import FormField from "@cloudscape-design/components/form-field";
import Select from "@cloudscape-design/components/select";
import Input from "@cloudscape-design/components/input";
import StatusIndicator from "@cloudscape-design/components/status-indicator";
import Table from "@cloudscape-design/components/table";
import Box from "@cloudscape-design/components/box";
import Alert from "@cloudscape-design/components/alert";
import { api } from "../api/client";
import type {
  BlueGreenStatusResponse,
  BlueGreenSubmitResponse,
  BlueGreenVariant,
  RegistryResponse,
} from "../api/types";
import Identifier from "../components/Identifier";
import ErrorAlert from "../components/ErrorAlert";
import { isReadOnly } from "../config/deployment";

interface Props {
  registry: RegistryResponse | null;
}

const STATUS_TERMINAL = ["InService", "Failed", "RollingBack"];

function pickGreenCandidate(registry: RegistryResponse | null): string | null {
  if (!registry) return null;
  const approved = registry.versions
    .filter((v) => v.approvalStatus === "Approved")
    .sort((a, b) => b.version - a.version);
  return approved.length > 0 ? approved[0].arn : null;
}

export default function BlueGreenTab({ registry }: Props) {
  const [mode, setMode] = useState<"canary" | "linear">("canary");
  const [canaryPercent, setCanaryPercent] = useState("10");
  const [linearStepPercent, setLinearStepPercent] = useState("20");
  const [bakeTimeSeconds, setBakeTimeSeconds] = useState("120");
  const [submit, setSubmit] = useState<BlueGreenSubmitResponse | null>(null);
  const [status, setStatus] = useState<BlueGreenStatusResponse | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const greenArn = pickGreenCandidate(registry);

  const clearTimer = () => {
    if (timer.current) {
      clearTimeout(timer.current);
      timer.current = null;
    }
  };
  useEffect(() => clearTimer, []);

  const pollStatus = async () => {
    try {
      const s = await api.getBlueGreenStatus();
      setStatus(s);
      if (!STATUS_TERMINAL.includes(s.status)) {
        timer.current = setTimeout(pollStatus, 5000);
      }
    } catch (e) {
      setError(e);
    }
  };

  const deploy = async () => {
    if (!greenArn) return;
    setLoading(true);
    setError(null);
    clearTimer();
    try {
      const res = await api.deployBlueGreen({
        modelPackageArn: greenArn,
        mode,
        canaryPercent: parseInt(canaryPercent, 10),
        linearStepPercent: parseInt(linearStepPercent, 10),
        bakeTimeSeconds: parseInt(bakeTimeSeconds, 10),
      });
      setSubmit(res);
      pollStatus();
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
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
            description="Promotes the newest Approved registry version (from the Pipeline & Registry tab) onto churnguard-realtime with a canary or linear traffic shift and a CloudWatch auto-rollback alarm."
          >
            Blue/Green deployment
          </Header>
        }
      >
        <SpaceBetween size="m">
          {greenArn ? (
            <Identifier label="Green candidate (newest Approved ARN)" value={greenArn} />
          ) : (
            <Alert type="info" header="No approved version available">
              Run the pipeline and approve a model version on the Pipeline &amp; Registry tab.
              The newest Approved package becomes the green candidate here.
            </Alert>
          )}

          <FormField label="Shift mode">
            <Select
              selectedOption={{ label: mode, value: mode }}
              options={[
                { label: "canary", value: "canary" },
                { label: "linear", value: "linear" },
              ]}
              onChange={({ detail }) => setMode(detail.selectedOption.value as "canary" | "linear")}
            />
          </FormField>

          {mode === "canary" ? (
            <FormField label="Canary percent (1–50)">
              <Input
                type="number"
                value={canaryPercent}
                onChange={({ detail }) => setCanaryPercent(detail.value)}
              />
            </FormField>
          ) : (
            <FormField label="Linear step percent (10–50)">
              <Input
                type="number"
                value={linearStepPercent}
                onChange={({ detail }) => setLinearStepPercent(detail.value)}
              />
            </FormField>
          )}

          <FormField label="Bake time seconds (30–1800)">
            <Input
              type="number"
              value={bakeTimeSeconds}
              onChange={({ detail }) => setBakeTimeSeconds(detail.value)}
            />
          </FormField>

          <Button
            variant="primary"
            onClick={deploy}
            loading={loading}
            disabled={isReadOnly || !greenArn}
          >
            Start blue/green deployment
          </Button>
        </SpaceBetween>
      </Container>

      <ErrorAlert error={error} />

      {submit && (
        <Container header={<Header variant="h2">Deployment</Header>}>
          <SpaceBetween size="m">
            <Identifier label="Endpoint" value={submit.endpoint} />
            <Identifier label="New model" value={submit.newModel} />
            <Identifier label="New endpoint config" value={submit.newConfig} />
            <SpaceBetween size="xxs">
              <strong>Mode / revision</strong>
              <span>
                {submit.mode} · rev {submit.rev}
              </span>
            </SpaceBetween>
          </SpaceBetween>
        </Container>
      )}

      {status && (
        <Container header={<Header variant="h2">Live traffic shift</Header>}>
          <SpaceBetween size="m">
            <SpaceBetween size="xxs">
              <strong>Endpoint status</strong>
              {status.status === "InService" ? (
                <StatusIndicator type="success">InService</StatusIndicator>
              ) : status.status === "Failed" || status.status === "RollingBack" ? (
                <StatusIndicator type="error">{status.status}</StatusIndicator>
              ) : (
                <StatusIndicator type="in-progress">{status.status}</StatusIndicator>
              )}
            </SpaceBetween>
            <Identifier label="Active endpoint config" value={status.endpointConfig} />
            {status.lastDeploymentStatus && (
              <SpaceBetween size="xxs">
                <strong>Last deployment status</strong>
                <span>{status.lastDeploymentStatus}</span>
              </SpaceBetween>
            )}
            {status.pendingDeploymentSummary ? (
              <Table<BlueGreenVariant>
                variant="embedded"
                header={<Header variant="h3">Pending deployment variants</Header>}
                items={status.pendingDeploymentSummary.variants}
                columnDefinitions={[
                  { id: "name", header: "Variant", cell: (v) => v.variantName },
                  { id: "cw", header: "Current weight", cell: (v) => v.currentWeight },
                  { id: "dw", header: "Desired weight", cell: (v) => v.desiredWeight },
                  { id: "ci", header: "Current instances", cell: (v) => v.currentInstanceCount },
                  { id: "di", header: "Desired instances", cell: (v) => v.desiredInstanceCount },
                ]}
                empty="No pending variants."
              />
            ) : (
              <Box color="text-body-secondary">
                No pending deployment — the shift has completed or has not started.
              </Box>
            )}
          </SpaceBetween>
        </Container>
      )}
    </SpaceBetween>
  );
}
