import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import PipelineRegistryTab from "../tabs/PipelineRegistryTab";
import type { RegistryResponse } from "../api/types";

// Stub the events fetch the tab fires on mount so nothing hits the network.
afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  vi.doUnmock("../config/deployment");
});

function stubEventsFetch() {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      text: () => Promise.resolve(JSON.stringify({ events: [] })),
    } as Response)
  );
}

describe("PipelineRegistryTab", () => {
  it("renders the explicit empty-registry state when there are no versions", async () => {
    stubEventsFetch();
    render(
      <PipelineRegistryTab
        registry={{ group: "churnguard-churn", versions: [] }}
        refreshRegistry={() => {}}
      />
    );
    expect(screen.getByText(/no versions yet/i)).toBeInTheDocument();
    await waitFor(() => expect(fetch).toHaveBeenCalled());
  });

  it("lists a registered version with its approval status", async () => {
    stubEventsFetch();
    render(
      <PipelineRegistryTab
        registry={{
          group: "churnguard-churn",
          versions: [
            {
              arn: "arn:aws:sagemaker:us-east-1:875692608981:model-package/churnguard-churn/1",
              version: 1,
              status: "Completed",
              approvalStatus: "PendingManualApproval",
              createdAt: "2025-01-01T00:00:00Z",
              metrics: { accuracy: 0.81, auc: 0.86 },
            },
          ],
        }}
        refreshRegistry={() => {}}
      />
    );
    expect(screen.getByText("PendingManualApproval")).toBeInTheDocument();
    expect(screen.getByText("0.8100")).toBeInTheDocument();
    await waitFor(() => expect(fetch).toHaveBeenCalled());
  });
});

// A registry with a pending version so the per-row Approve button renders.
const pendingRegistry: RegistryResponse = {
  group: "churnguard-churn",
  versions: [
    {
      arn: "arn:aws:sagemaker:us-east-1:875692608981:model-package/churnguard-churn/1",
      version: 1,
      status: "Completed",
      approvalStatus: "PendingManualApproval",
      createdAt: "2025-01-01T00:00:00Z",
      metrics: { accuracy: 0.81, auc: 0.86 },
    },
  ],
};

// isReadOnly is read at module load; flip the mocked value and re-import the tab
// per case via vi.resetModules() + dynamic import.
async function renderPipelineRegistryTab(readOnly: boolean) {
  vi.resetModules();
  vi.doMock("../config/deployment", () => ({ isReadOnly: readOnly }));
  const { default: Tab } = await import("../tabs/PipelineRegistryTab");
  render(<Tab registry={pendingRegistry} refreshRegistry={() => {}} />);
}

describe("PipelineRegistryTab read-only gating", () => {
  it("disables Run pipeline and Approve and shows the banner when read-only", async () => {
    stubEventsFetch();
    await renderPipelineRegistryTab(true);
    expect(screen.getByRole("button", { name: /run pipeline/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /approve/i })).toBeDisabled();
    expect(screen.getByText(/read-only demo/i)).toBeInTheDocument();
    // Viewing/refresh controls stay interactive.
    expect(screen.getByRole("button", { name: /^refresh$/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /refresh events/i })).toBeEnabled();
  });

  it("keeps Run pipeline and Approve enabled and hides the banner by default", async () => {
    stubEventsFetch();
    await renderPipelineRegistryTab(false);
    expect(screen.getByRole("button", { name: /run pipeline/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /approve/i })).toBeEnabled();
    expect(screen.queryByText(/read-only demo/i)).not.toBeInTheDocument();
  });
});
