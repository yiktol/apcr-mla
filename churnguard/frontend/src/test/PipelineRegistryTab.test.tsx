import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import PipelineRegistryTab from "../tabs/PipelineRegistryTab";

// Stub the events fetch the tab fires on mount so nothing hits the network.
afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
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
