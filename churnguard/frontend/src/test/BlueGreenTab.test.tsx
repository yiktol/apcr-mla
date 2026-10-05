import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import type { RegistryResponse } from "../api/types";

// A registry with an Approved version so greenArn is set and the button's only
// remaining gate is isReadOnly.
const registry: RegistryResponse = {
  group: "churnguard-churn",
  versions: [
    {
      arn: "arn:aws:sagemaker:us-east-1:875692608981:model-package/churnguard-churn/1",
      version: 1,
      status: "Completed",
      approvalStatus: "Approved",
      createdAt: "2025-01-01T00:00:00Z",
      metrics: { accuracy: 0.81, auc: 0.86 },
    },
  ],
};

function stubFetch() {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      text: () => Promise.resolve(JSON.stringify({})),
    } as Response)
  );
}

async function renderBlueGreenTab(readOnly: boolean) {
  vi.resetModules();
  vi.doMock("../config/deployment", () => ({ isReadOnly: readOnly }));
  const { default: BlueGreenTab } = await import("../tabs/BlueGreenTab");
  render(<BlueGreenTab registry={registry} />);
}

beforeEach(() => {
  stubFetch();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  vi.doUnmock("../config/deployment");
});

describe("BlueGreenTab read-only gating", () => {
  it("disables Start blue/green deployment and shows the banner when read-only", async () => {
    await renderBlueGreenTab(true);
    expect(
      screen.getByRole("button", { name: /start blue\/green deployment/i })
    ).toBeDisabled();
    expect(screen.getByText(/read-only demo/i)).toBeInTheDocument();
  });

  it("keeps Start blue/green deployment enabled and hides the banner by default", async () => {
    await renderBlueGreenTab(false);
    expect(
      screen.getByRole("button", { name: /start blue\/green deployment/i })
    ).toBeEnabled();
    expect(screen.queryByText(/read-only demo/i)).not.toBeInTheDocument();
  });
});
