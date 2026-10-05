import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";

// isReadOnly is read at module load, so flip the mocked value and re-import the
// tab for each case via vi.resetModules() + dynamic import.
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

async function renderBatchTab(readOnly: boolean) {
  vi.resetModules();
  vi.doMock("../config/deployment", () => ({ isReadOnly: readOnly }));
  const { default: BatchTab } = await import("../tabs/BatchTab");
  render(<BatchTab />);
}

beforeEach(() => {
  stubFetch();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  vi.doUnmock("../config/deployment");
});

describe("BatchTab read-only gating", () => {
  it("disables Start batch transform and shows the banner when read-only", async () => {
    await renderBatchTab(true);
    expect(screen.getByRole("button", { name: /start batch transform/i })).toBeDisabled();
    expect(screen.getByText(/read-only demo/i)).toBeInTheDocument();
  });

  it("keeps Start batch transform enabled and hides the banner by default", async () => {
    await renderBatchTab(false);
    expect(screen.getByRole("button", { name: /start batch transform/i })).toBeEnabled();
    expect(screen.queryByText(/read-only demo/i)).not.toBeInTheDocument();
  });
});
