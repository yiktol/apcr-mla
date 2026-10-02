import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ServerlessTab from "../tabs/ServerlessTab";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("ServerlessTab", () => {
  it("labels coldStartLikely as a heuristic, not an observed fact", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              endpoint: "churnguard-serverless",
              predictions: [{ churnProbability: 0.6, churn: true }],
              latencyMs: 2300,
              coldStartLikely: true,
              coldStartThresholdMs: 1500,
            })
          ),
      } as Response)
    );

    render(<ServerlessTab />);
    await userEvent.click(screen.getByRole("button", { name: /score/i }));

    expect(await screen.findByText(/cold start likely/i)).toBeInTheDocument();
    // The heuristic disclaimer must be present so the UI never presents cold start
    // as an observed fact.
    expect(screen.getByText(/heuristic only/i)).toBeInTheDocument();
  });
});
