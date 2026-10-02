import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "../api/client";

function mockFetchOnce(status: number, body: unknown) {
  const text = typeof body === "string" ? body : JSON.stringify(body);
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: status >= 200 && status < 300,
      status,
      text: () => Promise.resolve(text),
    } as Response)
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("api client", () => {
  it("returns the parsed realtime payload on success", async () => {
    mockFetchOnce(200, {
      endpoint: "churnguard-realtime",
      predictions: [{ churnProbability: 0.73, churn: true }],
      latencyMs: 42,
    });
    const res = await api.predictRealtime({ tenure: 1 } as never);
    expect(res.endpoint).toBe("churnguard-realtime");
    expect(res.predictions[0].churn).toBe(true);
  });

  it("surfaces the backend error envelope as an ApiError", async () => {
    mockFetchOnce(409, {
      error: {
        code: "MODEL_NOT_READY",
        message: "feature schema not loaded",
        aws_request_id: "req-123",
        detail: null,
      },
    });
    await expect(api.predictRealtime({ tenure: 1 } as never)).rejects.toMatchObject({
      code: "MODEL_NOT_READY",
      awsRequestId: "req-123",
      httpStatus: 409,
    });
  });

  it("renders the empty-registry response verbatim", async () => {
    mockFetchOnce(200, { group: "churnguard-churn", versions: [] });
    const res = await api.getRegistry();
    expect(res.versions).toHaveLength(0);
  });

  it("wraps a network failure in an ApiError", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("boom")));
    await expect(api.getHealth()).rejects.toBeInstanceOf(ApiError);
  });
});
