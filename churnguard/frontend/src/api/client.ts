// Typed fetch wrapper over every /api route in the FEAT-003 contract. Each
// method returns the real backend payload; on a non-2xx response it throws an
// ApiError carrying the backend's error envelope so the UI can surface the real
// AWS code/message/request-id rather than inventing one.

import type {
  ApproveResponse,
  ArchitectureResponse,
  AsyncResultResponse,
  AsyncSubmitResponse,
  BatchStatusResponse,
  BatchSubmitResponse,
  BlueGreenParams,
  BlueGreenStatusResponse,
  BlueGreenSubmitResponse,
  ChurnFeatures,
  ErrorEnvelope,
  EventsResponse,
  HealthResponse,
  MultiModelListResponse,
  MultiModelPredictResponse,
  PipelineRunResponse,
  PipelineStatusResponse,
  RealtimeResponse,
  RegistryResponse,
  ServerlessResponse,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000/api";

export class ApiError extends Error {
  code: string;
  awsRequestId: string | null;
  detail: unknown;
  httpStatus: number;

  constructor(httpStatus: number, envelope: ErrorEnvelope["error"] | null, fallback?: string) {
    super(envelope?.message ?? fallback ?? `Request failed with status ${httpStatus}`);
    this.name = "ApiError";
    this.httpStatus = httpStatus;
    this.code = envelope?.code ?? "UNKNOWN";
    this.awsRequestId = envelope?.aws_request_id ?? null;
    this.detail = envelope?.detail ?? null;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...(init?.headers ?? {}),
      },
    });
  } catch (networkErr) {
    throw new ApiError(0, null, `Network error: ${(networkErr as Error).message}`);
  }

  const text = await res.text();
  let body: unknown = null;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = null;
    }
  }

  if (!res.ok) {
    const envelope = (body as ErrorEnvelope | null)?.error ?? null;
    throw new ApiError(res.status, envelope, text || undefined);
  }

  return body as T;
}

export const api = {
  getHealth: () => request<HealthResponse>("/health"),
  getArchitecture: () => request<ArchitectureResponse>("/architecture"),

  predictRealtime: (features: ChurnFeatures) =>
    request<RealtimeResponse>("/predict/realtime", {
      method: "POST",
      body: JSON.stringify({ features }),
    }),

  predictServerless: (features: ChurnFeatures) =>
    request<ServerlessResponse>("/predict/serverless", {
      method: "POST",
      body: JSON.stringify({ features }),
    }),

  predictAsync: (records: ChurnFeatures[]) =>
    request<AsyncSubmitResponse>("/predict/async", {
      method: "POST",
      body: JSON.stringify({ records }),
    }),

  getAsyncResult: (outputLocation: string) =>
    request<AsyncResultResponse>(
      `/predict/async/result?outputLocation=${encodeURIComponent(outputLocation)}`
    ),

  runBatch: (inputS3Uri?: string) =>
    request<BatchSubmitResponse>("/batch", {
      method: "POST",
      body: JSON.stringify(inputS3Uri ? { inputS3Uri } : {}),
    }),

  getBatchStatus: (jobName: string) =>
    request<BatchStatusResponse>(`/batch/${encodeURIComponent(jobName)}`),

  runPipeline: () =>
    request<PipelineRunResponse>("/pipeline/run", { method: "POST", body: "{}" }),

  getPipelineStatus: (arn: string) =>
    request<PipelineStatusResponse>(`/pipeline/status?arn=${encodeURIComponent(arn)}`),

  getRegistry: () => request<RegistryResponse>("/registry"),

  approveModel: (modelPackageArn: string) =>
    request<ApproveResponse>("/registry/approve", {
      method: "POST",
      body: JSON.stringify({ modelPackageArn }),
    }),

  getRecentEvents: (limit = 20) =>
    request<EventsResponse>(`/events/recent?limit=${limit}`),

  deployBlueGreen: (params: BlueGreenParams) =>
    request<BlueGreenSubmitResponse>("/deploy/bluegreen", {
      method: "POST",
      body: JSON.stringify(params),
    }),

  getBlueGreenStatus: () =>
    request<BlueGreenStatusResponse>("/deploy/bluegreen/status"),

  getMultiModelModels: () =>
    request<MultiModelListResponse>("/hosting/multimodel/models"),

  predictMultiModel: (targetModel: string, features: ChurnFeatures) =>
    request<MultiModelPredictResponse>("/hosting/multimodel", {
      method: "POST",
      body: JSON.stringify({ targetModel, features }),
    }),
};
