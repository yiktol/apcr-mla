// Response/request types mirroring the FEAT-003 backend route contract. These
// are the exact shapes the real FastAPI backend returns; no field is invented
// by the frontend.

export interface ChurnFeatures {
  tenure: number;
  MonthlyCharges: number;
  TotalCharges: number;
  SeniorCitizen: 0 | 1;
  gender: "Female" | "Male";
  Partner: "Yes" | "No";
  Dependents: "Yes" | "No";
  PhoneService: "Yes" | "No";
  MultipleLines: "Yes" | "No" | "No phone service";
  InternetService: "DSL" | "Fiber optic" | "No";
  OnlineSecurity: "Yes" | "No" | "No internet service";
  OnlineBackup: "Yes" | "No" | "No internet service";
  DeviceProtection: "Yes" | "No" | "No internet service";
  TechSupport: "Yes" | "No" | "No internet service";
  StreamingTV: "Yes" | "No" | "No internet service";
  StreamingMovies: "Yes" | "No" | "No internet service";
  Contract: "Month-to-month" | "One year" | "Two year";
  PaperlessBilling: "Yes" | "No";
  PaymentMethod:
    | "Electronic check"
    | "Mailed check"
    | "Bank transfer (automatic)"
    | "Credit card (automatic)";
}

export interface ErrorEnvelope {
  error: {
    code: string;
    message: string;
    aws_request_id: string | null;
    detail: unknown;
  };
}

export interface Prediction {
  churnProbability: number;
  churn: boolean;
}

export interface HealthResponse {
  region: string;
  endpoints: { realtime: string; serverless: string; async: string; mme: string };
  buckets: { data: string; models: string; async: string; batch: string };
  snsTopicArn: string | null;
  featureSchemaLoaded: boolean;
  endpointStatus: Record<string, string>;
}

export interface ArchitectureNode {
  id: string;
  label: string;
  type: string;
}

export interface ArchitectureEdge {
  from: string;
  to: string;
}

export interface ArchitectureResponse {
  nodes: ArchitectureNode[];
  edges: ArchitectureEdge[];
}

export interface RealtimeResponse {
  endpoint: string;
  predictions: Prediction[];
  latencyMs: number;
}

export interface ServerlessResponse {
  endpoint: string;
  predictions: Prediction[];
  latencyMs: number;
  coldStartLikely: boolean;
  coldStartThresholdMs: number;
}

export interface AsyncSubmitResponse {
  endpoint: string;
  inferenceId: string;
  outputLocation: string;
  inputLocation: string;
  snsTopicArn: string | null;
  rowCount: number;
  status: "InProgress";
}

export type AsyncResultResponse =
  | { status: "Failed"; reason: string }
  | { status: "InProgress" }
  | { status: "Completed"; predictions: Prediction[] };

export interface BatchSubmitResponse {
  jobName: string;
  status: "InProgress";
  outputLocation: string;
}

export interface BatchStatusResponse {
  jobName: string;
  status: string;
  outputLocation: string;
  rowCount?: number;
  failureReason?: string;
  predictions?: Prediction[];
}

export interface PipelineRunResponse {
  pipelineExecutionArn: string;
  status: string;
}

export interface PipelineStep {
  name: string;
  status: string;
  startTime?: string | null;
  endTime?: string | null;
  failureReason?: string | null;
}

export interface PipelineStatusResponse {
  arn: string;
  status: string;
  steps: PipelineStep[];
}

export interface RegistryVersion {
  arn: string;
  version: number;
  status: string;
  approvalStatus: string;
  createdAt: string;
  metrics: { accuracy?: number; auc?: number };
}

export interface RegistryResponse {
  group: string;
  versions: RegistryVersion[];
}

export interface ApproveResponse {
  modelPackageArn: string;
  approvalStatus: string;
}

export interface EventRecord {
  timestamp: string;
  modelPackageArn: string;
  message: string;
}

export interface EventsResponse {
  events: EventRecord[];
}

export interface BlueGreenSubmitResponse {
  endpoint: string;
  newConfig: string;
  newModel: string;
  mode: string;
  rev: number;
  status: string;
}

export interface BlueGreenVariant {
  variantName: string;
  currentWeight: number;
  desiredWeight: number;
  currentInstanceCount: number;
  desiredInstanceCount: number;
  variantStatus: unknown[];
}

export interface BlueGreenStatusResponse {
  status: string;
  endpointConfig: string;
  lastDeploymentStatus?: string;
  pendingDeploymentSummary?: {
    endpointConfigName: string;
    variants: BlueGreenVariant[];
  };
}

export interface MultiModelEntry {
  targetModel: string;
  key: string;
  size: number;
}

export interface MultiModelListResponse {
  endpoint: string;
  models: MultiModelEntry[];
}

export interface MultiModelPredictResponse {
  endpoint: string;
  targetModel: string;
  prediction: Prediction;
  latencyMs: number;
}

export interface BlueGreenParams {
  modelPackageArn: string;
  mode: "canary" | "linear";
  canaryPercent?: number;
  linearStepPercent?: number;
  bakeTimeSeconds?: number;
}
