# ChurnGuard — Technical Design

APCR Machine Learning Engineer – Associate · Content Review Session 4 · Domain 3: Deployment and Orchestration of ML Workflows.

## Overview

ChurnGuard is a single, real, end-to-end demo that serves one customer-churn model through every SageMaker deployment and orchestration mechanism the Session 4 deck teaches. One XGBoost model, trained for real on the public Telco Customer Churn dataset via a SageMaker training job, is hosted and orchestrated seven ways: real-time endpoint, serverless endpoint, asynchronous endpoint (with SNS), batch transform, a SageMaker Pipeline (Processing → Training → Evaluation → Condition → RegisterModel) feeding the Model Registry, an EventBridge rule on model-package approval, blue/green endpoint updates with canary and linear traffic shifting, and a multi-model endpoint. A FastAPI backend wraps these resources with real boto3 calls (region `us-east-1`), and a React + Vite + TypeScript + Cloudscape frontend presents a landing page plus seven tabs, each wired to a real backend route showing real resource identifiers.

The design maps 1:1 to the deck so it doubles as a teaching artifact: Model Registry (slides 9–11), the four inference options (12–21), real-time hosting / multi-model (22–25), autoscaling (27), IaC with `AWS::SageMaker::*` resources (29), SageMaker Pipelines and the slide-33 DAG (31–33), EventBridge (34), blue/green deployment guardrails with canary/linear (36), and the small MLOps architecture (40–41).

**Hard constraint restated and honored throughout:** no simulation, no fake data, no mocks, no "simulated mode," no placeholder predictors. Every route invokes a real AWS resource; the model is trained by a real training job on the real dataset. Where a resource cannot exist yet (e.g. no model version approved), routes return an honest, explicit "not ready" state with the real reason — never a fabricated result.

## Technology stack (locked once approved)

- **ML / training:** SageMaker built-in XGBoost algorithm, image `683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.7-1` (us-east-1 built-in XGBoost account). Training job type `ml.m5.large`. Framework version pinned to `1.7-1` everywhere (training, pipeline, inference containers) so the model artifact and serving container agree.
- **Processing / evaluation:** SageMaker Processing with the SKLearn processor image (us-east-1 account `683313688378`, `sagemaker-scikit-learn`, framework version `1.2-1`) running project-owned scripts. `ml.m5.large`. **The ECR tag is NOT the bare `1.2-1` — it is platform/python-qualified (e.g. `1.2-1-cpu-py3`), so the image URI is SDK-resolved, never hard-coded (Finding 3).** The offline `ml/` scripts (`build_pipeline.py`, `run_training.py` sub-step 5a) resolve it at authoring/deploy time via `sagemaker.image_uris.retrieve(framework="sklearn", region="us-east-1", version="1.2-1", image_scope="processing", instance_type="ml.m5.large")` and pass the resolved URI into the processing job and the pipeline `ProcessingStep` definition. By contrast the XGBoost tag `1.7-1` IS literal and is used verbatim; the SKLearn tag differs and must be resolved.
- **Backend:** FastAPI (Python 3.11) + Uvicorn, boto3 ≥ 1.34, pinned. All boto3 clients created with `region_name="us-east-1"` via a single factory. SageMaker Python SDK (`sagemaker` ≥ 2.200) is used **only** in the offline `ml/` provisioning scripts (for image-uri resolution and pipeline authoring convenience); the FastAPI runtime depends on boto3 only.
- **Frontend:** React 18 + Vite 5 + TypeScript 5 + `@cloudscape-design/components` and `@cloudscape-design/global-styles`. Routing via `@cloudscape-design/components` `AppLayout` + a tab-based content area (no heavy router needed; single page with Cloudscape `Tabs`).
- **IaC:** AWS CloudFormation YAML only (no CDK). Multiple nested-free stacks orchestrated by a thin `deploy.sh` / `teardown.sh` wrapper.
- **Dataset:** IBM Telco Customer Churn CSV, source URL `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv` (verified reachable, 7043 rows, 21 columns, target `Churn` ∈ {Yes,No}).

## Environment & account facts (verified, do not re-verify by mutating)

- Account `875692608981`, region `us-east-1`.
- Caller: `AWSReservedSSO_AdministratorAccess` (admin). Credentials already active.
- **Execution role (verified): `arn:aws:iam::875692608981:role/AmazonSageMaker-ExecutionRole`.** Trust policy allows `sagemaker.amazonaws.com` to assume it; attached managed policies are `AdministratorAccess` + `AmazonSageMakerFullAccess`. This covers SageMaker, S3, SNS, and ECR pull. **No new IAM role is created for SageMaker.** The role ARN is a CloudFormation parameter (`ExecutionRoleArn`) defaulting to this value; dated fallbacks (`AmazonSageMaker-ExecutionRole-2025...`) can be substituted by overriding the parameter.
- The machine's AWS CLI default region is `ap-southeast-1`. **Every** CLI call carries `--region us-east-1`; **every** boto3 client sets `region_name="us-east-1"`. This is enforced in code by a single `boto_client()` factory and in scripts by exporting `AWS_DEFAULT_REGION=us-east-1` and still passing `--region` explicitly.

## Naming conventions and concrete resource names

A single prefix `churnguard` and a short deploy id keep names stable and teardown-able. Buckets must be globally unique, so they embed the account id. Fixed names (not account-scoped) are used for SageMaker resources so the backend can address them without extra lookups; the stacks own them.

S3 buckets (region us-east-1):
- Data: `churnguard-data-875692608981-us-east-1`
- Model artifacts: `churnguard-models-875692608981-us-east-1`
- Async output: `churnguard-async-875692608981-us-east-1`
- Batch output: `churnguard-batch-875692608981-us-east-1`

Key S3 prefixes:
- `s3://churnguard-data-.../raw/Telco-Customer-Churn.csv` — raw dataset
- `s3://churnguard-data-.../processed/{train,validation,test}/` — processing-step output (libsvm/CSV for XGBoost)
- `s3://churnguard-models-.../training/<job>/output/model.tar.gz` — training artifact
- `s3://churnguard-models-.../mme/` — multi-model endpoint model prefix (holds `churn-v1.tar.gz`, `churn-v2.tar.gz`)
- `s3://churnguard-async-.../input/`, `.../output/` — async payloads and results
- `s3://churnguard-batch-.../input/batch.csv`, `.../output/` — batch transform IO

SageMaker (fixed names, all **CloudFormation-owned** unless noted):
- Model package group: `churnguard-churn`
- Models: `churnguard-realtime-model`, `churnguard-serverless-model`, `churnguard-async-model`, `churnguard-mme-model`
- EndpointConfigs: `churnguard-realtime-config`, `churnguard-serverless-config`, `churnguard-async-config`, `churnguard-mme-config` (the realtime config has a single production variant **`AllTraffic`**)
- Endpoints: `churnguard-realtime`, `churnguard-serverless`, `churnguard-async`, `churnguard-mme`
- **Runtime (NOT CloudFormation-owned), created by `POST /api/deploy/bluegreen`** — all under the distinct `churnguard-bg-` sub-prefix so teardown can sweep them without touching stack-owned resources: `churnguard-bg-model-<rev>`, `churnguard-bg-config-<rev>`, where `<rev>` is unix epoch seconds (`int(time.time())`), unique per update. These are the **only** `churnguard-*` SageMaker resources created outside CloudFormation.
- Pipeline: `churnguard-pipeline`
- SNS topic: `churnguard-async-notifications`
- EventBridge rule: `churnguard-model-approved`
- Lambda target: `churnguard-model-approved-handler`
- CloudWatch log group (EventBridge demonstration target & Lambda logs): `/aws/lambda/churnguard-model-approved-handler` and `/churnguard/events`

CloudFormation stack names: `churnguard-foundation`, `churnguard-registry-pipeline`, `churnguard-realtime`, `churnguard-serverless`, `churnguard-async`, `churnguard-mme`, `churnguard-events`.

## Dataset and feature handling

The raw CSV has a `customerID` (dropped), 19 feature columns (3 numeric: `tenure`, `MonthlyCharges`, `TotalCharges`; the rest categorical), and target `Churn`. `TotalCharges` contains 11 blank strings (known data quirk) coerced to numeric with those rows dropped. The processing script owns all preprocessing deterministically so training and inference agree:

- Drop `customerID`.
- Coerce `TotalCharges` to float; drop the ~11 non-numeric rows.
- Map `Churn` Yes→1 / No→0 as the **first column** (XGBoost built-in expects label first, no header).
- One-hot encode categoricals with a **fixed, sorted column order** persisted to `s3://churnguard-data-.../processed/feature_columns.json`. **This file contains the feature columns only, in inference order, excluding the label `Churn`.** The label is emitted separately as column 0 of the headerless training CSV; the features follow in the `feature_columns.json` order. The backend loads this ordering to build inference payloads so a request's feature vector always matches training. This file is the single source of truth for the inference schema and is the enforcement point for input→vector alignment. **Invariant (owned and asserted by the preprocessing step):** `len(feature_columns) == training_csv_width - 1`.
- 70/15/15 train/validation/test split with fixed `random_state=42`.
- Emit headerless CSV (label first) to `train/`, `validation/`, `test/`.

The resulting feature vector width (~45 columns after one-hot) is written into `feature_columns.json`; the backend never hardcodes it.

## CloudFormation stack decomposition

Seven stacks, deployed in order by `infra/deploy.sh`. Decomposition is driven by lifecycle and billing: foundation is cheap and persistent; endpoint stacks are the billable ones torn down first; registry/pipeline is control-plane only.

1. **`churnguard-foundation`** (`infra/01-foundation.yaml`) — the four S3 buckets (with `BucketPolicy` for TLS-only, versioning off to ease teardown, `DeletionPolicy: Delete`), the SNS topic `churnguard-async-notifications` with an email subscription parameter (optional), and SSM Parameter Store entries publishing bucket names / role ARN / topic ARN for the other stacks and the backend to read. Parameters: `ExecutionRoleArn`, `NotificationEmail` (optional).

2. **`churnguard-registry-pipeline`** (`infra/02-registry-pipeline.yaml`) — `AWS::SageMaker::ModelPackageGroup` (`churnguard-churn`) and `AWS::SageMaker::Pipeline` (`churnguard-pipeline`). The pipeline definition (slide-33 DAG) is authored as JSON by `ml/build_pipeline.py`, which **uploads it to the data bucket** at `s3://churnguard-data-875692608981-us-east-1/pipeline/pipeline-definition.json` **before** this stack is deployed. The template references it via `AWS::SageMaker::Pipeline` `PipelineDefinition.PipelineDefinitionS3Location: {Bucket, Key}` — **not** `PipelineDefinitionBody`. Only the small S3 key (`pipeline/pipeline-definition.json`) and bucket name are passed as stack parameters, which stay well under the 4096-byte CloudFormation `String` parameter limit (a 5-step DAG serializes to 6–15 KB and cannot be passed inline). This keeps the pipeline as real `AWS::SageMaker::Pipeline` IaC, satisfying the slide-29 teaching point. `deploy.sh` sequences `ml/build_pipeline.py` (upload) → `foundation` must already exist so the bucket is present; the ordering below reflects this (foundation → dataset upload → build+upload pipeline def → registry-pipeline stack).

3. **`churnguard-realtime`** (`infra/03-realtime.yaml`) — `AWS::SageMaker::Model` (points at the trained `model.tar.gz` + XGBoost serving image), `AWS::SageMaker::EndpointConfig` with a single named `ProductionVariant` **`AllTraffic`** (`ml.m5.large`, **`InitialInstanceCount: 3`**), `AWS::SageMaker::Endpoint` (`churnguard-realtime`). The instance count is **3, not 1**, specifically so blue/green canary and linear traffic shifts resolve to whole instances and produce observable intermediate states (the slide-36 teaching point). With 3 instances: a 10% `CANARY` step rounds to 1 instance (≈33% of fleet, the smallest whole-instance step), and `LINEAR` with a 20%–50% step size produces 2–3 visible steps. A 1-instance fleet would collapse every shift to a single step and SageMaker may reject sub-instance CAPACITY_PERCENT steps, defeating the demo — so the extra cost is accepted deliberately (reconciled in the Cost section below: realtime is 3 × `ml.m5.large`). A `AWS::ApplicationAutoScaling::ScalableTarget` + `ScalingPolicy` (TargetTracking on `SageMakerVariantInvocationsPerInstance`, `MinCapacity: 3`, `MaxCapacity: 6`) is included to make slide-27 autoscaling real. The model artifact S3 URI is a parameter supplied after training. This stack also **fully defines the blue/green auto-rollback alarm `churnguard-realtime-ModelError`** (consumed by `POST /api/deploy/bluegreen`'s `AutoRollbackConfiguration`), specified concretely so the slide-36 rollback guardrail is reproducible rather than left to the implementer:

   ```yaml
   RealtimeModelErrorAlarm:
     Type: AWS::CloudWatch::Alarm
     Properties:
       AlarmName: churnguard-realtime-ModelError
       Namespace: AWS/SageMaker
       MetricName: Invocation5XXErrors
       Dimensions:
         - { Name: EndpointName, Value: churnguard-realtime }
         - { Name: VariantName, Value: AllTraffic }
       Statistic: Sum
       Period: 60
       EvaluationPeriods: 1
       Threshold: 1
       ComparisonOperator: GreaterThanOrEqualToThreshold
       TreatMissingData: notBreaching
   ```

   The metric is pinned to `Invocation5XXErrors` (the direct "model is erroring" signal SageMaker emits per endpoint/variant), with dimensions `EndpointName=churnguard-realtime` + `VariantName=AllTraffic` so it scopes to this endpoint's live variant. `TreatMissingData: notBreaching` is deliberate: a freshly created alarm — and a quiet green fleet with no traffic — sits in `INSUFFICIENT_DATA`, and without `notBreaching` SageMaker could treat the deployment's health as indeterminate and either stall the shift or spuriously roll back a healthy green fleet. With `notBreaching`, no traffic means "not in alarm," so rollback fires only on a real ≥1 5XX error during the bake window. The alarm exists in stack 3 (not created at deploy time by the backend) so it is guaranteed present and in a known state before any `update_endpoint` references it.

4. **`churnguard-serverless`** (`infra/04-serverless.yaml`) — `AWS::SageMaker::Model` + `EndpointConfig` with `ServerlessConfig` (`MemorySizeInMB: 2048`, `MaxConcurrency: 5`) + `Endpoint` (`churnguard-serverless`). No instance billing when idle.

5. **`churnguard-async`** (`infra/05-async.yaml`) — `AWS::SageMaker::Model` + `EndpointConfig` with a single named `ProductionVariant` **`AllTraffic`**, `AsyncInferenceConfig` (`OutputConfig.S3OutputPath` → async bucket `output/`, `NotificationConfig` setting **both `SuccessTopic` AND `ErrorTopic`** to the `churnguard-async-notifications` topic ARN (Finding 6) — the same ARN for both so neither the success nor the failure signal the UI relies on is ever lost) + `Endpoint` (`churnguard-async`, `ml.m5.large`, `InitialInstanceCount: 1`). An `AWS::ApplicationAutoScaling::ScalableTarget` with `MinCapacity: 0`, `MaxCapacity: 2` carries **two separate, both-required scaling policies** (slide-15 "scale to 0 and back"). The named variant is required for the scalable target to resolve.

   **Why two policies are mandatory (Finding 1).** A target-tracking policy alone can scale the async fleet *down* to 0 when idle but **cannot scale it back up from 0**: once instance count is 0, no instance emits `ApproximateBacklogSizePerInstance`, so the target-tracking alarm never fires and the endpoint is stranded at 0. The only CloudWatch signal available at 0 instances is `HasBacklogWithoutCapacity`, so scale-up-from-0 requires a dedicated step-scaling policy driven by that metric. Without it the async tab works exactly once (while the initial warm instance lasts), then every later `invoke_endpoint_async` queues against a 0-instance fleet, the `.out` never appears, and the backend's honest `InProgress` polling spins forever — silently breaking the slide-15 teaching point on the demo's second use. The two policies are:

   - **Scale-in to 0 (target-tracking):** an `AWS::ApplicationAutoScaling::ScalingPolicy` of `PolicyType: TargetTrackingScaling` with `CustomizedMetricSpecification` on `ApproximateBacklogSizePerInstance` (namespace `AWS/SageMaker`, dimension `EndpointName: churnguard-async`), a modest `TargetValue` (e.g. `5`), and `ScaleInCooldown`/`ScaleOutCooldown` of 300s. This drains the fleet to 0 when the backlog clears.
   - **Scale-up from 0 (step-scaling):** an `AWS::ApplicationAutoScaling::ScalingPolicy` of `PolicyType: StepScaling` on the *same* `ScalableTarget`, with `StepScalingPolicyConfiguration` (`AdjustmentType: ChangeInCapacity`, `Cooldown: 300`, `MetricAggregationType: Average`, a single step `MetricIntervalLowerBound: 0 / ScalingAdjustment: 1`). It is triggered by an `AWS::CloudWatch::Alarm` on metric `HasBacklogWithoutCapacity` — namespace `AWS/SageMaker`, dimension `EndpointName: churnguard-async`, `Statistic: Average`, `Period: 60`, `EvaluationPeriods: 1`, `Threshold: 0`, `ComparisonOperator: GreaterThanThreshold`, `TreatMissingData: notBreaching` — whose `AlarmActions` reference the step-scaling policy ARN. When a request lands on the 0-instance fleet, `HasBacklogWithoutCapacity` goes `> 0`, the alarm fires, and the step policy adds 1 instance so the queued request is served.

   State clearly: scale-to-0 (target-tracking) and scale-up-from-0 (step-scaling on `HasBacklogWithoutCapacity`) are independent policies on one scalable target, and both must exist for the slide-15 behavior to actually work across repeated invocations.

**Autoscaling resource references (Finding 8, applies to stacks 3 and 5):** each `ScalableTarget` uses `ServiceNamespace: sagemaker`, `ResourceId: endpoint/<endpointName>/variant/AllTraffic` (e.g. `endpoint/churnguard-realtime/variant/AllTraffic`, `endpoint/churnguard-async/variant/AllTraffic`), and `ScalableDimension: sagemaker:variant:DesiredInstanceCount`. The production variant is named `AllTraffic` in every `EndpointConfig` so these references resolve; the realtime target is `MinCapacity: 3 / MaxCapacity: 6`, the async target `MinCapacity: 0 / MaxCapacity: 2`.

6. **`churnguard-mme`** (`infra/06-mme.yaml`) — `AWS::SageMaker::Model` with `Mode: MultiModel` and `ModelDataUrl` set to the `mme/` prefix, `EndpointConfig` with a single named `ProductionVariant` **`AllTraffic`** (`ml.m5.large`, `InitialInstanceCount: 1`), `Endpoint` (`churnguard-mme`). Hosts `churn-v1.tar.gz` and `churn-v2.tar.gz`. No autoscaling target on MME (single instance is sufficient for the demo).

   **MME container-capability must be verified for real before this stack is locked (Finding 2).** `Mode: MultiModel` requires the serving container to implement the SageMaker multi-model (MMS) contract (model load/unload + `TargetModel` routing). The image URI/account/tag are verified correct, but whether the *built-in* `sagemaker-xgboost:1.7-1` serving image advertises the multi-model capability is NOT assumed — the implementer MUST confirm it, in `us-east-1`, by exactly one of:
   - **(a) A throwaway MME probe (preferred, authoritative).** Using a clean virtualenv (the local `sagemaker` module is broken), create a model with `Mode: MultiModel` on `683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.7-1` pointing at the `mme/` prefix, create a throwaway endpoint, and call `invoke_endpoint(..., TargetModel="churn-v1.tar.gz", ...)`. All CLI calls and boto3 clients set `--region us-east-1`. Tear the probe resources down afterward. If `CreateModel`/`CreateEndpoint` succeed and the `TargetModel` invoke returns a real probability, the built-in image is confirmed MME-capable and is pinned as-is.
   - **(b) Cite the authoritative AWS doc** confirming the built-in XGBoost container supports multi-model endpoints, recorded here in the design.

   **Fallback if the built-in image is NOT MME-capable:** switch the `churnguard-mme` model's container to the **open-source XGBoost framework inference container** (the SageMaker XGBoost *framework* image, which ships the Multi Model Server / MMS contract and supports `Mode: MultiModel`). Resolve that image URI with `sagemaker.image_uris.retrieve(framework="xgboost", region="us-east-1", version="1.7-1", image_scope="inference", instance_type="ml.m5.large")` (framework container, distinct from the built-in algorithm image) and pin the resolved URI into `infra/06-mme.yaml`. The `churn-v1.tar.gz`/`churn-v2.tar.gz` artifacts remain the same real trained models; only the MME serving image changes, and it must still agree with the pinned `text/csv`-in / `text/csv`-out inference contract (if the framework container requires a project-owned `inference.py` entry point to honor that contract, it is added under `ml/` and packaged into the model tarballs). The MME tab is one of the seven required real hosting paths — it must be a working SageMaker resource, so this verify-or-fallback step is mandatory, not optional. The result of (a)/(b) and whether the fallback was taken is recorded in the README at implementation time.

7. **`churnguard-events`** (`infra/07-events.yaml`) — `AWS::Events::Rule` (`churnguard-model-approved`) matching the real event pattern below, a small `AWS::Lambda::Function` (`churnguard-model-approved-handler`, Python 3.12, inline code) that logs the approved package ARN to CloudWatch and publishes to the SNS topic, the `AWS::Lambda::Permission` for EventBridge, and the rule `Target`. The stack **explicitly creates the `/churnguard/events` `AWS::Logs::LogGroup`** (with a `RetentionInDays`) so the group exists before any event fires and `GET /api/events/recent` never hits a missing group on a fresh deploy; the Lambda writes its structured event records there (in addition to its own `/aws/lambda/...` group). This makes the registry→deployment handoff real and observable.

EventBridge rule pattern (real SageMaker event):
```json
{
  "source": ["aws.sagemaker"],
  "detail-type": ["SageMaker Model Package State Change"],
  "detail": {
    "ModelPackageGroupName": ["churnguard-churn"],
    "ModelApprovalStatus": ["Approved"]
  }
}
```

### Provisioning order (why training happens between stacks)

`AWS::SageMaker::Model` needs a real `model.tar.gz`, which only exists after a training job. The `AWS::SageMaker::Pipeline` stack needs the pipeline definition JSON in S3 first. So `deploy.sh` sequences:
1. Deploy `foundation` (creates the data bucket the next two steps write to).
2. `ml/upload_dataset.py` → raw CSV to data bucket (real download from the IBM URL, then upload).
3. `ml/build_pipeline.py` → author the slide-33 DAG JSON and upload it to `s3://churnguard-data-.../pipeline/pipeline-definition.json`.
4. Deploy `registry-pipeline` (references the uploaded definition via `PipelineDefinitionS3Location`).
5. `ml/run_training.py` → run the standalone (non-pipeline) **baseline** path used to stand up the always-on endpoints without requiring a live pipeline run (the pipeline run is then demonstrated separately from the UI). **The baseline path is NOT just a training job — it runs the identical preprocessing first, as two explicit sub-steps so that `feature_columns.json` and the processed splits always exist on a fresh deploy:**
   - **5a. Preprocess (baseline).** `ml/run_training.py` runs the **same `ml/scripts/preprocessing.py`** against the raw CSV (`s3://churnguard-data-.../raw/Telco-Customer-Churn.csv`) — executed as a real SageMaker Processing job using the SDK-resolved SKLearn processing image (see Technology stack / Finding 3), byte-for-byte the same script the pipeline's `ProcessingStep "DataProcessing"` runs. It writes the identical `processed/{train,validation,test}/` headerless label-first splits **and** `processed/feature_columns.json` to the data bucket. This is the single producer of the backend's inference schema on first provisioning. Because the baseline artifact and the pipeline artifact come from byte-identical preprocessing, the two are interchangeable for the endpoints.
   - **5b. Train (baseline).** The standalone built-in XGBoost training job then trains on the `processed/train/` and `processed/validation/` splits produced by 5a (same hyperparameters as the pipeline `TrainingStep`), writing `model.tar.gz`.
   `run_training.py` writes the resulting `model.tar.gz` S3 URI to SSM (`/churnguard/model-artifact-uri`). **5a must complete before 5b**, and 5a is the reason the backend's lazy-load of `feature_columns.json` resolves (rather than returning `409 MODEL_NOT_READY` forever) after `deploy.sh` on a fresh account. (The pipeline's `ProcessingStep` remains the producer when the pipeline is run live; the baseline 5a simply guarantees the schema exists before the first pipeline run.)
6. `ml/build_mme_models.py` → copy the baseline artifact to `mme/churn-v1.tar.gz` and a second variant (different hyperparameters, really retrained) to `mme/churn-v2.tar.gz`.
7. Deploy `realtime`, `serverless`, `async`, `mme`, `events`, passing the artifact URI parameter.

## SageMaker Pipeline definition (slide-33 DAG)

Authored in `ml/build_pipeline.py`, written locally to `infra/pipeline-definition.json` for inspection and uploaded to `s3://churnguard-data-.../pipeline/pipeline-definition.json`, then referenced by stack 2 via `PipelineDefinitionS3Location` (see stack 2; the definition is far too large for a CloudFormation parameter). Steps:

1. **ProcessingStep "DataProcessing"** — SKLearn processor (image URI SDK-resolved via `image_uris.retrieve(framework="sklearn", version="1.2-1", image_scope="processing", ...)` — not a literal `1.2-1` tag, see Technology stack / Finding 3) runs `ml/scripts/preprocessing.py` on the raw CSV; outputs `train/`, `validation/`, `test/` and `feature_columns.json`.
2. **TrainingStep "ModelTraining"** — built-in XGBoost on the processing outputs. Hyperparameters: `objective=binary:logistic`, `num_round=150`, `max_depth=5`, `eta=0.2`, `subsample=0.8`, `eval_metric=auc`. `ml.m5.large`.
3. **ProcessingStep "ModelEvaluation"** — runs `ml/scripts/evaluation.py` loading the model + `test/`, computing accuracy and AUC, writing `evaluation.json` ( `{"metrics": {"accuracy": {"value": ...}, "auc": {"value": ...}}}` ). A **`PropertyFile`** named `EvaluationReport` captures this output.
4. **ConditionStep "CheckAccuracy"** — `ConditionGreaterThanOrEqualTo` using **`JsonGet`** on the `PropertyFile` at path `metrics.accuracy.value` vs threshold `0.78` (realistic for Telco churn; slide shows 90% as illustrative but Telco churn tops out ~0.80–0.82, so a real, honest threshold is used and documented). On true → RegisterModel; on false → pipeline ends without registration (honest failure, surfaced in the UI).
5. **RegisterModel "RegisterChurnModel"** — registers into group `churnguard-churn` with `ModelApprovalStatus=PendingManualApproval`, inference spec referencing the XGBoost image, `content_types=["text/csv"]`, instance types for real-time + transform. Evaluation metrics attached as `model_metrics`.

This directly implements the slide-33 teaching point (PropertyFile + JsonGet for the conditional), the same slide-33 DAG referenced in the Overview.

## FastAPI backend — route contract

Base: `/api`. All responses include real identifiers. Errors use a shared envelope `{"error": {"code": str, "message": str, "aws_request_id": str|null, "detail": any}}` with appropriate HTTP status. A single `boto_client(service)` factory pins `region_name="us-east-1"`. Config (endpoint names, bucket names, topic ARN) is read from SSM at startup with env-var overrides; names also have the fixed defaults above.

Common error mapping (applied by a FastAPI exception handler): `botocore.exceptions.ClientError` → map `ValidationException`/`ModelError` to 400, `ResourceNotFound`/`ValidationError` for missing endpoint to 404, throttling (`ThrottlingException`) to 429, everything else to 502 with the AWS request id. All backend errors are logged at `ERROR` with the AWS request id; client input validation failures are logged at `WARNING`. Successful inference calls log at `INFO` with endpoint name + latency, never the full payload (PII in the churn features).

### Input validation (applied to every scoring request)

A `ChurnFeatures` Pydantic model mirrors the 19 raw columns with types and constraints: numeric fields (`tenure` int ≥ 0 ≤ 120, `MonthlyCharges` float ≥ 0, `TotalCharges` float ≥ 0) and categorical fields constrained to `Literal` enums of the real dataset values (e.g. `Contract ∈ {"Month-to-month","One year","Two year"}`). Required: all 19. On failure FastAPI returns 422 with field-level detail (recoverable; caller fixes input). The validated record is one-hot encoded in the backend using `feature_columns.json`; the ordering file is the invariant owner for payload width. **Loading is lazy with retry, not startup-cached-once:** the backend attempts the S3 `get_object` on the first scoring request if the schema is not already in memory, caches it on success, and reuses the cache thereafter. It returns `409 MODEL_NOT_READY` **only** when the object is genuinely absent (S3 `NoSuchKey`), and retries on the next request. This avoids the foot-gun where a value cached at startup — before `deploy.sh` training writes the file — would make the backend return 409 forever until restarted. The cache invalidation rule is "load once after the object is present"; the demo does not need hot reload after that. `GET /api/health` exposes `featureSchemaLoaded: bool` so the operator can see whether the schema has been loaded. Feature encoding never fabricates a score — a missing schema is an honest 409.

**Feature schema contract (Finding 6 invariant).** `feature_columns.json` lists the **feature columns only, in inference order, excluding the label `Churn`**. The preprocessing script writes the label `Churn` separately as **column 0** of the headerless training CSV (XGBoost built-in label-first convention), and the feature columns follow. The invariant the preprocessing step owns and asserts: `len(feature_columns) == training_csv_width - 1` (the training CSV is one column wider than the inference vector, that extra column being the label). The backend builds every inference payload as `len(feature_columns)` values in that exact order — never including a label slot — so inference vectors can never be misaligned by the label column.

### Inference response contract (pinned — applies to realtime, serverless, mme, batch, async)

The SageMaker built-in XGBoost serving container (`sagemaker-xgboost:1.7-1`), trained with `objective=binary:logistic`, returns a **probability** per input row. The contract is pinned so every parser agrees:

- **Request to the container:** `ContentType="text/csv"`, `Accept="text/csv"`. Body is one CSV line per input row, each line being the `len(feature_columns)` feature values in `feature_columns.json` order, **no header, no label**.
- **Response from the container:** `text/csv` body containing **one float probability per input row, in input order**, separated by newlines (a single-row request returns one float, optionally with a trailing newline). The backend parses by splitting on whitespace/newlines/commas, discarding empties, and `float()`-casting each token. The i-th float maps to the i-th input record in `predictions[]`.
- **Probability → label decision:** pinned as a single named backend constant **`CHURN_DECISION_THRESHOLD = 0.5`**, with `churn = churnProbability >= CHURN_DECISION_THRESHOLD`. 0.5 is the default binary cutoff; it is intentionally **not** tuned for the mild class imbalance (~27% churn) to keep the demo output stable and explainable. The constant lives in one place (`backend/app/inference.py`) so every route — realtime, serverless, mme, batch, async — computes the boolean identically. Any future change to the cutoff is a one-line edit, documented in the README.
- **Batch and async `.out` parsing:** identical rule. The Batch Transform / async output `.out` object is `text/csv`, one probability per input row in input order; the backend reads it from S3 and applies the same split/cast/threshold mapping to build `predictions[]`. There is no separate parser.

### Routes

**`POST /api/predict/realtime`**
- Request: `{"features": ChurnFeatures}` (or `{"records": [ChurnFeatures,...]}` for a small batch, max 100).
- Action: `sagemaker-runtime.invoke_endpoint(EndpointName="churnguard-realtime", ContentType="text/csv", Accept="text/csv", Body=<csv>)` per the pinned inference contract above.
- Response: `{"endpoint":"churnguard-realtime","predictions":[{"churnProbability":float,"churn":bool}],"latencyMs":int}` (`churn` computed with `CHURN_DECISION_THRESHOLD`; `predictions[i]` maps to input record `i`).
- Failures: endpoint not `InService` → 409 with endpoint status; payload too large → 400.

**`POST /api/predict/serverless`** — identical contract against `EndpointName="churnguard-serverless"`. SageMaker returns **no** cold-start signal for serverless inference, so the response does **not** carry a `coldStart` boolean that could read as measured ground truth (that would risk the exact fabrication the NO-SIMULATION constraint forbids). Instead the response reports the real **`latencyMs`** (measured round-trip) plus a clearly-derived **`coldStartLikely: bool`** computed as `latencyMs > 1500` with the threshold surfaced in the response as `coldStartThresholdMs: 1500`. The UI labels it as a heuristic ("likely — inferred from latency > 1500 ms"), never as an observed fact. This keeps the slide-19 cold-start teaching point while staying honest about what is measured vs inferred.

**`POST /api/predict/async`**
- Request: `{"records":[ChurnFeatures,...]}` (designed for larger payloads).
- Action: write CSV to `s3://churnguard-async-.../input/<uuid>.csv`; `invoke_endpoint_async(EndpointName="churnguard-async", InputLocation=<s3 input uri>)`. **The backend records the number of rows it wrote to the input CSV**, keyed by `inferenceId`, so the result route (Finding 4) can gate `Completed` on `len(predictions) == rows`.
- Response: `{"endpoint":"churnguard-async","inferenceId":str,"outputLocation":"s3://.../output/<id>.out","inputLocation":"s3://...","rowCount":int,"status":"InProgress"}`.
- SNS: the async EndpointConfig posts success/error to `churnguard-async-notifications`; the UI shows the topic ARN.

**`GET /api/predict/async/result?outputLocation=<s3uri>`**
- Action: `s3.head_object` / `get_object` on the real output location, with a **pinned check order and completion rule (Finding 4)** so a failed or mid-write job can never be reported as a successful-looking-but-wrong `Completed`:
  1. **Check `<outputLocation>.out.failure` FIRST.** If present → `{"status":"Failed","reason":<real `.out.failure` body>}`. This is checked before `.out` precisely so a job that failed but also left a partial/empty `.out` is reported as `Failed`, never `Completed`.
  2. **Else check `<outputLocation>.out`.** If absent (S3 `NoSuchKey`/404) → `{"status":"InProgress"}`.
  3. **If `.out` is present, parse it and gate completion on row count.** The backend recorded the number of input rows submitted when it wrote the `input/<uuid>.csv` payload (it owns the CSV it generated). Report `{"status":"Completed","predictions":[...]}` **only when the parsed probability-token count equals that recorded input row count.** If the parse yields fewer tokens than rows (empty or short — e.g. a read-after-write race on a freshly `PutObject`'d `.out`), keep `{"status":"InProgress"}` and let the next poll retry. **An empty or short parse is NEVER reported as `Completed`.**
- The input row count is persisted alongside the inference id (in-memory map keyed by `inferenceId`, backed by the input CSV in S3 so it survives a backend restart — the backend can re-derive the count by reading the `input/<uuid>.csv` line count if the in-memory entry is missing). Fatal errors return the real SageMaker failure body, never a placeholder.

**`POST /api/batch`**
- Request: optional `{"inputS3Uri": str}`; default uses the test split already in S3. Backend ensures a real CSV exists at `s3://churnguard-batch-.../input/batch.csv`.
- **Default-path input alignment (Finding 5).** The processing `test/` split is headerless, **label-first** CSV (column 0 is the `Churn` label, as written by preprocessing). It CANNOT be fed to the transform job as-is — that would send the label as a feature and misalign every vector by one column, producing real-but-wrong probabilities. So the default path **strips column 0 (the label) from the test split before writing `batch.csv`**, producing `len(feature_columns)`-wide, headerless, feature-ordered rows that match the pinned inference contract exactly (no header, no label). The backend **records the row count** of the written `batch.csv` so it can assert `len(predictions) == rows` when the result is fetched (position-based `predictions[i]` → input row `i` alignment). If a caller supplies `inputS3Uri`, that object **must already be label-free, feature-ordered, headerless CSV**; the backend does not strip or re-order a caller-supplied input (documented in the route and README), and it reads the caller input's row count to apply the same `len(predictions) == rows` assertion on fetch.
- Action: `sagemaker.create_transform_job(TransformJobName="churnguard-batch-<ts>", ModelName=<batchModelName>, TransformInput={S3 CSV}, TransformOutput={batch bucket output/}, TransformResources={ml.m5.large,1})`. `<batchModelName>` is read from config/SSM (`/churnguard/batch-model-name`, default `churnguard-realtime-model`), not hard-coded, so a future rename of the realtime model doesn't silently break batch. **Batch depends on stack 3 (`churnguard-realtime`) being deployed** since it reuses that model; this dependency is documented in the README. Blue/green leaves `churnguard-realtime-model` in place (it creates new `churnguard-bg-*` models), so batch keeps working across blue/green runs.
- Response: `{"jobName":str,"status":"InProgress","outputLocation":"s3://..."}`.

**`GET /api/batch/{jobName}`** — `describe_transform_job`; returns `{"jobName","status","outputLocation","rowCount"?,"failureReason"?}`; when `Completed`, a follow-on fetch of the `.out` from S3 returns parsed predictions **only if `len(predictions)` equals the recorded input row count (Finding 5)**; a short/empty parse against a `Completed` job is surfaced as a real error (not a fabricated partial success), not silently returned.

**`POST /api/pipeline/run`**
- Action: `sagemaker.start_pipeline_execution(PipelineName="churnguard-pipeline", PipelineParameters=[...])` with a run-id tag.
- Response: `{"pipelineExecutionArn":str,"status":"Executing"}`.

**`GET /api/pipeline/status?arn=<execArn>`**
- Action: `describe_pipeline_execution` + `list_pipeline_execution_steps`.
- Response: `{"arn","status","steps":[{"name","status","startTime","endTime","failureReason"?}]}` — real per-step status driving the UI DAG view. If the Condition step fails the threshold, that's surfaced as a real `Failed`/`Succeeded`-without-registration state.

**`GET /api/registry`**
- Action: `list_model_package_groups` (filtered to `churnguard-churn`) + `list_model_packages(ModelPackageGroupName=...)`.
- Response: `{"group":"churnguard-churn","versions":[{"arn","version","status","approvalStatus","createdAt","metrics":{"accuracy"?,"auc"?}}]}`.

**`POST /api/registry/approve`**
- Request: `{"modelPackageArn":str}`.
- Action: `update_model_package(ModelPackageArn=..., ModelApprovalStatus="Approved")`. This is the real trigger that emits the `SageMaker Model Package State Change` event the EventBridge rule consumes.
- Response: `{"modelPackageArn","approvalStatus":"Approved"}`.
- Validation: ARN must belong to group `churnguard-churn` (checked via `describe_model_package`), else 400. Approving an already-approved version is idempotent (returns current state, logged WARNING).

**`GET /api/events/recent`** — reads the last N entries from the `/churnguard/events` CloudWatch log group (`filter_log_events`) so the UI can show the real EventBridge→Lambda handoff after an approval. The log group is created by stack 7, but the route still treats a `ResourceNotFoundException` (group not yet present, e.g. during provisioning) as **"no events yet" — returns `{"events": []}` with HTTP 200**, never a 502. Returns `{"events":[{"timestamp","modelPackageArn","message"}]}`.

**`POST /api/deploy/bluegreen`**
- Request: `{"modelPackageArn":str,"mode":"canary"|"linear","canaryPercent":int?=10,"linearStepPercent":int?=20,"bakeTimeSeconds":int?=120}`.
- Action (real endpoint update with deployment guardrails). All runtime-created resources use the distinct **`churnguard-bg-`** sub-prefix (see teardown ownership boundary below) and a `<rev>` suffix defined as the unix epoch seconds at request time (`int(time.time())`), guaranteeing uniqueness across repeated demo runs so `create_endpoint_config`/`create_model` never collide on a duplicate name:
  1. `create_model("churnguard-bg-model-<rev>")` from the approved package's inference spec (artifact + XGBoost image).
  2. `create_endpoint_config("churnguard-bg-config-<rev>")` pointing at the new model, `ml.m5.large`, `InitialInstanceCount: 3` (matches the live fleet so the shift math is whole-instance).
  3. `update_endpoint(EndpointName="churnguard-realtime", EndpointConfigName=<new>, DeploymentConfig={BlueGreenUpdatePolicy:{TrafficRoutingConfiguration:{Type: CANARY|LINEAR, CanarySize/LinearStepSize:{Type:CAPACITY_PERCENT,Value:...}, WaitIntervalInSeconds: bakeTime}, TerminationWaitInSeconds: 300, MaximumExecutionTimeoutInSeconds: 1800}, AutoRollbackConfiguration:{Alarms:[{AlarmName:"churnguard-realtime-ModelError"}]}})`.
- Response: `{"endpoint":"churnguard-realtime","newConfig":str,"newModel":str,"mode":str,"rev":int,"status":"Updating"}`.
- The auto-rollback CloudWatch alarm (`churnguard-realtime-ModelError`) is created and **fully specified by the realtime stack** (stack 3: `AWS/SageMaker` `Invocation5XXErrors`, dimensions `EndpointName=churnguard-realtime` + `VariantName=AllTraffic`, `Sum`/`Period 60`/`EvaluationPeriods 1`/`Threshold 1`/`GreaterThanOrEqualToThreshold`/`TreatMissingData notBreaching` — see stack 3) so rollback is real and reproducible (slide-36 built-in safeguards). Mode maps directly to slide-36 Canary / Linear. Because the fleet is 3 instances, `CAPACITY_PERCENT` steps resolve to whole instances: CANARY shifts ~1 instance first then the remainder; LINEAR with a 20%–50% step produces multiple observable steps. This is why the realtime fleet is sized at 3 (Finding 2).
- Validation: `modelPackageArn` must be `Approved` (else 409 `NOT_APPROVED` — enforces the registry→deploy invariant at the API layer, because approval is the business gate); `canaryPercent` 1–50, `linearStepPercent` 10–50, `bakeTimeSeconds` 30–1800.

**`GET /api/deploy/bluegreen/status`** — `describe_endpoint(EndpointName="churnguard-realtime")` → `{"status","endpointConfig","lastDeploymentStatus","pendingDeploymentSummary"?}` exposing real traffic-shift progress and rollback state.

**`POST /api/hosting/multimodel`**
- Request: `{"targetModel":"churn-v1.tar.gz"|"churn-v2.tar.gz","features":ChurnFeatures}`.
- Action: `invoke_endpoint(EndpointName="churnguard-mme", ContentType="text/csv", Accept="text/csv", TargetModel=<name>, Body=<csv>)` — `Accept="text/csv"` included to match the pinned inference contract and the realtime/serverless routes verbatim.
- Response: `{"endpoint":"churnguard-mme","targetModel":str,"prediction":{"churnProbability":float,"churn":bool},"latencyMs":int}`.
- `GET /api/hosting/multimodel/models` lists the real `*.tar.gz` objects under the `mme/` prefix (`list_objects_v2`).

**`GET /api/health`** — returns backend config echo: region, resolved endpoint names, bucket names, `featureSchemaLoaded: bool` (whether `feature_columns.json` has been loaded into the backend cache), and a per-endpoint `describe_endpoint` status map so the UI can show what's live. No secrets echoed.

**`GET /api/architecture`** — static JSON describing nodes/edges for the landing-page architecture diagram (derived from this design; it's descriptive metadata, not simulated data).

## Frontend — tab → route map

Single Cloudscape `AppLayout`. Landing page (default content) shows the scenario overview and an architecture diagram (rendered from `/api/architecture`, using Cloudscape layout primitives / an embedded SVG). A top `Tabs` component (or side navigation) exposes seven tabs:

1. **Live scoring** → `POST /api/predict/realtime`. A Cloudscape `Form` of the 19 features (with sensible defaults from a real sample row), a "Score" button, result shows churn probability, the endpoint name `churnguard-realtime`, and latency.
2. **Bulk / Batch** → `POST /api/batch` + poll `GET /api/batch/{jobName}`. Shows job name, status, output S3 URI, and parsed results when complete.
3. **Async report** → `POST /api/predict/async` then poll `GET /api/predict/async/result`. Shows inference id, input/output S3 URIs, SNS topic ARN, live status.
4. **Serverless** → `POST /api/predict/serverless`. Same form as Live scoring; highlights cold-start flag and endpoint `churnguard-serverless`.
5. **Pipeline & Registry** → `POST /api/pipeline/run`, `GET /api/pipeline/status`, `GET /api/registry`, `POST /api/registry/approve`, `GET /api/events/recent`. Renders the slide-33 DAG with live per-step status, lists registry versions with metrics and approval status, an **Approve** action, and an events panel showing the EventBridge→Lambda handoff after approval. **Expected empty-registry state (Finding 7):** only the pipeline's `RegisterModel` step registers a model version — the baseline (non-pipeline) artifact that stands up the endpoints is deliberately NOT registered. So on a fresh deploy the Model Registry is legitimately EMPTY until the first live pipeline run registers a version; the tab renders this as an explicit "no versions yet — run the pipeline" empty state, not an error. An empty registry on fresh deploy is expected, not a bug.
6. **Blue/Green** → `POST /api/deploy/bluegreen`, `GET /api/deploy/bluegreen/status`. **Consumes the approved version from tab 5 as the green candidate** (the newest `Approved` package ARN from `/api/registry`), lets the user pick Canary or Linear and parameters, triggers the real `update_endpoint`, and shows live traffic-shift progress and rollback state.
7. **Hosting (Multi-Model)** → `GET /api/hosting/multimodel/models` + `POST /api/hosting/multimodel`. Model picker (`churn-v1`/`churn-v2`), scoring form, shows `TargetModel` routing and endpoint `churnguard-mme`.

**Continuous flow (explicit):** Pipeline run (tab 5) → real model version registered (Pending) → Approve (tab 5) → real `update_model_package` emits the real EventBridge event → events panel shows the real Lambda handoff → tab 6 reads the newly Approved ARN and uses it as the green candidate for the real blue/green `update_endpoint`. No step is simulated; each UI transition reflects a real resource state change.

Real identifiers (endpoint names, model package ARNs, pipeline execution ARNs, S3 URIs, SNS topic ARN) are shown with Cloudscape `Box`/`CopyToClipboard` throughout.

Frontend config: `VITE_API_BASE` env var (default `http://localhost:8000/api`). Dev proxy in `vite.config.ts` forwards `/api` to the backend to avoid CORS; backend also enables permissive CORS for localhost origins only.

## Error handling (per fallible operation)

- **Endpoint not InService** (any invoke route): recoverable; `describe_endpoint` gives the real status; return 409 `ENDPOINT_NOT_READY` with status; logged WARNING. UI shows a retry hint.
- **Invalid feature input**: recoverable; 422 with field detail; logged WARNING; no AWS call made.
- **Model not trained / `feature_columns.json` missing**: recoverable (operator runs training); 409 `MODEL_NOT_READY`. The schema is lazy-loaded on the first scoring request and retried on every subsequent request until present (no startup-once cache), so the backend self-heals once training writes the file — no restart needed. Logged WARNING on each missing-schema request; `/api/health` reports `featureSchemaLoaded: false` until it loads.
- **S3 async result not yet present**: expected; treated as `InProgress`, not an error.
- **Async/batch job failure**: fatal for that job; backend returns the real `FailureReason` / `.out.failure` body; logged ERROR. Never replaced with a fabricated success.
- **Pipeline condition below threshold**: not an error — a real business outcome; surfaced as "not registered" with the real metric value.
- **Approve on non-churnguard ARN**: 400 `INVALID_PACKAGE`; logged WARNING.
- **Blue/green on unapproved ARN**: 409 `NOT_APPROVED`.
- **Blue/green rollback**: the real CloudWatch alarm triggers SageMaker auto-rollback; `/status` reports it; backend does not intervene.
- **boto3 throttling**: 429; UI backs off. **Credential/region misconfig**: 502 with a clear message naming the expected region `us-east-1`.
- **CloudFormation deploy failure**: `deploy.sh` stops on first non-zero `aws cloudformation deploy`, prints the stack events (`describe-stack-events`), and exits non-zero. No partial "pretend success."

## Testability

- **Unit (no AWS):** feature encoding against a fixed `feature_columns.json` fixture; Pydantic validation (valid/invalid rows); error-envelope mapping from synthetic `ClientError`s (botocore stubs); request/response schema (FastAPI `TestClient`). boto3 calls are exercised with `botocore.stub.Stubber` — stubbing the AWS transport in a unit test is not the same as faking a demo feature; the running demo always hits real AWS. This distinction is stated so reviewers don't read stubs as "mocks" violating the constraint.
- **Integration (real AWS, opt-in via `RUN_INTEGRATION=1`):** `pytest` suite that provisions nothing but asserts real endpoints are `InService`, runs one real `invoke_endpoint`, starts and polls a real batch job, lists the real registry, and runs a real pipeline execution to completion. Guarded so it only runs after `deploy.sh`.
- **Frontend:** component tests (Vitest + Testing Library) for form validation and state rendering against a mocked `fetch` layer (UI-layer test only); an optional Playwright smoke test hitting a running backend.
- **Infra:** `aws cloudformation validate-template` on every YAML in CI; `cfn-lint` if available.
- Hard-to-test async flows (SNS, EventBridge) are made observable via the `/api/events/recent` CloudWatch read and the async `/result` S3 read, so the demo is verifiable end-to-end without hidden state.

## Cost & teardown

- Billable always-on compute: **realtime endpoint = 3 × `ml.m5.large`** (sized at 3 for observable blue/green canary/linear shifts, Finding 2), async endpoint = 1 × `ml.m5.large` (scales to 0 when idle), mme endpoint = 1 × `ml.m5.large`. At ~$0.115/instance-hr that is up to ~5 × $0.115 ≈ **$0.58/hr** when the async endpoint is warm (~$0.46/hr with async idle at 0). Serverless and batch bill per use; pipeline/training jobs are transient `ml.m5.large`. The 3-instance realtime cost is a deliberate tradeoff for a working blue/green demo and is called out here so there is no surprise. **Transient peak during a blue/green run:** `update_endpoint` stands up a parallel green fleet equal to the live fleet before shifting traffic, so for the minutes a deployment is in progress the realtime variant runs 3 (blue) + 3 (green) = 6 instances, and peak concurrent `ml.m5.large` is up to 6 (realtime) + 2 (async) + 1 (mme) = **~9 instances (~$1.04/hr)**, not 5. This is well within the us-east-1 ml.m5.large endpoint-usage quota (16, verified), so it affects cost disclosure only, not feasibility — but it is noted here so the estimate is not an underestimate during the exact blue/green demo this design centers on.
- Autoscaling: async endpoint `MinCapacity: 0 / MaxCapacity: 2` with **both** a target-tracking policy (scale-in to 0) and a step-scaling policy on `HasBacklogWithoutCapacity` (scale-up from 0) — both required for the slide-15 scale-to-0-and-back behavior to survive repeated invocations (Finding 1); realtime target-tracking `MinCapacity: 3 / MaxCapacity: 6` (3 is the floor so it can serve the live demo and keep the blue/green fleet whole-instance). Serverless scales to zero inherently.
- **Ownership boundary (explicit):** every `churnguard-*` resource is CloudFormation-owned **except** the blue/green runtime models and endpoint configs, which use the `churnguard-bg-` sub-prefix. Teardown deletes stack-owned resources **only** via `delete-stack`, and sweeps **only** the `churnguard-bg-` sub-prefix with boto3 — the two prefixes never overlap, so the runtime sweep can never delete a stack-managed resource and cause stack-delete drift.
- **`infra/teardown.sh`:** (1) first sweeps runtime blue/green leftovers — `list_endpoint_configs`/`list_models` filtered to the `churnguard-bg-` prefix, then `delete_endpoint_config`/`delete_model` each (the live `churnguard-realtime` endpoint is restored to its stack config or left for `delete-stack`); (2) deletes endpoint stacks (`events`, `mme`, `async`, `serverless`, `realtime`), then `registry-pipeline`, then `foundation` — reverse of deploy; (3) before deleting `foundation`, empties all four S3 buckets (`aws s3 rm --recursive`) so bucket deletion succeeds; (4) deletes any standalone training/transform job artifacts left in S3 (covered by the bucket empty). Each delete passes `--region us-east-1`. The script waits on `stack-delete-complete` and reports anything left.
- **README** (`churnguard/README.md`) documents: prerequisites, `deploy.sh` (with the training-takes-~10-min note), running backend (`uvicorn`) + frontend (`npm run dev`), the demo walkthrough across the seven tabs, cost estimate (including the 3-instance realtime fleet), and `teardown.sh` with a verification step (`aws sagemaker list-endpoints --region us-east-1` returns no `churnguard-*`). The README **must also carry the condition-threshold mapping (Finding 13):** the deck's slide shows 90% accuracy as illustrative, but real Telco-churn XGBoost tops out ~0.80–0.82, so the pipeline's ConditionStep gate is `0.78`. This is written in the README so the demo audience understands the 90%→0.78 mapping and does not read it as a bug.

## Assumptions

- The baseline (non-pipeline) training path is used at first provisioning so the always-on endpoints have an artifact before anyone runs the pipeline live; this does not violate the no-fake rule (it's a real training job), it just decouples endpoint provisioning from the live pipeline demo.
- Accuracy threshold for the condition step is `0.78`, chosen because real Telco churn XGBoost tops out ~0.80; the deck's 90% is illustrative. Documented so the condition can genuinely pass.
- The async email SNS subscription is optional (parameter); if omitted, SNS still fires and success/error `.out`/`.out.failure` objects in S3 are the authoritative result the UI reads.
- A second MME variant (`churn-v2`) is a genuinely retrained model with different hyperparameters (e.g. `max_depth=7, eta=0.1`), not a copy, so multi-model routing shows real behavioral difference.

## Design review responses

Responses to the review at `.agents/tasks/design-review.md` (verdict CHANGES_REQUESTED: 2 HIGH, 5 MEDIUM, 6 NIT). All 13 findings are addressed in-place above; summarized here.

1. **HIGH — Pipeline definition too large for a CFN parameter. ADDRESSED.** Stack 2 now references the definition via `AWS::SageMaker::Pipeline` `PipelineDefinition.PipelineDefinitionS3Location`; `ml/build_pipeline.py` uploads `pipeline-definition.json` to `s3://churnguard-data-.../pipeline/` before stack 2 deploys. Only the small S3 key/bucket are stack parameters (under 4096 bytes). The pipeline stays real `AWS::SageMaker::Pipeline` IaC, so the slide-29 teaching point is preserved. Provisioning order updated (foundation → dataset → build+upload pipeline def → registry-pipeline stack).

2. **HIGH — Canary/Linear on a single instance isn't a real shift. ADDRESSED.** Realtime `EndpointConfig` is now `InitialInstanceCount: 3` (named variant `AllTraffic`), with autoscaling `MinCapacity: 3`. CAPACITY_PERCENT steps now resolve to whole instances and produce observable canary/linear states. The blue/green runtime config is also created at 3 instances to match the live fleet. Reconciled with the Cost section (realtime = 3 × `ml.m5.large`, ~$0.58/hr peak) and stated as a deliberate tradeoff.

3. **MEDIUM — `feature_columns.json` cached once at startup. ADDRESSED.** Changed to lazy-load-with-retry: attempt on first scoring request, cache on success, return 409 only when the object is genuinely absent, retry on subsequent requests. `/api/health` exposes `featureSchemaLoaded`. No restart needed after training writes the file.

4. **MEDIUM — Decision threshold unspecified. ADDRESSED.** Pinned `CHURN_DECISION_THRESHOLD = 0.5` as a single named backend constant in `backend/app/inference.py`, used identically by every route. 0.5 is intentionally untuned for the mild imbalance to keep demo output stable; documented in the README.

5. **MEDIUM — XGBoost text/csv response contract unspecified. ADDRESSED.** Added a pinned inference-response contract: `Accept="text/csv"`, body is one probability float per input row in input order, `predictions[i]` maps to input record `i`. Same split/cast/threshold rule applied explicitly to batch and async `.out` parsing.

6. **MEDIUM — Label column ambiguity in `feature_columns.json`. ADDRESSED.** Stated explicitly that `feature_columns.json` contains feature columns only, in inference order, excluding the label; the label is training-CSV column 0. Added the invariant `len(feature_columns) == training_csv_width - 1`, owned and asserted by the preprocessing step.

7. **MEDIUM — Serverless `coldStart` reads as fabricated. ADDRESSED.** Dropped the measured-looking `coldStart` boolean. Response now reports measured `latencyMs` plus a clearly-derived `coldStartLikely` (`latencyMs > 1500`) with `coldStartThresholdMs: 1500` surfaced and the UI labeling it a heuristic.

8. **NIT — ScalableTarget ResourceId/variant unspecified. ADDRESSED.** Every `EndpointConfig` names its production variant `AllTraffic`; stated `ServiceNamespace: sagemaker`, `ResourceId: endpoint/<name>/variant/AllTraffic`, `ScalableDimension: sagemaker:variant:DesiredInstanceCount`, with realtime `3/6` and async `0/2` capacities.

9. **NIT — Prefix-based teardown collides with CFN resources. ADDRESSED.** Blue/green runtime resources now use the distinct `churnguard-bg-` sub-prefix; teardown sweeps only that prefix with boto3 and deletes all other `churnguard-*` resources via `delete-stack` only. Ownership boundary stated explicitly.

10. **NIT — `<rev>` undefined. ADDRESSED.** `<rev>` is defined as unix epoch seconds (`int(time.time())`), unique per update, so repeated blue/green runs never collide on `create_endpoint_config`/`create_model`.

11. **NIT — `/churnguard/events` log group creation/timing. ADDRESSED.** Stack 7 now explicitly creates the `/churnguard/events` `AWS::Logs::LogGroup`; `GET /api/events/recent` treats `ResourceNotFoundException` as an empty list (HTTP 200), never a 502.

12. **NIT — Batch hard-codes the model name. ADDRESSED.** Batch reads `<batchModelName>` from SSM (`/churnguard/batch-model-name`, default `churnguard-realtime-model`); documented that batch depends on stack 3 and survives blue/green.

13. **NIT — 0.78 vs deck 90%. ADDRESSED (documentation).** The intentional, justified deviation is retained; the README is now required to carry the 90%→0.78 mapping so the demo audience understands it.

### Iteration 2 responses

Responses to the second review at `.agents/tasks/design-review.md` (verdict CHANGES_REQUESTED: 1 HIGH, 1 MEDIUM, 3 NIT). All five addressed in-place above; summarized here.

1. **HIGH — Async `MinCapacity: 0` with only a target-tracking policy can't scale up from 0. ADDRESSED.** Stack 5 now defines **two** scaling policies on the one scalable target: the existing target-tracking policy (scale-in to 0 on `ApproximateBacklogSizePerInstance`) *plus* a required step-scaling policy (`ChangeInCapacity`, step `MetricIntervalLowerBound: 0 / ScalingAdjustment: 1`, `Cooldown: 300`) driven by a new `AWS::CloudWatch::Alarm` on `HasBacklogWithoutCapacity` (`AWS/SageMaker`, `EndpointName=churnguard-async`, `Period 60`, `EvaluationPeriods 1`, `Threshold 0`, `GreaterThanThreshold`, `TreatMissingData notBreaching`). The design now states explicitly that scale-to-0 and scale-up-from-0 are separate, both-required policies, so the async tab works on its second and later uses. The Cost section's autoscaling note is updated to match.

2. **MEDIUM — Rollback alarm `churnguard-realtime-ModelError` referenced but never specified. ADDRESSED.** Stack 3 now fully defines the alarm: `Namespace AWS/SageMaker`, `MetricName Invocation5XXErrors`, `Dimensions EndpointName=churnguard-realtime` + `VariantName=AllTraffic`, `Statistic Sum`, `Period 60`, `EvaluationPeriods 1`, `Threshold 1`, `ComparisonOperator GreaterThanOrEqualToThreshold`, `TreatMissingData notBreaching`. The design documents that `notBreaching` is deliberate (so a quiet green fleet is not rolled back for lack of traffic) and names the exact metric so the rollback demo is reproducible. The bluegreen route note points at this pinned spec.

3. **NIT — MME invoke omitted the pinned `Accept` header. ADDRESSED.** `POST /api/hosting/multimodel` now calls `invoke_endpoint(..., Accept="text/csv", ...)`, matching the pinned inference contract and the realtime/serverless routes.

4. **NIT — Conflicting slide citations (31–33 vs 53/54) for PropertyFile/JsonGet. ADDRESSED.** Standardized on **slide-33** in both the Overview and the SageMaker Pipeline section, consistent with the DAG references elsewhere.

5. **NIT — Cost section understated blue/green peak instance count. ADDRESSED.** The Cost section now notes the transient doubling of the realtime fleet during a blue/green run (3 blue + 3 green = 6 realtime, ~9 total ≈ $1.04/hr for the minutes a deployment is in progress), still within the verified quota of 16.

### Iteration 3 responses

Responses to the third review at `.agents/tasks/design-review.md` (verdict CHANGES_REQUESTED: 2 HIGH, 3 MEDIUM, 2 NIT). All seven addressed in-place above; summarized here.

1. **HIGH — Baseline (non-pipeline) path never stated it runs preprocessing, so `feature_columns.json`/splits might never exist. ADDRESSED.** The "Provisioning order" step 5 is now split into explicit sub-steps **5a (preprocess)** and **5b (train)**: `ml/run_training.py` runs the identical `ml/scripts/preprocessing.py` (as a real SageMaker Processing job, SDK-resolved SKLearn image) against the raw CSV first, writing the same `processed/{train,validation,test}/` splits **and** `processed/feature_columns.json` to the data bucket, *before* the standalone training job trains on those splits. This guarantees the backend's lazy-loaded schema resolves after `deploy.sh` on a fresh account instead of returning `409 MODEL_NOT_READY` forever. The pipeline `ProcessingStep` remains the live producer; 5a just guarantees the schema exists before the first pipeline run.

2. **HIGH — MME capability of the built-in XGBoost image asserted without evidence. ADDRESSED.** Stack 6 now mandates a real verification before the stack is locked: either (a) a throwaway `create-model (Mode: MultiModel)` + `create-endpoint` + `invoke_endpoint(TargetModel=...)` probe in `us-east-1` (clean virtualenv, region-pinned, torn down after), or (b) an authoritative AWS doc citation, with the result recorded in the README. A precise fallback is defined: if the built-in image is not MME-capable, switch the MME model's container to the **open-source XGBoost framework inference container** (ships the MMS multi-model contract), resolve its URI via `image_uris.retrieve(framework="xgboost", version="1.7-1", image_scope="inference", ...)`, pin the resolved URI, and (if required) add a project-owned `inference.py` to honor the pinned `text/csv` contract. The MME tab stays a working real hosting path.

3. **MEDIUM — SKLearn processing image hard-coded as a bare `1.2-1` tag. ADDRESSED.** The Technology-stack section now states the SKLearn tag is platform/python-qualified (e.g. `1.2-1-cpu-py3`) and is **SDK-resolved, not literal**, via `image_uris.retrieve(framework="sklearn", region="us-east-1", version="1.2-1", image_scope="processing", instance_type="ml.m5.large")`; the pipeline `ProcessingStep` reference was updated to match. XGBoost's `1.7-1` is noted as literal and used verbatim.

4. **MEDIUM — Async result route could report `Completed` on a failed/partial `.out`. ADDRESSED.** `GET /api/predict/async/result` now has a pinned check order: `.out.failure` is checked **first** (→ `Failed` with the real body); otherwise `Completed` is reported **only** when the parsed probability-token count equals the recorded input row count, else it stays `InProgress`. An empty or short parse is never reported as `Completed`. The POST route now records (and returns) the input `rowCount`.

5. **MEDIUM — Batch default path fed the label-first test split as features. ADDRESSED.** `POST /api/batch` now states the default path **strips column 0 (the `Churn` label)** before writing `batch.csv`, producing `len(feature_columns)`-wide headerless feature-ordered rows matching the pinned contract, and records the row count so `GET /api/batch/{jobName}` asserts `len(predictions) == rows` on fetch. Caller-supplied `inputS3Uri` must already be label-free, feature-ordered, headerless CSV.

6. **NIT — Async `NotificationConfig` success/error topics unspecified. ADDRESSED.** Stack 5's `AsyncInferenceConfig.NotificationConfig` now explicitly sets **both** `SuccessTopic` and `ErrorTopic` to the `churnguard-async-notifications` ARN, so neither signal is lost.

7. **NIT — Empty registry on fresh deploy could read as a bug. ADDRESSED.** The Pipeline & Registry tab description now states the registry is legitimately EMPTY until the first live pipeline run registers a version (the baseline artifact is intentionally not registered), rendered as an explicit empty state rather than treated as a failure.
