# ChurnGuard

A real-AWS SageMaker demo that serves **one** customer-churn XGBoost model
**seven ways** — real-time, serverless, async, batch, pipeline+registry,
blue/green, and multi-model — behind a FastAPI/boto3 backend and a
React/Vite/Cloudscape frontend. Everything deploys to a live AWS account; there
is no simulation and no fake data. It doubles as a teaching artifact that maps
to the APCR-MLA Content Review deck.

Account `875692608981`, region **`us-east-1`** (the single pinned region for
every AWS call). Dataset: the IBM Telco Customer Churn CSV.

## Architecture

```
                         ┌─────────────────────────────┐
            React/Vite ──│  FastAPI + boto3 backend     │
            Cloudscape   │  (one boto_client factory,   │
            (7 tabs)     │   region pinned us-east-1)   │
                         └──────────────┬──────────────┘
                                        │ invoke / describe / start
         ┌──────────────┬───────────────┼───────────────┬──────────────┐
         ▼              ▼               ▼               ▼              ▼
  churnguard-      churnguard-     churnguard-     batch transform  churnguard-mme
   realtime        serverless        async         (per-use job)   (MultiModel:
  (3×m5.large     (serverless     (m5.large,                        churn-v1 /
   AllTraffic)     2048MB)         scales 0↔2)                      churn-v2)
         │                                                               
         │ blue/green canary/linear update (churnguard-bg-<rev> config)  
         ▼
  SageMaker Pipeline (churnguard-pipeline) ─► Model Registry (churnguard-churn)
     DataProcessing→Training→Evaluation→CheckAccuracy(≥0.78)→RegisterModel
                              │ ApprovalStatus=Approved
                              ▼
     EventBridge rule (churnguard-model-approved) ─► Lambda
     (churnguard-model-approved-handler) ─► SNS + /churnguard/events log
```

Seven CloudFormation stacks: `churnguard-foundation` (4 S3 buckets + SNS + SSM),
`churnguard-registry-pipeline` (ModelPackageGroup + Pipeline),
`churnguard-realtime`, `churnguard-serverless`, `churnguard-async`,
`churnguard-mme`, `churnguard-events`. Blue/green runtime resources use the
distinct `churnguard-bg-` prefix and are created outside CloudFormation by the
backend.

## Prerequisites

- An AWS account with the SageMaker execution role
  `arn:aws:iam::875692608981:role/AmazonSageMaker-ExecutionRole` and admin-level
  caller credentials, active for us-east-1.
- `ml.m5.large` endpoint-usage quota ≥ 9 in us-east-1 (the demo peaks at ~9
  during a blue/green run; quota is 16).
- Python 3.11+ and Node 18+ with npm.
- The SDK work must use the dedicated ML virtualenv (the system `sagemaker`
  module is broken): `bash ml/setup-venv.sh` builds `ml/.venv`.

> Credential note: if your CLI uses SSO and the venv's botocore cannot refresh
> the SSO token directly, export short-lived credentials before running
> deploy/scripts: `eval "$(aws configure export-credentials --format env)"`.
> Every call still targets real AWS in us-east-1.

## Deploy

```bash
bash ml/setup-venv.sh                 # one-time: build ml/.venv
export AWS_DEFAULT_REGION=us-east-1
bash infra/deploy.sh                  # end-to-end, ~25–35 min
```

`deploy.sh` runs, in order: foundation → upload the real Telco CSV →
build+upload the pipeline definition + processing code → registry-pipeline →
baseline preprocess+train → build the two MME artifacts → **MME-capability
probe** → realtime, serverless, async, mme, events. It stops on the first
non-zero step and dumps `describe-stack-events` on a CloudFormation failure.

> **Training takes ~10 minutes.** The baseline preprocess (SageMaker Processing)
> plus the baseline + v2 training jobs together run ~10 min before the endpoint
> stacks deploy. The endpoint stacks then take several more minutes each to
> reach InService. Budget ~25–35 min for a cold full deploy.

### MME-capability result (confirmed)

The mandatory probe `ml/verify_mme_capability.py` stood up a real throwaway
`Mode: MultiModel` endpoint, served `invoke_endpoint(TargetModel=churn-v1.tar.gz)`,
and tore it down. Confirmed on 2026-10-02:

```
MME_BUILTIN_SUPPORTED=true
MME_IMAGE_URI=683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.7-1
```

So `infra/06-mme.yaml` is pinned to the **built-in** `sagemaker-xgboost:1.7-1`
image. (If the probe ever reported the built-in image as not multi-model
capable, deploy.sh would override `ImageUri` with the SDK-resolved open-source
XGBoost framework-inference URI.)

## Run the backend + frontend

Backend (FastAPI via uvicorn):

```bash
cd backend
python -m venv .venv && .venv/bin/pip install -r requirements.txt
# real-AWS calls need creds in the environment:
eval "$(aws configure export-credentials --format env)"
export AWS_DEFAULT_REGION=us-east-1
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Frontend (Vite dev server, proxies `/api` → `http://localhost:8000`):

```bash
cd frontend
npm install
npm run dev
```

Open the printed Vite URL. The landing page renders the architecture diagram
(from `GET /api/architecture`) and a live health panel; the seven tabs each call
their real backend route(s).

## Demo walkthrough — 7 tabs → deck slides

The demo maps 1:1 to the deck so you can present straight from the UI.

| Tab | What it shows | Backend route(s) | Deck slides |
| --- | --- | --- | --- |
| 1. Live scoring | Real-time invoke on `churnguard-realtime`; probability + latency | `POST /api/predict/realtime` | 12–14, 22–23 |
| 2. Bulk / Batch | Batch Transform job on the 1056-row test split; job name/status/output S3/parsed results | `POST /api/batch`, `GET /api/batch/{job}` | 16–18, 20–21 |
| 3. Async report | Async invoke + input/output S3 + SNS topic; scale-0↔2 behavior | `POST /api/predict/async`, `GET /api/predict/async/result` | 15, 19 |
| 4. Serverless | Serverless invoke; cold-start **heuristic** (latency > 1500 ms), never an observed fact | `POST /api/predict/serverless` | 19, 24 |
| 5. Pipeline & Registry | slide-33 DAG with live per-step status; registry versions + approve; EventBridge→Lambda handoff | `POST /api/pipeline/run`, `GET /api/pipeline/status`, `GET /api/registry`, `POST /api/registry/approve`, `GET /api/events/recent` | 9–11, 32–34, 45–48 |
| 6. Blue/Green | Canary/Linear traffic shift on the newest Approved package; live traffic + rollback state | `POST /api/deploy/bluegreen`, `GET /api/deploy/bluegreen/status` | 35–36, 49–51 |
| 7. Hosting (Multi-Model) | `TargetModel` routing to churn-v1 / churn-v2 on `churnguard-mme` | `POST /api/hosting/multimodel`, `GET /api/hosting/multimodel/models` | 22, 25, 52–54 |

The continuous flow is explicit across tabs 5 → 6: run the pipeline, approve the
registered version, watch the EventBridge→Lambda event, then promote that exact
Approved package via blue/green.

**Expected empty-registry state:** only the pipeline's RegisterModel step
registers a version. The baseline artifact that stands up the endpoints is not
registered, so on a fresh deploy the registry is legitimately empty until the
first pipeline run — the tab shows a "no versions yet — run the pipeline" state,
not an error.

## The 90% → 0.78 condition-threshold mapping

The deck's pipeline slide shows a **90%** accuracy gate as *illustrative*. Real
Telco-churn XGBoost tops out around 0.80–0.82 accuracy, so an honest, passable
gate is used: the pipeline's `CheckAccuracy` ConditionStep evaluates
`metrics.accuracy.value >= 0.78` (JsonGet on the evaluation PropertyFile). This
is a deliberate, documented deviation — not a bug — so the condition can
genuinely pass on real data. (Observed baseline validation AUC ≈ 0.849.)

The decision threshold that turns a probability into a churn label is
`CHURN_DECISION_THRESHOLD = 0.5`, defined once in `backend/app/inference.py` and
used by every scoring route.

## Cost estimate

At ~$0.115 per `ml.m5.large` instance-hour (us-east-1):

- **Always-on:** realtime = **3 × m5.large** (sized at 3 so blue/green
  canary/linear shifts resolve to whole instances and show observable
  intermediate states), async = 1 × m5.large (scales to 0 when idle), mme =
  1 × m5.large. Warm total ≈ **$0.58/hr** (~$0.46/hr with async idle at 0).
- **Transient blue/green peak:** `update_endpoint` stands up a parallel green
  fleet equal to the live fleet, so for the minutes a deployment runs the
  realtime variant is 3 (blue) + 3 (green) = 6 instances; peak concurrent is
  up to 6 + 2 (async) + 1 (mme) = **~9 × m5.large ≈ $1.04/hr**. Well within the
  us-east-1 endpoint-usage quota of 16.
- Serverless and batch bill per use; pipeline/training jobs are transient
  `ml.m5.large`.

The 3-instance realtime fleet is a deliberate tradeoff for a working blue/green
demo.

## Teardown

Endpoints bill **hourly**, so the live demo is left running until you explicitly
tear it down.

```bash
export AWS_DEFAULT_REGION=us-east-1
bash infra/teardown.sh
```

`teardown.sh` reverses the deploy: it first sweeps the `churnguard-bg-` runtime
resources (blue/green models + endpoint configs created outside CloudFormation —
only that prefix is touched), then deletes the stacks in reverse (events, mme,
async, serverless, realtime → registry-pipeline), empties all four S3 buckets,
and deletes foundation last. It waits on `stack-delete-complete` and prints a
leftovers report.

**Verify teardown** — this should return no `churnguard-*` endpoints:

```bash
aws sagemaker list-endpoints --region us-east-1 \
  --query "Endpoints[?starts_with(EndpointName,'churnguard')].EndpointName" --output text
# (empty output = fully torn down)
```

## Tests

- Backend: `cd backend && .venv/bin/pytest -q` (Stubber-only unit tests; the
  running demo always hits real AWS).
- Frontend: `cd frontend && npm run test` (Vitest against a mocked fetch layer)
  and `npm run build` (type-check + bundle).

## Evidence

A full real-AWS evidence capture (all nine serving/MLOps items with live ARNs,
job names, S3 URIs, and probabilities) is in **`docs/EVIDENCE.md`**, with the
detailed capture log and applied-fix narrative in
`.agents/tasks/verification.md`.
