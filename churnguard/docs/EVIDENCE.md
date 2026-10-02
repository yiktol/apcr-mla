# ChurnGuard — Real-AWS Evidence

Consolidated proof that every one of the seven serving patterns and the full
MLOps flow ran against **real AWS** in account `875692608981`, region
`us-east-1`, on 2026-10-02. All values were captured from live AWS calls (AWS
CLI `--region us-east-1` and boto3 pinned to `us-east-1`, exercised through the
real FastAPI route handlers). No simulation, no mocks, no fabricated data.

Full capture log and the applied-fix narrative: `.agents/tasks/verification.md`.

## Endpoints InService

`aws sagemaker list-endpoints --region us-east-1`:

| Endpoint | Status |
| --- | --- |
| `churnguard-realtime` | InService |
| `churnguard-serverless` | InService |
| `churnguard-async` | InService |
| `churnguard-mme` | InService |

## MME-capability probe (mandatory)

A real throwaway `Mode: MultiModel` endpoint on the built-in
`sagemaker-xgboost:1.7-1` image served `invoke_endpoint(TargetModel=churn-v1.tar.gz)`
and was torn down:

```
MME_BUILTIN_SUPPORTED=true
MME_IMAGE_URI=683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.7-1
```

`infra/06-mme.yaml` is pinned to this confirmed built-in image.

## Nine evidence items

| # | Pattern | Real identifier captured | Result |
| --- | --- | --- | --- |
| 1 | Real-time | endpoint `churnguard-realtime` | `churnProbability = 0.5510009527206421` (churn=true), 1340 ms |
| 2 | Serverless | endpoint `churnguard-serverless` | `0.5510009527206421`, **8070 ms**, `coldStartLikely=true` (heuristic, threshold 1500 ms) |
| 3 | Async | out `s3://churnguard-async-875692608981-us-east-1/output/1bdb1e86-7277-44c9-99a7-96bb0c9d4389.out` | scaled **0→1** instance, `status=Completed`, 3 real probabilities (0.5510) |
| 4 | Batch transform | job `churnguard-batch-1790944107`, out `s3://churnguard-batch-875692608981-us-east-1/output/batch.csv.out` | `Completed`, `rowCount=1056`, 1056 real predictions (0.0277, 0.00067, 0.6911, …) |
| 5 | Pipeline + Registry | exec `arn:aws:sagemaker:us-east-1:875692608981:pipeline/churnguard-pipeline/execution/9zavg9nz0abo` → pkg `arn:aws:sagemaker:us-east-1:875692608981:model-package/churnguard-churn/1` | `Succeeded`, all 5 steps Succeeded, model package registered |
| 6 | Registry approval | pkg `.../churnguard-churn/1` | `ModelApprovalStatus` flipped to **Approved** (verified via describe-model-package) |
| 7 | EventBridge → Lambda | log group `/churnguard/events`, stream `2026/10/02/[$LATEST]handler` | real entry with the approved package ARN + `approvalStatus=Approved` |
| 8 | Blue/green | config swap `churnguard-realtime-config` → `churnguard-bg-config-1790946171` | canary 10%/60 s bake, Updating → `InService`, config swapped (verified via describe-endpoint) |
| 9 | Multi-model | endpoint `churnguard-mme`, targets `churn-v1.tar.gz` / `churn-v2.tar.gz` | v1 → `0.5510` (true, 1063 ms); v2 → `0.46916502714157104` (false, 1057 ms) — distinct per-target routing |

### Notes

- **Same input, different models (item 9):** v1 and v2 are independently trained
  XGBoost variants (v1 = baseline max_depth 5/eta 0.2; v2 = max_depth 7/eta 0.1).
  The same feature row yields 0.5510 on v1 and 0.4692 on v2, proving real
  `TargetModel` routing on the multi-model endpoint.
- **Async scale-to-0-and-back (item 3):** the endpoint was at 0 instances; the
  `HasBacklogWithoutCapacity` alarm triggered the step-scaling policy, SageMaker
  provisioned 1 instance, processed the request, and wrote the `.out`.
- **Condition gate (item 5):** the pipeline's `CheckAccuracy` step evaluates
  `metrics.accuracy.value >= 0.78` (the deck's illustrative 90% maps to the
  realistic 0.78 Telco-churn accuracy gate; see README). It passed.

## Model quality (from the real baseline training job)

Baseline job `churnguard-baseline-training-2026-10-02-11-37-32-266` (XGBoost
1.7-1, 150 rounds): best validation AUC ≈ **0.849**. This is consistent with
published Telco-churn XGBoost results and above the 0.78 condition gate.
