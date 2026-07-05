# POD Verifier — Engineering & QA Report

**Author:** Forward-deployed engineering
**Date:** 2026-07-05
**Scope:** Single-invocation redesign of the POD scoring pipeline + local validation + model-evaluation harness.
**Verdict so far:** Engineering & test phases **PASS**. Product ship/no-ship call is **BLOCKED** on one input — the gold-standard image *bytes* (see §6).

---

## 1. What was built (Development)

| Area | File | Change |
|---|---|---|
| Handler | `lambda_scorer/handler.py` | Rewritten: single invocation covers the whole dataset. Concurrent downloads (`ThreadPoolExecutor`), bounded-memory windowed loop, resume-from-checkpoint, idempotent upsert, per-input outcome recording, clock-aware self-continuation, coverage metrics, ImageNet-normalization fix. |
| Schema | `infra/schema.sql` | `status` + `failure_reason` columns; **unique `(awb, pod_link, run_date)`** for idempotent upsert; flag view excludes failures so ops never penalise a failed fetch. |
| Infra | `infra/stack.yaml` | Memory 3008→**10240 MB** (~6 vCPU); ephemeral 5120→512 MB; **async retries (2) + SQS DLQ**; single-trigger empty event; new tuning params; CloudWatch + DLQ IAM. |
| Provision | `provision-stack.sh` | New overrides (`MaxDownloadWorkers`, `WindowSize`, `ImagenetNormalize`). |
| Eval | `eval/evaluate_model.py` | Turnkey harness (local-dir or download mode) — per-attribute + operational metrics, both preprocessings, threshold sweep, penalty-safe detection. |
| Design | `SINGLE_INVOCATION_DESIGN.md` | Architecture + layered resilience/fallback (§6). |

Resilience answer to "what if a stage overruns the 15-min wall": the run is **resumable + auto-retried**. Scored rows are the checkpoint; a timeout counts as a failure so AWS re-invokes; each retry resumes; a clock-aware continuation hands off *before* the wall; exhausted retries land in a DLQ with a coverage alarm. A slow Metabase fetch or an overrun degrades to "the next attempt finishes," never a lost day.

---

## 2. Test results (Testing)

`pytest tests/` → **12 passed**. Run in-sandbox on torch 2.2.2 (CPU, aarch64).

| Test | Proves |
|---|---|
| `test_expand_dedup_and_filter` | Comma-split expansion, dedup, non-URL filtering. |
| `test_preprocess_shape_and_normalization` | Correct CHW shape; normalization actually applied. |
| `test_download_and_prepare_outcomes` | Success / HTTP-error / exception each produce a recorded outcome. |
| `test_download_window_partitions_all_inputs` | **Nothing dropped** — successes + failures == inputs. |
| `test_upsert_sql_is_idempotent` | SQL uses `ON CONFLICT (awb, pod_link, run_date) DO UPDATE`. |
| `test_load_done_keys` | Resume checkpoint reads scored rows. |
| `test_handler_covers_whole_dataset_one_invocation` | 2,500 images fully covered in ONE invocation, no continuation. |
| `test_handler_records_failures_as_outcomes` | scored + failed == total (coverage guaranteed). |
| `test_handler_resume_skips_done` | Re-run processes only the remainder. |
| `test_handler_continuation_near_wall` | Near the wall: exactly ONE continuation queued, status `continuing`. |
| `test_score_prepared_real_math` | Real torch inference wiring: logits→sigmoid→weighted composite. |
| `test_ephemeral_peak_mb` | `/tmp` sizing helper. |

---

## 3. Model runtime — validated

- `best.pt` loads into `MultiHeadEfficientNet` with **0 missing / 0 unexpected** keys → it is a genuine trained EfficientNet-B0 4-head checkpoint (not a dummy, not `PODNet`). Checkpoint is a bare `state_dict` (no training metadata).
- Forward pass runs; produces the four attribute probabilities + composite `pod_score`.
- Eval harness ran **end-to-end on synthetic images** (load gold → local images → both preprocessings → per-attribute + AWB-level metrics → report files). Pipeline is proven; only real pixels are missing.

---

## 4. Throughput — measured, with a real implication

Measured on this sandbox (**4 vCPU, CPU torch**): EfficientNet-B0 @224px ≈ **19 img/s** (batch 64).

| Volume | Inference-only @ this box | Note |
|---|---|---|
| 5,000 | ~260 s | fits one invocation |
| 20,000 | ~1,050 s | **exceeds the 900 s wall on this class of CPU** |

Reality check: this sandbox CPU is constrained and un-optimized (aarch64, no MKL). Lambda x86 at 10 GB / ~6 vCPU will be faster, but this is a real signal: **at the top of the 5k–20k range, EfficientNet CPU inference is the binding cost, not download.** That is exactly why the design has the **clock-aware continuation** (coverage still guaranteed across two invocations) and why **Lambda Managed Instances (16 vCPU)** is the recommended lever if 20k must finish in a single window. Action: measure duration on the first real Lambda run and, if it trends toward the wall, move to Managed Instances or trim input size.

---

## 5. Environment constraints hit during local testing (transparency)

- **POD image host unreachable from the build sandbox** — `growsimplee-sarathy-stg.blitznow.in` returns `403` through the egress proxy and `web_fetch` times out. So real images can't be downloaded here.
- **Metabase MCP** reaches the data plane (databases enumerated), confirming the *data* path; it does not return image bytes.
- **torch** had to be installed CPU-only (the PyTorch CPU index is also proxy-blocked; the current aarch64 PyPI wheel is a CUDA build) — resolved by pinning `torch==2.2.2` + `numpy<2`.

None of these are code defects; they are sandbox network limits.

---

## 6. What's blocking the ship decision (Product sign-off)

The ship/no-ship call on the operations penalise-table **requires real model metrics** (precision/recall/F1 on the gold set) — and those require the gold **image bytes**, which are not reachable from here and have not been uploaded yet.

To finish, provide the ~1,100 gold images (named by URL basename, e.g. `37507436_BZNYB2527064_1777731609190_1.png`) as a folder/zip. Then:

```
python eval/evaluate_model.py \
  --xlsx Data_annotation.xlsx \
  --model lambda_scorer/model/best.pt \
  --images-dir <uploaded_images_dir> \
  --out eval_out --preproc both
```

I will run this, then complete the **Product sign-off** using the agreed rule (operational bad-POD = weighted composite < 0.7) against these gates:

- **Penalty-safe precision** on the FLAG (bad) class at AWB level — a wrongly-flagged good POD is an unfair penalty, so precision is the priority. Target ≥ 0.90 before recommending push.
- **Recall** at that operating point — how many bad PODs we still catch.
- **Per-attribute F1** — whether all four heads are trustworthy or only some.
- Preprocessing check — whether normalized or raw scoring matches the trained checkpoint (the harness decides empirically).

---

## 7. Sign-off status

| Phase | Status |
|---|---|
| Development | ✅ Complete |
| Testing (logic + coverage + resilience) | ✅ Complete — 12/12 pass |
| Model runtime validation | ✅ Complete |
| Model quality metrics on gold set | ⛔ Pending gold image bytes |
| QA sign-off | ✅ Engineering/architecture; ⛔ model-quality gate pending |
| Product sign-off (ship / no-ship) | ⛔ Pending metrics |
