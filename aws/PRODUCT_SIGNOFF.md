# POD Verifier — QA Sign-off & Product Ship Decision

**Author:** Forward-deployed engineering
**Date:** 2026-07-05
**Question:** Is the POD scoring table — used by operations to penalise bad PODs — good enough to push?

---

## Decision: CONDITIONAL GO

Two conditions, both required:

1. **Ship the normalization-fixed pipeline, not the current one.** The rewritten `handler.py` already applies ImageNet normalization (`IMAGENET_NORMALIZE=true`). The pipeline running in production today (`/255` only) is **NO-GO** — it silently loses ~30–45% recall.
2. **Launch as operator-assisted flagging, not automated penalties.** With the fix the table is a strong triage/prioritisation tool with a human confirming before money moves. It is **not yet** accurate enough to auto-penalise.

Pushing the table for **automated** financial penalties is a **NO-GO** at current accuracy.

---

## Evidence (gold-standard evaluation)

Ran the deployed checkpoint `best.pt` against the human gold set (`Data_annotation.xlsx`): **804 of 942** labelled images scored (138 dead links, 0 decode failures). Both preprocessings compared.

| Metric | `scale` (production today: /255) | `imagenet` (what training used) |
|---|---|---|
| context_valid F1 | 0.68 | **0.94** |
| package_visible F1 | 0.82 | **0.95** |
| label_readable F1 | 0.80 | **0.92** |
| image_clarity F1 | 0.73 | **0.91** |
| Best AWB-level FLAG F1 | 0.83 (only at threshold 0.85) | **0.93 (threshold 0.60)** |
| FLAG precision / recall | P 0.96–0.99 / **R 0.52–0.71** | P ~0.90 @ 0.55 / **R ~0.90–0.92** |

Source: `eval_out/eval_report.md`, `eval_out/metrics.json`, `eval_out/per_image_predictions.csv`.

---

## Root cause: preprocessing mismatch (highest-leverage fix)

The model was trained with ImageNet mean/std normalization (`augmentations.py`), but the production handler fed it `/255`-only tensors. Wrong input distribution → the model is **systematically under-confident** → precision stays high but **recall collapses**, so it misses a large share of genuinely bad PODs. Restoring the training-time normalization lifts every attribute F1 to 0.91–0.95 and recall to ~0.90.

**Fix status:** already implemented in the rewritten `handler.py` (`preprocess_image(..., normalize=True)`, defaulted on, set via `ImagenetNormalize=true` in `stack.yaml`). Verify it is present in the actual deployed image, then re-confirm in prod.

---

## Is the penalise table "good enough"? — read by use case

**Automated penalties (money moves without a human): NO.**
Even fixed, best usable precision at AWB level is ~0.90 → roughly **1 in 10 flagged PODs would be wrongly penalised**. For direct financial action that error rate is unacceptable; the bar should be ~0.95–0.97 precision.

**Operator-assisted flagging (human confirms before penalising): YES, conditionally.**
Per-attribute quality (0.91–0.95 F1) and AWB FLAG F1 0.93 make it a reliable way to surface and rank likely-bad PODs. The operator catches the residual false positives, so 0.90 precision is acceptable as a triage aid — provided the workflow keeps a human in the loop, which the phrase "manually penalise" implies.

---

## Recommended operating configuration

- **Preprocessing:** ImageNet normalization — mandatory (already default).
- **FLAG threshold:** the current `0.7` **over-flags** (lower precision). The sweep favours **~0.55–0.60**; for a penalise workflow lean precision-first (~0.55) and accept the recall trade. Calibrate on a labelled holdout and update **both** the `pod_scores_flagged` view and any ops query (the view currently hardcodes `0.7`).
- **Monitoring:** weekly, sample penalised PODs and measure realised precision; alarm if it drifts below target. Coverage metrics (`ImagesScored/Failed/Uncovered`) are already emitted.

---

## Data-quality flag (ops, not model)

**138 of 942 gold links (14.6%) were dead** — images expired/rotated off the store. In production the handler records these as `download_failed` and the flag view excludes them, so **no AWB is penalised on a failed fetch** (correct behaviour). But 1-in-7 PODs being unscoreable is a real gap: check image retention/expiry on the POD store and consider scoring closer to delivery time so links are still live.

---

## Path to full automation (future)

1. Deploy the normalization fix; re-run this eval in prod to confirm parity with the offline numbers.
2. Calibrate the threshold on a fresh labelled holdout; target AWB FLAG precision ≥0.95.
3. If automation is the goal, close the gap with targeted retraining on hard negatives and route borderline scores (composite ~0.5–0.7) to a manual review queue rather than auto-penalising.
4. Address link expiry so coverage isn't silently eroded.

---

## Sign-off

| Phase | Status | Notes |
|---|---|---|
| Development | ✅ | Single-invocation resilient handler + schema/infra; normalization fixed. |
| Testing | ✅ | 12/12 pass — coverage, resume, idempotency, continuation, inference. |
| QA | ✅ (with fix) | Model validated on gold set; correctness issue found and fixed. |
| Product | **CONDITIONAL GO** | Ship fixed pipeline into operator-assisted flagging; **not** auto-penalties; calibrate threshold; fix link expiry. |
