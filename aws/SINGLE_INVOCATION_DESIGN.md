# POD Verifier — Single-Invocation Redesign

**Goal:** One EventBridge trigger fires **one** Lambda invocation that classifies the **entire** day's POD dataset (5k–20k images) every time, reliably, within Lambda's runtime limits. Retire the self-invoking batch chain.

**Status:** Design / plan. No code changed yet.

---

## 1. Why change the current design

Today the Lambda is *self-invoking*: EventBridge sends `{"i":0}`, and each invocation downloads 500 images, scores them, writes to Postgres, then asynchronously re-invokes itself with `i += 500` until the dataset is exhausted. State (the URL manifest) is passed through S3.

Two structural problems:

- **No coverage guarantee.** The chain is a series of *async* self-invokes with no orchestrator, no DLQ, and no completion check. If any link fails — a throttle, an unhandled exception, an image host hiccup — the chain silently stops and the rest of the day is never scored, with no signal. Separately, `download_batch_to_memory` silently `continue`s past every failed/short/undecodable image, so those inputs vanish with no record.
- **Serial download is the real cost, and it's paid per link.** Images are fetched one at a time with a blocking `session.get`. That, not the model, dominates wall-clock time (see §3).

The good news: at 5k–20k images the whole job fits comfortably in a **single** invocation once the download is parallelized and memory is bounded. That is the redesign.

---

## 2. The hard constraints (the "wall")

These are the Lambda limits the design must live inside.

| Knob | Lambda (default compute) | Lambda Managed Instances |
|---|---|---|
| Max memory | 10,240 MB | 32,768 MB |
| vCPUs | ~6 (at 10,240 MB; 1 vCPU per 1,769 MB, scales with memory) | up to 16; **configurable** memory:vCPU ratio (2:1, 4:1, 8:1) |
| Max timeout | **900 s (15 min) — hard** | EC2-backed, no cold start; verify per-invoke duration cap before relying on >15 min |
| /tmp ephemeral | up to 10,240 MB | n/a (in-memory design uses no disk) |
| Concurrency model | 1 invoke per environment | multi-concurrent per environment |

The 900 s ceiling on default compute is the number that matters: **the whole dataset must finish inside one 15-minute window.** The design targets ~2–4 min of wall-clock for 20k images, leaving a wide margin.

The other lever is the **memory↔vCPU coupling**: on default compute, CPU is bought with memory. The current 3,008 MB gives only ~1.7 vCPU, which throttles inference. Sizing memory up is how you "manage the runtime."

---

## 3. Bottleneck analysis (ranked)

Numbers below assume ~300 ms average per-image download latency and a batched EfficientNet-B0 forward pass at 224px. Treat them as order-of-magnitude; validate against CloudWatch after the first real run (§8).

### #1 — Image download (network I/O) — *dominant, and the one to fix first*

| Approach | Effective throughput | 20,000 images |
|---|---|---|
| Sequential (current) | ~3 img/s | **~100 min → impossible** |
| Thread pool, 64 workers | ~210 img/s | ~95 s |
| Thread pool, 128 workers | ~425 img/s | ~47 s |

Downloads are I/O-bound, so concurrency buys near-linear speedup until the source host or the network path saturates. This single change moves download from "impossible in 15 min" to "~1 min."

**Control:** a `ThreadPoolExecutor` (tunable `MAX_DOWNLOAD_WORKERS`, start at 64) with the `urllib3` connection-pool `maxsize` matched to the worker count. Decode + resize happen inside the worker, right after download, so CPU decode overlaps network wait and only the 224×224 tensor is retained. Watch for HTTP 429 from the image host and for NAT-gateway throughput if running default compute in a VPC.

### #2 — Inference (CPU) — *gated by memory→vCPU sizing*

| Compute | vCPUs | ~throughput | 20,000 images |
|---|---|---|---|
| Current 3,008 MB | ~1.7 | ~45 img/s | ~440 s (7+ min) |
| Default, 10,240 MB | ~6 | ~150–250 img/s | ~80–130 s |
| Managed Instances, 32 GB @ 2:1 | 16 | ~400+ img/s | ~50 s |

Inference time is inversely proportional to vCPUs, and vCPUs come from memory. This is why the current 3 GB config is itself a bottleneck. **Control:** set `torch.set_num_threads()` to the vCPU count and size memory to 10 GB (default) or move to Managed Instances.

### #3 — Decode / resize (CPU)
`cv2.imdecode` + resize is ~10 ms/image and runs inside the download workers, so it overlaps network I/O and adds ~no serial time. Storing the resized `224×224×3 uint8` (150 KB) instead of the full-res image is also the main memory saver.

### #4 — Postgres writes
`execute_values` bulk insert over one persistent connection; ~20k rows flush in ~10–30 s. Minor. Keep the connection open for the whole invocation and flush per window.

### #5 — Metabase fetch + model load
One Metabase query (~10–30 s for a large payload) and one ~19 MB checkpoint load per cold container. Both one-time; negligible against the batch.

---

## 4. Target architecture

```
EventBridge Scheduler ──(once/day, empty event)──▶  ONE Lambda invocation
                                                         │
        ┌────────────────────────────────────────────────┘
        ▼
  1. Fetch POD data from Metabase  ─►  expand links  ─►  total = N  (dedup)
        │
        ▼   producer/consumer pipeline (bounded queues = backpressure = flat memory)
   ┌──────────────────────────────────────────────────────────────┐
   │  W download+decode+resize threads  ──►  bounded tensor queue  │
   │                                              │                │
   │                     main thread pulls fixed inference batches │
   │                     ─► EfficientNet score ─► accumulate       │
   │                     ─► flush window to Postgres (upsert)      │
   └──────────────────────────────────────────────────────────────┘
        │  loop until every one of N inputs has an outcome
        ▼
  2. Assert processed == N  ─►  emit coverage metric  ─►  done (no self-invoke)
```

Key properties:

- **One invocation, no cursor, no self-invoke.** EventBridge sends an empty/simple event; the handler owns the full run start-to-finish.
- **Bounded memory regardless of dataset size.** The pipeline never holds all N images. A sliding window (e.g. 1,000–2,000 in flight) is downloaded, scored, flushed, and dropped. Bounded queues apply backpressure so a fast downloader can't outrun the scorer and blow up memory. Peak ≈ one window of tensors (~150–300 MB) + model + runtime (~1–2 GB), comfortably inside 10 GB.
- **Download and inference overlap.** Because network I/O and CPU inference run concurrently through the queue, wall-clock ≈ `max(download_time, inference_time)` plus a small tail — not their sum. That's the ~2–4 min figure for 20k.
- **S3 becomes optional.** The manifest in S3 only existed to pass state between self-invocations. With a single invocation there's no cross-invocation state, so S3 can be dropped entirely (or kept solely to support the resume safety-valve in §6).

---

## 5. Lambda sizing — pick by volume

The **code is identical** for both options; only the deployment target and config differ.

**Option A — Default compute, maxed (recommended for ≤ ~15k, simplest):**

| Setting | Value |
|---|---|
| MemorySize | 10,240 MB (≈6 vCPU) |
| Timeout | 900 s |
| EphemeralStorage | 512 MB (in-memory design, no /tmp) |
| Reserved concurrency | 1 (only one run at a time) |
| `torch` threads | = vCPU count |
| `MAX_DOWNLOAD_WORKERS` | 64 (tune) |

**Option B — Lambda Managed Instances (headroom / growth / top-of-range volume):**

- 32 GB @ 2:1 → **16 vCPU** on current-gen (e.g. Graviton4) with high-bandwidth networking, **no cold start**.
- Best when volume rides the upper end of 5k–20k, is trending up, or when you want maximum margin. Multi-concurrency and EC2-based pricing (instance cost + 15% management fee, Savings-Plan eligible) apply — evaluate cost vs. a ~3-min/day default-compute run.
- **Verify** the max execution duration for Managed Instances before assuming it lifts the 15-min wall; the design finishes in ~1–3 min either way, so it stays safe under any cap.

**Recommendation:** Start on **Option A at 10 GB**. It clears 20k with margin. Move to **Option B** only if measured runs approach the window, volume grows past ~20k, or you want the no-cold-start / networking benefits. Right-size from the CloudWatch numbers in §8 rather than guessing.

---

## 6. Coverage + resilience — a single invocation must still be recoverable

A single invocation removes the *silent-chain-death* failure of the self-invoke design, but it introduces its own risk the user correctly flagged: **it's one shot against the 900 s wall.** If Metabase is slow, downloads stall, or volume spikes, that one invocation can run out of time and lose the day. Making the run faster lowers the probability but doesn't remove the failure mode. So the design must be **resumable and automatically retried**, not just fast.

The governing principle: **any failure should cost at most a partial re-run, never the whole day, and recovery must be automatic.** That's achieved in layers — Layer 0 is mandatory and makes every higher layer cheap.

### Layer 0 — Make failure cheap to recover from (foundation, mandatory)

Without this, no retry is safe; with it, a retry is just "resume."

1. **Idempotent upsert.** Unique key `(awb, pod_link, run_date)` + `INSERT … ON CONFLICT DO UPDATE`. (Today: `SERIAL` PK, no unique constraint → re-runs duplicate.) This is what makes a retry safe — it fills gaps instead of double-counting.
2. **Resume from checkpoint.** On start, compute the work set as *expanded links for `run_date` not already in `pod_scores`*. A fresh run does everything; a retry after a partial run does only the remainder. The scored rows themselves are the checkpoint.
3. **Every input ends with an outcome.** Replace the blanket `continue` on download/decode failure with a recorded result: after `urllib3 Retry` still fails, write the row with a NULL score + `status`/`failure_reason`. This gives a real denominator (`total`) and a coverage check `scored + failed == total` — you can't detect an incomplete run if failures vanish silently.
4. **Cache the expanded manifest** (the one good use of S3 here). Fetch Metabase + expand **once**, write the manifest to S3; retries load it instead of re-hitting the slow Metabase call. Fetch is done first and cheap, so if Metabase is the thing that's slow, you fail early and cheaply rather than after burning the download/inference budget.

### Layer 1 — Automatic retry of the whole invocation (cheap, no new services)

- **Async invoke config + DLQ.** Trigger the function asynchronously with `MaximumRetryAttempts = 2` and an on-failure destination (SQS DLQ). A **timeout counts as a failure**, so AWS re-invokes automatically; with Layer 0 each retry resumes and advances. The DLQ captures the case where even the retries didn't finish, for alerting.
- **Coverage metric + alarm.** Publish `images_total / scored / failed`; alarm when `scored + failed < total` at end-of-run. This is the signal that turns a silent shortfall into a page.

### Layer 2 — Bounded, clock-aware continuation (self-invoke, done right)

This is the old idea fixed. Instead of blindly re-invoking every 500 images, the handler watches its own clock via `context.get_remaining_time_in_millis()`. **Only if** time is about to run out with work remaining does it flush, then trigger **exactly one** continuation that resumes from the Layer-0 checkpoint. Guardrails: a `max_continuations` cap and an alarm so it can never become a runaway chain. Result: the happy path is a single invocation; continuation fires solely as a safety valve when genuinely needed — bounded, checkpointed, and observable.

### Layer 3 — Orchestration (production-grade, optional)

For the strongest guarantee, wrap the Lambda in a **Step Functions** state machine triggered once by EventBridge: `Retry` with exponential backoff, `Catch` → fallback state, and an explicit "is everything covered?" `Choice` that loops until `scored + failed == total` or raises. This externalizes all retry/continuation logic from the function, adds full run visibility, and is the right call if this pipeline is business-critical or volume outgrows a comfortable single-run window. Cost: one more piece of infra.

### Recommended stack

**Layer 0 + Layer 1** are the baseline — mandatory and nearly free. Add **Layer 2** so the system self-heals within its own trigger without extra services. Reach for **Layer 3 (Step Functions)** only when criticality or volume justifies the added infra. In all cases the specific worry — a slow Metabase fetch or an overrun — degrades to "the next automatic attempt resumes and finishes," never a lost day.

---

## 7. Correctness fixes to fold in

The rewrite is the moment to close three issues found in the current code:

- **Inference normalization gap.** Training (`augmentations.py`) normalizes with ImageNet mean/std; the in-memory inference path only does `/255.0`. If `best.pt` was trained with normalization, apply the same mean/std at inference — otherwise scores are systematically off. Verify against how the checkpoint was actually trained.
- **Model identity.** Comments call `PODNet` "production," but the handler loads `MultiHeadEfficientNet` (EfficientNet-B0). Make the deployed model explicit and consistent so the checkpoint and the class always match.
- **Unique constraint.** Add it (see §6.3) so re-runs and retries are safe.

---

## 8. Rollout & validation

1. Implement the pipeline handler (concurrent download + bounded window loop + upsert + coverage metrics).
2. Deploy **Option A at 10 GB**, reserved concurrency 1, EventBridge → single trigger (empty event).
3. Run once on a real day. From CloudWatch read: **max memory used**, **duration**, per-stage timings, and `scored + failed == total`.
4. Tune: raise/lower `MAX_DOWNLOAD_WORKERS` (watch host 429s and NAT throughput), adjust window size, right-size memory down if headroom is large or up (or to Managed Instances) if duration approaches the window.
5. Keep the safety-valve alarm wired so any incomplete day is visible and auto-recoverable via re-run.

---

## 9. Risks & watch-items

- **Source image host rate-limits** the higher download concurrency (429s). Mitigate with worker-count tuning, backoff (already in the Retry adapter), and politeness.
- **VPC egress** on default compute goes via NAT — NAT bandwidth/cost can cap download throughput. Managed Instances' high-bandwidth networking helps; consider a VPC endpoint / gateway if the host is on AWS.
- **Volume growth beyond ~20k.** The bounded-window loop still covers it, but wall-clock scales with N; if a single run nears 900 s, move to Managed Instances or reintroduce a *bounded, orchestrated* continuation (Step Functions), not the old fire-and-forget self-invoke.
- **Managed Instances duration cap** — unverified here; confirm before depending on runs longer than 15 min.

---

*Sources: [Configure Lambda memory](https://docs.aws.amazon.com/lambda/latest/dg/configuration-memory.html), [Lambda quotas](https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html), [10 GB / 6 vCPU announcement](https://aws.amazon.com/about-aws/whats-new/2020/12/aws-lambda-supports-10gb-memory-6-vcpu-cores-lambda-functions/), [32 GB / 16 vCPU Managed Instances announcement (Mar 2026)](https://aws.amazon.com/about-aws/whats-new/2026/03/lambda-32-gb-memory-16-vcpus/), [Lambda Managed Instances docs](https://docs.aws.amazon.com/lambda/latest/dg/lambda-managed-instances.html).*
