"""
POD Pipeline Lambda — Single-Invocation, Resilient Batch Processor.

EventBridge triggers ONCE with an empty event. A single invocation classifies
the entire day's POD dataset:

  Metabase fetch  →  expand POD links  →  (resume: skip already-scored)  →
  windowed loop { concurrent download+decode+resize → EfficientNet score →
                  idempotent upsert to Postgres → free memory } →
  coverage check.

Resilience (see aws/SINGLE_INVOCATION_DESIGN.md §6):
  * Idempotent upsert on (awb, pod_link, run_date) — retries fill gaps, never dup.
  * Resume-from-checkpoint — scored rows in Postgres ARE the checkpoint.
  * Every input gets an outcome row (scored OR recorded-failed) — real denominator.
  * Clock-aware continuation — if the invocation is about to hit the wall with work
    left, it flushes and triggers exactly ONE checkpointed continuation (bounded by
    MAX_CONTINUATIONS). The happy path is a single invocation.
  * Coverage metric — emits total/scored/failed; alarm when scored+failed < total.

Memory stays flat regardless of dataset size: only one WINDOW_SIZE of images is
ever in memory. Bounded download pool applies backpressure.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import time
import uuid
from datetime import date
from typing import Any, Optional

import boto3
import cv2
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import requests
import torch

from src.model import ATTRIBUTE_NAMES, ATTRIBUTE_WEIGHTS, MultiHeadEfficientNet

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

METABASE_URL = os.environ.get("METABASE_URL", "")
METABASE_API_KEY = os.environ.get("METABASE_API_KEY", "")
METABASE_CARD_ID = int(os.environ.get("METABASE_CARD_ID", "10989"))

INFERENCE_BATCH_SIZE = int(os.environ.get("INFERENCE_BATCH_SIZE", "64"))
FLAG_THRESHOLD = float(os.environ.get("FLAG_THRESHOLD", "0.7"))

# Pipeline tuning
MAX_DOWNLOAD_WORKERS = int(os.environ.get("MAX_DOWNLOAD_WORKERS", "64"))
WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", "800"))
DOWNLOAD_TIMEOUT = int(os.environ.get("DOWNLOAD_TIMEOUT", "15"))
MIN_CONTENT_BYTES = int(os.environ.get("MIN_CONTENT_BYTES", "500"))

# Resilience tuning
CONTINUATION_SAFETY_MS = int(os.environ.get("CONTINUATION_SAFETY_MS", "90000"))
MAX_CONTINUATIONS = int(os.environ.get("MAX_CONTINUATIONS", "5"))

PG_HOST = os.environ.get("PG_HOST", "")
PG_PORT = os.environ.get("PG_PORT", "5432")
PG_DATABASE = os.environ.get("PG_DATABASE", "pod_classifier")
PG_USER = os.environ.get("PG_USER", "postgres")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "")

MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/model/best.pt")
INPUT_SIZE = int(os.environ.get("INPUT_SIZE", "224"))
# Match training-time preprocessing. ImageNet stats are the timm/EfficientNet
# default; eval/evaluate_model.py empirically confirms the correct setting.
IMAGENET_NORMALIZE = os.environ.get("IMAGENET_NORMALIZE", "true").lower() == "true"

LAMBDA_FUNCTION_NAME = os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "")
CW_NAMESPACE = os.environ.get("CW_NAMESPACE", "PODPipeline")

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ---------------------------------------------------------------------------
# Model cache (warm across invocations within same container)
# ---------------------------------------------------------------------------

_model: Optional[MultiHeadEfficientNet] = None
_device: Optional[torch.device] = None


def get_model() -> tuple[MultiHeadEfficientNet, torch.device]:
    """Load model once per container lifetime; use every available vCPU."""
    global _model, _device
    if _model is None:
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        _device = torch.device("cpu")
        _model = MultiHeadEfficientNet(num_attributes=4, pretrained=False)
        checkpoint = torch.load(MODEL_PATH, map_location=_device, weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        _model.load_state_dict(state_dict)
        _model.to(_device)
        _model.eval()
        logger.info("Model loaded from %s (threads=%d)", MODEL_PATH, torch.get_num_threads())
    return _model, _device


# ---------------------------------------------------------------------------
# HTTP session
# ---------------------------------------------------------------------------


def build_session() -> requests.Session:
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=MAX_DOWNLOAD_WORKERS,
        pool_maxsize=MAX_DOWNLOAD_WORKERS,
        max_retries=requests.adapters.Retry(
            total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504]
        ),
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# ---------------------------------------------------------------------------
# Metabase data fetch + link expansion
# ---------------------------------------------------------------------------


def fetch_pod_data(session: requests.Session) -> pd.DataFrame:
    """Fetch today's POD data from the configured Metabase card."""
    url = f"{METABASE_URL}/api/card/{METABASE_CARD_ID}/query/json"
    resp = session.post(url, headers={"x-api-key": METABASE_API_KEY}, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return pd.DataFrame(data) if data else pd.DataFrame()


def expand_pod_links(df: pd.DataFrame) -> pd.DataFrame:
    """Expand comma-separated POD URLs into one row per image link."""
    pod_col = next((c for c in ("POD", "pod", "pod_link") if c in df.columns), None)
    if pod_col is None:
        raise ValueError("POD column not found in DataFrame")

    awb_col = "AWB" if "AWB" in df.columns else "awb"
    trip_col = "Trip Id" if "Trip Id" in df.columns else "trip_id"

    rows = []
    for _, row in df.iterrows():
        raw = row.get(pod_col, "")
        if not isinstance(raw, str) or not raw.strip():
            continue
        for link in (l.strip() for l in raw.split(",")):
            if link.startswith("http"):
                rows.append({
                    "awb": str(row.get(awb_col, "")),
                    "trip_id": str(row.get(trip_col, "")),
                    "pod_link": link,
                })
    return pd.DataFrame(rows).drop_duplicates(subset=["awb", "pod_link"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Preprocessing + concurrent download
# ---------------------------------------------------------------------------


def preprocess_image(img_rgb: np.ndarray, size: int = INPUT_SIZE,
                     normalize: bool = IMAGENET_NORMALIZE) -> np.ndarray:
    """Resize → float[0,1] → (optional) ImageNet normalize → CHW float32."""
    img = cv2.resize(img_rgb, (size, size)).astype(np.float32) / 255.0
    if normalize:
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.transpose(img, (2, 0, 1)).astype(np.float32)


def download_and_prepare(session: requests.Session, row: dict) -> dict:
    """
    Download + decode + resize a single image. ALWAYS returns an outcome dict:
      success →  {..., "status": "scored",           "chw": np.ndarray}
      failure →  {..., "status": "download_failed",  "failure_reason": str}
    Decode/resize run here so CPU work overlaps network wait.
    """
    base = {"awb": row["awb"], "trip_id": row["trip_id"], "pod_link": row["pod_link"]}
    try:
        resp = session.get(row["pod_link"], timeout=DOWNLOAD_TIMEOUT)
        if resp.status_code != 200:
            return {**base, "status": "download_failed", "failure_reason": f"http_{resp.status_code}"}
        if len(resp.content) < MIN_CONTENT_BYTES:
            return {**base, "status": "download_failed", "failure_reason": "too_small"}
        arr = np.frombuffer(resp.content, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return {**base, "status": "download_failed", "failure_reason": "decode_failed"}
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return {**base, "status": "scored", "chw": preprocess_image(img_rgb)}
    except Exception as e:  # noqa: BLE001
        return {**base, "status": "download_failed", "failure_reason": type(e).__name__}


def download_window(session: requests.Session, rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Concurrently download+prepare a window. Returns (successes, failures)."""
    successes, failures = [], []
    workers = min(MAX_DOWNLOAD_WORKERS, max(1, len(rows)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for out in ex.map(lambda r: download_and_prepare(session, r), rows):
            (successes if out["status"] == "scored" else failures).append(out)
    return successes, failures


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_prepared(model: MultiHeadEfficientNet, device: torch.device,
                   successes: list[dict]) -> list[dict]:
    """Run inference on prepared CHW arrays; attach probs + composite pod_score."""
    if not successes:
        return []
    weights = ATTRIBUTE_WEIGHTS
    results: list[dict] = []
    for start in range(0, len(successes), INFERENCE_BATCH_SIZE):
        chunk = successes[start:start + INFERENCE_BATCH_SIZE]
        batch = torch.from_numpy(np.stack([s["chw"] for s in chunk])).to(device)
        with torch.no_grad():
            logits = model(batch)
        probs = {name: torch.sigmoid(logits[name]).cpu() for name in ATTRIBUTE_NAMES}
        composite = sum(probs[ATTRIBUTE_NAMES[i]] * weights[i] for i in range(4))
        for j, s in enumerate(chunk):
            results.append({
                "awb": s["awb"], "trip_id": s["trip_id"], "pod_link": s["pod_link"],
                "status": "scored", "failure_reason": None,
                "pod_score": round(float(composite[j]), 6),
                "context_valid_prob": round(float(probs["context_valid"][j]), 6),
                "package_visible_prob": round(float(probs["package_visible"][j]), 6),
                "label_readable_prob": round(float(probs["label_readable"][j]), 6),
                "image_clarity_prob": round(float(probs["image_clarity"][j]), 6),
            })
    return results


# ---------------------------------------------------------------------------
# Postgres — checkpoint (resume) + idempotent upsert
# ---------------------------------------------------------------------------


def get_db_connection():
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, database=PG_DATABASE,
        user=PG_USER, password=PG_PASSWORD, connect_timeout=10,
    )


def load_done_keys(conn, run_date: str) -> set:
    """Resume checkpoint: (awb, pod_link) already scored today."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT awb, pod_link FROM pod_scores WHERE run_date = %s AND status = 'scored'",
            (run_date,),
        )
        return {(a, p) for a, p in cur.fetchall()}


def upsert_results(conn, rows: list[dict], run_date: str) -> int:
    """Idempotent bulk upsert on (awb, pod_link, run_date). Returns row count."""
    if not rows:
        return 0
    sql = """
        INSERT INTO pod_scores
            (awb, trip_id, pod_link, run_date, status, failure_reason,
             pod_score, context_valid_prob, package_visible_prob,
             label_readable_prob, image_clarity_prob)
        VALUES %s
        ON CONFLICT (awb, pod_link, run_date) DO UPDATE SET
            status = EXCLUDED.status,
            failure_reason = EXCLUDED.failure_reason,
            pod_score = EXCLUDED.pod_score,
            context_valid_prob = EXCLUDED.context_valid_prob,
            package_visible_prob = EXCLUDED.package_visible_prob,
            label_readable_prob = EXCLUDED.label_readable_prob,
            image_clarity_prob = EXCLUDED.image_clarity_prob,
            scored_at = NOW()
    """
    tuples = [
        (r["awb"], r.get("trip_id"), r["pod_link"], run_date, r["status"],
         r.get("failure_reason"), r.get("pod_score"),
         r.get("context_valid_prob"), r.get("package_visible_prob"),
         r.get("label_readable_prob"), r.get("image_clarity_prob"))
        for r in rows
    ]
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, tuples, page_size=200)
    conn.commit()
    return len(tuples)


# ---------------------------------------------------------------------------
# Observability + self-continuation
# ---------------------------------------------------------------------------


def emit_coverage(total: int, scored: int, failed: int) -> None:
    logger.info("COVERAGE total=%d scored=%d failed=%d covered=%s",
                total, scored, failed, scored + failed >= total)
    try:
        boto3.client("cloudwatch").put_metric_data(
            Namespace=CW_NAMESPACE,
            MetricData=[
                {"MetricName": "ImagesTotal", "Value": total},
                {"MetricName": "ImagesScored", "Value": scored},
                {"MetricName": "ImagesFailed", "Value": failed},
                {"MetricName": "ImagesUncovered", "Value": max(0, total - scored - failed)},
            ],
        )
    except Exception as e:  # noqa: BLE001 — metrics are best-effort
        logger.warning("CloudWatch put_metric_data failed: %s", e)


def invoke_continuation(run_id: str, run_date: str, continuation: int) -> None:
    """Trigger exactly one checkpointed continuation (Layer 2 safety valve)."""
    boto3.client("lambda").invoke(
        FunctionName=LAMBDA_FUNCTION_NAME,
        InvocationType="Event",
        Payload=json.dumps({
            "run_id": run_id, "run_date": run_date, "continuation": continuation,
        }).encode("utf-8"),
    )
    logger.warning("Triggered continuation #%d for run %s (approaching time limit)",
                   continuation, run_id)


def _remaining_ms(context: Any) -> int:
    try:
        return int(context.get_remaining_time_in_millis())
    except Exception:  # noqa: BLE001 — local/tests have no real context
        return 10 ** 9


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


def handler(event: dict, context: Any) -> dict:
    """Single-invocation POD scoring with resume + bounded self-continuation."""
    t_start = time.time()
    run_date = event.get("run_date") or date.today().isoformat()
    run_id = event.get("run_id") or f"{run_date}_{uuid.uuid4().hex[:8]}"
    continuation = int(event.get("continuation", 0))

    if not METABASE_URL or not METABASE_API_KEY:
        return {"statusCode": 500, "body": json.dumps({"error": "Missing Metabase config"})}

    session = build_session()

    # Fetch + expand (cheap, one shot; fails fast if Metabase is slow).
    raw = fetch_pod_data(session)
    if raw.empty:
        return {"statusCode": 200, "body": json.dumps({"message": "No data", "run_id": run_id})}
    expanded = expand_pod_links(raw)
    total = len(expanded)
    if total == 0:
        return {"statusCode": 200, "body": json.dumps({"message": "No POD links", "run_id": run_id})}

    model, device = get_model()
    conn = get_db_connection()
    scored_total = failed_total = 0
    hit_time_limit = False
    try:
        # Resume: skip rows already scored today.
        done = load_done_keys(conn, run_date)
        pending = [r for r in expanded.to_dict("records")
                   if (r["awb"], r["pod_link"]) not in done]
        logger.info("run=%s total=%d already_scored=%d pending=%d continuation=%d",
                    run_id, total, len(done), len(pending), continuation)

        for w in range(0, len(pending), WINDOW_SIZE):
            # Safety valve: stop before the wall, hand off one continuation.
            if _remaining_ms(context) < CONTINUATION_SAFETY_MS:
                hit_time_limit = True
                break

            window = pending[w:w + WINDOW_SIZE]
            successes, failures = download_window(session, window)
            scored = score_prepared(model, device, successes)
            upsert_results(conn, scored + failures, run_date)
            scored_total += len(scored)
            failed_total += len(failures)
            del successes, failures, scored
            logger.info("window %d-%d done (scored_total=%d failed_total=%d)",
                        w, w + len(window), scored_total, failed_total)
    finally:
        conn.close()

    if hit_time_limit and continuation < MAX_CONTINUATIONS:
        invoke_continuation(run_id, run_date, continuation + 1)

    # Recompute true coverage from DB-independent counters for this run.
    covered = len(done) + scored_total + failed_total if not hit_time_limit else None
    if not hit_time_limit:
        emit_coverage(total, len(done) + scored_total, failed_total)

    summary = {
        "run_id": run_id, "run_date": run_date, "total_images": total,
        "scored_this_invocation": scored_total, "failed_this_invocation": failed_total,
        "already_done_at_start": len(done),
        "status": "continuing" if hit_time_limit else "complete",
        "continuation": continuation,
        "invocation_duration_s": round(time.time() - t_start, 3),
    }
    logger.info("=== Pipeline %s === %s", summary["status"].upper(), json.dumps(summary))
    return {"statusCode": 200, "body": json.dumps(summary)}
