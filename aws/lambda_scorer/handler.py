"""
POD scoring Lambda — single invocation, resilient.

One daily EventBridge trigger scores the whole day's Proof-of-Delivery photos:

    read POD rows from Postgres  ->  expand links (kept bound to awb/trip)  ->
    skip already-scored (resume)  ->  for each window:
        download images concurrently into memory  ->  EfficientNet score  ->
        idempotent upsert to Postgres  ->  free memory
    ->  if near the time limit, hand off one checkpointed continuation.

Design guarantees:
  * bounded memory  — only one WINDOW_SIZE of images is held at a time.
  * full coverage   — every input ends as 'scored' or 'download_failed'.
  * idempotent      — upsert on (awb, pod_link, run_date); safe to re-run/resume.
  * time-safe       — a clock check hands off before the 15-minute wall.
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

# --------------------------------------------------------------------------- #
# Configuration (all from environment; set by CloudFormation)
# --------------------------------------------------------------------------- #

# Results database (where pod_scores lives).
PG_HOST = os.environ.get("PG_HOST", "")
PG_PORT = os.environ.get("PG_PORT", "5432")
PG_DATABASE = os.environ.get("PG_DATABASE", "pod_classifier")
PG_USER = os.environ.get("PG_USER", "postgres")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "")

# Source of the day's POD rows: a SQL query run against Postgres. It must return
# an AWB column, a trip-id column, and a column of POD image link(s) named one of
# POD / pod / pod_link (comma-separated links are expanded). The source DB falls
# back to the results DB unless SOURCE_PG_* is set separately.
SOURCE_QUERY = os.environ.get("SOURCE_QUERY", "")
SOURCE_PG_HOST = os.environ.get("SOURCE_PG_HOST", PG_HOST)
SOURCE_PG_PORT = os.environ.get("SOURCE_PG_PORT", PG_PORT)
SOURCE_PG_DATABASE = os.environ.get("SOURCE_PG_DATABASE", PG_DATABASE)
SOURCE_PG_USER = os.environ.get("SOURCE_PG_USER", PG_USER)
SOURCE_PG_PASSWORD = os.environ.get("SOURCE_PG_PASSWORD", PG_PASSWORD)

# Model / scoring.
MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/model/best.pt")
INPUT_SIZE = int(os.environ.get("INPUT_SIZE", "224"))
INFERENCE_BATCH_SIZE = int(os.environ.get("INFERENCE_BATCH_SIZE", "64"))
FLAG_THRESHOLD = float(os.environ.get("FLAG_THRESHOLD", "0.7"))
# Must match training preprocessing — do not disable in production.
IMAGENET_NORMALIZE = os.environ.get("IMAGENET_NORMALIZE", "true").lower() == "true"

# Download / memory tuning.
MAX_DOWNLOAD_WORKERS = int(os.environ.get("MAX_DOWNLOAD_WORKERS", "64"))
WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", "800"))
DOWNLOAD_TIMEOUT = int(os.environ.get("DOWNLOAD_TIMEOUT", "15"))
MIN_CONTENT_BYTES = int(os.environ.get("MIN_CONTENT_BYTES", "500"))

# Resilience.
CONTINUATION_SAFETY_MS = int(os.environ.get("CONTINUATION_SAFETY_MS", "90000"))
MAX_CONTINUATIONS = int(os.environ.get("MAX_CONTINUATIONS", "5"))

LAMBDA_FUNCTION_NAME = os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "")
CW_NAMESPACE = os.environ.get("CW_NAMESPACE", "PODPipeline")

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Model (loaded once per warm container)
# --------------------------------------------------------------------------- #

_model: Optional[MultiHeadEfficientNet] = None
_device: Optional[torch.device] = None


def get_model() -> tuple[MultiHeadEfficientNet, torch.device]:
    global _model, _device
    if _model is None:
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        _device = torch.device("cpu")
        _model = MultiHeadEfficientNet(num_attributes=4, pretrained=False)
        ckpt = torch.load(MODEL_PATH, map_location=_device, weights_only=True)
        _model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        _model.to(_device).eval()
        logger.info("Model loaded from %s", MODEL_PATH)
    return _model, _device


# --------------------------------------------------------------------------- #
# Postgres helpers
# --------------------------------------------------------------------------- #

def _connect(host, port, database, user, password):
    return psycopg2.connect(host=host, port=port, database=database,
                            user=user, password=password, connect_timeout=10)


def get_db_connection():
    """Connection to the results database (pod_scores)."""
    return _connect(PG_HOST, PG_PORT, PG_DATABASE, PG_USER, PG_PASSWORD)


def fetch_pod_data() -> pd.DataFrame:
    """Run SOURCE_QUERY against the source Postgres DB and return the rows."""
    conn = _connect(SOURCE_PG_HOST, SOURCE_PG_PORT, SOURCE_PG_DATABASE,
                    SOURCE_PG_USER, SOURCE_PG_PASSWORD)
    try:
        with conn.cursor() as cur:
            cur.execute(SOURCE_QUERY)
            columns = [d[0] for d in cur.description]
            rows = cur.fetchall()
    finally:
        conn.close()
    return pd.DataFrame(rows, columns=columns)


def expand_pod_links(df: pd.DataFrame) -> pd.DataFrame:
    """Explode comma-separated POD links into one row each, keeping awb + trip_id."""
    pod_col = next((c for c in ("POD", "pod", "pod_link") if c in df.columns), None)
    if pod_col is None:
        raise ValueError("No POD link column (POD / pod / pod_link) in source rows")

    awb_col = "AWB" if "AWB" in df.columns else "awb"
    trip_col = "Trip Id" if "Trip Id" in df.columns else "trip_id"

    rows = []
    for _, row in df.iterrows():
        raw = row.get(pod_col, "")
        if not isinstance(raw, str) or not raw.strip():
            continue
        for link in (l.strip() for l in raw.split(",")):
            if link.startswith("http"):
                rows.append({"awb": str(row.get(awb_col, "")),
                             "trip_id": str(row.get(trip_col, "")),
                             "pod_link": link})
    return (pd.DataFrame(rows)
            .drop_duplicates(subset=["awb", "pod_link"])
            .reset_index(drop=True))


def load_done_keys(conn, run_date: str) -> set:
    """Resume checkpoint: (awb, pod_link) already scored today."""
    with conn.cursor() as cur:
        cur.execute("SELECT awb, pod_link FROM pod_scores "
                    "WHERE run_date = %s AND status = 'scored'", (run_date,))
        return {(awb, link) for awb, link in cur.fetchall()}


def upsert_results(conn, rows: list[dict], run_date: str) -> int:
    """Idempotent bulk upsert on (awb, pod_link, run_date)."""
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
    tuples = [(r["awb"], r.get("trip_id"), r["pod_link"], run_date, r["status"],
               r.get("failure_reason"), r.get("pod_score"),
               r.get("context_valid_prob"), r.get("package_visible_prob"),
               r.get("label_readable_prob"), r.get("image_clarity_prob"))
              for r in rows]
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, tuples, page_size=200)
    conn.commit()
    return len(tuples)


# --------------------------------------------------------------------------- #
# Download + preprocess + score
# --------------------------------------------------------------------------- #

def build_session() -> requests.Session:
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=MAX_DOWNLOAD_WORKERS, pool_maxsize=MAX_DOWNLOAD_WORKERS,
        max_retries=requests.adapters.Retry(
            total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504]),
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def preprocess_image(img_rgb: np.ndarray, size: int = INPUT_SIZE,
                     normalize: bool = IMAGENET_NORMALIZE) -> np.ndarray:
    """Resize -> float[0,1] -> (optional) ImageNet normalize -> CHW float32."""
    img = cv2.resize(img_rgb, (size, size)).astype(np.float32) / 255.0
    if normalize:
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.transpose(img, (2, 0, 1)).astype(np.float32)


def download_and_prepare(session: requests.Session, row: dict) -> dict:
    """Download + decode + resize one image. Always returns an outcome dict."""
    base = {"awb": row["awb"], "trip_id": row["trip_id"], "pod_link": row["pod_link"]}
    try:
        resp = session.get(row["pod_link"], timeout=DOWNLOAD_TIMEOUT)
        if resp.status_code != 200:
            return {**base, "status": "download_failed", "failure_reason": f"http_{resp.status_code}"}
        if len(resp.content) < MIN_CONTENT_BYTES:
            return {**base, "status": "download_failed", "failure_reason": "too_small"}
        img = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return {**base, "status": "download_failed", "failure_reason": "decode_failed"}
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return {**base, "chw": preprocess_image(img_rgb)}   # prepared tensor = success
    except Exception as e:  # noqa: BLE001
        return {**base, "status": "download_failed", "failure_reason": type(e).__name__}


def download_window(session: requests.Session, rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Concurrently download a window. Returns (prepared, failures)."""
    prepared, failures = [], []
    workers = min(MAX_DOWNLOAD_WORKERS, max(1, len(rows)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for out in pool.map(lambda r: download_and_prepare(session, r), rows):
            (prepared if "chw" in out else failures).append(out)
    return prepared, failures


def score_prepared(model, device, successes: list[dict]) -> list[dict]:
    """Run inference; attach per-attribute probs + weighted composite pod_score."""
    results = []
    for start in range(0, len(successes), INFERENCE_BATCH_SIZE):
        chunk = successes[start:start + INFERENCE_BATCH_SIZE]
        batch = torch.from_numpy(np.stack([s["chw"] for s in chunk])).to(device)
        with torch.no_grad():
            logits = model(batch)
        probs = {name: torch.sigmoid(logits[name]).cpu() for name in ATTRIBUTE_NAMES}
        composite = sum(probs[ATTRIBUTE_NAMES[i]] * ATTRIBUTE_WEIGHTS[i] for i in range(4))
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


# --------------------------------------------------------------------------- #
# Observability + self-continuation
# --------------------------------------------------------------------------- #

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
        logger.warning("put_metric_data failed: %s", e)


def invoke_continuation(run_id: str, run_date: str, continuation: int) -> None:
    """Queue exactly one checkpointed continuation before the time limit."""
    boto3.client("lambda").invoke(
        FunctionName=LAMBDA_FUNCTION_NAME, InvocationType="Event",
        Payload=json.dumps({"run_id": run_id, "run_date": run_date,
                            "continuation": continuation}).encode("utf-8"),
    )
    logger.warning("Queued continuation #%d for run %s", continuation, run_id)


def _remaining_ms(context: Any) -> int:
    try:
        return int(context.get_remaining_time_in_millis())
    except Exception:  # noqa: BLE001 — local/tests have no real context
        return 10 ** 9


# --------------------------------------------------------------------------- #
# Handler
# --------------------------------------------------------------------------- #

def handler(event: dict, context: Any) -> dict:
    t_start = time.time()
    run_date = event.get("run_date") or date.today().isoformat()
    run_id = event.get("run_id") or f"{run_date}_{uuid.uuid4().hex[:8]}"
    continuation = int(event.get("continuation", 0))

    if not SOURCE_QUERY:
        return {"statusCode": 500, "body": json.dumps({"error": "Missing SOURCE_QUERY"})}
    if not PG_HOST or not PG_PASSWORD:
        return {"statusCode": 500, "body": json.dumps({"error": "Missing Postgres config"})}

    # Read + expand the day's POD rows (cheap, one shot).
    raw = fetch_pod_data()
    if raw.empty:
        return {"statusCode": 200, "body": json.dumps({"message": "No data", "run_id": run_id})}
    expanded = expand_pod_links(raw)
    total = len(expanded)
    if total == 0:
        return {"statusCode": 200, "body": json.dumps({"message": "No POD links", "run_id": run_id})}

    model, device = get_model()
    session = build_session()
    conn = get_db_connection()
    scored_total = failed_total = 0
    hit_time_limit = False
    try:
        done = load_done_keys(conn, run_date)
        pending = [r for r in expanded.to_dict("records")
                   if (r["awb"], r["pod_link"]) not in done]
        logger.info("run=%s total=%d done=%d pending=%d continuation=%d",
                    run_id, total, len(done), len(pending), continuation)

        for w in range(0, len(pending), WINDOW_SIZE):
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
            logger.info("window %d-%d done (scored=%d failed=%d)",
                        w, w + len(window), scored_total, failed_total)
    finally:
        conn.close()

    if hit_time_limit and continuation < MAX_CONTINUATIONS:
        invoke_continuation(run_id, run_date, continuation + 1)
        status = "continuing"
    else:
        # Done, or out of continuations: emit coverage so an alarm can fire if
        # scored + failed < total (i.e. we gave up with work remaining).
        emit_coverage(total, len(done) + scored_total, failed_total)
        status = "incomplete" if hit_time_limit else "complete"

    summary = {
        "run_id": run_id, "run_date": run_date, "total_images": total,
        "scored_this_invocation": scored_total, "failed_this_invocation": failed_total,
        "already_done_at_start": len(done),
        "status": status,
        "continuation": continuation,
        "invocation_duration_s": round(time.time() - t_start, 3),
    }
    logger.info("Pipeline %s: %s", status.upper(), json.dumps(summary))
    return {"statusCode": 200, "body": json.dumps(summary)}
