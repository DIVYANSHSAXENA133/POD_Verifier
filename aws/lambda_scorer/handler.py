"""
POD Pipeline Lambda — Self-Invoking Batch Processor.

EventBridge triggers with {"i": 0}. The Lambda processes images in batches,
re-invoking itself until all images are scored. Each invocation downloads
images to memory, scores them with EfficientNet, writes results directly
to Postgres, then invokes itself for the next batch.

Flow:
  i=0        → Metabase fetch, expand POD links, store links manifest in S3,
               download batch to memory, score, write to Postgres, invoke self.
  0<i<total  → Load links manifest from S3, download next batch to memory,
               score, write to Postgres, invoke self.
  i>=total   → All batches done. Clean up S3 manifest.

No images or results touch S3 — only the URL manifest (which URLs to
download) is stored in S3 because it can exceed the 256KB async invoke
payload limit. Scoring is purely in-memory, Postgres is the final store.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import date
from typing import Any
from urllib.parse import urlparse

import boto3
import cv2
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import requests
import torch
from torch.utils.data import DataLoader, Dataset

from src.model import ATTRIBUTE_NAMES, ATTRIBUTE_WEIGHTS, MultiHeadEfficientNet

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

METABASE_URL = os.environ.get("METABASE_URL", "")
METABASE_API_KEY = os.environ.get("METABASE_API_KEY", "")
METABASE_CARD_ID = int(os.environ.get("METABASE_CARD_ID", "10989"))

FETCH_BATCH_SIZE = int(os.environ.get("FETCH_BATCH_SIZE", "500"))
INFERENCE_BATCH_SIZE = int(os.environ.get("INFERENCE_BATCH_SIZE", "64"))
FLAG_THRESHOLD = float(os.environ.get("FLAG_THRESHOLD", "0.7"))

PG_HOST = os.environ.get("PG_HOST", "")
PG_PORT = os.environ.get("PG_PORT", "5432")
PG_DATABASE = os.environ.get("PG_DATABASE", "pod_classifier")
PG_USER = os.environ.get("PG_USER", "postgres")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "")

MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/model/best.pt")
INPUT_SIZE = int(os.environ.get("INPUT_SIZE", "224"))

S3_BUCKET = os.environ.get("S3_STATE_BUCKET", "")
LAMBDA_FUNCTION_NAME = os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "")

# ---------------------------------------------------------------------------
# Model cache (warm across invocations within same container)
# ---------------------------------------------------------------------------

_model: MultiHeadEfficientNet | None = None
_device: torch.device | None = None


def get_model() -> tuple[MultiHeadEfficientNet, torch.device]:
    """Load model once per container lifetime."""
    global _model, _device
    if _model is None:
        _device = torch.device("cpu")
        _model = MultiHeadEfficientNet(num_attributes=4, pretrained=False)
        checkpoint = torch.load(MODEL_PATH, map_location=_device, weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        _model.load_state_dict(state_dict)
        _model.to(_device)
        _model.eval()
        logger.info("Model loaded from %s", MODEL_PATH)
    return _model, _device


# ---------------------------------------------------------------------------
# HTTP session
# ---------------------------------------------------------------------------


def build_session() -> requests.Session:
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        max_retries=requests.adapters.Retry(
            total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504]
        )
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# ---------------------------------------------------------------------------
# Metabase data fetch
# ---------------------------------------------------------------------------


def fetch_pod_data(session: requests.Session) -> pd.DataFrame:
    """Fetch today's POD data from Metabase card."""
    url = f"{METABASE_URL}/api/card/{METABASE_CARD_ID}/query/json"
    headers = {"X-Metabase-Session": METABASE_API_KEY}
    resp = session.post(url, headers=headers, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# POD link expansion
# ---------------------------------------------------------------------------


def expand_pod_links(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand comma-separated POD URLs into individual rows.
    Returns DataFrame with columns: original columns + 'pod_link'.
    """
    pod_col = None
    for candidate in ("POD", "pod"):
        if candidate in df.columns:
            pod_col = candidate
            break
    if pod_col is None:
        raise ValueError("POD column not found in DataFrame")

    rows = []
    for _, row in df.iterrows():
        raw = row.get(pod_col, "")
        if not isinstance(raw, str) or not raw.strip():
            continue
        links = [link.strip() for link in raw.split(",")]
        for link in links:
            if link.startswith("http"):
                new_row = row.to_dict()
                new_row["pod_link"] = link
                rows.append(new_row)

    result = pd.DataFrame(rows)
    if pod_col in result.columns:
        result = result.drop(columns=[pod_col])
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Image download (in-memory)
# ---------------------------------------------------------------------------


def download_batch_to_memory(
    session: requests.Session, batch: pd.DataFrame
) -> list[dict[str, Any]]:
    """
    Download a batch of images into memory as numpy arrays.
    Each entry: {"awb": str, "trip_id": str, "pod_link": str, "image": np.ndarray}
    """
    samples = []
    awb_col = "AWB" if "AWB" in batch.columns else "awb"
    trip_col = "Trip Id" if "Trip Id" in batch.columns else "trip_id"

    for _, row in batch.iterrows():
        url = row["pod_link"]
        awb = str(row.get(awb_col, ""))
        trip_id = str(row.get(trip_col, ""))

        try:
            resp = session.get(url, timeout=15)
            if resp.status_code != 200 or len(resp.content) < 500:
                continue
            img_array = np.frombuffer(resp.content, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            samples.append({
                "awb": awb,
                "trip_id": trip_id,
                "pod_link": url,
                "image": img_rgb,
            })
        except Exception as e:
            logger.warning("Download failed for %s: %s", url, e)
            continue

    return samples


# ---------------------------------------------------------------------------
# In-memory dataset and scoring
# ---------------------------------------------------------------------------


class InMemoryImageDataset(Dataset):
    """PyTorch dataset wrapping in-memory numpy images."""

    def __init__(self, images: list[np.ndarray], input_size: int = 224):
        self.images = images
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = self.images[idx]
        img = cv2.resize(img, (self.input_size, self.input_size))
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return tensor


def score_samples_in_memory(
    model: MultiHeadEfficientNet,
    device: torch.device,
    samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Run inference on in-memory samples. Returns scored results with
    per-attribute probabilities and composite pod_score.
    """
    if not samples:
        return []

    images = [s["image"] for s in samples]
    dataset = InMemoryImageDataset(images, input_size=INPUT_SIZE)
    loader = DataLoader(dataset, batch_size=INFERENCE_BATCH_SIZE, shuffle=False)

    all_logits = {name: [] for name in ATTRIBUTE_NAMES}

    with torch.no_grad():
        for batch_tensor in loader:
            batch_tensor = batch_tensor.to(device)
            logits = model(batch_tensor)
            for name in ATTRIBUTE_NAMES:
                all_logits[name].append(logits[name].cpu())

    cat_logits = {name: torch.cat(all_logits[name]) for name in ATTRIBUTE_NAMES}
    probs = {name: torch.sigmoid(cat_logits[name]) for name in ATTRIBUTE_NAMES}
    weights = ATTRIBUTE_WEIGHTS
    composite = sum(probs[ATTRIBUTE_NAMES[i]] * weights[i] for i in range(4))

    results = []
    for idx, sample in enumerate(samples):
        results.append({
            "awb": sample["awb"],
            "trip_id": sample["trip_id"],
            "pod_link": sample["pod_link"],
            "pod_score": round(composite[idx].item(), 6),
            "context_valid_prob": round(probs["context_valid"][idx].item(), 6),
            "package_visible_prob": round(probs["package_visible"][idx].item(), 6),
            "label_readable_prob": round(probs["label_readable"][idx].item(), 6),
            "image_clarity_prob": round(probs["image_clarity"][idx].item(), 6),
        })

    return results


# ---------------------------------------------------------------------------
# S3 state management (cross-invocation persistence)
# ---------------------------------------------------------------------------


def _s3_client():
    return boto3.client("s3")


def _run_key(run_id: str, name: str) -> str:
    return f"pod-pipeline-runs/{run_id}/{name}"


def store_expanded_links(run_id: str, df: pd.DataFrame) -> None:
    """Persist expanded links manifest to S3 (URLs + metadata only, no images)."""
    s3 = _s3_client()
    body = df.to_json(orient="records")
    s3.put_object(Bucket=S3_BUCKET, Key=_run_key(run_id, "expanded_links.json"), Body=body)
    logger.info("Stored %d expanded links manifest to S3 for run %s", len(df), run_id)


def load_expanded_links(run_id: str) -> pd.DataFrame:
    """Load expanded links manifest from S3."""
    s3 = _s3_client()
    obj = s3.get_object(Bucket=S3_BUCKET, Key=_run_key(run_id, "expanded_links.json"))
    data = json.loads(obj["Body"].read().decode("utf-8"))
    return pd.DataFrame(data)


def cleanup_s3_state(run_id: str) -> None:
    """Remove the links manifest from S3 after all batches are processed."""
    s3 = _s3_client()
    try:
        s3.delete_object(Bucket=S3_BUCKET, Key=_run_key(run_id, "expanded_links.json"))
    except Exception:
        pass
    logger.info("Cleaned up S3 manifest for run %s", run_id)


# ---------------------------------------------------------------------------
# Self-invocation
# ---------------------------------------------------------------------------


def invoke_self(event_payload: dict) -> None:
    """Asynchronously invoke this Lambda with updated state."""
    client = boto3.client("lambda")
    client.invoke(
        FunctionName=LAMBDA_FUNCTION_NAME,
        InvocationType="Event",
        Payload=json.dumps(event_payload).encode("utf-8"),
    )
    logger.info(
        "Self-invoked with i=%d, total_count=%d",
        event_payload.get("i", 0),
        event_payload.get("total_count", 0),
    )


# ---------------------------------------------------------------------------
# PostgreSQL flush
# ---------------------------------------------------------------------------


def get_db_connection():
    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        database=PG_DATABASE,
        user=PG_USER,
        password=PG_PASSWORD,
        connect_timeout=10,
    )


def write_to_postgres(conn, results: list[dict], run_date: str) -> None:
    """Bulk insert all scored results into pod_scores table."""
    if not results:
        logger.info("No results to write to Postgres")
        return

    insert_sql = """
        INSERT INTO pod_scores
            (awb, trip_id, pod_score, pod_link,
             context_valid_prob, package_visible_prob,
             label_readable_prob, image_clarity_prob, run_date)
        VALUES %s
    """
    tuples = [
        (
            r["awb"], r["trip_id"], r["pod_score"], r["pod_link"],
            r["context_valid_prob"], r["package_visible_prob"],
            r["label_readable_prob"], r["image_clarity_prob"],
            run_date,
        )
        for r in results
    ]

    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, insert_sql, tuples, page_size=200)
    conn.commit()
    logger.info("Flushed %d rows to Postgres for run_date=%s", len(tuples), run_date)


def _flush_results_to_postgres(results: list[dict], run_date: str) -> float:
    """Connect to Postgres and write all results. Returns elapsed seconds."""
    t0 = time.time()
    conn = get_db_connection()
    try:
        write_to_postgres(conn, results, run_date)
    finally:
        conn.close()
    return time.time() - t0


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def log_scoring_summary(results: list[dict]) -> None:
    if not results:
        return
    scores = [r["pod_score"] for r in results]
    flagged = sum(1 for s in scores if s < FLAG_THRESHOLD)
    logger.info(
        "Batch scoring summary: %d images, avg=%.3f, min=%.3f, max=%.3f, flagged=%d (<%s)",
        len(scores),
        np.mean(scores),
        np.min(scores),
        np.max(scores),
        flagged,
        FLAG_THRESHOLD,
    )


# ---------------------------------------------------------------------------
# Legacy helpers (kept for test compatibility)
# ---------------------------------------------------------------------------


def download_batch_to_dir(
    session: requests.Session, batch: pd.DataFrame, img_dir: str, start_idx: int = 0
) -> list[dict]:
    """Download images to disk. Legacy — used by tests only."""
    os.makedirs(img_dir, exist_ok=True)
    manifest = []
    awb_col = "AWB" if "AWB" in batch.columns else "awb"
    trip_col = "Trip Id" if "Trip Id" in batch.columns else "trip_id"

    for row_idx, (_, row) in enumerate(batch.iterrows()):
        url = row["pod_link"]
        try:
            resp = session.get(url, timeout=15)
            if resp.status_code != 200 or len(resp.content) < 500:
                continue
        except Exception:
            continue

        parsed = urlparse(url)
        ext = os.path.splitext(parsed.path)[1]
        if not ext or len(ext) > 5:
            ext = ".png"

        filename = f"pod_{start_idx + row_idx}{ext}"
        filepath = os.path.join(img_dir, filename)
        with open(filepath, "wb") as f:
            f.write(resp.content)

        manifest.append({
            "filename": filename,
            "awb": str(row.get(awb_col, "")),
            "trip_id": str(row.get(trip_col, "")),
            "pod_link": url,
        })

    return manifest


class TmpImageDataset(Dataset):
    """Legacy disk-based dataset. Kept for test compatibility."""

    def __init__(self, paths: list[str], input_size: int = 224):
        self.paths = paths
        self.input_size = input_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = cv2.imread(path)
        if img is None:
            tensor = torch.zeros(3, self.input_size, self.input_size)
            return tensor, path
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size, self.input_size))
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return tensor, path


def score_batch(model, device, paths: list[str]) -> pd.DataFrame:
    """Legacy disk-based scoring. Kept for test compatibility."""
    ds = TmpImageDataset(paths, input_size=INPUT_SIZE)
    loader = DataLoader(ds, batch_size=INFERENCE_BATCH_SIZE, shuffle=False)

    records = []
    with torch.no_grad():
        for batch_tensor, batch_paths in loader:
            batch_tensor = batch_tensor.to(device)
            logits = model(batch_tensor)
            probs = {k: torch.sigmoid(v).cpu() for k, v in logits.items()}
            weights = ATTRIBUTE_WEIGHTS
            composite = sum(
                probs[ATTRIBUTE_NAMES[j]] * weights[j] for j in range(4)
            )
            for k in range(batch_tensor.shape[0]):
                records.append({
                    "image_path": batch_paths[k],
                    "pod_score": composite[k].item(),
                    "context_valid_prob": probs["context_valid"][k].item(),
                    "package_visible_prob": probs["package_visible"][k].item(),
                    "label_readable_prob": probs["label_readable"][k].item(),
                    "image_clarity_prob": probs["image_clarity"][k].item(),
                })

    return pd.DataFrame(records)


def process_one_batch(model, device, manifest, img_dir, batch_idx, threshold, run_date):
    """Legacy per-batch process. Kept for test compatibility."""
    paths = [os.path.join(img_dir, m["filename"]) for m in manifest]
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        return {"scored": 0, "flagged": 0, "written": 0}

    df = score_batch(model, device, paths)
    flagged = int((df["pod_score"] < threshold).sum())
    return {"scored": len(df), "flagged": flagged, "written": len(df)}


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------

BATCH_SIZE = FETCH_BATCH_SIZE  # alias for test compatibility


def handler(event: dict, context: Any) -> dict:
    """
    Self-invoking Lambda handler for POD scoring pipeline.

    Event schema:
      i           : int  — cumulative images processed so far (0 on first call)
      total_count : int  — total images to process (set after expand_pod_links)
      run_id      : str  — unique identifier for this pipeline run
    """
    t_start = time.time()
    run_date = date.today().isoformat()

    i = event.get("i", 0)
    total_count = event.get("total_count")
    run_id = event.get("run_id")

    # ------------------------------------------------------------------
    # Validate environment
    # ------------------------------------------------------------------
    if not METABASE_URL or not METABASE_API_KEY:
        logger.error("Missing METABASE_URL or METABASE_API_KEY")
        return {"statusCode": 500, "body": json.dumps({"error": "Missing Metabase config"})}

    if not S3_BUCKET:
        logger.error("Missing S3_STATE_BUCKET environment variable")
        return {"statusCode": 500, "body": json.dumps({"error": "Missing S3_STATE_BUCKET"})}

    # ------------------------------------------------------------------
    # PHASE 1: First invocation (i=0) — fetch and expand
    # ------------------------------------------------------------------
    if i == 0:
        run_id = f"{run_date}_{uuid.uuid4().hex[:8]}"
        logger.info("=== Pipeline START === run_id=%s", run_id)

        session = build_session()

        # Fetch from Metabase
        logger.info("Fetching POD data from Metabase card %d", METABASE_CARD_ID)
        raw_df = fetch_pod_data(session)
        if raw_df.empty:
            logger.info("Metabase returned no data")
            return {
                "statusCode": 200,
                "body": json.dumps({"message": "No data", "run_id": run_id}),
            }

        # Expand POD links
        expanded = expand_pod_links(raw_df)
        total_count = len(expanded)
        logger.info("Expanded to %d POD image links", total_count)

        if total_count == 0:
            return {
                "statusCode": 200,
                "body": json.dumps({"message": "No POD links", "run_id": run_id}),
            }

        # Store links manifest in S3 so subsequent invocations know what to download
        store_expanded_links(run_id, expanded)

    # ------------------------------------------------------------------
    # PHASE 2: Download batch to memory → Score → Write to Postgres
    # ------------------------------------------------------------------
    logger.info("Processing batch: i=%d, total_count=%d, run_id=%s", i, total_count, run_id)

    expanded = load_expanded_links(run_id)

    batch_start = i
    batch_end = min(i + FETCH_BATCH_SIZE, total_count)
    batch_df = expanded.iloc[batch_start:batch_end]
    scored_count = 0

    if not batch_df.empty:
        # Download images directly into memory (labeled with trip_id + awb)
        session = build_session()
        samples = download_batch_to_memory(session, batch_df)
        logger.info(
            "Downloaded %d/%d images to memory (batch %d-%d)",
            len(samples), len(batch_df), batch_start, batch_end,
        )

        # Score with EfficientNet — all in memory, no disk
        if samples:
            model, device = get_model()
            batch_results = score_samples_in_memory(model, device, samples)
            log_scoring_summary(batch_results)
            scored_count = len(batch_results)

            # Delete images from memory
            del samples

            # Write this batch's results directly to Postgres
            _flush_results_to_postgres(batch_results, run_date)
            logger.info("Wrote %d scored rows to Postgres", scored_count)

    # Update cumulative counter
    new_i = batch_end

    # ------------------------------------------------------------------
    # PHASE 3: Decide — invoke self or done
    # ------------------------------------------------------------------
    if new_i >= total_count:
        # All images processed — clean up S3 manifest
        cleanup_s3_state(run_id)

        duration = time.time() - t_start
        summary = {
            "run_id": run_id,
            "run_date": run_date,
            "total_images": total_count,
            "final_i": new_i,
            "scored_this_invocation": scored_count,
            "invocation_duration_s": round(duration, 3),
            "status": "complete",
        }
        logger.info("=== Pipeline COMPLETE === %s", json.dumps(summary))
        return {"statusCode": 200, "body": json.dumps(summary)}

    else:
        # More images remain — invoke self with updated i
        next_event = {
            "i": new_i,
            "total_count": total_count,
            "run_id": run_id,
        }
        invoke_self(next_event)

        duration = time.time() - t_start
        summary = {
            "run_id": run_id,
            "run_date": run_date,
            "total_images": total_count,
            "processed_so_far": new_i,
            "scored_this_invocation": scored_count,
            "invocation_duration_s": round(duration, 3),
            "status": "continuing",
            "next_i": new_i,
        }
        logger.info("Batch done, invoking next: %s", json.dumps(summary))
        return {"statusCode": 200, "body": json.dumps(summary)}
