"""
POD Pipeline Lambda (merged Fetcher + Scorer).

Triggered by EventBridge Scheduler. Fetches POD links from Metabase, downloads
images to /tmp in batches (bounded by FETCH_BATCH_SIZE and ephemeral storage),
runs EfficientNet inference, writes scores to PostgreSQL, then cleans staging.
"""

import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone

import cv2
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import requests
from requests.adapters import HTTPAdapter
from torch.utils.data import DataLoader, Dataset
from urllib3.util.retry import Retry
import torch

from src.model import ATTRIBUTE_NAMES, ATTRIBUTE_WEIGHTS, MultiHeadEfficientNet

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# --- Metabase ---
METABASE_URL = os.environ.get("METABASE_URL", "").rstrip("/")
METABASE_API_KEY = os.environ.get("METABASE_API_KEY", "")
METABASE_CARD_ID = int(os.environ.get("METABASE_CARD_ID", "10989"))

# --- Pipeline ---
FETCH_BATCH_SIZE = int(os.environ.get("FETCH_BATCH_SIZE", "500"))
FLAG_THRESHOLD = float(os.environ.get("FLAG_THRESHOLD", "0.7"))
STAGING_ROOT = os.environ.get("STAGING_TMP_PATH", "/tmp")

# --- PostgreSQL ---
PG_HOST = os.environ.get("PG_HOST", "localhost")
PG_PORT = int(os.environ.get("PG_PORT", "5432"))
PG_DATABASE = os.environ.get("PG_DATABASE", "pod_classifier")
PG_USER = os.environ.get("PG_USER", "postgres")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "")

MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/model/best.pt")
INPUT_SIZE = int(os.environ.get("INPUT_SIZE", "224"))
BATCH_SIZE = int(os.environ.get("INFERENCE_BATCH_SIZE", "64"))

_model = None
_device = None


def build_session(retries: int = 3, backoff: float = 0.5) -> requests.Session:
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_pod_data(session: requests.Session) -> pd.DataFrame:
    """Fetch AWB + POD data from Metabase card."""
    url = f"{METABASE_URL}/api/card/{METABASE_CARD_ID}/query/json"
    headers = {
        "X-API-Key": METABASE_API_KEY,
        "Content-Type": "application/json",
    }
    logger.info("Querying Metabase card %s...", METABASE_CARD_ID)
    resp = session.post(url, headers=headers, json={"parameters": []}, timeout=120)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    logger.info("Fetched %s rows from Metabase", len(df))
    return df


def expand_pod_links(df: pd.DataFrame) -> pd.DataFrame:
    """Expand comma-separated POD links into individual rows."""
    pod_col = "POD" if "POD" in df.columns else "pod"
    if pod_col not in df.columns:
        raise ValueError(f"POD column not found. Available: {list(df.columns)}")

    rows = []
    for _, row in df.iterrows():
        links = str(row[pod_col]) if pd.notna(row[pod_col]) else ""
        for link in links.split(","):
            link = link.strip()
            if link.startswith("http"):
                new_row = row.to_dict()
                new_row["pod_link"] = link
                rows.append(new_row)

    expanded = pd.DataFrame(rows)
    logger.info("Expanded to %s individual POD links", len(expanded))
    return expanded


def download_batch_to_dir(
    session: requests.Session,
    batch_df: pd.DataFrame,
    image_dir: str,
    start_idx: int,
) -> list:
    """Download a batch of images to image_dir. Returns manifest entry dicts."""
    os.makedirs(image_dir, exist_ok=True)
    manifest_entries = []

    for i, (_, row) in enumerate(batch_df.iterrows()):
        url = row["pod_link"]
        idx = start_idx + i
        ext = url.split(".")[-1].split("?")[0]
        if len(ext) > 5:
            ext = "png"
        filename = f"pod_{idx}.{ext}"
        filepath = os.path.join(image_dir, filename)

        awb_col = "AWB" if "AWB" in row.index else "awb"
        trip_col = "Trip Id" if "Trip Id" in row.index else "trip_id"

        try:
            resp = session.get(url, timeout=15)
            if resp.status_code == 200 and len(resp.content) > 500:
                with open(filepath, "wb") as f:
                    f.write(resp.content)
                manifest_entries.append({
                    "filename": filename,
                    "awb": str(row.get(awb_col, "")),
                    "trip_id": str(row.get(trip_col, "")),
                    "pod_link": url,
                })
        except Exception as e:
            logger.warning("Failed to download %s: %s", url, e)

    return manifest_entries


class TmpImageDataset(Dataset):
    """Reads images from local paths for inference."""

    def __init__(self, image_paths: list, input_size: int = 224):
        self.image_paths = image_paths
        self.input_size = input_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = cv2.imread(path)
        if image is None:
            image = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.input_size, self.input_size))

        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        image = np.transpose(image, (2, 0, 1))
        return torch.from_numpy(image), path


def get_model():
    """Load model (cached across warm starts)."""
    global _model, _device
    if _model is not None:
        return _model, _device

    _device = torch.device("cpu")
    _model = MultiHeadEfficientNet(num_attributes=4, pretrained=False)
    checkpoint = torch.load(MODEL_PATH, map_location=_device, weights_only=False)
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.eval()
    _model.to(_device)
    logger.info("Model loaded from %s", MODEL_PATH)
    return _model, _device


def get_db_connection():
    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        database=PG_DATABASE,
        user=PG_USER,
        password=PG_PASSWORD,
    )


@torch.no_grad()
def score_batch(model, device, image_paths: list) -> pd.DataFrame:
    dataset = TmpImageDataset(image_paths, input_size=INPUT_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    weights = ATTRIBUTE_WEIGHTS.to(device)
    results = []

    for images, paths in loader:
        images = images.to(device)
        logits = model(images)
        probs = {k: torch.sigmoid(v) for k, v in logits.items()}
        scores = sum(probs[ATTRIBUTE_NAMES[i]] * weights[i] for i in range(4))

        for idx in range(len(paths)):
            row = {
                "image_path": paths[idx],
                "pod_score": scores[idx].item(),
                "context_valid_prob": probs["context_valid"][idx].item(),
                "package_visible_prob": probs["package_visible"][idx].item(),
                "label_readable_prob": probs["label_readable"][idx].item(),
                "image_clarity_prob": probs["image_clarity"][idx].item(),
            }
            results.append(row)

    return pd.DataFrame(results)


def write_to_postgres(conn, results: list, run_date: str):
    insert_sql = """
        INSERT INTO pod_scores (
            awb, trip_id, pod_score, pod_link,
            context_valid_prob, package_visible_prob,
            label_readable_prob, image_clarity_prob,
            run_date, scored_at
        ) VALUES %s
    """
    scored_at = datetime.now(timezone.utc)
    values = [
        (
            r["awb"], r["trip_id"], r["pod_score"], r["pod_link"],
            r["context_valid_prob"], r["package_visible_prob"],
            r["label_readable_prob"], r["image_clarity_prob"],
            run_date, scored_at,
        )
        for r in results
    ]
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, insert_sql, values, page_size=200)
    conn.commit()


def log_scoring_summary(results: list, batch_id: int, threshold: float, elapsed: float):
    scores = [r["pod_score"] for r in results]
    scores_arr = np.array(scores)

    n_pass = sum(1 for s in scores if s >= threshold)
    n_flag = len(scores) - n_pass
    pct_flag = (n_flag / len(scores) * 100) if scores else 0

    ctx_avg = np.mean([r["context_valid_prob"] for r in results])
    pkg_avg = np.mean([r["package_visible_prob"] for r in results])
    lbl_avg = np.mean([r["label_readable_prob"] for r in results])
    clr_avg = np.mean([r["image_clarity_prob"] for r in results])

    worst = sorted(results, key=lambda r: r["pod_score"])[:5]
    worst_str = ", ".join(f"{r['awb']} ({r['pod_score']:.4f})" for r in worst)

    logger.info("[BATCH %s] Scored %s images in %.1fs", batch_id, len(results), elapsed)
    logger.info(
        "  Scores: mean=%.4f | median=%.4f | min=%.4f | max=%.4f",
        scores_arr.mean(), np.median(scores_arr), scores_arr.min(), scores_arr.max(),
    )
    logger.info(
        "  PASS: %s (%.1f%%) | FLAG: %s (%.1f%%) | threshold=%s",
        n_pass, 100 - pct_flag, n_flag, pct_flag, threshold,
    )
    logger.info(
        "  Attr avg: context=%.2f | package=%.2f | label=%.2f | clarity=%.2f",
        ctx_avg, pkg_avg, lbl_avg, clr_avg,
    )
    logger.info("  Worst: %s", worst_str)


def _cleanup_batch_dir(batch_dir: str):
    """Remove staging directory after batch completes."""
    try:
        shutil.rmtree(batch_dir, ignore_errors=True)
    except OSError as e:
        logger.warning("Cleanup failed for %s: %s", batch_dir, e)


def process_one_batch(
    model,
    device,
    manifest_entries: list,
    image_dir: str,
    batch_id: int,
    threshold: float,
    run_date: str,
):
    """Score images referenced by manifest_entries under image_dir; persist and cleanup."""
    image_paths = []
    metadata = []
    for entry in manifest_entries:
        filepath = os.path.join(image_dir, entry["filename"])
        if os.path.exists(filepath):
            image_paths.append(filepath)
            metadata.append(entry)

    if not image_paths:
        logger.warning("Batch %s: no valid images on disk, skipping inference", batch_id)
        _cleanup_batch_dir(image_dir)
        return 0

    logger.info("Batch %s: scoring %s images", batch_id, len(image_paths))

    inference_start = time.time()
    scores_df = score_batch(model, device, image_paths)
    inference_elapsed = time.time() - inference_start

    results = []
    for (_, row), meta in zip(scores_df.iterrows(), metadata):
        results.append({
            "awb": meta["awb"],
            "trip_id": meta["trip_id"],
            "pod_score": row["pod_score"],
            "pod_link": meta["pod_link"],
            "context_valid_prob": row["context_valid_prob"],
            "package_visible_prob": row["package_visible_prob"],
            "label_readable_prob": row["label_readable_prob"],
            "image_clarity_prob": row["image_clarity_prob"],
        })

    log_scoring_summary(results, batch_id, threshold, inference_elapsed)

    conn = get_db_connection()
    try:
        write_to_postgres(conn, results, run_date)
    finally:
        conn.close()

    logger.info("Batch %s: written %s rows to PostgreSQL", batch_id, len(results))

    _cleanup_batch_dir(image_dir)

    return len(results)


def handler(event, context):
    start_time = time.time()
    req_id = getattr(context, "aws_request_id", "local")
    run_date = datetime.utcnow().strftime("%Y-%m-%d")

    logger.info("=== POD Pipeline Lambda (%s) ===", req_id)
    logger.info(
        "Fetch batch=%s | inference_bs=%s | threshold=%s | staging=%s",
        FETCH_BATCH_SIZE, BATCH_SIZE, FLAG_THRESHOLD, STAGING_ROOT,
    )

    if not METABASE_URL or not METABASE_API_KEY:
        logger.error("METABASE_URL and METABASE_API_KEY must be set")
        return {"statusCode": 500, "body": "Missing Metabase credentials"}

    staging_base = os.path.join(STAGING_ROOT, "pod_scoring", req_id)

    session = build_session()
    df = fetch_pod_data(session)
    if df.empty:
        logger.warning("No data returned from Metabase")
        return {"statusCode": 200, "body": json.dumps({"message": "No data", "run_date": run_date})}

    expanded = expand_pod_links(df)
    if expanded.empty:
        logger.warning("No valid POD links found")
        return {"statusCode": 200, "body": json.dumps({"message": "No POD links", "run_date": run_date})}

    total_images = len(expanded)
    n_batches = (total_images + FETCH_BATCH_SIZE - 1) // FETCH_BATCH_SIZE
    total_downloaded = 0
    total_scored = 0

    model, device = get_model()

    for batch_id in range(n_batches):
        batch_start = batch_id * FETCH_BATCH_SIZE
        batch_end = min(batch_start + FETCH_BATCH_SIZE, total_images)
        batch_df = expanded.iloc[batch_start:batch_end]
        image_dir = os.path.join(staging_base, run_date, f"batch_{batch_id}")

        logger.info("Batch %s: downloading %s images...", batch_id, len(batch_df))
        manifest_entries = download_batch_to_dir(session, batch_df, image_dir, start_idx=batch_start)
        total_downloaded += len(manifest_entries)

        if not manifest_entries:
            logger.warning("Batch %s: no downloads, skipping", batch_id)
            continue

        scored = process_one_batch(
            model, device, manifest_entries, image_dir, batch_id, FLAG_THRESHOLD, run_date,
        )
        total_scored += scored

    # Remove empty parent dirs left under staging
    try:
        shutil.rmtree(staging_base, ignore_errors=True)
    except OSError:
        pass

    elapsed = time.time() - start_time
    logger.info(
        "=== Pipeline complete | POD links=%s | downloaded=%s | scored=%s | %.1fs ===",
        total_images, total_downloaded, total_scored, elapsed,
    )

    return {
        "statusCode": 200,
        "body": json.dumps({
            "run_date": run_date,
            "total_pod_links": total_images,
            "total_downloaded": total_downloaded,
            "total_scored_rows": total_scored,
            "batches_processed": n_batches,
            "duration_seconds": round(elapsed, 1),
        }),
    }
