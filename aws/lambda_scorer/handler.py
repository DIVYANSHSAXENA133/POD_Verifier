"""
POD Pipeline Lambda — single invocation, in-memory batch loop.

EventBridge invokes with {"i": 0} (i is accepted but the full run happens in one call).

1. Fetch Metabase once → expand → total_images = len(expanded) (kept in memory).
2. Loop while i < total_images:
     download up to FETCH_BATCH_SIZE images into RAM (decoded RGB arrays),
     score with EfficientNet directly from memory,
     append results to an in-memory list, release batch memory.
3. When i == total_images: write all results to PostgreSQL once.

No S3 for shipments or images. /tmp is not used for image staging.
"""

import json
import logging
import os
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

METABASE_URL = os.environ.get("METABASE_URL", "").rstrip("/")
METABASE_API_KEY = os.environ.get("METABASE_API_KEY", "")
METABASE_CARD_ID = int(os.environ.get("METABASE_CARD_ID", "10989"))

FETCH_BATCH_SIZE = int(os.environ.get("FETCH_BATCH_SIZE", "500"))
FLAG_THRESHOLD = float(os.environ.get("FLAG_THRESHOLD", "0.7"))

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


def download_batch_to_memory(session: requests.Session, batch_df: pd.DataFrame) -> list:
    """
    Download a batch of POD images into RAM as RGB uint8 arrays.
    Returns list of dicts: awb, trip_id, pod_link, image (H,W,3) or None if failed.
    """
    samples = []
    for _, row in batch_df.iterrows():
        url = row["pod_link"]
        awb_col = "AWB" if "AWB" in row.index else "awb"
        trip_col = "Trip Id" if "Trip Id" in row.index else "trip_id"
        meta = {
            "awb": str(row.get(awb_col, "")),
            "trip_id": str(row.get(trip_col, "")),
            "pod_link": url,
            "image": None,
        }
        try:
            resp = session.get(url, timeout=15)
            if resp.status_code == 200 and len(resp.content) > 500:
                buf = np.frombuffer(resp.content, dtype=np.uint8)
                bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if bgr is not None:
                    meta["image"] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception as ex:
            logger.warning("Failed to download %s: %s", url, ex)
        samples.append(meta)
    return samples


def download_batch_to_dir(
    session: requests.Session,
    batch_df: pd.DataFrame,
    image_dir: str,
    start_idx: int,
) -> list:
    """Disk-based download (kept for unit tests)."""
    os.makedirs(image_dir, exist_ok=True)
    manifest_entries = []
    for j, (_, row) in enumerate(batch_df.iterrows()):
        url = row["pod_link"]
        idx = start_idx + j
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
        except Exception as ex:
            logger.warning("Failed to download %s: %s", url, ex)
    return manifest_entries


class InMemoryImageDataset(Dataset):
    """EfficientNet input tensors built from in-memory RGB images."""

    def __init__(self, images_rgb: list, input_size: int = 224):
        self.images_rgb = images_rgb
        self.input_size = input_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.images_rgb)

    def __getitem__(self, idx):
        image = self.images_rgb[idx]
        if image is None:
            image = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        else:
            image = cv2.resize(image, (self.input_size, self.input_size))

        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        image = np.transpose(image, (2, 0, 1))
        return torch.from_numpy(image), idx


class TmpImageDataset(Dataset):
    """Reads images from local paths (unit tests)."""

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
    """Score images from disk paths (unit tests)."""
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


@torch.no_grad()
def score_samples_in_memory(model, device, samples: list) -> list:
    """Score in-memory RGB images; returns list of result dicts (no Postgres)."""
    valid = [(i, s) for i, s in enumerate(samples) if s.get("image") is not None]
    if not valid:
        return []

    indices = [i for i, _ in valid]
    images_rgb = [samples[i]["image"] for i in indices]

    dataset = InMemoryImageDataset(images_rgb, input_size=INPUT_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    weights = ATTRIBUTE_WEIGHTS.to(device)

    scored_rows = []
    for images, batch_idx in loader:
        images = images.to(device)
        logits = model(images)
        probs = {k: torch.sigmoid(v) for k, v in logits.items()}
        scores = sum(probs[ATTRIBUTE_NAMES[i]] * weights[i] for i in range(4))

        for j in range(len(batch_idx)):
            src = samples[indices[batch_idx[j].item()]]
            scored_rows.append({
                "awb": src["awb"],
                "trip_id": src["trip_id"],
                "pod_score": scores[j].item(),
                "pod_link": src["pod_link"],
                "context_valid_prob": probs["context_valid"][j].item(),
                "package_visible_prob": probs["package_visible"][j].item(),
                "label_readable_prob": probs["label_readable"][j].item(),
                "image_clarity_prob": probs["image_clarity"][j].item(),
            })
    return scored_rows


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
    if not results:
        return
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

    logger.info("[BATCH i=%s] Scored %s images in %.1fs", batch_id, len(results), elapsed)
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


def _normalize_event(raw):
    if isinstance(raw, str):
        raw = json.loads(raw)
    if raw is None or not isinstance(raw, dict):
        return {}
    if "body" in raw and isinstance(raw["body"], str):
        try:
            inner = json.loads(raw["body"])
            if isinstance(inner, dict):
                return inner
        except json.JSONDecodeError:
            pass
    return raw


def _parse_event(raw):
    evt = _normalize_event(raw)
    i = int(evt.get("i", evt.get("e", 0)))
    return evt, i


def _flush_results_to_postgres(results: list, run_date: str) -> float:
    if not results:
        logger.warning("No scored rows to write to PostgreSQL")
        return 0.0
    pg_start = time.perf_counter()
    conn = get_db_connection()
    try:
        write_to_postgres(conn, results, run_date)
    finally:
        conn.close()
    elapsed = time.perf_counter() - pg_start
    logger.info("Wrote %s rows to PostgreSQL (run_date=%s)", len(results), run_date)
    return elapsed


def process_one_batch(
    model,
    device,
    manifest_entries: list,
    image_dir: str,
    i_offset: int,
    threshold: float,
    run_date: str,
):
    """Disk-based batch (unit tests)."""
    import os
    import shutil

    def _cleanup(batch_dir: str):
        shutil.rmtree(batch_dir, ignore_errors=True)

    image_paths = []
    metadata = []
    for entry in manifest_entries:
        filepath = os.path.join(image_dir, entry["filename"])
        if os.path.exists(filepath):
            image_paths.append(filepath)
            metadata.append(entry)

    if not image_paths:
        _cleanup(image_dir)
        return {"scored": 0, "inference_s": 0.0, "postgres_s": 0.0, "cleanup_s": 0.0}

    t0 = time.perf_counter()
    scores_df = score_batch(model, device, image_paths)
    inference_s = time.perf_counter() - t0

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

    log_scoring_summary(results, i_offset, threshold, inference_s)
    pg_s = _flush_results_to_postgres(results, run_date)
    _cleanup(image_dir)
    return {"scored": len(results), "inference_s": inference_s, "postgres_s": pg_s, "cleanup_s": 0.0}


def handler(event, context):
    start = time.perf_counter()
    req_id = getattr(context, "aws_request_id", "local")
    timing: dict = {}

    evt, start_i = _parse_event(event)
    logger.info("=== POD Pipeline (in-memory) | request=%s ===", req_id)

    if start_i != 0:
        logger.warning(
            "Ignoring start_i=%s — full pipeline runs in a single invocation from i=0",
            start_i,
        )

    if not METABASE_URL or not METABASE_API_KEY:
        logger.error("METABASE_URL and METABASE_API_KEY must be set")
        return {"statusCode": 500, "body": "Missing Metabase credentials"}

    logger.info(
        "FETCH_BATCH_SIZE=%s | inference_bs=%s | threshold=%s",
        FETCH_BATCH_SIZE, BATCH_SIZE, FLAG_THRESHOLD,
    )

    session = build_session()
    run_date = datetime.utcnow().strftime("%Y-%m-%d")

    t0 = time.perf_counter()
    df = fetch_pod_data(session)
    timing["metabase_fetch_s"] = round(time.perf_counter() - t0, 3)

    if df.empty:
        return {"statusCode": 200, "body": json.dumps({"message": "No data"})}

    t0 = time.perf_counter()
    expanded = expand_pod_links(df)
    timing["expand_pod_links_s"] = round(time.perf_counter() - t0, 3)

    total_images = len(expanded)
    if total_images == 0:
        return {"statusCode": 200, "body": json.dumps({"message": "No POD links"})}

    logger.info("total_images=%s — processing in memory with batch_size=%s", total_images, FETCH_BATCH_SIZE)

    t0 = time.perf_counter()
    model, device = get_model()
    timing["model_load_s"] = round(time.perf_counter() - t0, 3)

    all_results: list = []
    i = 0
    batch_count = 0
    timing["download_images_s"] = 0.0
    timing["efficientnet_inference_s"] = 0.0
    total_downloaded = 0

    while i < total_images:
        end = min(i + FETCH_BATCH_SIZE, total_images)
        batch_df = expanded.iloc[i:end]
        batch_count += 1

        logger.info("Iteration %s: rows [%s,%s) of %s", batch_count, i, end, total_images)

        t_dl = time.perf_counter()
        samples = download_batch_to_memory(session, batch_df)
        timing["download_images_s"] += time.perf_counter() - t_dl
        total_downloaded += sum(1 for s in samples if s.get("image") is not None)

        t_inf = time.perf_counter()
        chunk_results = score_samples_in_memory(model, device, samples)
        inf_elapsed = time.perf_counter() - t_inf
        timing["efficientnet_inference_s"] += inf_elapsed

        if chunk_results:
            log_scoring_summary(chunk_results, i, FLAG_THRESHOLD, inf_elapsed)
            all_results.extend(chunk_results)

        # Release batch image buffers before next iteration
        del samples
        i = end

    timing["download_images_s"] = round(timing["download_images_s"], 3)
    timing["efficientnet_inference_s"] = round(timing["efficientnet_inference_s"], 3)

    timing["postgres_write_s"] = round(_flush_results_to_postgres(all_results, run_date), 3)

    wall = time.perf_counter() - start
    timing["wall_clock_total_s"] = round(wall, 3)

    logger.info(
        "[timing] total_images=%s batches=%s downloaded=%s scored=%s wall=%.3fs",
        total_images, batch_count, total_downloaded, len(all_results), wall,
    )

    body = {
        "run_date": run_date,
        "total_images": total_images,
        "total_downloaded": total_downloaded,
        "total_scored_rows": len(all_results),
        "batches_processed": batch_count,
        "final_i": total_images,
        "postgres_flushed": True,
        "duration_seconds": round(wall, 1),
        "phase_timings_seconds": timing,
    }
    return {"statusCode": 200, "body": json.dumps(body, default=str)}
