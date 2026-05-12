"""
Lambda 2 — Scorer.

Triggered asynchronously by the Fetcher Lambda.
Reads images from shared EFS, runs EfficientNet inference,
writes per-image results to PostgreSQL, and logs scoring summary.
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
import torch
from torch.utils.data import DataLoader, Dataset

from src.model import ATTRIBUTE_NAMES, ATTRIBUTE_WEIGHTS, MultiHeadEfficientNet

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# --- Configuration via environment ---
PG_HOST = os.environ.get("PG_HOST", "localhost")
PG_PORT = int(os.environ.get("PG_PORT", "5432"))
PG_DATABASE = os.environ.get("PG_DATABASE", "pod_classifier")
PG_USER = os.environ.get("PG_USER", "postgres")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "")
MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/model/best.pt")
INPUT_SIZE = int(os.environ.get("INPUT_SIZE", "224"))
BATCH_SIZE = int(os.environ.get("INFERENCE_BATCH_SIZE", "64"))

# Global model cache (persists across warm invocations)
_model = None
_device = None


class EFSImageDataset(Dataset):
    """Reads images from EFS paths for inference."""

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
    logger.info(f"Model loaded from {MODEL_PATH}")
    return _model, _device


def get_db_connection():
    """Create PostgreSQL connection."""
    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        database=PG_DATABASE,
        user=PG_USER,
        password=PG_PASSWORD,
    )


@torch.no_grad()
def score_batch(model, device, image_paths: list) -> pd.DataFrame:
    """Score a list of images and return results DataFrame."""
    dataset = EFSImageDataset(image_paths, input_size=INPUT_SIZE)
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
    """Batch insert scored results into PostgreSQL."""
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
    """Log batch scoring summary to CloudWatch."""
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

    logger.info(f"[BATCH {batch_id}] Scored {len(results)} images in {elapsed:.1f}s")
    logger.info(
        f"  Scores: mean={scores_arr.mean():.4f} | "
        f"median={np.median(scores_arr):.4f} | "
        f"min={scores_arr.min():.4f} | max={scores_arr.max():.4f}"
    )
    logger.info(
        f"  PASS: {n_pass} ({100 - pct_flag:.1f}%) | "
        f"FLAG: {n_flag} ({pct_flag:.1f}%) | threshold={threshold}"
    )
    logger.info(
        f"  Attr avg: context={ctx_avg:.2f} | package={pkg_avg:.2f} | "
        f"label={lbl_avg:.2f} | clarity={clr_avg:.2f}"
    )
    logger.info(f"  Worst: {worst_str}")


def handler(event, context):
    """Lambda entry point."""
    start_time = time.time()

    efs_image_dir = event["efs_image_dir"]
    manifest_path = event["manifest_path"]
    threshold = float(event.get("threshold", 0.7))
    run_date = event["run_date"]
    batch_id = int(event.get("batch_id", 0))

    logger.info(f"=== POD Scorer Lambda | Batch {batch_id} | {run_date} ===")
    logger.info(f"Image dir: {efs_image_dir}")
    logger.info(f"Threshold: {threshold}")

    # 1. Read manifest
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    logger.info(f"Manifest loaded: {len(manifest)} entries")

    # 2. Build image paths and metadata
    image_paths = []
    metadata = []
    for entry in manifest:
        filepath = os.path.join(efs_image_dir, entry["filename"])
        if os.path.exists(filepath):
            image_paths.append(filepath)
            metadata.append(entry)

    if not image_paths:
        logger.warning("No valid images found on EFS. Exiting.")
        return {"statusCode": 200, "body": "No images to score"}

    logger.info(f"Found {len(image_paths)} images on EFS")

    # 3. Load model
    model, device = get_model()

    # 4. Score
    inference_start = time.time()
    scores_df = score_batch(model, device, image_paths)
    inference_elapsed = time.time() - inference_start

    # 5. Merge with metadata
    results = []
    for i, row in scores_df.iterrows():
        meta = metadata[i]
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

    # 6. Log summary
    log_scoring_summary(results, batch_id, threshold, inference_elapsed)

    # 7. Write to PostgreSQL
    try:
        conn = get_db_connection()
        write_to_postgres(conn, results, run_date)
        conn.close()
        logger.info(f"Written {len(results)} rows to PostgreSQL")
    except Exception as e:
        logger.error(f"PostgreSQL write failed: {e}")
        raise

    # 8. Cleanup batch images from EFS (optional)
    for path in image_paths:
        try:
            os.remove(path)
        except OSError:
            pass
    try:
        os.remove(manifest_path)
    except OSError:
        pass

    elapsed = time.time() - start_time
    logger.info(f"=== Scorer Complete | Batch {batch_id} | {elapsed:.1f}s total ===")

    return {
        "statusCode": 200,
        "body": json.dumps({
            "batch_id": batch_id,
            "images_scored": len(results),
            "inference_time_seconds": round(inference_elapsed, 1),
            "total_time_seconds": round(elapsed, 1),
        }),
    }
