"""
Lambda 1 — Fetcher.

Triggered by EventBridge Scheduler at 23:55 daily.
Fetches POD links from Metabase, downloads images to shared EFS,
and invokes the Scorer Lambda asynchronously per batch.
"""

import json
import logging
import os
import time
from datetime import datetime

import boto3
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# --- Configurable via environment variables (nothing hardcoded) ---
METABASE_URL = os.environ.get("METABASE_URL", "").rstrip("/")
METABASE_API_KEY = os.environ.get("METABASE_API_KEY", "")
METABASE_CARD_ID = int(os.environ.get("METABASE_CARD_ID", "10989"))
SCORER_LAMBDA_ARN = os.environ.get("SCORER_LAMBDA_ARN", "")
FETCH_BATCH_SIZE = int(os.environ.get("FETCH_BATCH_SIZE", "500"))
FLAG_THRESHOLD = float(os.environ.get("FLAG_THRESHOLD", "0.7"))
EFS_MOUNT_PATH = os.environ.get("EFS_MOUNT_PATH", "/mnt/efs")


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

    logger.info(f"Querying Metabase card {METABASE_CARD_ID}...")
    resp = session.post(url, headers=headers, json={"parameters": []}, timeout=120)
    resp.raise_for_status()

    data = resp.json()
    df = pd.DataFrame(data)
    logger.info(f"Fetched {len(df)} rows from Metabase")
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
    logger.info(f"Expanded to {len(expanded)} individual POD links")
    return expanded


def download_batch_to_efs(
    session: requests.Session,
    batch_df: pd.DataFrame,
    image_dir: str,
    start_idx: int,
) -> list:
    """Download a batch of images to EFS. Returns list of manifest entries."""
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
            logger.warning(f"Failed to download {url}: {e}")

    return manifest_entries


def invoke_scorer(lambda_client, payload: dict):
    """Invoke the Scorer Lambda asynchronously."""
    lambda_client.invoke(
        FunctionName=SCORER_LAMBDA_ARN,
        InvocationType="Event",
        Payload=json.dumps(payload).encode(),
    )


def handler(event, context):
    """Lambda entry point."""
    start_time = time.time()
    run_date = datetime.utcnow().strftime("%Y-%m-%d")

    logger.info(f"=== POD Fetcher Lambda ===")
    logger.info(f"Run date: {run_date}")
    logger.info(f"Batch size: {FETCH_BATCH_SIZE} | Threshold: {FLAG_THRESHOLD}")

    if not METABASE_URL or not METABASE_API_KEY:
        logger.error("METABASE_URL and METABASE_API_KEY must be set")
        return {"statusCode": 500, "body": "Missing Metabase credentials"}

    session = build_session()

    # 1. Fetch data from Metabase
    df = fetch_pod_data(session)
    if df.empty:
        logger.warning("No data returned from Metabase")
        return {"statusCode": 200, "body": "No data to process"}

    # 2. Expand POD links
    expanded = expand_pod_links(df)
    if expanded.empty:
        logger.warning("No valid POD links found")
        return {"statusCode": 200, "body": "No POD links to process"}

    # 3. Create date directory on EFS
    image_dir = os.path.join(EFS_MOUNT_PATH, "images", run_date)
    os.makedirs(image_dir, exist_ok=True)

    # 4. Process in batches
    lambda_client = boto3.client("lambda")
    total_images = len(expanded)
    n_batches = (total_images + FETCH_BATCH_SIZE - 1) // FETCH_BATCH_SIZE
    batches_dispatched = 0
    total_downloaded = 0

    for batch_id in range(n_batches):
        batch_start = batch_id * FETCH_BATCH_SIZE
        batch_end = min(batch_start + FETCH_BATCH_SIZE, total_images)
        batch_df = expanded.iloc[batch_start:batch_end]

        logger.info(f"Batch {batch_id}: downloading {len(batch_df)} images...")
        manifest_entries = download_batch_to_efs(
            session, batch_df, image_dir, start_idx=batch_start
        )
        total_downloaded += len(manifest_entries)

        if not manifest_entries:
            logger.warning(f"Batch {batch_id}: no images downloaded, skipping")
            continue

        # Write manifest to EFS
        manifest_path = os.path.join(image_dir, f"manifest_{batch_id}.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest_entries, f)

        # Invoke Scorer Lambda asynchronously
        payload = {
            "efs_image_dir": image_dir,
            "manifest_path": manifest_path,
            "threshold": FLAG_THRESHOLD,
            "run_date": run_date,
            "batch_id": batch_id,
        }
        invoke_scorer(lambda_client, payload)
        batches_dispatched += 1
        logger.info(f"Batch {batch_id}: dispatched to Scorer ({len(manifest_entries)} images)")

    elapsed = time.time() - start_time
    logger.info(f"=== Fetcher Complete ===")
    logger.info(f"  Total POD links: {total_images}")
    logger.info(f"  Downloaded: {total_downloaded}")
    logger.info(f"  Batches dispatched: {batches_dispatched}")
    logger.info(f"  Duration: {elapsed:.1f}s")

    return {
        "statusCode": 200,
        "body": json.dumps({
            "run_date": run_date,
            "total_pod_links": total_images,
            "total_downloaded": total_downloaded,
            "batches_dispatched": batches_dispatched,
            "duration_seconds": round(elapsed, 1),
        }),
    }
