#!/usr/bin/env python3
"""
Rough staging peak on Lambda /tmp for the merged POD pipeline (one fetch batch).

This ignores model weights, overlays, manifest JSON overhead, etc. —
keep headroom (~30–40%) versus TmpEphemeralMB in template.yaml.

Usage:
  python3 ephemeral_peak_mb.py --fetch-batch-size 500 --avg-kb-per-image 200
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate /tmp MB needed for largest fetch batch staging.",
    )
    parser.add_argument(
        "--fetch-batch-size",
        type=int,
        default=500,
        help="FETCH_BATCH_SIZE (images written before scoring + PostgreSQL persist)",
    )
    parser.add_argument(
        "--avg-kb-per-image",
        type=float,
        default=200.0,
        help="Arithmetic mean POD image payload (KB)",
    )
    parser.add_argument(
        "--headroom",
        type=float,
        default=1.35,
        help="Multiplicative safety factor for OS/Python scratch",
    )
    args = parser.parse_args()

    naive_mb = (args.fetch_batch_size * args.avg_kb_per_image) / 1024.0
    with_headroom_mb = naive_mb * args.headroom

    print(
        f"fetch_batch_size={args.fetch_batch_size} avg_kb={args.avg_kb_per_image} "
        f"-> naive={naive_mb:.0f} MiB scaled={with_headroom_mb:.0f} MiB "
        f"(headroom x{args.headroom})"
    )
    staging_floor = max(512, int(with_headroom_mb) + 1)
    print(f"Staging-floor TmpEphemeralMB (rounded up): {staging_floor}")
    print(
        "(Stack default is often 3072–10240 MiB — include extra for OpenCV/decoding jitter.)",
    )


if __name__ == "__main__":
    main()
