#!/usr/bin/env bash
# One-command gold-standard evaluation.
# Run this on a machine/network that can reach the POD image host
# (the build sandbox is allowlist-blocked from blitznow.in, so it must run here).
#
# It: downloads all gold images from the links in the xlsx to a temp dir,
#     runs the model, prints precision/recall/F1, then DELETES the images.
#
#   cd aws/eval && ./run_gold_eval.sh /path/to/Data_annotation.xlsx
set -euo pipefail

XLSX="${1:-Data_annotation.xlsx}"
MODEL="${2:-../lambda_scorer/model/best.pt}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> installing deps (CPU torch)"
pip install --quiet \
  torch torchvision --index-url https://download.pytorch.org/whl/cpu || \
  pip install --quiet torch torchvision
pip install --quiet timm opencv-python-headless pandas "numpy<2" requests openpyxl

echo "==> download -> evaluate -> cleanup"
python3 "${HERE}/evaluate_model.py" \
  --xlsx "${XLSX}" \
  --model "${MODEL}" \
  --download-dir "${HERE}/_gold_images_tmp" \
  --cleanup \
  --preproc both \
  --workers 48 \
  --out "${HERE}/eval_out"

echo "==> done. Report: ${HERE}/eval_out/eval_report.md  (metrics.json alongside)"
echo "    Send me eval_out/metrics.json and I'll finish the ship/no-ship sign-off."
