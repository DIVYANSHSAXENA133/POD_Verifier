#!/usr/bin/env python3
"""
POD model evaluation harness — measure the deployed model against a human gold set.

WHAT IT DOES
  1. Loads the gold annotation workbook (all sheets), unifying columns:
       awb, trip_id, url, context_valid, package_visible, label_readable, image_clarity
  2. Downloads each POD image concurrently (ThreadPoolExecutor + retries), recording
     every failure so eval coverage is known (never silently dropped).
  3. Loads the deployed MultiHeadEfficientNet checkpoint (best.pt).
  4. Runs inference under BOTH preprocessing variants so the normalization question
     is settled empirically:
        --preproc scale     -> resize, /255  (exactly what handler.py does today)
        --preproc imagenet  -> resize, /255, then ImageNet mean/std (what training used)
        --preproc both       -> run both and print the delta (default)
  5. Reports, for each attribute head: precision / recall / F1 / accuracy + confusion
     (prob > --attr-threshold, default 0.5).
  6. Composite pod_score = weighted sum of the 4 attribute probs. Sweeps the FLAG
     threshold at IMAGE and AWB level (AWB = best/max image, mirroring pod_scores_flagged),
     reporting precision/recall/F1 for the FLAG ("bad POD") class at each threshold, and
     highlighting (a) the F1-optimal threshold and (b) the lowest threshold that still
     achieves precision >= --penalty-precision (default 0.90) — the penalty-safe point.
  7. Writes: eval_report.md, metrics.json, per_image_predictions.csv into --out.

USAGE
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  pip install timm opencv-python-headless pandas numpy requests openpyxl
  python evaluate_model.py \
      --xlsx Data_annotation.xlsx \
      --model ../lambda_scorer/model/best.pt \
      --out ./eval_out --workers 32 --preproc both

  # If the gold "bad POD" definition differs from "composite below threshold",
  # pass a CSV of gold AWB labels with --gold-awb-labels awb,is_bad.

NOTE: This must run where the image host is reachable. In the build sandbox that host
was network-blocked, so this harness is delivered ready-to-run rather than pre-run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

ATTRS = ["context_valid", "package_visible", "label_readable", "image_clarity"]
WEIGHTS = {"context_valid": 0.35, "package_visible": 0.30, "label_readable": 0.20, "image_clarity": 0.15}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Gold loading
# --------------------------------------------------------------------------- #
def load_gold(xlsx_path: str) -> pd.DataFrame:
    xl = pd.ExcelFile(xlsx_path)
    frames = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        df.columns = [c.strip() for c in df.columns]
        url_col = "image_path" if "image_path" in df.columns else "pod"
        if url_col not in df.columns:
            continue
        df = df.rename(columns={url_col: "url"})
        keep = ["awb", "trip_id", "url"] + [a for a in ATTRS if a in df.columns]
        df = df[keep].copy()
        df["sheet"] = sheet
        frames.append(df)
    g = pd.concat(frames, ignore_index=True)
    for a in ATTRS:
        g[a] = (
            g[a].astype(str).str.strip().str.lower()
            .map({"true": 1, "false": 0, "1": 1, "0": 0}).fillna(0).astype(int)
        )
    g = g.dropna(subset=["url"]).drop_duplicates(subset=["url"]).reset_index(drop=True)
    return g


# --------------------------------------------------------------------------- #
# Concurrent download
# --------------------------------------------------------------------------- #
def build_session():
    import requests
    s = requests.Session()
    retry = requests.adapters.Retry(
        total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504]
    )
    ad = requests.adapters.HTTPAdapter(max_retries=retry, pool_maxsize=64, pool_connections=64)
    s.mount("https://", ad)
    s.mount("http://", ad)
    return s


def download_all(urls: list[str], workers: int) -> dict[str, bytes]:
    import requests
    session = build_session()

    def fetch(u: str):
        try:
            r = session.get(u, timeout=20)
            if r.status_code != 200 or len(r.content) < 500:
                return u, None
            return u, r.content
        except Exception:
            return u, None

    out: dict[str, bytes] = {}
    failures = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fetch, u) for u in urls]
        for i, f in enumerate(as_completed(futs), 1):
            u, content = f.result()
            if content is None:
                failures += 1
            else:
                out[u] = content
            if i % 100 == 0:
                print(f"  downloaded {i}/{len(urls)} (failures so far: {failures})", flush=True)
    print(f"Download complete: {len(out)} ok, {failures} failed of {len(urls)}", flush=True)
    return out


def download_to_disk(urls: list[str], out_dir: str, workers: int) -> str:
    """Download every URL to out_dir/<basename> concurrently (with retries).
    Returns out_dir. Images are removed later if --cleanup is set."""
    import requests
    os.makedirs(out_dir, exist_ok=True)
    session = build_session()

    def fetch(u: str):
        base = u.split("/")[-1]
        dst = os.path.join(out_dir, base)
        if os.path.exists(dst) and os.path.getsize(dst) >= 500:
            return True  # resume: already have it
        try:
            r = session.get(u, timeout=20)
            if r.status_code != 200 or len(r.content) < 500:
                return False
            with open(dst, "wb") as fh:
                fh.write(r.content)
            return True
        except Exception:
            return False

    ok = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, done in enumerate(ex.map(fetch, urls), 1):
            ok += 1 if done else 0
            if i % 100 == 0:
                print(f"  downloaded {i}/{len(urls)} (ok so far: {ok})", flush=True)
    print(f"Download-to-disk complete: {ok}/{len(urls)} saved to {out_dir}", flush=True)
    return out_dir


def load_local(urls: list[str], images_dir: str) -> dict[str, bytes]:
    """Load image bytes from a local dir, matching each URL by its basename.
    Use this when the image host is unreachable but images were provided locally."""
    index = {}
    for root, _, files in os.walk(images_dir):
        for f in files:
            index[f] = os.path.join(root, f)
    out, missing = {}, 0
    for u in urls:
        base = u.split("/")[-1]
        path = index.get(base)
        if path is None:
            missing += 1
            continue
        with open(path, "rb") as fh:
            content = fh.read()
        if len(content) >= 500:
            out[u] = content
        else:
            missing += 1
    print(f"Local load: {len(out)} ok, {missing} missing/too-small of {len(urls)}", flush=True)
    return out


# --------------------------------------------------------------------------- #
# Model + inference
# --------------------------------------------------------------------------- #
def load_model(model_path: str):
    import torch
    # make src.model importable relative to the repo layout
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "..", "lambda_scorer"))
    from src.model import MultiHeadEfficientNet  # noqa

    device = torch.device("cpu")
    model = MultiHeadEfficientNet(num_attributes=4, pretrained=False)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.to(device).eval()
    torch.set_num_threads(os.cpu_count() or 4)
    return model, device


def decode_resize(content: bytes, size: int) -> np.ndarray | None:
    import cv2
    arr = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    return img


def run_inference(model, device, images: list[np.ndarray], preproc: str, size: int, batch: int):
    import torch
    probs = {a: [] for a in ATTRS}
    with torch.no_grad():
        for i in range(0, len(images), batch):
            chunk = images[i : i + batch]
            x = np.stack(chunk).astype(np.float32) / 255.0
            if preproc == "imagenet":
                x = (x - IMAGENET_MEAN) / IMAGENET_STD
            t = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous().float().to(device)
            logits = model(t)
            for a in ATTRS:
                probs[a].append(torch.sigmoid(logits[a]).cpu().numpy())
    return {a: np.concatenate(probs[a]) for a in ATTRS}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / max(1, (tp + fp + fn + tn))
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "accuracy": round(acc, 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "support_pos": int((y_true == 1).sum())}


def evaluate(preproc_name, probs, gold, attr_threshold, penalty_precision):
    res = {"preproc": preproc_name, "per_attribute": {}, "composite_sweep": []}

    # per-attribute (positive class = attribute TRUE)
    for a in ATTRS:
        y_true = gold[a].to_numpy()
        y_pred = (probs[a] > attr_threshold).astype(int)
        res["per_attribute"][a] = binary_metrics(y_true, y_pred)

    # composite pod_score
    comp = sum(probs[a] * WEIGHTS[a] for a in ATTRS)
    gold_comp = sum(gold[a].to_numpy() * WEIGHTS[a] for a in ATTRS)
    df = gold.copy()
    df["model_comp"] = comp
    df["gold_comp"] = gold_comp

    # AWB level: best (max) image per AWB, mirroring pod_scores_flagged view
    awb = df.groupby("awb").agg(model_comp=("model_comp", "max"),
                                gold_comp=("gold_comp", "max")).reset_index()

    # FLAG = "bad POD" = composite below threshold. positive class = FLAG.
    for t in [round(x, 2) for x in np.arange(0.30, 0.91, 0.05)]:
        img_true = (df["gold_comp"].to_numpy() < t).astype(int)
        img_pred = (df["model_comp"].to_numpy() < t).astype(int)
        awb_true = (awb["gold_comp"].to_numpy() < t).astype(int)
        awb_pred = (awb["model_comp"].to_numpy() < t).astype(int)
        res["composite_sweep"].append({
            "threshold": t,
            "image_flag": binary_metrics(img_true, img_pred),
            "awb_flag": binary_metrics(awb_true, awb_pred),
        })

    # highlight thresholds
    sweeps = res["composite_sweep"]
    res["f1_optimal_awb"] = max(sweeps, key=lambda s: s["awb_flag"]["f1"])
    penalty_safe = [s for s in sweeps if s["awb_flag"]["precision"] >= penalty_precision]
    res["penalty_safe_awb"] = (max(penalty_safe, key=lambda s: s["awb_flag"]["recall"])
                               if penalty_safe else None)
    return res


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def write_report(all_res, meta, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"meta": meta, "results": all_res}, f, indent=2)

    lines = ["# POD Model Evaluation Report", ""]
    lines.append(f"- Gold images evaluated: **{meta['evaluated']}** of {meta['gold_total']} "
                 f"({meta['download_failures']} download failures, "
                 f"{meta['decode_failures']} decode failures)")
    lines.append(f"- Unique AWBs: **{meta['unique_awb']}**")
    lines.append(f"- Model: `{meta['model']}` | input {meta['input_size']}px | "
                 f"attr threshold {meta['attr_threshold']}")
    lines.append("")
    for res in all_res:
        lines.append(f"## Preprocessing: `{res['preproc']}`")
        lines.append("")
        lines.append("### Per-attribute (positive = attribute TRUE)")
        lines.append("")
        lines.append("| attribute | precision | recall | F1 | accuracy | support+ |")
        lines.append("|---|---|---|---|---|---|")
        for a in ATTRS:
            m = res["per_attribute"][a]
            lines.append(f"| {a} | {m['precision']} | {m['recall']} | {m['f1']} | "
                         f"{m['accuracy']} | {m['support_pos']} |")
        lines.append("")
        lines.append("### Composite FLAG (bad POD) — AWB level, threshold sweep")
        lines.append("")
        lines.append("| threshold | precision | recall | F1 | flagged AWBs |")
        lines.append("|---|---|---|---|---|")
        for s in res["composite_sweep"]:
            m = s["awb_flag"]
            lines.append(f"| {s['threshold']} | {m['precision']} | {m['recall']} | "
                         f"{m['f1']} | {m['tp'] + m['fp']} |")
        lines.append("")
        f1o = res["f1_optimal_awb"]
        lines.append(f"- **F1-optimal AWB threshold:** {f1o['threshold']} "
                     f"(P={f1o['awb_flag']['precision']}, R={f1o['awb_flag']['recall']}, "
                     f"F1={f1o['awb_flag']['f1']})")
        ps = res["penalty_safe_awb"]
        if ps:
            lines.append(f"- **Penalty-safe threshold** (precision >= {meta['penalty_precision']}): "
                         f"{ps['threshold']} (P={ps['awb_flag']['precision']}, "
                         f"R={ps['awb_flag']['recall']})")
        else:
            lines.append(f"- **No threshold reaches precision >= {meta['penalty_precision']}** "
                         f"at AWB level — NOT penalty-safe as-is.")
        lines.append("")
    with open(os.path.join(out_dir, "eval_report.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--model", default="../lambda_scorer/model/best.pt")
    ap.add_argument("--out", default="./eval_out")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--attr-threshold", type=float, default=0.5)
    ap.add_argument("--penalty-precision", type=float, default=0.90)
    ap.add_argument("--preproc", choices=["scale", "imagenet", "both"], default="both")
    ap.add_argument("--limit", type=int, default=0, help="cap images for a quick smoke run")
    ap.add_argument("--images-dir", default="", help="load images locally (by URL basename) "
                    "instead of downloading — use when the image host is unreachable")
    ap.add_argument("--download-dir", default="", help="download all gold images to this dir "
                    "first, then evaluate from it (matches 'download locally then evaluate')")
    ap.add_argument("--cleanup", action="store_true",
                    help="delete --download-dir after evaluation (remove images from local)")
    args = ap.parse_args()

    t0 = time.time()
    gold = load_gold(args.xlsx)
    if args.limit:
        gold = gold.head(args.limit).reset_index(drop=True)
    print(f"Gold rows: {len(gold)} | unique AWB: {gold['awb'].nunique()}")

    if args.download_dir:
        download_to_disk(gold["url"].tolist(), args.download_dir, args.workers)
        blobs = load_local(gold["url"].tolist(), args.download_dir)
    elif args.images_dir:
        blobs = load_local(gold["url"].tolist(), args.images_dir)
    else:
        blobs = download_all(gold["url"].tolist(), args.workers)
    dl_fail = len(gold) - len(blobs)

    imgs, keep_idx, decode_fail = [], [], 0
    for idx, row in gold.iterrows():
        c = blobs.get(row["url"])
        if c is None:
            continue
        im = decode_resize(c, args.input_size)
        if im is None:
            decode_fail += 1
            continue
        imgs.append(im)
        keep_idx.append(idx)
    gold_eval = gold.loc[keep_idx].reset_index(drop=True)
    print(f"Decoded {len(imgs)} images; running inference...")

    model, device = load_model(args.model)
    preprocs = ["scale", "imagenet"] if args.preproc == "both" else [args.preproc]
    all_res = []
    for p in preprocs:
        probs = run_inference(model, device, imgs, p, args.input_size, args.batch)
        all_res.append(evaluate(p, probs, gold_eval, args.attr_threshold, args.penalty_precision))
        # dump predictions for the first/primary preproc
        if p == preprocs[0]:
            pred_df = gold_eval.copy()
            for a in ATTRS:
                pred_df[f"pred_{a}"] = probs[a].round(6)
            pred_df["pred_composite"] = sum(probs[a] * WEIGHTS[a] for a in ATTRS).round(6)
            os.makedirs(args.out, exist_ok=True)
            pred_df.to_csv(os.path.join(args.out, "per_image_predictions.csv"), index=False)

    meta = {
        "model": args.model, "gold_total": len(gold), "evaluated": len(imgs),
        "download_failures": dl_fail, "decode_failures": decode_fail,
        "unique_awb": int(gold_eval["awb"].nunique()), "input_size": args.input_size,
        "attr_threshold": args.attr_threshold, "penalty_precision": args.penalty_precision,
        "elapsed_s": round(time.time() - t0, 1),
    }
    write_report(all_res, meta, args.out)
    print(f"\nDone in {meta['elapsed_s']}s. Outputs in {args.out}/")

    if args.cleanup and args.download_dir and os.path.isdir(args.download_dir):
        import shutil
        shutil.rmtree(args.download_dir, ignore_errors=True)
        print(f"Cleaned up local images: removed {args.download_dir}")


if __name__ == "__main__":
    main()
