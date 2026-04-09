#!/usr/bin/env python3
"""Render validation images with ground truth and model predictions side by side.

Sorts images into aligned/ (model matches GT) and misaligned/ (disagreements).
"""

import cv2
import numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment

from ultralytics import YOLO

# ── Configuration ────────────────────────────────────────────────────────────
DATASET_DIR = Path("/mnt/trove/beesbook_feeder_model/feeder_bee_datasets_v1")
VARIANT = "cvat_only"
MODEL_PATH = Path("/home/johan/runs/polo_sweep/sweep_20260321_134359/weights/best.pt")
OUTPUT_DIR = Path("/home/johan/feeder-model-training/rendered_validation")

MATCH_RADIUS = 75.0  # pixels — same as evaluate.py default
IMGSZ = 640
CONF_THRESHOLD = 0.25
DOR = 0.8

CLASS_NAMES = ["UnmarkedBee", "MarkedBee", "BeeInCell", "UpsideDownBee"]
# BGR colors
GT_COLOR = (0, 200, 0)        # Green for ground truth
PRED_COLOR = (0, 100, 255)    # Orange for predictions
MATCH_COLOR = (200, 200, 0)   # Cyan for matched pairs
FP_COLOR = (0, 0, 255)        # Red for false positives
FN_COLOR = (255, 0, 255)      # Magenta for false negatives


def load_gt(label_path: Path, img_w: int, img_h: int):
    """Load POLO ground truth: returns list of (x, y, class_id)."""
    points = []
    if not label_path.exists():
        return points
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        class_id = int(parts[0])
        x = float(parts[2]) * img_w
        y = float(parts[3]) * img_h
        points.append((x, y, class_id))
    return points


def run_polo_model(model, img_path: Path):
    """Run POLO inference; returns list of (x, y, class_id, confidence)."""
    results = model(str(img_path), imgsz=IMGSZ, conf=CONF_THRESHOLD, verbose=False)
    preds = []
    for r in results:
        if r.locations is None or len(r.locations) == 0:
            continue
        locs = r.locations
        xy = locs.xy.cpu().numpy()        # (N, 2)
        confs = locs.conf.cpu().numpy()   # (N,)
        classes = locs.cls.cpu().numpy().astype(int)  # (N,)
        for i in range(len(classes)):
            preds.append((float(xy[i, 0]), float(xy[i, 1]), int(classes[i]), float(confs[i])))
    return preds


def match_gt_pred(gt_points, pred_points, match_radius):
    """Hungarian matching. Returns (matched_gt_idx, matched_pred_idx, unmatched_gt, unmatched_pred)."""
    if not gt_points or not pred_points:
        return [], [], list(range(len(gt_points))), list(range(len(pred_points)))

    gt_arr = np.array([(p[0], p[1]) for p in gt_points])
    pred_arr = np.array([(p[0], p[1]) for p in pred_points])
    dists = np.linalg.norm(gt_arr[:, None] - pred_arr[None, :], axis=2)

    gt_idx, pred_idx = linear_sum_assignment(dists)
    matched_gt, matched_pred = [], []
    used_gt, used_pred = set(), set()

    for gi, pi in zip(gt_idx, pred_idx):
        if dists[gi, pi] <= match_radius:
            matched_gt.append(gi)
            matched_pred.append(pi)
            used_gt.add(gi)
            used_pred.add(pi)

    unmatched_gt = [i for i in range(len(gt_points)) if i not in used_gt]
    unmatched_pred = [i for i in range(len(pred_points)) if i not in used_pred]
    return matched_gt, matched_pred, unmatched_gt, unmatched_pred


def draw_comparison(img, gt_points, pred_points, matched_gt, matched_pred, unmatched_gt, unmatched_pred):
    """Draw GT and predictions on image with match status."""
    overlay = img.copy()
    R = 30  # draw radius for visibility

    # Draw matched pairs (TP) — cyan circles + connecting line
    for gi, pi in zip(matched_gt, matched_pred):
        gx, gy, gc = int(gt_points[gi][0]), int(gt_points[gi][1]), gt_points[gi][2]
        px, py, pc = int(pred_points[pi][0]), int(pred_points[pi][1]), pred_points[pi][2]
        # GT: solid circle
        cv2.circle(overlay, (gx, gy), R, MATCH_COLOR, 2)
        cv2.putText(overlay, f"GT:{CLASS_NAMES[gc][:3]}", (gx + R + 2, gy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, MATCH_COLOR, 1)
        # Pred: dashed-style circle (thinner)
        cv2.circle(overlay, (px, py), R, MATCH_COLOR, 1)
        cv2.putText(overlay, f"P:{CLASS_NAMES[pc][:3]} {pred_points[pi][3]:.2f}",
                    (px + R + 2, py + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, MATCH_COLOR, 1)
        cv2.line(overlay, (gx, gy), (px, py), MATCH_COLOR, 1)

    # Draw false negatives (missed by model) — magenta
    for gi in unmatched_gt:
        gx, gy, gc = int(gt_points[gi][0]), int(gt_points[gi][1]), gt_points[gi][2]
        cv2.circle(overlay, (gx, gy), R, FN_COLOR, 2)
        cv2.putText(overlay, f"FN:{CLASS_NAMES[gc][:3]}", (gx + R + 2, gy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, FN_COLOR, 1)

    # Draw false positives (model hallucinations) — red
    for pi in unmatched_pred:
        px, py, pc = int(pred_points[pi][0]), int(pred_points[pi][1]), pred_points[pi][2]
        conf = pred_points[pi][3]
        cv2.circle(overlay, (px, py), R, FP_COLOR, 2)
        cv2.putText(overlay, f"FP:{CLASS_NAMES[pc][:3]} {conf:.2f}",
                    (px + R + 2, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, FP_COLOR, 1)

    return overlay


def main():
    img_dir = DATASET_DIR / "models" / "polo" / VARIANT / "valid" / "images"
    label_dir = DATASET_DIR / "models" / "polo" / VARIANT / "valid" / "labels"

    aligned_dir = OUTPUT_DIR / "aligned"
    close_dir = OUTPUT_DIR / "close"
    misaligned_dir = OUTPUT_DIR / "misaligned"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    close_dir.mkdir(parents=True, exist_ok=True)
    misaligned_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH), task="locate")

    img_files = sorted(img_dir.glob("*.png"))
    print(f"Found {len(img_files)} validation images")
    print(f"Match radius: {MATCH_RADIUS}px")
    print(f"Legend: cyan=TP (matched), magenta=FN (missed), red=FP (extra)\n")

    n_aligned, n_close, n_misaligned = 0, 0, 0
    total_tp, total_fp, total_fn = 0, 0, 0

    for i, img_path in enumerate(img_files, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        label_path = label_dir / (img_path.stem + ".txt")
        gt_points = load_gt(label_path, w, h)
        pred_points = run_polo_model(model, img_path)

        matched_gt, matched_pred, unmatched_gt, unmatched_pred = match_gt_pred(
            gt_points, pred_points, MATCH_RADIUS
        )

        rendered = draw_comparison(img, gt_points, pred_points,
                                   matched_gt, matched_pred, unmatched_gt, unmatched_pred)

        tp = len(matched_gt)
        fp = len(unmatched_pred)
        fn = len(unmatched_gt)
        errors = fp + fn

        if errors == 0:
            out_dir = aligned_dir
            n_aligned += 1
            status = "OK"
        elif errors <= 1:
            out_dir = close_dir
            n_close += 1
            status = "CLOSE"
        else:
            out_dir = misaligned_dir
            n_misaligned += 1
            status = "MISMATCH"

        cv2.imwrite(str(out_dir / img_path.name), rendered)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        print(f"[{i:4d}/{len(img_files)}] {status:8s}  TP={tp} FP={fp} FN={fn}  {img_path.name}")

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print(f"\n{'='*60}")
    print(f"Aligned:    {n_aligned:4d} images → {aligned_dir}")
    print(f"Close:      {n_close:4d} images → {close_dir}")
    print(f"Misaligned: {n_misaligned:4d} images → {misaligned_dir}")
    print(f"Total:      {n_aligned + n_close + n_misaligned:4d}")
    print(f"\n{'='*60}")
    print(f"TP: {total_tp}  FP: {total_fp}  FN: {total_fn}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1:        {f1:.3f}")


if __name__ == "__main__":
    main()
