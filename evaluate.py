#!/usr/bin/env python3
"""Evaluate a trained model on the test set.

Supports both POLO point-detection models and localizer heatmap models.

Usage
-----
  # Evaluate a POLO model
  python evaluate.py --type polo --model runs/polo/my_run/weights/best.pt \\
      --dataset /path/to/feeder_bee_datasets_v1

  # Evaluate a localizer model
  python evaluate.py --type localizer --model runs/localizer/my_run/weights/best.pt \\
      --dataset /path/to/feeder_bee_datasets_v1

  # Evaluate on validation split instead of test
  python evaluate.py --type polo --model best.pt --dataset /data --split valid
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

import mosaic.tracking.pose_training as pose
from mosaic.tracking.pose_training.localizer_inference import detect_locations
from mosaic.tracking.pose_training.localizer_model import LocalizerEncoder

import config


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a trained model on the feeder bee test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--model", required=True,
        help="Path to trained model weights (.pt)",
    )
    p.add_argument(
        "--dataset", required=True,
        help="Path to extracted feeder_bee_datasets_v1/ directory",
    )
    p.add_argument(
        "--type", required=True, choices=["polo", "localizer"],
        help="Model type to evaluate",
    )
    p.add_argument("--device", default=None, help="Device: '0' (cuda), 'mps', 'cpu'")
    p.add_argument("--dor", type=float, default=0.8, help="DoR threshold (POLO only)")
    p.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold (localizer only)")
    p.add_argument("--match-radius", type=float, default=75.0, help="Hungarian matching radius in pixels (localizer only)")
    p.add_argument("--min-distance", type=float, default=15.0, help="NMS min distance in pixels (localizer only)")
    p.add_argument("--split", default="test", choices=["test", "valid"], help="Which split to evaluate on")
    p.add_argument(
        "--polo-variant", default="cvat_only",
        help="POLO dataset variant for data.yaml path (test set is identical across variants)",
    )

    args = p.parse_args()

    if args.device is None:
        args.device = config.auto_device()

    return args


# ── Localizer evaluation helpers ─────────────────────────────────────────────

def parse_polo_labels(label_path: Path, img_w: int, img_h: int):
    """Parse a POLO label file to pixel coordinates.

    Label format: ``class_id radius x_rel y_rel``
    """
    points = []
    text = label_path.read_text().strip()
    if not text:
        return points
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        class_id = int(parts[0])
        x = float(parts[2]) * img_w
        y = float(parts[3]) * img_h
        points.append((x, y, class_id))
    return points


def hungarian_match(gt_points, pred_points, match_radius):
    """Match GT to predictions using Hungarian algorithm.

    Returns (tp, fp, fn, matched_distances).
    """
    if len(gt_points) == 0:
        return 0, len(pred_points), 0, []
    if len(pred_points) == 0:
        return 0, 0, len(gt_points), []

    gt_arr = np.array(gt_points)[:, :2].astype(float)
    pred_arr = np.array(pred_points)[:, :2].astype(float)

    dists = np.linalg.norm(gt_arr[:, None] - pred_arr[None, :], axis=2)
    gt_idx, pred_idx = linear_sum_assignment(dists)
    matched = dists[gt_idx, pred_idx] <= match_radius

    tp = int(matched.sum())
    matched_dists = dists[gt_idx[matched], pred_idx[matched]].tolist()
    fp = len(pred_points) - tp
    fn = len(gt_points) - tp
    return tp, fp, fn, matched_dists


def evaluate_localizer(args):
    """Evaluate localizer model on full images using Hungarian matching."""
    # Load model
    model = LocalizerEncoder(
        num_classes=config.NUM_CLASSES,
        initial_channels=config.INITIAL_CHANNELS,
    )
    pose.load_localizer_weights(model, args.model)
    model.eval()

    # Use POLO cvat_only dataset for canonical test images and GT labels
    data_yaml = config.resolve_polo_data(args.dataset, args.polo_variant)
    dataset_dir = data_yaml.parent
    images_dir = dataset_dir / args.split / "images"
    labels_dir = dataset_dir / args.split / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    eval_device = "0" if torch.cuda.is_available() else "cpu"

    total_tp, total_fp, total_fn = 0, 0, 0
    all_dists = []
    n_images = 0

    image_paths = sorted(images_dir.glob("*"))
    print(f"Evaluating {len(image_paths)} images from {args.split} split...")

    for img_path in image_paths:
        if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue
        img_h, img_w = image.shape[:2]
        n_images += 1

        # Parse GT from POLO labels (original image coordinates)
        label_path = labels_dir / (img_path.stem + ".txt")
        gt_points = parse_polo_labels(label_path, img_w, img_h) if label_path.exists() else []

        # Rescale image to match pretrained localizer's expected px/tag
        scale = config.LOCALIZER_SCALE_FACTOR
        scaled_img = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA,
        )

        # Run localizer on scaled image
        dets = detect_locations(
            model, scaled_img,
            thresholds=args.threshold,
            device=eval_device,
            min_distance=args.min_distance,
        )
        # Map detections back to original image coordinates
        pred_points = [(d["x"] / scale, d["y"] / scale, d["class_id"]) for d in dets]

        # Match
        tp, fp, fn, dists = hungarian_match(gt_points, pred_points, args.match_radius)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_dists.extend(dists)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    mean_dist = np.mean(all_dists) if all_dists else 0

    print(f"\n{'='*60}")
    print(f"Localizer Evaluation — {args.split} split")
    print(f"{'='*60}")
    print(f"  Model:     {args.model}")
    print(f"  Images:    {n_images}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Match radius: {args.match_radius}px")
    print(f"  Scale factor: {config.LOCALIZER_SCALE_FACTOR:.3f} "
          f"({config.PRETRAINED_PX_PER_TAG}/{config.FEEDER_CAM_PX_PER_TAG} px/tag)")
    print(f"  TP: {total_tp}  FP: {total_fp}  FN: {total_fn}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1:        {f1:.3f}")
    print(f"  Mean dist: {mean_dist:.1f}px")


def evaluate_polo(args):
    """Evaluate POLO model using built-in validation."""
    data_yaml = config.resolve_polo_data(args.dataset, args.polo_variant)

    print(f"POLO Evaluation — {args.split} split")
    print(f"  Model:    {args.model}")
    print(f"  data.yaml: {data_yaml}")
    print(f"  DoR:      {args.dor}")
    print()

    pose.validate_point_model(
        model_path=args.model,
        data_yaml=data_yaml,
        device=args.device,
        imgsz=config.POLO_DEFAULTS["imgsz"],
        dor=args.dor,
        split=args.split,
    )


def main():
    args = parse_args()

    if args.type == "polo":
        evaluate_polo(args)
    else:
        evaluate_localizer(args)


if __name__ == "__main__":
    main()
