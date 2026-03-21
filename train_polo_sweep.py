#!/usr/bin/env python3
"""W&B Bayesian sweep for POLO merged-variant hyperparameter search.

Sweeps model size (n/s/m), loc weight, learning rate, and individual
augmentation parameters while keeping epochs (200) and dor (0.8) fixed.
Maximises val/f1 across runs.

Usage
-----
  # Start a NEW sweep (first machine):
  python train_polo_sweep.py

  # Join an EXISTING sweep from a second machine:
  python train_polo_sweep.py <sweep_id>
"""

import sys
from datetime import datetime
from pathlib import Path

import wandb
from ultralytics.utils import SETTINGS

import config
import mosaic.tracking.pose_training as pose

SETTINGS["wandb"] = True

# ── Dataset (shared NFS mount, works on both cirrus and thria) ──────────────
DATASET_DIR = Path("/mnt/trove/beesbook_feeder_model/feeder_bee_datasets_v1")
VARIANT = "merged"

# ── Output (local to each machine — NFS mount is read-only) ─────────────────
OUTPUT_DIR = Path.home() / "runs" / "polo_sweep"

# ── Fixed training settings ─────────────────────────────────────────────────
EPOCHS = 200
# try these later on in different sweeps
IMGSZ = 640 # fixed — safe for n/s/m on 640px; model is swept instead
BATCH = 8   # fixed — safe for n/s/m on 8 GB VRAM; model is swept instead
DOR = 0.8
PATIENCE = 50
LOC_LOSS = "mse"

# ── Sweep configuration ─────────────────────────────────────────────────────
SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "val/f1", "goal": "maximize"},
    "parameters": {
        "model":        {"values": ["polo26n.yaml", "polo26s.yaml", "polo26m.yaml"]},
        "loc":          {"min": 1.0, "max": 10.0},
        "lr0":          {"min": 1e-4, "max": 1e-1, "distribution": "log_uniform_values"},
        "lrf":          {"min": 1e-3, "max": 1e-1, "distribution": "log_uniform_values"},
        "weight_decay": {"min": 1e-5, "max": 1e-2, "distribution": "log_uniform_values"},
        # ── Augmentation ────────────────────────────────────────────────
        "degrees":      {"min": 0.0, "max": 45.0},
        "translate":    {"min": 0.0, "max": 0.3},
        "scale":        {"min": 0.0, "max": 0.5},
        "flipud":       {"min": 0.0, "max": 0.5},
        "fliplr":       {"min": 0.0, "max": 1.0},
        "mosaic":       {"min": 0.0, "max": 1.0},
        "mixup":        {"min": 0.0, "max": 0.3},
        "hsv_s":        {"min": 0.0, "max": 0.7},
        "hsv_v":        {"min": 0.0, "max": 0.4},
    },
}

WANDB_PROJECT = "beesbook-feeder"


def train():
    with wandb.init() as run:
        cfg = run.config
        run_name = f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        aug = {
            "degrees": cfg.degrees,
            "translate": cfg.translate,
            "scale": cfg.scale,
            "flipud": cfg.flipud,
            "fliplr": cfg.fliplr,
            "mosaic": cfg.mosaic,
            "mixup": cfg.mixup,
            "hsv_s": cfg.hsv_s,
            "hsv_v": cfg.hsv_v,
        }

        print(f"\nmodel={cfg.model}, loc={cfg.loc}, lr0={cfg.lr0}, batch={BATCH}\n")

        data_yaml = config.resolve_polo_data(DATASET_DIR, VARIANT)
        output_dir = str(OUTPUT_DIR)

        results = pose.train_point_model(
            data_yaml=data_yaml,
            model=cfg.model,
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            device=config.auto_device(),
            project=output_dir,
            name=run_name,
            patience=PATIENCE,
            loc_loss=LOC_LOSS,
            loc=cfg.loc,
            dor=DOR,
            augmentation=aug,
            lr0=cfg.lr0,
            lrf=cfg.lrf,
            weight_decay=cfg.weight_decay,
        )

        # ── Extract POLO metrics (L = localization, not B = box) ────────
        if not hasattr(results, "results_dict") or not results.results_dict:
            wandb.log({"error": True})
            print("WARNING: Training returned no results_dict — skipping metric logging")
            return

        d = results.results_dict
        precision = d.get("metrics/precision(L)", 0)
        recall    = d.get("metrics/recall(L)", 0)
        map100    = d.get("metrics/mAP100(L)", 0)
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        wandb.log({
            "val/precision": precision,
            "val/recall":    recall,
            "val/f1":        f1,
            "val/mAP100":    map100,
        })
        print(f"\nP:{precision:.3f}  R:{recall:.3f}  F1:{f1:.3f}  mAP100:{map100:.3f}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Join existing sweep (for second machine)
        sweep_id = sys.argv[1]
        print(f"Joining sweep {sweep_id}")
    else:
        # Create new sweep
        sweep_id = wandb.sweep(SWEEP_CONFIG, project=WANDB_PROJECT)
        print(f"Created sweep: https://wandb.ai/johanidler-org/{WANDB_PROJECT}/sweeps/{sweep_id}")

    wandb.agent(sweep_id, function=train, project=WANDB_PROJECT)
