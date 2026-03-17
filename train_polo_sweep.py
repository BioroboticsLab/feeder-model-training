#!/usr/bin/env python3
"""W&B Bayesian sweep for POLO merged-variant hyperparameter search.

Sweeps loc weight, augmentation preset, and batch size while keeping
model (polo26n.yaml), epochs (200), and dor (0.8) fixed.  Maximises
val/f1 across runs.

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

import config
import mosaic.tracking.pose_training as pose

# ── Dataset (shared NFS mount, works on both cirrus and thria) ──────────────
DATASET_DIR = Path("/mnt/trove/beesbook_feeder_model/feeder_bee_datasets_v1")
VARIANT = "merged"

# ── Fixed training settings ─────────────────────────────────────────────────
MODEL = "polo26n.yaml"
EPOCHS = 200
IMGSZ = 640
DOR = 0.8
PATIENCE = 50
LOC_LOSS = "mse"

# ── Sweep configuration ─────────────────────────────────────────────────────
SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "val/f1", "goal": "maximize"},
    "parameters": {
        "loc":          {"values": [3.0, 5.0, 8.0]},
        "augmentation": {"values": ["none", "moderate", "heavy"]},
        "batch":        {"values": [8, 16]},
    },
}

WANDB_PROJECT = "beesbook-feeder"


def train():
    run = wandb.init()
    cfg = run.config
    run_name = f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\nloc={cfg.loc}, aug={cfg.augmentation}, batch={cfg.batch}\n")

    data_yaml = config.resolve_polo_data(DATASET_DIR, VARIANT)
    output_dir = str(Path.home() / "runs" / "polo_sweep")

    results = pose.train_point_model(
        data_yaml=data_yaml,
        model=MODEL,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=cfg.batch,
        device=config.auto_device(),
        project=output_dir,
        name=run_name,
        patience=PATIENCE,
        loc_loss=LOC_LOSS,
        loc=cfg.loc,
        dor=DOR,
        augmentation=cfg.augmentation,
    )

    # ── Extract POLO metrics (L = localization, not B = box) ────────────
    d = results.results_dict if hasattr(results, "results_dict") else {}

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

    wandb.agent(sweep_id, function=train, project=WANDB_PROJECT, count=999)
