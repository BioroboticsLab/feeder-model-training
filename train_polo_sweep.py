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

        # Ultralytics' W&B callback calls wb.run.finish() inside on_train_end,
        # so any wandb.log() after train_point_model() returns lands on a finished
        # run and is silently dropped (→ null on the dashboard).
        # Fix: inject val/f1 into trainer.metrics before each epoch is committed,
        # so it is logged while the run is still active.
        import ultralytics.utils.callbacks.wb as _wb_mod
        _orig_on_fit = _wb_mod.callbacks.get("on_fit_epoch_end")

        def _on_fit_with_f1(trainer):
            p = trainer.metrics.get("metrics/precision(L)", 0)
            r = trainer.metrics.get("metrics/recall(L)", 0)
            trainer.metrics["val/f1"] = 2 * p * r / (p + r) if (p + r) > 0 else 0
            if _orig_on_fit:
                _orig_on_fit(trainer)

        _wb_mod.callbacks["on_fit_epoch_end"] = _on_fit_with_f1

        try:
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
        finally:
            # Restore so the next sweep run starts clean.
            if _orig_on_fit is not None:
                _wb_mod.callbacks["on_fit_epoch_end"] = _orig_on_fit
            else:
                _wb_mod.callbacks.pop("on_fit_epoch_end", None)

        # Print final metrics locally (run is already finished by ultralytics).
        if hasattr(results, "results_dict") and results.results_dict:
            d = results.results_dict
            p = d.get("metrics/precision(L)", 0)
            r = d.get("metrics/recall(L)", 0)
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            print(f"\nP:{p:.3f}  R:{r:.3f}  F1:{f1:.3f}  mAP100:{d.get('metrics/mAP100(L)', 0):.3f}")
        else:
            print("WARNING: Training returned no results_dict")


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
