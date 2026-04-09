#!/usr/bin/env python3
"""W&B Bayesian sweep for POLO feeder-only hyperparameter search.

Sweeps model size (s/m), loc weight, learning rate, and augmentation
while keeping epochs (200) and dor (0.8) fixed.  Ranges tightened from
prior sweep results: polo26m + medium/heavy aug + lr0 ~0.003-0.015
performed best.  Valid/test are feeder-cam only to match deployment.
Maximises val/f1 across runs.

Usage
-----
  # Start a NEW sweep (first machine):
  python train_polo_sweep.py

  # Join an EXISTING sweep from a second machine:
  python train_polo_sweep.py <sweep_id>
"""

import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import wandb
from ultralytics.utils import SETTINGS

import config
import mosaic.tracking.pose_training as pose

SETTINGS["wandb"] = True

# ── Dataset (local symlinked variant — feeder-only valid/test) ─────────────
DATA_YAML = Path(__file__).resolve().parent / "data" / "feeder_only" / "data.yaml"

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
        "model":        {"values": ["polo26s.yaml", "polo26m.yaml"]},
        "loc":          {"min": 3.0, "max": 7.0},
        "lr0":          {"min": 1e-3, "max": 3e-2, "distribution": "log_uniform_values"},
        "lrf":          {"min": 1e-3, "max": 2e-2, "distribution": "log_uniform_values"},
        "weight_decay": {"min": 1e-4, "max": 2e-3, "distribution": "log_uniform_values"},
        "augmentation": {"values": ["medium", "heavy"]},
    },
}

WANDB_PROJECT = "beesbook-feeder"

MAX_RETRIES  = 3
RETRY_DELAY  = 60  # seconds between attempts — gives NFS/GPU time to recover


def _is_transient(exc: BaseException) -> bool:
    """Return True for infrastructure errors that are worth retrying."""
    if isinstance(exc, (OSError, IOError)):
        return True
    if isinstance(exc, RuntimeError) and "CUDA" in str(exc):
        return True
    return False


def train():
    with wandb.init() as run:
        cfg = run.config
        run_name = f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        print(f"\nmodel={cfg.model}, aug={cfg.augmentation}, loc={cfg.loc}, lr0={cfg.lr0}, batch={BATCH}\n")

        data_yaml = config.ensure_absolute_data_yaml(DATA_YAML)
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
            results = None
            for attempt in range(1, MAX_RETRIES + 1):
                # Fresh directory name per attempt so partial runs don't conflict.
                attempt_name = run_name if attempt == 1 else f"{run_name}_retry{attempt - 1}"
                try:
                    results = pose.train_point_model(
                        data_yaml=data_yaml,
                        model=cfg.model,
                        epochs=EPOCHS,
                        imgsz=IMGSZ,
                        batch=BATCH,
                        device=config.auto_device(),
                        project=output_dir,
                        name=attempt_name,
                        patience=PATIENCE,
                        loc_loss=LOC_LOSS,
                        loc=cfg.loc,
                        dor=DOR,
                        augmentation=cfg.augmentation,
                        lr0=cfg.lr0,
                        lrf=cfg.lrf,
                        weight_decay=cfg.weight_decay,
                    )
                    break  # success — exit retry loop
                except Exception as exc:
                    if _is_transient(exc) and attempt < MAX_RETRIES:
                        print(f"\nWARNING: attempt {attempt}/{MAX_RETRIES} failed ({type(exc).__name__}: {exc})")
                        print(traceback.format_exc())
                        print(f"Retrying in {RETRY_DELAY}s...\n")
                        time.sleep(RETRY_DELAY)
                    else:
                        raise  # non-transient error or out of retries — let sweep mark it crashed
        finally:
            # Restore so the next sweep run starts clean.
            if _orig_on_fit is not None:
                _wb_mod.callbacks["on_fit_epoch_end"] = _orig_on_fit
            else:
                _wb_mod.callbacks.pop("on_fit_epoch_end", None)

        # Print final metrics locally (run is already finished by ultralytics).
        if results is not None and hasattr(results, "results_dict") and results.results_dict:
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
