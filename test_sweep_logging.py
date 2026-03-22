#!/usr/bin/env python3
"""Quick smoke-test: 2-epoch run to verify val/f1 is logged to W&B.

Run with:
  python test_sweep_logging.py
"""
from pathlib import Path

import wandb
from ultralytics.utils import SETTINGS

import config
import mosaic.tracking.pose_training as pose

SETTINGS["wandb"] = True

DATASET_DIR = Path("/mnt/trove/beesbook_feeder_model/feeder_bee_datasets_v1")
VARIANT     = "merged"
OUTPUT_DIR  = Path.home() / "runs" / "polo_sweep_test"


def main():
    data_yaml  = config.resolve_polo_data(DATASET_DIR, VARIANT)

    with wandb.init(project="beesbook-feeder", name="smoke-test-f1", tags=["smoke-test"]) as run:
        # ── same monkey-patch as train_polo_sweep.py ────────────────────
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
                model="polo26n.yaml",   # smallest/fastest
                epochs=2,
                imgsz=640,
                batch=8,
                device=config.auto_device(),
                project=str(OUTPUT_DIR),
                name="smoke_test",
                patience=0,             # no early stopping
                loc_loss="mse",
                loc=5.0,
                dor=0.8,
                augmentation="light",
            )
        finally:
            if _orig_on_fit is not None:
                _wb_mod.callbacks["on_fit_epoch_end"] = _orig_on_fit
            else:
                _wb_mod.callbacks.pop("on_fit_epoch_end", None)

        # ── verify ──────────────────────────────────────────────────────
        if hasattr(results, "results_dict") and results.results_dict:
            d  = results.results_dict
            p  = d.get("metrics/precision(L)", 0)
            r  = d.get("metrics/recall(L)", 0)
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            print(f"\nP:{p:.3f}  R:{r:.3f}  F1:{f1:.3f}  mAP100:{d.get('metrics/mAP100(L)', 0):.3f}")
            print("PASS — check W&B run for val/f1 chart")
        else:
            print("WARNING: results_dict empty")


if __name__ == "__main__":
    main()
