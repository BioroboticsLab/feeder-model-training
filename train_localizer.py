#!/usr/bin/env python3
"""Train the BeesBook localizer model on the feeder bee dataset.

The localizer is a lightweight fully-convolutional heatmap model (~248K params)
that classifies 128x128 grayscale patches as containing a bee (positive) or not
(negative), with per-class output channels.

Usage
-----
  # Train on CVAT patches (default)
  python train_localizer.py --dataset /path/to/feeder_bee_datasets_v1

  # Train on merged patches (CVAT + HDF5)
  python train_localizer.py --dataset /data --variant merged

  # Fine-tune from pretrained weights
  python train_localizer.py --dataset /data --weights /path/to/localizer_2019_weights.pt
"""

import argparse
from datetime import datetime
from pathlib import Path

import mosaic.tracking.pose_training as pose
from mosaic.tracking.pose_training import LocalizerAugmentConfig

import config


def parse_args():
    p = argparse.ArgumentParser(
        description="Train the BeesBook localizer on feeder bee patch data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    d = config.LOCALIZER_DEFAULTS

    p.add_argument(
        "--dataset", required=True,
        help="Path to extracted feeder_bee_datasets_v1/ directory",
    )
    p.add_argument(
        "--variant", default="cvat", choices=["cvat", "merged"],
        help="Dataset variant to train on",
    )
    p.add_argument("--epochs", type=int, default=d["epochs"])
    p.add_argument("--batch-size", type=int, default=d["batch_size"])
    p.add_argument("--lr", type=float, default=d["lr"], help="Learning rate")
    p.add_argument(
        "--patience", type=int, default=d["early_stopping_patience"],
        help="Early stopping patience (epochs)",
    )
    p.add_argument(
        "--lr-patience", type=int, default=d["lr_patience"],
        help="ReduceLROnPlateau patience (epochs)",
    )
    p.add_argument("--device", default=None, help="Device: '0' (cuda), 'mps', 'cpu'")
    p.add_argument(
        "--freeze-encoder", action="store_true",
        help="Freeze encoder weights (train head only)",
    )
    p.add_argument(
        "--weights", default=None,
        help="Pretrained weights path (.pt or .h5). "
             "Keras .h5 files are auto-converted to PyTorch.",
    )
    p.add_argument("--name", default=None, help="Run name (auto-generated if omitted)")
    p.add_argument("--output-dir", default="runs/localizer", help="Base output directory")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    if args.device is None:
        args.device = config.auto_device()

    if args.name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.name = f"{args.variant}_{timestamp}"

    return args


def main():
    args = parse_args()

    # Resolve dataset
    dataset_dir = config.resolve_localizer_data(args.dataset, args.variant)
    print(f"Dataset: {dataset_dir}")
    print(f"Variant: {args.variant}")
    print(f"Device:  {args.device}")
    print()

    # Handle pretrained weights
    weights = args.weights
    if weights is not None:
        weights_path = Path(weights)
        if weights_path.suffix == ".h5":
            pt_path = weights_path.with_suffix(".pt")
            if pt_path.exists():
                print(f"Using existing PyTorch weights: {pt_path}")
                weights = str(pt_path)
            else:
                print(f"Converting Keras weights: {weights_path} -> {pt_path}")
                weights = str(
                    pose.convert_keras_weights(
                        weights_path,
                        output_pt_path=pt_path,
                        num_classes=config.NUM_CLASSES,
                        initial_channels=config.INITIAL_CHANNELS,
                    )
                )

    # Augmentation config (matches notebook defaults)
    augment = LocalizerAugmentConfig(flip_h=True, flip_v=True, rotate_90=True)

    # Train
    print(f"Starting training: {args.name}")
    print(f"Output: {args.output_dir}/{args.name}")
    print()

    result = pose.train_localizer(
        dataset_dir=dataset_dir,
        num_classes=config.NUM_CLASSES,
        initial_channels=config.INITIAL_CHANNELS,
        weights=weights,
        freeze_encoder=args.freeze_encoder,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        early_stopping_patience=args.patience,
        lr_patience=args.lr_patience,
        device=args.device,
        project=args.output_dir,
        name=args.name,
        seed=args.seed,
        augment=augment,
    )

    # Summary
    print(f"\nTraining complete.")
    print(f"  Best model: {result.best_model_path}")
    print(f"  Best epoch: {result.best_epoch + 1}")
    print(f"  Best val loss: {result.best_val_loss:.4f}")
    print(f"  Run dir: {result.run_dir}")


if __name__ == "__main__":
    main()
