#!/usr/bin/env python3
"""Train a POLO point-detection model on the feeder bee dataset.

Usage
-----
  # Train on merged dataset (default)
  python train_polo.py --dataset /path/to/feeder_bee_datasets_v1

  # Train on CVAT-only baseline
  python train_polo.py --dataset /path/to/feeder_bee_datasets_v1 --variant cvat_only

  # Custom hyperparameters
  python train_polo.py --dataset /data --epochs 300 --batch 8 --model polo26n.yaml
"""

import argparse
from datetime import datetime
from pathlib import Path

import mosaic.tracking.pose_training as pose

import config


def parse_args():
    p = argparse.ArgumentParser(
        description="Train a POLO point-detection model on feeder bee data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    d = config.POLO_DEFAULTS

    p.add_argument(
        "--dataset", required=True,
        help="Path to extracted feeder_bee_datasets_v1/ directory",
    )
    p.add_argument(
        "--variant", default="merged", choices=["merged", "cvat_only"],
        help="Dataset variant to train on",
    )
    p.add_argument("--model", default=d["model"], help="POLO architecture YAML")
    p.add_argument("--epochs", type=int, default=d["epochs"])
    p.add_argument("--imgsz", type=int, default=d["imgsz"], help="Input image size")
    p.add_argument(
        "--batch", type=int, default=None,
        help="Batch size (default: 16 for cvat_only, 8 for merged)",
    )
    p.add_argument("--patience", type=int, default=d["patience"], help="Early stopping patience")
    p.add_argument("--device", default=None, help="Device: '0' (cuda), 'mps', 'cpu'")
    p.add_argument("--loc", type=float, default=d["loc"], help="Localization loss weight")
    p.add_argument("--dor", type=float, default=d["dor"], help="Distance of Reference threshold")
    p.add_argument("--augmentation", default=d["augmentation"], help="Augmentation preset")
    p.add_argument("--name", default=None, help="Run name (auto-generated if omitted)")
    p.add_argument(
        "--output-dir", default=None,
        help="Base output directory (default: dataset models/polo/<variant>/runs/)",
    )
    p.add_argument(
        "--no-validate", action="store_true",
        help="Skip test-set validation after training",
    )

    args = p.parse_args()

    # Auto-detect device
    if args.device is None:
        args.device = config.auto_device()

    # Auto-adjust batch size for merged variant (more annotations per image)
    if args.batch is None:
        args.batch = 8 if args.variant == "merged" else 16

    # Auto-generate run name
    if args.name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.name = f"{args.variant}_{timestamp}"

    return args


def main():
    args = parse_args()

    # Resolve dataset paths (ensures absolute path in data.yaml)
    data_yaml = config.resolve_polo_data(args.dataset, args.variant)

    # Resolve output directory (default: under the dataset)
    if args.output_dir is None:
        output_dir = str(config.resolve_polo_output(args.dataset, args.variant))
    else:
        output_dir = str(Path(args.output_dir).resolve())

    print(f"Dataset: {data_yaml}")
    print(f"Variant: {args.variant}")
    print(f"Device:  {args.device}")
    print(f"Batch:   {args.batch}")
    print(f"Model:   {args.model}")
    print()

    # Show dataset summary
    pose.check_dataset(str(data_yaml.parent))

    # Train
    print(f"\nStarting training: {args.name}")
    print(f"Output: {output_dir}/{args.name}")
    print()

    results = pose.train_point_model(
        data_yaml=data_yaml,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=output_dir,
        name=args.name,
        patience=args.patience,
        loc_loss=config.POLO_DEFAULTS["loc_loss"],
        loc=args.loc,
        dor=args.dor,
        augmentation=args.augmentation,
    )

    # Find best model
    best_model = pose.find_best_model(output_dir)
    print(f"\nBest model: {best_model}")

    # Test-set validation
    if not args.no_validate and best_model is not None:
        print("\nRunning test-set validation...")
        pose.validate_point_model(
            model_path=best_model,
            data_yaml=data_yaml,
            device=args.device,
            imgsz=args.imgsz,
            dor=args.dor,
            split="test",
        )


if __name__ == "__main__":
    main()
