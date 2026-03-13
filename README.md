# Feeder Model Training

Training scripts for feeder bee detection models. Supports two model architectures:

- **POLO** -- Point-detection model based on YOLO (ultralytics fork). Predicts bee locations as single points with class-specific radii on full-resolution images.
- **Localizer** -- Lightweight fully-convolutional heatmap model (~248K params). Classifies 128x128 grayscale patches as containing a bee or background.

Both models detect four classes: `UnmarkedBee`, `MarkedBee`, `BeeInCell`, `UpsideDownBee`.

## Setup

Requires Python 3.10+ and PyTorch with CUDA support (see [pytorch.org](https://pytorch.org/get-started/locally/)).

```bash
# Install mosaic-behavior with POLO and localizer extras
pip install "mosaic-behavior[polo,localizer] @ git+https://github.com/ecodylicscience/mosaic.git"

# Clone this repo
git clone <repo-url>
cd feeder-model-training
```

## Dataset

The training data is distributed as a tarball (`feeder_bee_datasets_v1.tar.gz`), separate from this repo. Extract it on the training machine:

```bash
tar xzf feeder_bee_datasets_v1.tar.gz
```

This produces the following structure:

| Dataset | Path | Contents | Purpose |
|---------|------|----------|---------|
| A | `polo/cvat_only` | CVAT annotations only | POLO baseline |
| B | `polo/merged` | CVAT + HDF5 + pseudo-labels | POLO merged training |
| C | `localizer/cvat` | 128x128 patches from CVAT | Localizer baseline |
| D | `localizer/merged` | CVAT + HDF5 patches | Localizer merged |

All datasets share the same test set (CVAT images only) for fair model comparison. The train/valid/test split is stratified by camera type (feeder vs exit cam) to ensure proportional representation.

## Quick Start

```bash
# Train POLO on merged dataset (recommended)
python train_polo.py --dataset /path/to/feeder_bee_datasets_v1

# Train POLO on CVAT-only baseline
python train_polo.py --dataset /path/to/feeder_bee_datasets_v1 --variant cvat_only

# Train localizer on CVAT patches
python train_localizer.py --dataset /path/to/feeder_bee_datasets_v1

# Train localizer with pretrained weights
python train_localizer.py --dataset /path/to/feeder_bee_datasets_v1 \
    --weights /path/to/localizer_2019_weights.pt

# Evaluate a trained POLO model
python evaluate.py --type polo --dataset /path/to/feeder_bee_datasets_v1 \
    --model runs/polo/merged_20260313/weights/best.pt

# Evaluate a trained localizer
python evaluate.py --type localizer --dataset /path/to/feeder_bee_datasets_v1 \
    --model runs/localizer/cvat_20260313/weights/best.pt
```

## Configuration

All scripts accept `--help` for full argument documentation. Key parameters:

### train_polo.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | (required) | Path to `feeder_bee_datasets_v1/` |
| `--variant` | `merged` | `merged` or `cvat_only` |
| `--model` | `polo26n.yaml` | Architecture (nano/small/medium/large) |
| `--epochs` | 200 | Max training epochs |
| `--batch` | 16 (8 for merged) | Batch size |
| `--patience` | 50 | Early stopping patience |
| `--loc` | 5.0 | Localization loss weight |
| `--dor` | 0.8 | Distance of Reference threshold |
| `--augmentation` | `heavy` | Augmentation preset |
| `--device` | auto | `0` (cuda), `mps`, `cpu` |

### train_localizer.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | (required) | Path to `feeder_bee_datasets_v1/` |
| `--variant` | `cvat` | `cvat` or `merged` |
| `--epochs` | 300 | Max training epochs |
| `--batch-size` | 128 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--patience` | 40 | Early stopping patience |
| `--weights` | None | Pretrained weights (.pt or .h5) |
| `--freeze-encoder` | False | Train head only |
| `--device` | auto | `0` (cuda), `mps`, `cpu` |

## Output Structure

Training runs are saved to `runs/<model_type>/<run_name>/`:

```
runs/polo/merged_20260313_143022/
    weights/
        best.pt        # best checkpoint (by validation metric)
        last.pt        # final epoch checkpoint
    results.csv        # per-epoch metrics
    args.yaml          # training configuration
```

## Classes

| ID | Name | Notes |
|----|------|-------|
| 0 | UnmarkedBee | Most common class |
| 1 | MarkedBee | Rare in feeder cam images, more common in exit cam |
| 2 | BeeInCell | Absent from CVAT annotations (no comb cells at feeder). Present in HDF5/pseudo-label sources only. |
| 3 | UpsideDownBee | Bees walking upside down on feeder |
