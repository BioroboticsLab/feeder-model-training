"""Shared configuration, defaults, and dataset path helpers."""

from pathlib import Path

import torch
from mosaic.core.dataset import Dataset

# ── Classes ──────────────────────────────────────────────────────────────────

CLASS_NAMES = ["UnmarkedBee", "MarkedBee", "BeeInCell", "UpsideDownBee"]
RADII = {name: 100.0 for name in CLASS_NAMES}
NUM_CLASSES = len(CLASS_NAMES)
INITIAL_CHANNELS = 16  # localizer encoder width

# ── Localizer scale factor ───────────────────────────────────────────────────
# The 2019 pretrained localizer was trained at 38 px/tag (BeesBook colony cams).
# Feeder cam images have ~58 px/tag.  To fine-tune from pretrained weights the
# input must be downscaled so bees appear at the expected size.
PRETRAINED_PX_PER_TAG = 38.0
FEEDER_CAM_PX_PER_TAG = 58.0
LOCALIZER_SCALE_FACTOR = PRETRAINED_PX_PER_TAG / FEEDER_CAM_PX_PER_TAG  # ~0.655

# ── POLO defaults (from notebooks 03/07) ────────────────────────────────────

POLO_DEFAULTS = {
    "model": "polo26n.yaml",
    "epochs": 200,
    "imgsz": 640,
    "batch": 16,
    "patience": 50,
    "loc": 5.0,
    "loc_loss": "mse",
    "dor": 0.8,
    "augmentation": "heavy",
}

# ── Localizer defaults (from notebook 03_train_beesbook_localizer) ──────────

LOCALIZER_DEFAULTS = {
    "epochs": 300,
    "batch_size": 128,
    "lr": 1e-3,
    "early_stopping_patience": 40,
    "lr_patience": 25,
    "freeze_encoder": False,
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def auto_device() -> str:
    """Return the best available device string for training."""
    if torch.cuda.is_available():
        return "0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_polo_data(dataset_dir: str | Path, variant: str = "merged") -> Path:
    """Resolve the POLO data.yaml path from a mosaic dataset directory.

    Parameters
    ----------
    dataset_dir : path
        Root of the extracted ``feeder_bee_datasets_v1/`` directory.
    variant : str
        ``"merged"`` (CVAT + HDF5 + pseudo-labels) or ``"cvat_only"`` (CVAT baseline).

    Returns
    -------
    Path to the ``data.yaml`` file for the chosen variant.
    """
    ds = Dataset(Path(dataset_dir) / "dataset.yaml").load()
    data_yaml = ds.get_root("models") / "polo" / variant / "data.yaml"
    if not data_yaml.exists():
        available = [
            p.parent.name
            for p in (ds.get_root("models") / "polo").glob("*/data.yaml")
        ]
        raise FileNotFoundError(
            f"data.yaml not found for variant '{variant}'. "
            f"Available: {available}"
        )
    return data_yaml


def resolve_localizer_data(
    dataset_dir: str | Path, variant: str = "cvat"
) -> Path:
    """Resolve the localizer patches directory from a mosaic dataset directory.

    Parameters
    ----------
    dataset_dir : path
        Root of the extracted ``feeder_bee_datasets_v1/`` directory.
    variant : str
        ``"cvat"`` (CVAT patches) or ``"merged"`` (CVAT + HDF5).

    Returns
    -------
    Path to the localizer directory containing ``train/``, ``valid/``, ``test/``
    subdirectories with ``patches.npy`` and ``labels.npy``.
    """
    ds = Dataset(Path(dataset_dir) / "dataset.yaml").load()
    loc_dir = ds.get_root("models") / "localizer" / variant
    if not (loc_dir / "train" / "patches.npy").exists():
        available = [
            p.parent.parent.name
            for p in (ds.get_root("models") / "localizer").glob(
                "*/train/patches.npy"
            )
        ]
        raise FileNotFoundError(
            f"Localizer data not found for variant '{variant}'. "
            f"Available: {available}"
        )
    return loc_dir


def resolve_polo_output(dataset_dir: str | Path, variant: str = "merged") -> Path:
    """Return the POLO training-runs directory under the dataset."""
    ds = Dataset(Path(dataset_dir) / "dataset.yaml").load()
    return ds.get_root("models") / "polo" / variant / "runs"


def resolve_localizer_output(dataset_dir: str | Path, variant: str = "cvat") -> Path:
    """Return the localizer training-runs directory under the dataset."""
    ds = Dataset(Path(dataset_dir) / "dataset.yaml").load()
    return ds.get_root("models") / "localizer" / variant / "runs"
