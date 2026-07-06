"""Shared configuration, defaults, paths, and evaluation helpers.

This is the single shared Python module for the feeder-model-training repo; the
end-to-end workflow lives in the ``notebooks/`` directory and imports from here.

It holds:

* class / radius definitions and the localizer resolution-scale constant,
* the locked POLO training hyperparameters for the deployed ``polo26n`` model,
* the fixed evaluation settings (Hungarian matching at 75 px, conf 0.25, DoR 0.3),
* point-detection metric primitives (``load_gt``, ``match``, ``polo_predict``,
  ``localizer_predict``, ``run_point_eval``) reused by ``04_evaluation.ipynb``,
* small dataset-path helpers for the mosaic Dataset layout used by the notebooks.

The evaluation primitives mirror Johan's ``compare_models.py`` so the published
numbers are reproduced by the same class-agnostic Hungarian matching he used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

# ``torch`` is imported lazily (only auto_device / localizer inference need it) so
# the metric primitives below can be used in environments without a GPU stack.

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

# ── mosaic Dataset layout ────────────────────────────────────────────────────
# The notebooks write the converted POLO dataset under this model namespace.
# Named "polo" (the model is POLO/polo26, not YOLO11); matches Johan's
# feeder-model-training scripts, which used models/polo/...
POLO_MODEL_NAME = "polo"
CVAT_CONVERTED_NAME = "cvat_converted"  # CVAT -> POLO conversion output
FEEDER_ONLY_NAME = "feeder_only"        # session-split, feeder-val/test variant

# ── POLO training hyperparameters ────────────────────────────────────────────
# Locked configuration for the DEPLOYED nano model (polo26n / polov8n).  These
# are the best hyperparameters from Johan's W&B sweep, applied to the nano
# architecture (see 03_train_polo.ipynb and docs/model-comparison.md).
#
# NOTE on the model config string: Johan's final runs used "polo26n.yaml"; an
# earlier mosaic/POLO build used "polov8n.yaml".  Confirm which the installed
# POLO fork (mooch443/POLO) accepts for the nano `locate` task and set
# POLO_MODEL_CFG accordingly.
POLO_MODEL_CFG = "polo26n.yaml"

POLO_FINAL = {
    "model": POLO_MODEL_CFG,
    "epochs": 200,
    "imgsz": 640,
    "batch": 8,
    "patience": 50,
    "loc_loss": "mse",
    "dor": 0.3,            # deployment/eval DoR (NOT the 0.8 used during the sweep)
    "loc": 4.86,
    "lr0": 0.0044,
    "lrf": 0.0072,
    "weight_decay": 0.000139,
    "augmentation": "medium",
}

# Sweep variants — the two architectures Johan tuned alongside nano.  Same
# locked hyperparameters; only the model config differs.  Used by the optional
# "selected retrain" cell in 03_train_polo.ipynb.
POLO_SWEEP_MODELS = {
    "polo26n": "polo26n.yaml",
    "polo26s": "polo26s.yaml",
    "polo26m": "polo26m.yaml",
}

# ── Localizer defaults ───────────────────────────────────────────────────────

LOCALIZER_DEFAULTS = {
    "epochs": 300,
    "batch_size": 128,
    "lr": 1e-3,
    "early_stopping_patience": 40,
    "lr_patience": 25,
    "freeze_encoder": False,
}

# ── Evaluation settings (fixed; reproduce the published numbers) ──────────────
IMGSZ = 640
CONF = 0.25                 # detection confidence threshold

# Near-detection suppression. We DON'T rely on POLO's internal DoR-NMS (its
# suppression distance is dor*radius, and the per-class radii come from a data.yaml
# baked into the checkpoint — which is absent off the training machine, collapsing
# the radius to ~1px and disabling suppression). Instead every model's RAW
# detections are de-duplicated with one explicit, class-agnostic, confidence-ranked
# radius NMS (point_nms) at NMS_RADIUS px. 30px == the deployed POLO setting
# (DoR 0.3 x 100px radius), so polo26n still reproduces F1 ~0.929.
NMS_RADIUS = 30.0           # explicit point-NMS suppression radius, pixels
DOR = 0.0                   # POLO internal DoR-NMS disabled (suppression is explicit)
MATCH_RADIUS = 75.0         # Hungarian GT-matching radius, pixels (distinct purpose)
LOCALIZER_THRESHOLD = 0.5   # localizer detection threshold
LOCALIZER_MIN_DISTANCE = 15.0  # localizer peak min-distance (finer than NMS_RADIUS)


# ── Generic helpers ──────────────────────────────────────────────────────────

def auto_device() -> str:
    """Return the best available device string for training/inference."""
    import torch

    if torch.cuda.is_available():
        return "0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ensure_absolute_data_yaml(data_yaml: Path) -> Path:
    """Rewrite ``data.yaml`` so its ``path`` key is absolute.

    ultralytics resolves the image directories relative to ``path``; making it
    absolute lets the data.yaml be used regardless of the working directory.
    """
    data_yaml = Path(data_yaml)
    with open(data_yaml) as f:
        data = yaml.safe_load(f)

    dataset_root = str(data_yaml.parent.resolve())
    if data.get("path") == dataset_root:
        return data_yaml

    data["path"] = dataset_root
    with open(data_yaml, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False)
    return data_yaml


# ── Dataset-path helpers (mosaic Dataset layout) ─────────────────────────────

def cvat_converted_dir(ds) -> Path:
    """Directory of the CVAT -> POLO converted dataset for a mosaic Dataset."""
    return ds.get_root("models") / POLO_MODEL_NAME / "_polo_data" / CVAT_CONVERTED_NAME


def feeder_only_dir(ds) -> Path:
    """Directory of the feeder-only (session-split) POLO dataset variant."""
    return ds.get_root("models") / POLO_MODEL_NAME / "_polo_data" / FEEDER_ONLY_NAME


# ── Point-detection metric primitives ────────────────────────────────────────

def load_gt(label_path: str | Path, img_w: int, img_h: int) -> list[tuple[float, float, int]]:
    """Load POLO ground-truth labels as ``(x, y, class_id)`` in pixel coords.

    POLO label format per line: ``class_id radius x_rel y_rel`` (relative coords).
    """
    label_path = Path(label_path)
    points: list[tuple[float, float, int]] = []
    if not label_path.exists():
        return points
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        class_id = int(parts[0])
        x = float(parts[2]) * img_w
        y = float(parts[3]) * img_h
        points.append((x, y, class_id))
    return points


def match(gt_points, pred_points, match_radius: float = MATCH_RADIUS):
    """Class-agnostic Hungarian matching of GT to predictions by 2-D distance.

    Returns ``(matched_gt_idx, matched_pred_idx, unmatched_gt_idx,
    unmatched_pred_idx)``.  A GT/pred pair counts as matched only if its optimal
    assignment distance is within ``match_radius`` pixels.
    """
    if not gt_points or not pred_points:
        return [], [], list(range(len(gt_points))), list(range(len(pred_points)))

    gt_arr = np.array([(p[0], p[1]) for p in gt_points])
    pred_arr = np.array([(p[0], p[1]) for p in pred_points])
    dists = np.linalg.norm(gt_arr[:, None] - pred_arr[None, :], axis=2)

    gt_idx, pred_idx = linear_sum_assignment(dists)
    matched_gt, matched_pred = [], []
    used_gt, used_pred = set(), set()
    for gi, pi in zip(gt_idx, pred_idx):
        if dists[gi, pi] <= match_radius:
            matched_gt.append(int(gi))
            matched_pred.append(int(pi))
            used_gt.add(int(gi))
            used_pred.add(int(pi))

    unmatched_gt = [i for i in range(len(gt_points)) if i not in used_gt]
    unmatched_pred = [i for i in range(len(pred_points)) if i not in used_pred]
    return matched_gt, matched_pred, unmatched_gt, unmatched_pred


def point_nms(points, radius: float = NMS_RADIUS, class_agnostic: bool = True):
    """Greedy, confidence-ranked radius suppression of point detections.

    ``points`` is a list of ``(x, y, class_id, conf)``.  Iterating from highest to
    lowest confidence, each kept detection suppresses all lower-confidence
    detections within ``radius`` pixels; returns the surviving tuples.  This is the
    single near-detection-suppression definition applied to every model in the eval
    (ported from the production ``point_nms`` in bb_pipeline).

    class_agnostic
        If True (default), suppress across all classes.  If False, only suppress
        detections sharing the same class.
    """
    if not points:
        return []
    xy = np.array([(p[0], p[1]) for p in points], dtype=float)
    conf = np.array([p[3] for p in points], dtype=float)
    cls = [p[2] for p in points]

    order = np.argsort(-conf)
    tree = cKDTree(xy)
    suppressed = np.zeros(len(points), dtype=bool)
    for idx in order:
        if suppressed[idx]:
            continue
        for nb in tree.query_ball_point(xy[idx], r=radius):
            if nb == idx or suppressed[nb]:
                continue
            if class_agnostic or cls[nb] == cls[idx]:
                suppressed[nb] = True
    return [points[i] for i in range(len(points)) if not suppressed[i]]


def polo_predict(model, img_path, imgsz: int = IMGSZ, conf: float = CONF, dor: float = DOR):
    """Run a POLO model on one image; return suppressed ``[(x, y, class_id, conf), ...]``.

    ``dor=0`` disables POLO's internal DoR-NMS (radii/data.yaml-independent), so we
    get the model's RAW detections, then apply the shared :func:`point_nms`.
    """
    results = model(str(img_path), imgsz=imgsz, conf=conf, dor=dor, verbose=False)
    preds: list[tuple[float, float, int, float]] = []
    for r in results:
        locs = getattr(r, "locations", None)
        if locs is None or len(locs) == 0:
            continue
        xy = locs.xy.cpu().numpy()
        confs = locs.conf.cpu().numpy()
        classes = locs.cls.cpu().numpy().astype(int)
        for i in range(len(classes)):
            preds.append((float(xy[i, 0]), float(xy[i, 1]), int(classes[i]), float(confs[i])))
    return point_nms(preds, NMS_RADIUS)


def localizer_predict(
    model,
    image,
    scale: float = LOCALIZER_SCALE_FACTOR,
    threshold: float = LOCALIZER_THRESHOLD,
    min_distance: float = LOCALIZER_MIN_DISTANCE,
    device: str | None = None,
):
    """Run the localizer on a (downscaled) image; coords mapped back to original.

    ``image`` is a BGR numpy array (as read by ``cv2.imread``).  Returns the
    suppressed ``[(x, y, class_id, conf), ...]`` in original-image pixel
    coordinates, de-duplicated with the same :func:`point_nms` used for POLO.
    """
    import cv2
    import torch
    from mosaic.tracking.pose_training.localizer_inference import detect_locations

    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"
    scaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    dets = detect_locations(
        model, scaled, thresholds=threshold, device=device, min_distance=min_distance,
    )
    preds = [
        (d["x"] / scale, d["y"] / scale, int(d["class_id"]), float(d.get("confidence", 1.0)))
        for d in dets
    ]
    return point_nms(preds, NMS_RADIUS)


def session_of(filename: str) -> str:
    """Recording session = filename up to ``__frame_`` (matches the split logic)."""
    return Path(filename).name.rsplit("__frame_", 1)[0]


def camera_of(filename: str) -> str:
    """Camera type from filename: ``"exit"`` if it contains ``exitcam`` else ``"feeder"``."""
    return "exit" if "exitcam" in Path(filename).name else "feeder"


@dataclass
class PointEvalResult:
    """Aggregated point-detection metrics over a set of images."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    gt_class_counts: np.ndarray = field(default_factory=lambda: np.zeros(NUM_CLASSES, int))
    detected_class_counts: np.ndarray = field(default_factory=lambda: np.zeros(NUM_CLASSES, int))
    correct_class_counts: np.ndarray = field(default_factory=lambda: np.zeros(NUM_CLASSES, int))
    confusion: np.ndarray = field(default_factory=lambda: np.zeros((NUM_CLASSES, NUM_CLASSES), int))
    per_image: list[dict] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def classification_accuracy(self) -> float:
        detected = int(self.detected_class_counts.sum())
        return float(self.correct_class_counts.sum()) / detected if detected else 0.0


def run_point_eval(image_paths, label_dir, predict_fn, match_radius: float = MATCH_RADIUS):
    """Evaluate a predictor over images, returning a :class:`PointEvalResult`.

    Parameters
    ----------
    image_paths : iterable of Path
        Images to evaluate (already filtered by split/camera).
    label_dir : Path
        Directory holding the matching ``<stem>.txt`` POLO label files.
    predict_fn : callable
        ``predict_fn(img_path, image_bgr) -> [(x, y, class_id, conf), ...]``.
        Receives both the path and the loaded BGR image (some predictors need
        the array, some the path).
    match_radius : float
        Hungarian matching radius in pixels.
    """
    import cv2

    label_dir = Path(label_dir)
    res = PointEvalResult()
    for img_path in image_paths:
        img_path = Path(img_path)
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]

        gt_points = load_gt(label_dir / (img_path.stem + ".txt"), w, h)
        pred_points = predict_fn(img_path, image)

        m_gt, m_pred, u_gt, u_pred = match(gt_points, pred_points, match_radius)
        tp, fp, fn = len(m_gt), len(u_pred), len(u_gt)
        res.tp += tp
        res.fp += fp
        res.fn += fn

        for gi in range(len(gt_points)):
            res.gt_class_counts[gt_points[gi][2]] += 1
        for gi, pi in zip(m_gt, m_pred):
            gc = gt_points[gi][2]
            pc = pred_points[pi][2]
            res.detected_class_counts[gc] += 1
            res.confusion[gc][pc] += 1
            if gc == pc:
                res.correct_class_counts[gc] += 1

        res.per_image.append({
            "name": img_path.name,
            "session": session_of(img_path.name),
            "camera": camera_of(img_path.name),
            "tp": tp, "fp": fp, "fn": fn, "errors": fp + fn,
            "n_gt": len(gt_points), "n_pred": len(pred_points),
        })
    return res
