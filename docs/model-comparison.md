# Model & Hyperparameter Comparison

Static reference for the paper: the architectures and hyperparameters compared
when selecting the deployed feeder bee detector, and the headline results.
These numbers are reproduced by [`notebooks/03_evaluation.ipynb`](../notebooks/03_evaluation.ipynb)
(confidence ≥ 0.25 → explicit class-agnostic 30 px `point_nms` de-duplication →
class-agnostic Hungarian matching at 75 px — one suppression definition shared by
POLO and the localizer); this file records the final selection for citation.

## What was compared

POLO is a point-detection fork of YOLO (ultralytics) that predicts bee
**locations** as single points with class-specific radii — the `locate` task —
rather than bounding boxes. Selection proceeded in three stages:

1. **Architecture** — a YOLOv8n baseline (no augmentation, F1 ≈ 0.49) → the
   `polo26` family (nano / small / medium).
2. **Hyperparameters** — a Bayesian W&B sweep over `polo26s` / `polo26m`,
   converging on *medium* augmentation and a tight learning-rate / loc-loss band
   (sweep F1 0.841–0.854, no clear winner between small and medium).
3. **Model size** — all three sizes retrained with the best hyperparameters and
   evaluated on the same feeder-only val/test sets.

The legacy 2019 BeesBook **localizer** (a ~1.0M-param heatmap model) is included
as the pre-POLO baseline.

## Dataset

Feeder-only split, by recording session (no temporal leakage). Exit-cam images
are training-only; validation and test are feeder-only to match deployment.

| Split | Feeder | Exit | Total | Role |
|-------|-------:|-----:|------:|------|
| Train | 788 | 189 | 977 | model fits on this |
| Valid | 133 | 0 | 133 | sweep optimised against |
| Test  | 136 | 0 | 136 | held out |

Frozen in [`split_assignment.json`](../split_assignment.json) (1246 entries, seed 42).

## Augmentation presets

| Preset | hsv_h | hsv_s | hsv_v | degrees | translate | scale | mosaic | mixup |
|--------|------:|------:|------:|--------:|----------:|------:|-------:|------:|
| light  | 0.015 | 0.4 | 0.4 | 5  | 0.05 | 0.2 | 0.0 | 0.0 |
| **medium** | 0.015 | 0.5 | 0.3 | 15 | 0.15 | 0.4 | 1.0 | 0.1 |
| heavy  | 0.015 | 0.7 | 0.4 | 25 | 0.20 | 0.5 | 1.0 | 0.15 |

*Medium* augmentation dominated the top sweep runs. (light/heavy values are
indicative of the preset family; medium is the one used for the final models.)

## Sweep search space (Bayesian, metric = val F1)

| Parameter | Range / values |
|-----------|----------------|
| `model` | `polo26s.yaml`, `polo26m.yaml` |
| `loc` (localization loss weight) | 3.0 – 7.0 |
| `lr0` | 1e-3 – 3e-2 (log) |
| `lrf` | 1e-3 – 2e-2 (log) |
| `weight_decay` | 1e-4 – 2e-3 (log) |
| `augmentation` | medium, heavy |
| *fixed* | epochs 200, imgsz 640, batch 8, patience 50, loc_loss mse, DoR 0.3 |

> **DoR caveat.** Several sweep runs logged validation F1 at **DoR 0.8**, which
> depresses F1 to ≈ 0.85. Deployment and all numbers below use **DoR 0.3**.

## Final locked hyperparameters (deployed `polo26n`)

Defined in [`config.py`](../config.py) as `POLO_FINAL`:

| Param | Value | Param | Value |
|-------|-------|-------|-------|
| model | `polo26n.yaml` | lr0 | 0.0044 |
| epochs | 200 | lrf | 0.0072 |
| imgsz | 640 | weight_decay | 0.000139 |
| batch | 8 | loc | 4.86 |
| patience | 50 | loc_loss | mse |
| augmentation | medium | dor (deploy/eval) | 0.30 |

## Old localizer vs POLO

| Model | Split | F1 | P | R | Class. acc. |
|-------|-------|---:|---:|---:|---:|
| localizer (2019) | test | 0.519 | 0.532 | 0.507 | 83.8% |
| polo26n | test | 0.929 | 0.941 | 0.917 | 99.9% |
| localizer (2019) | valid | 0.531 | 0.507 | 0.557 | 86.8% |
| polo26n | valid | 0.990 | 0.990 | 0.990 | 99.9% |

POLO nearly doubles F1 and lifts classification from ≈ 85% to ≈ 100%.

## Model-size comparison

**Test set (136 images)**

| Model | F1 | P | R | Class. acc. |
|-------|---:|---:|---:|---:|
| polo26n | 0.929 | 0.941 | 0.917 | 99.9% |
| polo26s | 0.932 | 0.942 | 0.922 | 99.9% |
| polo26m | 0.926 | 0.921 | 0.931 | 99.8% |

**Validation set (133 images) + cost**

| Model | F1 | P | R | Class. acc. | Params | GFLOPs | GPU ms/img¹ | Weight |
|-------|---:|---:|---:|---:|-------:|-------:|------------:|-------:|
| polo26n | 0.990 | 0.990 | 0.990 | 99.9% | 3.0 M | 6.1 | 5.9 (4.1×) | 18.6 MB |
| polo26s | 0.989 | 0.990 | 0.987 | 100.0% | 11.1 M | 22.8 | 11.1 (2.2×) | 67.3 MB |
| polo26m | 0.983 | 0.981 | 0.985 | 100.0% | 24.5 M | 75.4 | 24.1 (1.0×) | 148 MB |

¹ RTX 3050, imgsz 640. Speed is hardware-specific — re-benchmark and cite the
GPU used. The deployed/exported `polo26n` checkpoint
(`bb_hpc_dev/polo26_feedercams.pt`) is ~5.9 MB.

**Selection.** Model size makes essentially no difference to quality (test F1
spread 0.6 pp; classification 99.8–99.9%). The val→test gap (≈ 0.99 → 0.93) is
detection, not classification, and is largely driven by one crowded,
poorly-lit session whose dense clusters are hard even to annotate. **`polo26n`**
is deployed: it matches the larger models while being 4.1× faster and 3.6×
smaller.

## Classification

Near-perfect across all models (99.8–99.9%). `UnmarkedBee` (~97% of annotations)
is essentially 100% accurate; `MarkedBee` (~2.5%) is well handled (28/29 detected,
27/28 correctly classified on test). `BeeInCell` and `UpsideDownBee` have no
feeder-only samples and remain untested on feeder cameras.

---

*Source: Johan's training report and W&B sweep notes. Reproduce/refresh any
number with `notebooks/03_evaluation.ipynb` pointed at the final dataset +
weights.*
