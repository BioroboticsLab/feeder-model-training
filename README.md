# Feeder Model Training

End-to-end training and evaluation for the **feeder bee detector** used by the
BeesBook system. The pipeline goes from raw video to a deployable
point-detection model and a reproducible evaluation:

```
video → frames → CVAT annotation → POLO labels → feeder-only split → train → evaluate
```

The detector is **POLO** — a point-detection fork of YOLO (ultralytics) whose
`locate` task predicts bee locations as single points with class-specific radii.
The deployed model is `polo26n`. The legacy 2019 BeesBook **localizer** is kept
as a baseline for comparison. Four classes: `UnmarkedBee`, `MarkedBee`,
`BeeInCell`, `UpsideDownBee`.

[`mosaic`](https://github.com/ecodylicscience/mosaic) does the heavy lifting
(frame extraction, CVAT conversion, POLO/localizer training); this repo is the
thin, reproducible workflow on top of it.

## Workflow (notebooks)

Run in order; each notebook has a `DATASET_BASE` config cell at the top.

| Notebook | Does |
|----------|------|
| [`notebooks/01_frame_extraction_and_annotation_prep.ipynb`](notebooks/01_frame_extraction_and_annotation_prep.ipynb) | index videos + k-means frame sampling, stage for CVAT, convert CVAT XML → POLO labels, build the feeder-only split |
| [`notebooks/02_train_polo.ipynb`](notebooks/02_train_polo.ipynb) | train deployed `polo26n`; optional n/s/m retrain, sweep reference, localizer baseline |
| [`notebooks/03_evaluation.ipynb`](notebooks/03_evaluation.ipynb) | **definitive evaluation** — the paper's numbers |

[`config.py`](config.py) is the single shared module: classes, radii, the locked
training hyperparameters (`POLO_FINAL`), the fixed evaluation settings, and the
point-detection helpers (`load_gt`, `point_nms`, `match`, `polo_predict`,
`localizer_predict`, `run_point_eval`).

## Setup

Requires Python ≥ 3.10 and (for training) a CUDA GPU.

```bash
pip install -e ".[notebooks,wandb]"
# pulls mosaic-behavior[polo,localizer] from git, plus scipy/matplotlib/pandas
```

`polo` and `pose` extras both ship under the `ultralytics` name and conflict —
this repo needs `[polo]` (the mooch443/POLO fork providing the `locate` task),
which `mosaic-behavior[polo,localizer]` already pins.

## Definitive evaluation

[`03_evaluation.ipynb`](notebooks/03_evaluation.ipynb) is the single source of
truth for the paper. Each model's raw detections (confidence ≥ 0.25) are
de-duplicated with **one explicit, class-agnostic, confidence-ranked radius NMS**
(`config.point_nms`, **30 px**), then scored against ground truth by
**class-agnostic Hungarian matching** (**75 px**). The 30 px suppression equals the
deployed POLO setting (DoR 0.3 × 100 px radius) and is applied uniformly to POLO and
the localizer, so the comparison is apples-to-apples (we don't use POLO's internal
DoR-NMS, which needs a `data.yaml` baked into the checkpoint). It produces three
outputs:

1. **Old localizer vs POLO**
2. **Model-size comparison** (polo26n / s / m): F1, P/R, classification, params, speed
3. **Deployed-model deep dive** (polo26n): per-class, confusion, per-session,
   per-image error histogram, failure gallery

Configure it at the top:

- `SPLIT` — `test` (default) | `valid` | `train`
- `CAMERA_FILTER` — `feeder` (default, deployment) | `exit` | `all`
  - **Caveat:** exit-cam images are *train-only* in the split, so evaluating on
    them is **not** held out. The notebook flags this.
- `MODELS` / `LOCALIZER_MODELS` — paths to trained weights. Any model left
  `None` falls back to Johan's reported numbers, shown labelled `reported`
  alongside the recomputed rows for cross-check.

The deployment-matched run is `SPLIT='test'`, `CAMERA_FILTER='feeder'`; it should
reproduce **test F1 ≈ 0.929** (P 0.941 / R 0.917, classification ≈ 99.9%), and
`SPLIT='valid'` → **F1 ≈ 0.990**. See [`docs/model-comparison.md`](docs/model-comparison.md)
for the full hyperparameter/architecture tables and headline numbers.

## Data & weights you provide

Neither datasets nor weights are committed. Point the notebooks at:

- a **mosaic `Dataset`** (the `DATASET_BASE` root) with the feeder media /
  CVAT-converted labels;
- trained **weights** for `04` — the final `polo26n` (deployed:
  `bb_hpc_dev/polo26_feedercams.pt`), and optionally `polo26s` / `polo26m` /
  the localizer to recompute those rows.

The published numbers come from the **final feeder-only dataset** (1246 images,
test 136) and the final weights on the training machine. An earlier local
snapshot is useful only for smoke-testing the pipeline wiring (smaller split,
no exit cams) — its numbers will not match the report.

## Reproducibility notes

- The split is **frozen** in `split_assignment.json`; rebuilding with seed 42 only
  matches given the identical CVAT image set.
- Evaluation settings are fixed in `config.py` (imgsz 640, conf 0.25, explicit
  `point_nms` suppression at 30 px, Hungarian match radius 75 px). POLO's internal
  DoR-NMS is disabled (`DOR = 0.0`) in favor of the shared `point_nms`.
- **Training DoR caveat:** during the *sweep*, val F1 was logged at DoR 0.8 (≈ 0.85);
  the deployed model and the 30 px explicit-NMS eval give test ≈ 0.93 / val ≈ 0.99.
- **Model config string:** `config.POLO_MODEL_CFG = "polo26n.yaml"`. An earlier
  POLO build used `polov8n.yaml` — switch it if the installed POLO fork rejects
  the config.
- Inference speed is hardware-specific; `04` records the device — cite the GPU.

## Classes

| ID | Name | Notes |
|----|------|-------|
| 0 | UnmarkedBee | most common (~97%) |
| 1 | MarkedBee | rare on feeder cams (~2.5%), more common on exit cams |
| 2 | BeeInCell | no feeder-cam samples (untested on feeders) |
| 3 | UpsideDownBee | bees walking upside down on the feeder |
