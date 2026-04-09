# POLO Feeder Model Training -- Raw Notes for Report

## 1. Project Overview

- POLO = point-detection variant of YOLO (ultralytics), task="locate"
- Uses ultralytics YOLO framework with `task="locate"` (point localization, not bounding boxes)
- Deployment target: feeder cam for the beesbook project
- 4 classes with uniform detection radii (100px each):
  - 0: UnmarkedBee
  - 1: MarkedBee
  - 2: BeeInCell
  - 3: UpsideDownBee
- Training script: `train_polo_sweep.py` (W&B Bayesian sweep)
- Dataset construction: `build_feeder_only_dataset.py` (symlinks to NFS)
- Split generation: `generate_split.py` (session-based splitting)
- Manual validation: `render_validation_data.py` (Hungarian matching evaluation)

## 2. Dataset Details

- Source NFS path: `/mnt/trove/beesbook_feeder_model/feeder_bee_datasets_v1/models/polo/cvat_only`
- Images: 1920x646 color PNG frames, human-annotated in CVAT
- Total: 1246 images

### Split Assignment (from split_assignment.json)

| cam_type | split | count |
|----------|-------|-------|
| exit     | train | 189   |
| feeder   | train | 788   |
| feeder   | valid | 133   |
| feeder   | test  | 136   |

- Train total: 977 (78.4%) -- 788 feeder + 189 exit cam
- Valid total: 133 (10.7%) -- feeder only
- Test total: 136 (10.9%) -- feeder only
- Grand total: 1246

### Split Design Decisions

- Exit cam images go to training only (extra learning data, but not the deployment domain)
- Valid and test are feeder-cam only (matches deployment scenario)
- Split by recording session (filename prefix before `__frame_`) to avoid temporal leakage
- Target ratios: ~75% train, ~12.5% valid, ~12.5% test (feeder images only)
- Seed: 42 for reproducibility
- Previous "merged" variant included 128x128 HDF5 patches that polluted metrics (wrong domain) -- this dataset is the cleaned "cvat_only" variant

## 3. Fixed Training Settings

| Parameter   | Value  |
|-------------|--------|
| EPOCHS      | 200    |
| IMGSZ       | 640    |
| BATCH       | 8      |
| PATIENCE    | 50     |
| LOC_LOSS    | "mse"  |
| DOR         | 0.3    |

- Output directory: `~/runs/polo_sweep`
- Data YAML: `data/feeder_only/data.yaml`
- F1 metric injected into trainer.metrics before each epoch (workaround for ultralytics W&B callback calling `run.finish()` too early)
- Retry logic: up to 3 retries for transient errors (NFS/CUDA), 60s delay between attempts

## 4. Sweep Configuration

- Method: Bayesian optimization
- Metric: maximize `val/f1`
- W&B Project: `beesbook-feeder`
- Sweep ID: `aj5kmrkb`
- URL: https://wandb.ai/johanidler-org/beesbook-feeder/sweeps/aj5kmrkb

### Swept Parameters

| Parameter    | Type               | Range / Values                    |
|--------------|--------------------|-----------------------------------|
| model        | categorical        | polo26s.yaml, polo26m.yaml        |
| loc          | continuous         | 3.0 -- 7.0                        |
| lr0          | log_uniform_values | 1e-3 -- 3e-2                      |
| lrf          | log_uniform_values | 1e-3 -- 2e-2                      |
| weight_decay | log_uniform_values | 1e-4 -- 2e-3                      |
| augmentation | categorical        | medium, heavy                     |

### Rationale for Ranges

- From prior sweep: polo26m + medium/heavy aug + lr0 0.003-0.015 performed best
- This sweep tightened ranges accordingly
- Both polo26s and polo26m included to compare model sizes

## 5. W&B Sweep Results

- Sweep state: RUNNING (as of 2026-04-09)
- Total runs: 23
- Best run: `cool-sweep-21` (ID: f8gyznbr) -- state: failed (but metrics logged before failure)
- Best run model parameters: 24,548,050 (polo26m), 87.99 GFLOPs, 9.2ms PyTorch inference

### All Runs Sorted by val/F1

| Rank | Run Name               | F1     | Model    | Aug    | lr0      | lrf      | weight_decay | loc  | State    |
|------|------------------------|--------|----------|--------|----------|----------|--------------|------|----------|
| 1    | cool-sweep-21          | 0.8538 | polo26m  | medium | 0.00302  | 0.00106  | 0.000282     | 4.32 | failed   |
| 2    | proud-sweep-15         | 0.8528 | polo26s  | medium | 0.00206  | 0.00292  | 0.000470     | 6.62 | finished |
| 3    | zesty-sweep-2          | 0.8523 | polo26m  | heavy  | 0.01007  | 0.00362  | 0.001528     | 4.14 | finished |
| 4    | genial-sweep-11        | 0.8514 | polo26s  | medium | 0.01315  | 0.01936  | 0.001546     | 6.75 | finished |
| 5    | soft-sweep-8           | 0.8513 | polo26s  | heavy  | 0.00135  | 0.01034  | 0.000353     | 5.27 | finished |
| 6    | sandy-sweep-22         | 0.8511 | polo26s  | medium | 0.00776  | 0.00150  | 0.001299     | 5.29 | finished |
| 7    | happy-sweep-17         | 0.8506 | polo26m  | heavy  | 0.00414  | 0.00682  | 0.001399     | 4.18 | finished |
| 8    | wise-sweep-12          | 0.8503 | polo26m  | medium | 0.00304  | 0.00215  | 0.000105     | 3.32 | finished |
| 9    | sparkling-sweep-18     | 0.8500 | polo26m  | medium | 0.00307  | 0.01250  | 0.000752     | 6.83 | finished |
| 10   | young-sweep-19         | 0.8497 | polo26s  | medium | 0.00239  | 0.00308  | 0.000928     | 6.06 | finished |
| 11   | denim-sweep-1          | 0.8497 | polo26s  | medium | 0.00188  | 0.00478  | 0.000659     | 3.66 | finished |
| 12   | worthy-sweep-4         | 0.8496 | polo26s  | medium | 0.00441  | 0.00722  | 0.000139     | 4.86 | finished |
| 13   | pious-sweep-13         | 0.8489 | polo26s  | medium | 0.00122  | 0.00140  | 0.001889     | 3.84 | finished |
| 14   | robust-sweep-7         | 0.8486 | polo26m  | medium | 0.00110  | 0.00484  | 0.000556     | 5.12 | finished |
| 15   | polished-sweep-23      | 0.8474 | polo26m  | heavy  | 0.00258  | 0.00285  | 0.000137     | 3.31 | crashed  |
| 16   | volcanic-sweep-9       | 0.8471 | polo26m  | medium | 0.00140  | 0.01215  | 0.000972     | 4.18 | finished |
| 17   | misunderstood-sweep-5  | 0.8465 | polo26s  | heavy  | 0.01655  | 0.00187  | 0.000128     | 5.64 | finished |
| 18   | toasty-sweep-3         | 0.8457 | polo26m  | heavy  | 0.00420  | 0.00957  | 0.000402     | 6.65 | finished |
| 19   | frosty-sweep-20        | 0.8438 | polo26m  | heavy  | 0.02812  | 0.00125  | 0.001703     | 6.65 | finished |
| 20   | pleasant-sweep-6       | 0.8425 | polo26m  | heavy  | 0.00878  | 0.00227  | 0.000885     | 3.91 | finished |
| 21   | lilac-sweep-14         | 0.8420 | polo26s  | heavy  | 0.00477  | 0.00465  | 0.001288     | 6.60 | finished |
| 22   | whole-sweep-16         | 0.8416 | polo26m  | heavy  | 0.00511  | 0.00483  | 0.001794     | 4.22 | finished |
| 23   | snowy-sweep-10         | 0.8413 | polo26m  | heavy  | 0.00115  | 0.00725  | 0.000438     | 5.94 | failed   |

### Observations from Sweep Results

- F1 range across all runs: 0.841 -- 0.854 (very narrow, ~1.3 percentage points spread)
- Best run (cool-sweep-21) metrics: P=0.999, R=0.746, mAP100=0.868
- Both polo26s and polo26m appear in top positions -- no clear winner on model size
- "medium" augmentation tends to appear more in top runs vs "heavy"
- lr0 sweet spot roughly 0.002--0.013
- 2 runs failed, 1 crashed (out of 23 total)
- 20 runs finished successfully

### Best Run Details (cool-sweep-21)

- Precision(L): 0.99855
- Recall(L): 0.74571
- mAP100(L): 0.86767
- val/f1: 0.8538
- val/cls_loss: 0.1647
- val/loc_loss: 0.14266
- Model: polo26m.yaml (24.5M params, 88 GFLOPs)
- Inference speed: 9.2ms (PyTorch)
- Runtime: ~4.5 hours (16053 seconds)

## 6. Key Finding: Metrics Discrepancy (Sweep F1 vs Real-World F1)

### The Problem

- Sweep reported best F1 ~ 0.854
- Manual evaluation on the SAME validation set showed F1 = 0.988
- Gap: ~13 percentage points

### Root Causes

1. **DOR (Detection-Overlap-Radius) mismatch**: During the sweep, DOR was set to 0.8 (in earlier runs before the fix), which is too aggressive for NMS during validation. This suppresses nearby detections that are actually correct. The deployment default is DOR=0.3.
   - NOTE: The current sweep config shows DOR=0.3 fixed. The discrepancy may stem from earlier runs or from how ultralytics internally evaluates during training vs the manual evaluation approach.
2. **Matching method**: Ultralytics uses class-aware NMS-based matching. The manual evaluation (`render_validation_data.py`) uses Hungarian matching (scipy `linear_sum_assignment`) with a 75px match radius, which is class-agnostic for spatial matching.
3. **Confidence threshold**: Manual eval uses conf=0.25; training validation may use different thresholds.
4. **The very high precision (0.999) but lower recall (0.746) in sweep metrics** suggests NMS is being too aggressive -- removing valid predictions that are close to each other.

### Resolution

- DOR set to 0.3 in sweep config to match deployment defaults
- Manual validation confirms the model is actually performing much better than sweep metrics suggest

## 7. Manual Validation Results

- Script: `render_validation_data.py`
- Model used: `/home/johan/runs/polo_sweep/sweep_20260409_033533/weights/best.pt`
- Validation set: 133 feeder-only images from `data/feeder_only/valid/images`
- Match radius: 75px (Hungarian matching via `scipy.optimize.linear_sum_assignment`)
- Confidence threshold: 0.25
- DOR: 0.3
- IMGSZ: 640

### Image-Level Results

| Category    | Count | Description              |
|-------------|-------|--------------------------|
| Aligned     | 118   | 0 errors (perfect match) |
| Close       | 9     | 1 error (FP or FN)       |
| Misaligned  | 6     | 2+ errors                |
| Total       | 133   |                          |

- 88.7% of images have zero errors
- 95.5% of images have at most 1 error

### Detection-Level Results

| Metric    | Value |
|-----------|-------|
| TP        | 943   |
| FP        | 10    |
| FN        | 12    |
| Precision | 0.990 |
| Recall    | 0.987 |
| F1        | 0.988 |

### Rendered output

- Aligned images: `rendered_validation/aligned/`
- Close images (1 error): `rendered_validation/close/`
- Misaligned images (2+ errors): `rendered_validation/misaligned/`
- Color coding: cyan=TP, magenta=FN, red=FP

## 8. Architecture & Infrastructure Notes

- Dataset uses symlinks from local `data/feeder_only/` to NFS (NFS is read-only)
- Output goes to `~/runs/polo_sweep` (local to each machine)
- Multiple machines can join the same sweep: `python train_polo_sweep.py <sweep_id>`
- W&B integration via ultralytics callback with custom F1 injection hack
- VRAM requirement: 8 GB (batch=8 at imgsz=640)
- `config.py` provides `auto_device()` and `ensure_absolute_data_yaml()` utilities

## 9. Key Files

| File                          | Purpose                                           |
|-------------------------------|---------------------------------------------------|
| train_polo_sweep.py           | W&B Bayesian sweep driver                         |
| generate_split.py             | Session-based train/valid/test split generation    |
| build_feeder_only_dataset.py  | Symlink-based dataset construction from NFS        |
| render_validation_data.py     | Visual validation with Hungarian matching          |
| split_assignment.json         | Per-image split assignments (1246 entries)          |
| data/feeder_only/data.yaml    | Ultralytics dataset config (generated)             |
| config.py                     | Shared device/path utilities                       |

## 10. Open Questions / TODO

- Sweep is still RUNNING -- more runs may complete
- Best run (cool-sweep-21) has state "failed" -- metrics were logged before failure, but should investigate if it completed training or failed mid-way
- polished-sweep-23 crashed -- infrastructure issue?
- Per-class evaluation not yet done (manual eval is class-agnostic for matching)
- No test-set evaluation reported yet (only validation)
- Consider whether polo26s is sufficient given similar F1 to polo26m (fewer params, faster inference)
