# Wildfire Pipeline: Phase 1 + Phase 2 + Phase 3 (Spatial Baseline)

This repository currently supports:
- Download NDWS data from Kaggle.
- Build two separate loaders from the same data source.
- Enforce deterministic 70/15/15 Train/Calibration/Test splits with seed `42`.
- Train a LightGBM tabular baseline and export probabilities for calibration/test.
- Train a Tiny U-Net spatial baseline and export probability heatmaps for calibration/test.

## 1. Install Dependencies

```bash
uv sync
```

If running on macOS with Apple Silicon, LightGBM may also need OpenMP:

```bash
brew install libomp
```

## 2. Download Dataset (Kaggle)

```bash
uv run fire-download-ndws --dataset fantineh/next-day-wildfire-spread --output-dir data/raw
```

Prerequisite:
- Kaggle credentials configured (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` + `KAGGLE_KEY`).
- Either Kaggle CLI (`kaggle`) or Python package (`pip install kaggle`).
- Optional: set `KAGGLE_CONFIG_DIR` if you want credentials read from a custom path.

## 3. Prepare Splits + Smoke Test Both Loaders

```bash
uv run fire-prepare-phase1 \
  --data-root data/raw \
  --input-format auto \
  --split-manifest data/splits/ndws_seed42_70_15_15.json \
  --tabular-sample-rate 0.05
```

What this validates:
- `train/calibration/test = 70/15/15` sample split.
- Module 1A tabular flattening with **5% pixel sampling**.
- Module 1B PyTorch spatial `Dataset` + `DataLoader` output shapes.

## 4. Phase 2: Train Tabular Baseline (LightGBM)

```bash
uv run fire-train-tabular-baseline \
  --data-root data/raw \
  --input-format auto \
  --split-manifest data/splits/ndws_seed42_70_15_15.json \
  --train-pixel-sample-rate 0.05 \
  --eval-pixel-sample-rate 0.05 \
  --output-dir data/predictions/tabular_baseline
```

Outputs:
- `data/predictions/tabular_baseline/lightgbm_model.pkl`
- `data/predictions/tabular_baseline/calibration_probabilities.csv`
- `data/predictions/tabular_baseline/test_probabilities.csv`
- `data/predictions/tabular_baseline/run_metadata.json`

`calibration_probabilities.csv` and `test_probabilities.csv` contain per-pixel:
- `split`
- `sample_id`
- `row`, `col`
- `target` (ground truth binary label)
- `probability` (positive class from `predict_proba()`)

## 5. Phase 3: Train Spatial Baseline (Tiny U-Net)

```bash
uv run fire-train-spatial-baseline \
  --data-root data/raw \
  --input-format auto \
  --split-manifest data/splits/ndws_seed42_70_15_15.json \
  --epochs 10 \
  --batch-size 8 \
  --output-dir data/predictions/spatial_baseline
```

Outputs:
- `data/predictions/spatial_baseline/tiny_unet_checkpoint.pt`
- `data/predictions/spatial_baseline/calibration_probability_heatmaps.npz`
- `data/predictions/spatial_baseline/test_probability_heatmaps.npz`
- `data/predictions/spatial_baseline/run_metadata.json`

Each `*_probability_heatmaps.npz` contains:
- `sample_ids`
- `targets` with shape `(N, 64, 64)`
- `probabilities` with shape `(N, 64, 64)` and values in `[0.0, 1.0]`

For a focused usage guide, see `guide/train_unet.md`.

## Module Overview

- `fire/data/tabular_loader.py`:
  - Flattens each `64x64` map to per-pixel rows.
  - Randomly samples 5% of pixels per sample.
  - Returns `pandas.DataFrame` or NumPy payload.

- `fire/data/spatial_loader.py`:
  - `SpatialWildfireDataset` returns tensors as:
    - Inputs: `(C, 64, 64)`
    - Targets: `(1, 64, 64)`
  - Standard PyTorch `DataLoader` factory.

- `fire/data/readers.py`:
  - Supports `.tfrecord/.tfrecords` and `.tif/.tiff` map ingestion.

- `fire/data/splits.py`:
  - Deterministic split utility (seed `42`) and split manifest persistence.

- `fire/models/tabular_baseline.py`:
  - Trains `lightgbm.LGBMClassifier` on tabular pixel rows from train split.
  - Never trains on calibration/test.
  - Uses `.predict_proba()` and saves calibration/test probabilities.

- `fire/models/spatial_baseline.py`:
  - Defines a shallow Tiny U-Net (2 downsampling blocks) in PyTorch.
  - Uses BCE loss with Sigmoid final output.
  - Saves calibration/test probability heatmaps for CRC and safety evaluation.
