# Train Tiny U-Net (Phase 3)

This guide explains how to run the spatial baseline trainer when you are ready.

## Purpose

`fire-train-spatial-baseline` trains a shallow Tiny U-Net on the **train split only** and exports:

- `calibration_probability_heatmaps.npz`
- `test_probability_heatmaps.npz`

These exports contain raw probability maps (`0.0` to `1.0`) required for downstream conformal calibration and safety evaluation.

## Prerequisites

- Phase 1 split manifest exists, typically:
  - `data/splits/ndws_seed42_70_15_15.json`
- Dataset is available under `data/raw`.
- Dependencies installed:

```bash
uv sync
```

## Run Training

```bash
uv run fire-train-spatial-baseline \
  --data-root data/raw \
  --input-format auto \
  --split-manifest data/splits/ndws_seed42_70_15_15.json \
  --seed 42 \
  --epochs 10 \
  --batch-size 8 \
  --learning-rate 1e-3 \
  --output-dir data/predictions/spatial_baseline
```

## Main Arguments

- `--epochs`: Number of training epochs.
- `--batch-size`: Mini-batch size for train/calibration/test loaders.
- `--base-channels`: Model width for Tiny U-Net (default `32`).
- `--device`: `auto`, `cpu`, `cuda`, or `mps`.
- `--num-workers`: DataLoader workers (default `0` for deterministic behavior).

## Artifacts

After training, the output directory includes:

- `tiny_unet_checkpoint.pt`
- `calibration_probability_heatmaps.npz`
- `test_probability_heatmaps.npz`
- `run_metadata.json`

Both NPZ files include:

- `sample_ids`: NDWS sample identifier per map.
- `targets`: Binary target maps, shape `(N, 64, 64)`.
- `probabilities`: Model probability maps, shape `(N, 64, 64)`.

## Notes

- The architecture is intentionally small (2 downsampling blocks) to keep training fast.
- A fixed seed (`42` default) is used for reproducibility.
- Calibration and test splits are never used for parameter updates.
