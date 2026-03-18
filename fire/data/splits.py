"""Deterministic train/calibration/test splitting utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .constants import CALIBRATION_FRACTION, TRAIN_FRACTION, VALID_SPLITS


class SplitError(ValueError):
    """Raised when an invalid split is requested."""


def build_split_indices(
    n_samples: int,
    seed: int,
    train_fraction: float = TRAIN_FRACTION,
    calibration_fraction: float = CALIBRATION_FRACTION,
) -> dict[str, np.ndarray]:
    """Builds deterministic train/calibration/test indices."""
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be in (0, 1)")
    if train_fraction + calibration_fraction >= 1:
        raise ValueError("train_fraction + calibration_fraction must be < 1")

    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples, dtype=np.int64)
    rng.shuffle(indices)

    n_train = int(n_samples * train_fraction)
    n_cal = int(n_samples * calibration_fraction)
    n_test = n_samples - n_train - n_cal

    train = np.sort(indices[:n_train])
    calibration = np.sort(indices[n_train : n_train + n_cal])
    test = np.sort(indices[n_train + n_cal : n_train + n_cal + n_test])

    return {
        "train": train,
        "calibration": calibration,
        "test": test,
    }


def require_split(split: str) -> str:
    """Validates and returns split name."""
    normalized = split.strip().lower()
    if normalized not in VALID_SPLITS:
        valid = ", ".join(VALID_SPLITS)
        raise SplitError(f"Unknown split '{split}'. Expected one of: {valid}.")
    return normalized


def save_split_indices(split_indices: dict[str, np.ndarray], path: str | Path) -> None:
    """Saves split indices to JSON for reproducibility/auditing."""
    payload = {key: value.tolist() for key, value in split_indices.items()}
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


def load_split_indices(path: str | Path) -> dict[str, np.ndarray]:
    """Loads split indices from a JSON manifest."""
    payload = json.loads(Path(path).read_text())
    return {key: np.asarray(value, dtype=np.int64) for key, value in payload.items()}
