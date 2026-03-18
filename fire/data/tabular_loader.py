"""Module 1A: tabular loader with 5% pixel sampling."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .constants import DEFAULT_TARGET_FEATURE
from .readers import SampleBackend


def build_tabular_split(
    backend: SampleBackend,
    split_indices: Sequence[int] | np.ndarray,
    feature_names: list[str],
    pixel_sample_rate: float = 0.05,
    seed: int = 42,
    as_dataframe: bool = True,
    include_coords: bool = True,
    target_name: str = DEFAULT_TARGET_FEATURE,
):
    """Builds a tabular split by flattening maps and sampling pixels.

    Output rows represent sampled pixels. Sampling is done per-map, preserving
    the requested `pixel_sample_rate` while avoiding full 64x64 expansion.
    """
    if not 0 < pixel_sample_rate <= 1:
        raise ValueError("pixel_sample_rate must be in (0, 1].")

    split_indices = np.asarray(split_indices, dtype=np.int64)
    if split_indices.size == 0:
        raise ValueError("split_indices is empty")

    rng = np.random.default_rng(seed)

    feature_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    sample_id_blocks: list[np.ndarray] = []
    row_blocks: list[np.ndarray] = []
    col_blocks: list[np.ndarray] = []

    for idx in split_indices:
        inputs_hwc, target_hw, sample_id = backend.get_sample(int(idx))

        h, w, c = inputs_hwc.shape
        if c != len(feature_names):
            raise ValueError(
                f"Feature count mismatch for sample {sample_id}: got {c}, expected {len(feature_names)}"
            )

        n_pixels = h * w
        n_keep = max(1, int(round(n_pixels * pixel_sample_rate)))
        chosen = rng.choice(n_pixels, size=n_keep, replace=False)

        flat_features = inputs_hwc.reshape(n_pixels, c)
        flat_target = target_hw.reshape(n_pixels)

        feature_blocks.append(flat_features[chosen])
        target_blocks.append(flat_target[chosen])
        sample_id_blocks.append(np.full(shape=n_keep, fill_value=sample_id, dtype=object))

        if include_coords:
            rows, cols = np.unravel_index(chosen, (h, w))
            row_blocks.append(rows.astype(np.int16))
            col_blocks.append(cols.astype(np.int16))

    x = np.concatenate(feature_blocks, axis=0).astype(np.float32)
    y = np.concatenate(target_blocks, axis=0).astype(np.float32)
    sample_ids = np.concatenate(sample_id_blocks, axis=0)

    if not as_dataframe:
        payload = {
            "X": x,
            "y": y,
            "sample_id": sample_ids,
        }
        if include_coords:
            payload["row"] = np.concatenate(row_blocks, axis=0)
            payload["col"] = np.concatenate(col_blocks, axis=0)
        return payload

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - dependency check
        raise ImportError("Pandas output requested but pandas is not installed") from exc

    frame = pd.DataFrame(x, columns=feature_names)
    frame[target_name] = y
    frame["sample_id"] = sample_ids

    if include_coords:
        frame["row"] = np.concatenate(row_blocks, axis=0)
        frame["col"] = np.concatenate(col_blocks, axis=0)

    return frame
