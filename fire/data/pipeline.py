"""High-level data pipeline orchestration for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import DEFAULT_INPUT_FEATURES, DEFAULT_SAMPLE_SIZE, DEFAULT_SEED, DEFAULT_TARGET_FEATURE
from .readers import SampleBackend, build_backend
from .splits import build_split_indices, load_split_indices, require_split, save_split_indices


@dataclass
class NDWSPipeline:
    """Owns backend discovery and deterministic split assignment."""

    data_root: str | Path
    input_format: str = "auto"
    seed: int = DEFAULT_SEED
    sample_size: int = DEFAULT_SAMPLE_SIZE
    input_features: list[str] | None = None
    target_feature: str = DEFAULT_TARGET_FEATURE
    split_manifest_path: str | Path | None = None

    def __post_init__(self) -> None:
        self.input_features = self.input_features or list(DEFAULT_INPUT_FEATURES)
        self.backend: SampleBackend = build_backend(
            data_root=self.data_root,
            input_format=self.input_format,
            input_features=self.input_features,
            target_feature=self.target_feature,
            sample_size=self.sample_size,
        )

        if self.split_manifest_path and Path(self.split_manifest_path).exists():
            self.split_indices = load_split_indices(self.split_manifest_path)
        else:
            self.split_indices = build_split_indices(len(self.backend), seed=self.seed)
            if self.split_manifest_path:
                save_split_indices(self.split_indices, self.split_manifest_path)

    def get_split_indices(self, split: str) -> np.ndarray:
        split_name = require_split(split)
        return self.split_indices[split_name]

    def save_splits(self, path: str | Path) -> None:
        save_split_indices(self.split_indices, path)

    def split_sizes(self) -> dict[str, int]:
        return {name: len(indices) for name, indices in self.split_indices.items()}

    def build_tabular_split(
        self,
        split: str,
        pixel_sample_rate: float = 0.05,
        as_dataframe: bool = True,
        include_coords: bool = True,
        target_name: str = DEFAULT_TARGET_FEATURE,
    ):
        from .tabular_loader import build_tabular_split

        return build_tabular_split(
            backend=self.backend,
            split_indices=self.get_split_indices(split),
            feature_names=self.input_features,
            pixel_sample_rate=pixel_sample_rate,
            seed=self.seed,
            as_dataframe=as_dataframe,
            include_coords=include_coords,
            target_name=target_name,
        )

    def build_spatial_dataset(
        self,
        split: str,
        return_sample_id: bool = False,
    ):
        from .spatial_loader import SpatialWildfireDataset

        return SpatialWildfireDataset(
            backend=self.backend,
            indices=self.get_split_indices(split),
            return_sample_id=return_sample_id,
        )

    def build_spatial_dataloader(
        self,
        split: str,
        batch_size: int,
        shuffle: bool | None = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        return_sample_id: bool = False,
    ):
        from .spatial_loader import build_spatial_dataloader

        dataset = self.build_spatial_dataset(split, return_sample_id=return_sample_id)
        if shuffle is None:
            shuffle = split.strip().lower() == "train"
        return build_spatial_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
