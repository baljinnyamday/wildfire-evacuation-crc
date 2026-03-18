"""Module 1B: PyTorch Dataset/DataLoader for spatial wildfire maps."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .readers import SampleBackend


class SpatialWildfireDataset(Dataset):
    """PyTorch Dataset for 64x64 spatial maps."""

    def __init__(
        self,
        backend: SampleBackend,
        indices: np.ndarray,
        return_sample_id: bool = False,
    ) -> None:
        self.backend = backend
        self.indices = np.asarray(indices, dtype=np.int64)
        if self.indices.size == 0:
            raise ValueError("indices cannot be empty")
        self.return_sample_id = return_sample_id

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int):
        global_index = int(self.indices[item])
        inputs_hwc, target_hw, sample_id = self.backend.get_sample(global_index)

        # NDWS FireMask uses -1 for no-data pixels; mark them before clamping.
        valid_mask = (target_hw >= 0).astype(np.float32)
        target_hw = np.clip(target_hw, 0.0, 1.0)

        x = torch.from_numpy(np.transpose(inputs_hwc, (2, 0, 1))).to(torch.float32)
        y = torch.from_numpy(target_hw[np.newaxis, ...]).to(torch.float32)
        mask = torch.from_numpy(valid_mask[np.newaxis, ...]).to(torch.float32)

        if self.return_sample_id:
            return x, y, mask, sample_id
        return x, y, mask


def build_spatial_dataloader(
    dataset: SpatialWildfireDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """Constructs a standard PyTorch DataLoader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
