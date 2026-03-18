"""Dataset readers for NDWS TFRecord and TIFF sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from bisect import bisect_right
from pathlib import Path
import re
from typing import Literal

import numpy as np

from .constants import DEFAULT_INPUT_FEATURES, DEFAULT_SAMPLE_SIZE, DEFAULT_TARGET_FEATURE


class SampleBackend(ABC):
    """Common backend interface used by tabular and spatial loaders."""

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_sample(self, index: int) -> tuple[np.ndarray, np.ndarray, str]:
        """Returns (inputs_hwc, target_hw, sample_id)."""
        raise NotImplementedError


class TFRecordBackend(SampleBackend):
    """Loads NDWS records from TFRecord files."""

    def __init__(
        self,
        data_root: str | Path,
        input_features: list[str] | None = None,
        target_feature: str = DEFAULT_TARGET_FEATURE,
        sample_size: int = DEFAULT_SAMPLE_SIZE,
    ) -> None:
        try:
            from tfrecord.reader import tfrecord_loader
        except ImportError as exc:  # pragma: no cover - dependency check
            raise ImportError(
                "TFRecord support requires the 'tfrecord' package. Install with: pip install tfrecord"
            ) from exc

        self._tfrecord_loader = tfrecord_loader
        self.data_root = Path(data_root)
        self.input_features = input_features or list(DEFAULT_INPUT_FEATURES)
        self.target_feature = target_feature
        self.sample_size = sample_size

        self.files = self._discover_tfrecord_files(self.data_root)
        if not self.files:
            raise FileNotFoundError(
                f"No TFRecord files found under {self.data_root}. "
                "Expected *.tfrecord or *.tfrecords files."
            )

        self._description = {
            **{feature: "float" for feature in self.input_features},
            self.target_feature: "float",
        }
        self._counts = [self._count_records(path) for path in self.files]
        self._offsets = np.concatenate([[0], np.cumsum(np.asarray(self._counts, dtype=np.int64))])
        self._file_cache: dict[int, list[tuple[np.ndarray, np.ndarray, str]]] = {}

    @staticmethod
    def _discover_tfrecord_files(root: Path) -> list[Path]:
        patterns = ("*.tfrecord", "*.tfrecords")
        found: list[Path] = []
        for pattern in patterns:
            found.extend(root.rglob(pattern))
        unique_sorted = sorted({path.resolve() for path in found})
        return unique_sorted

    def _count_records(self, path: Path) -> int:
        return sum(1 for _ in self._tfrecord_loader(str(path), None, self._description))

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def get_sample(self, index: int) -> tuple[np.ndarray, np.ndarray, str]:
        n = len(self)
        if index < 0 or index >= n:
            raise IndexError(f"Index {index} out of range for {n} samples")

        file_idx = bisect_right(self._offsets, index) - 1
        local_idx = index - int(self._offsets[file_idx])

        if file_idx not in self._file_cache:
            self._file_cache[file_idx] = self._load_file(file_idx)

        return self._file_cache[file_idx][local_idx]

    def _load_file(self, file_idx: int) -> list[tuple[np.ndarray, np.ndarray, str]]:
        path = self.files[file_idx]
        decoded: list[tuple[np.ndarray, np.ndarray, str]] = []
        for record_idx, raw_record in enumerate(
            self._tfrecord_loader(str(path), None, self._description)
        ):
            record = self._normalize_keys(raw_record)
            inputs = np.stack(
                [self._decode_map(record, key) for key in self.input_features], axis=-1
            ).astype(np.float32)
            target = self._decode_map(record, self.target_feature).astype(np.float32)
            sample_id = f"{path.name}:{record_idx}"
            decoded.append((inputs, target, sample_id))
        return decoded

    @staticmethod
    def _normalize_keys(record: dict) -> dict[str, object]:
        normalized: dict[str, object] = {}
        for key, value in record.items():
            if isinstance(key, bytes):
                normalized[key.decode("utf-8")] = value
            else:
                normalized[str(key)] = value
        return normalized

    def _decode_map(self, record: dict[str, object], key: str) -> np.ndarray:
        if key not in record:
            known = ", ".join(sorted(record.keys()))
            raise KeyError(f"Feature '{key}' missing from TFRecord. Available keys: {known}")

        value = np.asarray(record[key], dtype=np.float32)
        if value.shape == (self.sample_size, self.sample_size):
            return value

        if value.size == self.sample_size * self.sample_size:
            return value.reshape(self.sample_size, self.sample_size)

        raise ValueError(
            f"Feature '{key}' has shape {value.shape}; "
            f"expected ({self.sample_size}, {self.sample_size}) or flat size {self.sample_size**2}."
        )


class TIFFBackend(SampleBackend):
    """Loads NDWS maps from TIFF files."""

    TARGET_PATTERN = re.compile(r"(target|label|mask|firemask|gt|y)$")
    INPUT_PATTERN = re.compile(r"(input|inputs|image|images|features|feat|x)$")

    def __init__(
        self,
        data_root: str | Path,
        input_features: list[str] | None = None,
        sample_size: int = DEFAULT_SAMPLE_SIZE,
    ) -> None:
        try:
            import tifffile
        except ImportError as exc:  # pragma: no cover - dependency check
            raise ImportError(
                "TIFF support requires the 'tifffile' package. Install with: pip install tifffile"
            ) from exc

        self._tifffile = tifffile
        self.data_root = Path(data_root)
        self.input_features = input_features or list(DEFAULT_INPUT_FEATURES)
        self.n_features = len(self.input_features)
        self.sample_size = sample_size

        self.entries = self._discover_entries(self.data_root)
        if not self.entries:
            raise FileNotFoundError(
                f"No TIFF files found under {self.data_root}. Expected *.tif or *.tiff files."
            )

    def __len__(self) -> int:
        return len(self.entries)

    def get_sample(self, index: int) -> tuple[np.ndarray, np.ndarray, str]:
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for {len(self)} samples")

        input_path, target_path, sample_id = self.entries[index]
        raw_input = np.asarray(self._tifffile.imread(str(input_path)), dtype=np.float32)

        if target_path is None:
            inputs, target = self._split_combined_tiff(raw_input)
        else:
            raw_target = np.asarray(self._tifffile.imread(str(target_path)), dtype=np.float32)
            inputs = self._coerce_input(raw_input)
            target = self._coerce_target(raw_target)

        return inputs, target, sample_id

    def _discover_entries(self, root: Path) -> list[tuple[Path, Path | None, str]]:
        files = sorted({*root.rglob("*.tif"), *root.rglob("*.tiff")})
        if not files:
            return []

        classified: dict[str, dict[str, Path]] = {}
        for file_path in files:
            sample_key = self._normalize_stem(file_path.stem)
            role = self._classify_file(file_path.stem)
            classified.setdefault(sample_key, {})
            classified[sample_key][role] = file_path

        paired_entries: list[tuple[Path, Path | None, str]] = []
        combined_entries: list[tuple[Path, Path | None, str]] = []

        for key, payload in sorted(classified.items()):
            input_path = payload.get("input") or payload.get("unknown")
            target_path = payload.get("target")
            if input_path is None and target_path is not None:
                continue
            if input_path is None:
                continue

            if target_path is not None:
                paired_entries.append((input_path, target_path, key))
            else:
                combined_entries.append((input_path, None, key))

        return paired_entries if paired_entries else combined_entries

    def _classify_file(self, stem: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", stem.lower()).strip("_")
        tokens = [token for token in cleaned.split("_") if token]
        if not tokens:
            return "unknown"

        last = tokens[-1]
        if self.TARGET_PATTERN.match(last):
            return "target"
        if self.INPUT_PATTERN.match(last):
            return "input"

        if any(token in {"label", "labels", "mask", "target", "firemask"} for token in tokens):
            return "target"
        if any(token in {"input", "inputs", "image", "images", "feature", "features"} for token in tokens):
            return "input"
        return "unknown"

    def _normalize_stem(self, stem: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", stem.lower()).strip("_")
        cleaned = re.sub(r"_(input|inputs|image|images|features|feature|feat|x)$", "", cleaned)
        cleaned = re.sub(r"_(target|label|labels|mask|firemask|gt|y)$", "", cleaned)
        return cleaned or stem.lower()

    def _split_combined_tiff(self, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hwc = self._to_hwc(arr)
        channels = hwc.shape[-1]
        if channels != self.n_features + 1:
            raise ValueError(
                f"Combined TIFF {arr.shape} has {channels} channels, expected {self.n_features + 1} "
                "(12 inputs + 1 target)."
            )

        inputs = hwc[..., : self.n_features]
        target = hwc[..., self.n_features]
        return inputs, target

    def _coerce_input(self, arr: np.ndarray) -> np.ndarray:
        hwc = self._to_hwc(arr)
        channels = hwc.shape[-1]
        if channels != self.n_features:
            raise ValueError(
                f"Input TIFF has {channels} channels; expected {self.n_features} channels."
            )
        return hwc

    def _coerce_target(self, arr: np.ndarray) -> np.ndarray:
        squeezed = np.asarray(arr, dtype=np.float32)
        if squeezed.ndim == 2:
            target = squeezed
        elif squeezed.ndim == 3:
            if squeezed.shape[-1] == 1:
                target = squeezed[..., 0]
            elif squeezed.shape[0] == 1:
                target = squeezed[0, ...]
            else:
                target = self._to_hwc(squeezed)[..., 0]
        else:
            raise ValueError(f"Target TIFF has unsupported shape: {arr.shape}")

        self._validate_hw(target)
        return target

    def _to_hwc(self, arr: np.ndarray) -> np.ndarray:
        data = np.asarray(arr, dtype=np.float32)
        if data.ndim == 2:
            data = data[..., np.newaxis]
        elif data.ndim == 3:
            if data.shape[0] == self.sample_size and data.shape[1] == self.sample_size:
                pass
            elif data.shape[1] == self.sample_size and data.shape[2] == self.sample_size:
                data = np.transpose(data, (1, 2, 0))
            else:
                raise ValueError(
                    f"TIFF tensor with shape {arr.shape} is not compatible with 64x64 maps."
                )
        else:
            raise ValueError(f"TIFF tensor with shape {arr.shape} is not supported.")

        self._validate_hw(data)
        return data

    def _validate_hw(self, arr: np.ndarray) -> None:
        if arr.shape[0] != self.sample_size or arr.shape[1] != self.sample_size:
            raise ValueError(
                f"Expected map size ({self.sample_size}, {self.sample_size}), got {arr.shape[:2]}."
            )


def build_backend(
    data_root: str | Path,
    input_format: Literal["auto", "tfrecord", "tif"] = "auto",
    input_features: list[str] | None = None,
    target_feature: str = DEFAULT_TARGET_FEATURE,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> SampleBackend:
    """Creates a backend for reading NDWS samples."""
    path = Path(data_root)
    if not path.exists():
        raise FileNotFoundError(f"Data root does not exist: {path}")

    fmt = input_format.lower().strip()
    if fmt == "tfrecord":
        return TFRecordBackend(path, input_features, target_feature, sample_size)
    if fmt == "tif":
        return TIFFBackend(path, input_features, sample_size)
    if fmt != "auto":
        raise ValueError("input_format must be one of: auto, tfrecord, tif")

    has_tfrecord = any(path.rglob("*.tfrecord")) or any(path.rglob("*.tfrecords"))
    has_tif = any(path.rglob("*.tif")) or any(path.rglob("*.tiff"))

    if has_tfrecord:
        return TFRecordBackend(path, input_features, target_feature, sample_size)
    if has_tif:
        return TIFFBackend(path, input_features, sample_size)

    raise FileNotFoundError(
        f"No supported NDWS files found in {path}. "
        "Expected TFRecords (*.tfrecord, *.tfrecords) or TIFFs (*.tif, *.tiff)."
    )
