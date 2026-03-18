"""Project-wide data constants."""

from __future__ import annotations

DEFAULT_INPUT_FEATURES = [
    "elevation",
    "th",
    "vs",
    "tmmn",
    "tmmx",
    "sph",
    "pr",
    "pdsi",
    "NDVI",
    "population",
    "erc",
    "PrevFireMask",
]
DEFAULT_TARGET_FEATURE = "FireMask"
DEFAULT_SAMPLE_SIZE = 64
DEFAULT_SEED = 42

TRAIN_FRACTION = 0.70
CALIBRATION_FRACTION = 0.15
TEST_FRACTION = 0.15

VALID_SPLITS = ("train", "calibration", "test")
