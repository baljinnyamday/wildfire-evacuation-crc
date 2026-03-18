"""Phase 1 driver: build deterministic splits and smoke-test both loaders."""

from __future__ import annotations

import argparse
from pathlib import Path

from .constants import DEFAULT_SEED
from .pipeline import NDWSPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--input-format",
        default="auto",
        choices=["auto", "tfrecord", "tif"],
        help="Source file type.",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("data/splits/ndws_seed42_70_15_15.json"),
        help="Where to store split indices for reproducibility.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--tabular-sample-rate",
        type=float,
        default=0.05,
        help="Fraction of pixels sampled per map in tabular loader.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = NDWSPipeline(
        data_root=args.data_root,
        input_format=args.input_format,
        seed=args.seed,
        split_manifest_path=args.split_manifest,
    )

    print(f"Total samples: {len(pipeline.backend)}")
    for split_name, size in pipeline.split_sizes().items():
        print(f"{split_name:12s}: {size}")

    train_loader = pipeline.build_spatial_dataloader(
        split="train",
        batch_size=args.batch_size,
        shuffle=True,
    )
    x_batch, y_batch, mask_batch = next(iter(train_loader))
    print(f"Spatial batch inputs shape: {tuple(x_batch.shape)}")
    print(f"Spatial batch targets shape: {tuple(y_batch.shape)}")
    print(f"Spatial batch masks shape:   {tuple(mask_batch.shape)}")

    tabular_train = pipeline.build_tabular_split(
        split="train",
        pixel_sample_rate=args.tabular_sample_rate,
        as_dataframe=True,
    )
    print(f"Tabular train rows: {len(tabular_train)}")
    print(f"Tabular columns: {list(tabular_train.columns)}")


if __name__ == "__main__":
    main()
