"""Phase 2: train LightGBM tabular baseline and export split probabilities."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from fire.data.constants import DEFAULT_SEED, DEFAULT_TARGET_FEATURE
from fire.data.pipeline import NDWSPipeline

if TYPE_CHECKING:
    from lightgbm import LGBMClassifier


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
        help="Deterministic split manifest generated during Phase 1.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--train-pixel-sample-rate",
        type=float,
        default=0.05,
        help="Pixel sampling rate for train split flattening.",
    )
    parser.add_argument(
        "--eval-pixel-sample-rate",
        type=float,
        default=0.05,
        help="Pixel sampling rate for calibration/test flattening.",
    )
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/predictions/tabular_baseline"),
        help="Where model and split probabilities are saved.",
    )
    return parser.parse_args()


def _to_binary_labels(target: pd.Series) -> np.ndarray:
    y = target.to_numpy(dtype=np.float32, copy=False)
    unique = np.unique(y)
    if np.all(np.isin(unique, [0.0, 1.0])):
        return y.astype(np.uint8)
    return (y > 0).astype(np.uint8)


def _split_frame_to_xy(
    frame: pd.DataFrame,
    feature_names: list[str],
    target_name: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    x = frame[feature_names]
    y = _to_binary_labels(frame[target_name])
    return x, y


def _save_split_probabilities(
    model: "LGBMClassifier",
    frame: pd.DataFrame,
    feature_names: list[str],
    target_name: str,
    split_name: str,
    output_path: Path,
) -> None:
    x_split, y_split = _split_frame_to_xy(frame, feature_names, target_name)
    probs = model.predict_proba(x_split)[:, 1].astype(np.float32)

    payload = pd.DataFrame(
        {
            "split": split_name,
            "sample_id": frame["sample_id"].to_numpy(),
            "row": frame["row"].to_numpy(),
            "col": frame["col"].to_numpy(),
            "target": y_split,
            "probability": probs,
        }
    )
    payload.to_csv(output_path, index=False)


def _load_lgbm_classifier():
    try:
        from lightgbm import LGBMClassifier
    except OSError as exc:
        raise RuntimeError(
            "LightGBM library load failed. On macOS, install OpenMP runtime with: brew install libomp"
        ) from exc
    return LGBMClassifier


def main() -> None:
    args = parse_args()
    target_name = DEFAULT_TARGET_FEATURE
    lgbm_classifier = _load_lgbm_classifier()

    pipeline = NDWSPipeline(
        data_root=args.data_root,
        input_format=args.input_format,
        seed=args.seed,
        split_manifest_path=args.split_manifest,
    )
    feature_names = list(pipeline.input_features or [])

    print("Building tabular train split...")
    train_frame = pipeline.build_tabular_split(
        split="train",
        pixel_sample_rate=args.train_pixel_sample_rate,
        as_dataframe=True,
        include_coords=True,
        target_name=target_name,
    )

    x_train, y_train = _split_frame_to_xy(train_frame, feature_names, target_name)
    print(f"Training rows: {x_train.shape[0]}, features: {x_train.shape[1]}")

    model = lgbm_classifier(
        objective="binary",
        random_state=args.seed,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        n_jobs=args.n_jobs,
    )
    model.fit(x_train, y_train)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "lightgbm_model.pkl").open("wb") as handle:
        pickle.dump(model, handle)

    print("Building calibration split and saving probabilities...")
    calibration_frame = pipeline.build_tabular_split(
        split="calibration",
        pixel_sample_rate=args.eval_pixel_sample_rate,
        as_dataframe=True,
        include_coords=True,
        target_name=target_name,
    )
    _save_split_probabilities(
        model=model,
        frame=calibration_frame,
        feature_names=feature_names,
        target_name=target_name,
        split_name="calibration",
        output_path=output_dir / "calibration_probabilities.csv",
    )

    print("Building test split and saving probabilities...")
    test_frame = pipeline.build_tabular_split(
        split="test",
        pixel_sample_rate=args.eval_pixel_sample_rate,
        as_dataframe=True,
        include_coords=True,
        target_name=target_name,
    )
    _save_split_probabilities(
        model=model,
        frame=test_frame,
        feature_names=feature_names,
        target_name=target_name,
        split_name="test",
        output_path=output_dir / "test_probabilities.csv",
    )

    metadata = {
        "seed": args.seed,
        "train_pixel_sample_rate": args.train_pixel_sample_rate,
        "eval_pixel_sample_rate": args.eval_pixel_sample_rate,
        "feature_names": feature_names,
        "target_name": target_name,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Saved model and probabilities in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
