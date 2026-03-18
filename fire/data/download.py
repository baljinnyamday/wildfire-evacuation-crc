"""Download helper for the Kaggle Next Day Wildfire Spread dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess

DEFAULT_DATASET = "fantineh/next-day-wildfire-spread"


def download_from_kaggle(dataset: str, output_dir: Path, force: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    kaggle_bin = shutil.which("kaggle")
    if kaggle_bin is not None:
        cmd = [
            kaggle_bin,
            "datasets",
            "download",
            "-d",
            dataset,
            "-p",
            str(output_dir),
            "--unzip",
        ]
        if force:
            cmd.append("--force")

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            return

        details = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(
            "Kaggle CLI download failed. Ensure KAGGLE_USERNAME/KAGGLE_KEY are configured. "
            f"Details: {details}"
        )

    # Fallback: Python API (works when the CLI binary is not installed).
    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", output_dir / ".kaggle"))
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["KAGGLE_CONFIG_DIR"] = str(config_dir.resolve())

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError(
            "Kaggle CLI binary was not found and Python Kaggle API is not installed. "
            "Install with: pip install kaggle"
        ) from exc

    try:
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(
            dataset=dataset,
            path=str(output_dir),
            unzip=True,
            force=force,
            quiet=False,
        )
    except Exception as exc:  # pragma: no cover - runtime auth/network dependency
        raise RuntimeError(
            "Kaggle API download failed. Ensure ~/.kaggle/kaggle.json exists "
            "or KAGGLE_USERNAME/KAGGLE_KEY are set."
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Kaggle dataset slug (<owner>/<dataset-name>).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory where NDWS files are downloaded.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing archive contents if present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_from_kaggle(args.dataset, args.output_dir, force=args.force)
    print(f"Downloaded dataset '{args.dataset}' to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
