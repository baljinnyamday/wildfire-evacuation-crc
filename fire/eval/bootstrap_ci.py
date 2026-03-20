"""Bootstrap confidence intervals on aggregate metrics.

Resamples test images with replacement, recomputes pixel-aggregate
FNR/coverage/set_size each time, reports 95% CIs.

Usage
-----
    uv run python -m fire.eval.bootstrap_ci
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _aggregate_fnr_coverage_setsize(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> tuple[float, float, float]:
    """Compute pixel-aggregate FNR, coverage, set_size from flat arrays."""
    preds = (probs >= threshold).astype(np.float64)
    n_fire = int((labels == 1).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    fnr = fn / n_fire if n_fire > 0 else 0.0
    set_size = float(preds.sum()) / len(probs) if len(probs) > 0 else 0.0
    return fnr, 1.0 - fnr, set_size


def bootstrap_aggregate_spatial(
    probabilities: np.ndarray,
    targets: np.ndarray,
    valid_masks: np.ndarray,
    threshold: float,
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Bootstrap aggregate metrics by resampling images."""
    n_images = probabilities.shape[0]
    rng = np.random.default_rng(seed)

    boot_fnr = np.empty(n_boot)
    boot_cov = np.empty(n_boot)
    boot_ss = np.empty(n_boot)

    for b in range(n_boot):
        idx = rng.choice(n_images, size=n_images, replace=True)
        probs_flat = probabilities[idx][valid_masks[idx].astype(bool)].astype(np.float64)
        labs_flat = targets[idx][valid_masks[idx].astype(bool)].astype(np.float64)
        boot_fnr[b], boot_cov[b], boot_ss[b] = _aggregate_fnr_coverage_setsize(
            probs_flat, labs_flat, threshold,
        )

    # Point estimates on full data
    all_probs = probabilities[valid_masks.astype(bool)].astype(np.float64)
    all_labs = targets[valid_masks.astype(bool)].astype(np.float64)
    fnr, cov, ss = _aggregate_fnr_coverage_setsize(all_probs, all_labs, threshold)

    def _ci(values: np.ndarray, point: float) -> dict[str, float]:
        return {
            "point": point,
            "ci_lo": float(np.percentile(values, 2.5)),
            "ci_hi": float(np.percentile(values, 97.5)),
        }

    return {
        "fnr": _ci(boot_fnr, fnr),
        "coverage": _ci(boot_cov, cov),
        "set_size": _ci(boot_ss, ss),
    }


def bootstrap_aggregate_tabular(
    csv_path: Path,
    threshold: float,
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Bootstrap aggregate metrics by resampling images (tabular)."""
    df = pd.read_csv(csv_path)

    # Group pixels by image
    grouped = df.groupby("sample_id")
    image_probs = []
    image_labels = []
    for _, group in grouped:
        image_probs.append(group["probability"].to_numpy(dtype=np.float64))
        image_labels.append(group["target"].to_numpy(dtype=np.float64))

    n_images = len(image_probs)
    rng = np.random.default_rng(seed)

    boot_fnr = np.empty(n_boot)
    boot_cov = np.empty(n_boot)
    boot_ss = np.empty(n_boot)

    for b in range(n_boot):
        idx = rng.choice(n_images, size=n_images, replace=True)
        probs_flat = np.concatenate([image_probs[i] for i in idx])
        labs_flat = np.concatenate([image_labels[i] for i in idx])
        boot_fnr[b], boot_cov[b], boot_ss[b] = _aggregate_fnr_coverage_setsize(
            probs_flat, labs_flat, threshold,
        )

    # Point estimates
    all_probs = df["probability"].to_numpy(dtype=np.float64)
    all_labs = df["target"].to_numpy(dtype=np.float64)
    fnr, cov, ss = _aggregate_fnr_coverage_setsize(all_probs, all_labs, threshold)

    def _ci(values: np.ndarray, point: float) -> dict[str, float]:
        return {
            "point": point,
            "ci_lo": float(np.percentile(values, 2.5)),
            "ci_hi": float(np.percentile(values, 97.5)),
        }

    return {
        "fnr": _ci(boot_fnr, fnr),
        "coverage": _ci(boot_cov, cov),
        "set_size": _ci(boot_ss, ss),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spatial-dir", type=Path,
                        default=Path("data/predictions/spatial_baseline"))
    parser.add_argument("--tabular-dir", type=Path,
                        default=Path("data/predictions/tabular_baseline"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/evaluation"))
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results_path = args.output_dir / "full_results.json"
    full_results = json.loads(results_path.read_text())

    print("=" * 60)
    print("  Bootstrap 95% CIs on Aggregate Metrics")
    print("  (resampling images, recomputing pixel-aggregate)")
    print("=" * 60)

    all_ci: dict[str, dict] = {}

    # --- U-Net ---
    print("\n--- U-Net ---")
    sp_data = np.load(args.spatial_dir / "test_probability_heatmaps.npz")

    for method_key, threshold_val, label in [
        ("U-Net_standard", 0.5, "U-Net (p>=0.5)"),
        ("U-Net_crc", None, "U-Net+CRC"),
    ]:
        threshold = (
            full_results[method_key]["lambda_hat"]
            if threshold_val is None
            else threshold_val
        )
        ci = bootstrap_aggregate_spatial(
            sp_data["probabilities"],
            sp_data["targets"].astype(np.float64),
            sp_data["valid_masks"],
            threshold,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        all_ci[label] = {"threshold": threshold, **ci}

        print(f"\n  {label} (λ={threshold:.4f})")
        print(f"    FNR:      {ci['fnr']['point']:.4f}  [{ci['fnr']['ci_lo']:.4f}, {ci['fnr']['ci_hi']:.4f}]")
        print(f"    Coverage: {ci['coverage']['point']:.4f}  [{ci['coverage']['ci_lo']:.4f}, {ci['coverage']['ci_hi']:.4f}]")
        print(f"    Set size: {ci['set_size']['point']:.4f}  [{ci['set_size']['ci_lo']:.4f}, {ci['set_size']['ci_hi']:.4f}]")

    # --- LightGBM ---
    print("\n--- LightGBM ---")
    tab_csv = args.tabular_dir / "test_probabilities.csv"

    for method_key, threshold_val, label in [
        ("LightGBM_standard", 0.5, "LightGBM (p>=0.5)"),
        ("LightGBM_crc", None, "LightGBM+CRC"),
    ]:
        threshold = (
            full_results[method_key]["lambda_hat"]
            if threshold_val is None
            else threshold_val
        )
        ci = bootstrap_aggregate_tabular(
            tab_csv, threshold,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        all_ci[label] = {"threshold": threshold, **ci}

        print(f"\n  {label} (λ={threshold:.4f})")
        print(f"    FNR:      {ci['fnr']['point']:.4f}  [{ci['fnr']['ci_lo']:.4f}, {ci['fnr']['ci_hi']:.4f}]")
        print(f"    Coverage: {ci['coverage']['point']:.4f}  [{ci['coverage']['ci_lo']:.4f}, {ci['coverage']['ci_hi']:.4f}]")
        print(f"    Set size: {ci['set_size']['point']:.4f}  [{ci['set_size']['ci_lo']:.4f}, {ci['set_size']['ci_hi']:.4f}]")

    # --- Save ---
    output_path = args.output_dir / "bootstrap_ci.json"
    output_path.write_text(json.dumps(all_ci, indent=2, default=str))
    print(f"\nSaved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
