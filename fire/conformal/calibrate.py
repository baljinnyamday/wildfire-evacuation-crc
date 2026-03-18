"""Phase 4: apply Conformal Risk Control to saved calibration probabilities.

Reads the calibration-set outputs produced by Phase 2 (tabular) and/or
Phase 3 (spatial) and computes CRC thresholds for each model.

Two modes:
  - Standard CRC:  binary fire/no-fire with FNR ≤ α guarantee.
  - Three-Way CRC: cost-sensitive SAFE / MONITOR / EVACUATE decisions
                    with shift-aware recalibration (from Dayan 2026).

Usage
-----
    fire-calibrate-crc                          # standard CRC, both models
    fire-calibrate-crc --three-way              # three-way CRC
    fire-calibrate-crc --three-way --cost-fn 5  # custom cost ratio
    fire-calibrate-crc --three-way --rho-lo 0.5 --rho-hi 2.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fire.conformal.crc import CRCResult, compute_crc_threshold, sweep_fnr
from fire.conformal.threeway import ThreeWayResult, three_way_crc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="both",
        choices=["tabular", "spatial", "both"],
        help="Which model's calibration outputs to process.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Target risk level. For standard CRC this is FNR bound (default 0.05). "
             "For --three-way, use a higher value (e.g. 0.30) since the cost-weighted "
             "risk scale is different.",
    )
    parser.add_argument(
        "--tabular-dir",
        type=Path,
        default=Path("data/predictions/tabular_baseline"),
    )
    parser.add_argument(
        "--spatial-dir",
        type=Path,
        default=Path("data/predictions/spatial_baseline"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/predictions/crc"),
        help="Where CRC results are saved.",
    )

    # Three-way CRC arguments
    parser.add_argument(
        "--three-way",
        action="store_true",
        help="Enable three-way decisions (SAFE / MONITOR / EVACUATE).",
    )
    parser.add_argument(
        "--cost-fn",
        type=float,
        default=5.0,
        help="False negative cost (missing fire). Default 5.0.",
    )
    parser.add_argument(
        "--cost-fp",
        type=float,
        default=1.0,
        help="False positive cost (false alarm). Default 1.0.",
    )
    parser.add_argument(
        "--rho-lo",
        type=float,
        default=0.5,
        help="Lower bound of shift-uncertainty interval.",
    )
    parser.add_argument(
        "--rho-hi",
        type=float,
        default=2.0,
        help="Upper bound of shift-uncertainty interval.",
    )
    return parser.parse_args()


# ------------------------------------------------------------------
# Loaders for the two calibration output formats
# ------------------------------------------------------------------

def _load_tabular_calibration(
    tabular_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (probabilities, labels) from LightGBM calibration CSV."""
    csv_path = tabular_dir / "calibration_probabilities.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Tabular calibration file not found: {csv_path}\n"
            "Run Phase 2 first: fire-train-tabular-baseline"
        )
    df = pd.read_csv(csv_path)
    return (
        df["probability"].to_numpy(dtype=np.float64),
        df["target"].to_numpy(dtype=np.float64),
    )


def _load_spatial_calibration(
    spatial_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (probabilities, labels, valid_mask) from U-Net NPZ."""
    npz_path = spatial_dir / "calibration_probability_heatmaps.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Spatial calibration file not found: {npz_path}\n"
            "Run Phase 3 first: fire-train-spatial-baseline"
        )
    data = np.load(npz_path)
    return (
        data["probabilities"],
        data["targets"].astype(np.float64),
        data["valid_masks"],
    )


# ------------------------------------------------------------------
# Pretty-print helpers
# ------------------------------------------------------------------

def _report_standard(model_name: str, result: CRCResult) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Standard CRC — {model_name}")
    print(f"{'=' * 60}")
    print(f"  α (target FNR bound)       : {result.alpha:.4f}")
    print(f"  λ̂ (safe threshold)         : {result.lambda_hat:.6f}")
    print(f"  Positive calibration pixels : {result.n_positive_pixels:,}")
    print(f"  Total calibration pixels    : {result.n_total_pixels:,}")
    print(f"  Cal-set FNR  (should ≤ α)   : {result.cal_fnr:.6f}")
    print(f"  Cal-set Coverage (1 − FNR)  : {result.cal_coverage:.6f}")
    print(f"  Cal-set flagged fraction    : {result.cal_set_size_frac:.6f}")
    print(f"{'=' * 60}")


def _report_threeway(model_name: str, result: ThreeWayResult) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Three-Way CRC — {model_name}")
    print(f"{'=' * 60}")
    print(f"  α (target risk)             : {result.alpha:.4f}")
    print(f"  α_safe (adjusted)           : {result.alpha_safe:.4f}")
    print(f"  c_fn / c_fp                 : {result.cost_fn:.1f} / {result.cost_fp:.1f}")
    print(f"  Shift interval [ρ_lo, ρ_hi] : [{result.rho_lo:.2f}, {result.rho_hi:.2f}]")
    print(f"  π₁ᵗʳ (fire prevalence)      : {result.pi_pos_train:.4f}")
    print(f"  B_pw (ours)                 : {result.B_pw:.4f}")
    print(f"  B_tv (standard)             : {result.B_tv:.4f}")
    print(f"  Bound improvement           : {result.bound_improvement_pct:.1f}%")
    print(f"  ε_max (shift penalty)       : {result.epsilon_max:.4f}")
    print(f"  ───────────────────────────────────────────────")
    print(f"  λ_min (SAFE boundary)       : {result.lambda_min:.6f}")
    print(f"  λ_max (EVACUATE boundary)   : {result.lambda_max:.6f}")
    print(f"  Monitor zone width          : {result.lambda_max - result.lambda_min:.6f}")
    print(f"  ───────────────────────────────────────────────")
    print(f"  Cal SAFE fraction           : {result.cal_safe_frac:.4f}")
    print(f"  Cal MONITOR fraction        : {result.cal_monitor_frac:.4f}")
    print(f"  Cal EVACUATE fraction       : {result.cal_evacuate_frac:.4f}")
    print(f"  Cal risk on decided         : {result.cal_risk_on_decided:.6f}")
    print(f"  Cal FNR on decided          : {result.cal_fnr_on_decided:.6f}")
    print(f"{'=' * 60}")


def _standard_to_dict(model_name: str, result: CRCResult) -> dict:
    return {
        "model": model_name,
        "method": "standard_crc",
        "alpha": result.alpha,
        "lambda_hat": result.lambda_hat,
        "n_positive_calibration_pixels": result.n_positive_pixels,
        "n_total_calibration_pixels": result.n_total_pixels,
        "calibration_fnr": result.cal_fnr,
        "calibration_coverage": result.cal_coverage,
        "calibration_flagged_fraction": result.cal_set_size_frac,
    }


def _threeway_to_dict(model_name: str, result: ThreeWayResult) -> dict:
    return {
        "model": model_name,
        "method": "three_way_crc",
        "alpha": result.alpha,
        "alpha_safe": result.alpha_safe,
        "cost_fn": result.cost_fn,
        "cost_fp": result.cost_fp,
        "rho_lo": result.rho_lo,
        "rho_hi": result.rho_hi,
        "pi_pos_train": result.pi_pos_train,
        "B_pw": result.B_pw,
        "B_tv": result.B_tv,
        "bound_improvement_pct": result.bound_improvement_pct,
        "epsilon_max": result.epsilon_max,
        "lambda_min": result.lambda_min,
        "lambda_max": result.lambda_max,
        "monitor_zone_width": result.lambda_max - result.lambda_min,
        "n_samples": result.n_samples,
        "cal_safe_frac": result.cal_safe_frac,
        "cal_monitor_frac": result.cal_monitor_frac,
        "cal_evacuate_frac": result.cal_evacuate_frac,
        "cal_risk_on_decided": result.cal_risk_on_decided,
        "cal_fnr_on_decided": result.cal_fnr_on_decided,
    }


# ------------------------------------------------------------------
# Run standard CRC for a single model
# ------------------------------------------------------------------

def _run_standard(
    model_name: str,
    probs: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    output_dir: Path,
    valid_mask: np.ndarray | None = None,
) -> dict:
    result = compute_crc_threshold(probs, labels, alpha=alpha, valid_mask=valid_mask)
    _report_standard(model_name, result)

    thresholds, fnr_vals = sweep_fnr(probs, labels, valid_mask=valid_mask)
    np.savez_compressed(
        output_dir / f"{model_name}_fnr_sweep.npz",
        thresholds=thresholds,
        fnr=fnr_vals,
    )
    return _standard_to_dict(model_name, result)


# ------------------------------------------------------------------
# Run three-way CRC for a single model
# ------------------------------------------------------------------

def _run_threeway(
    model_name: str,
    probs: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    cost_fn: float,
    cost_fp: float,
    rho_lo: float,
    rho_hi: float,
    output_dir: Path,
    valid_mask: np.ndarray | None = None,
) -> dict:
    result = three_way_crc(
        probs, labels,
        alpha=alpha,
        cost_fn=cost_fn,
        cost_fp=cost_fp,
        rho_lo=rho_lo,
        rho_hi=rho_hi,
        valid_mask=valid_mask,
    )
    _report_threeway(model_name, result)
    return _threeway_to_dict(model_name, result)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    models_data: dict[str, tuple] = {}

    if args.model in ("tabular", "both"):
        probs, labels = _load_tabular_calibration(args.tabular_dir)
        models_data["tabular"] = (probs, labels, None)

    if args.model in ("spatial", "both"):
        probs, labels, mask = _load_spatial_calibration(args.spatial_dir)
        models_data["spatial"] = (probs, labels, mask)

    all_results: dict[str, dict] = {}

    for name, (probs, labels, mask) in models_data.items():
        # Always run standard CRC
        std_dict = _run_standard(name, probs, labels, args.alpha, output_dir, mask)
        all_results[f"{name}_standard"] = std_dict

        # Optionally run three-way CRC
        if args.three_way:
            tw_dict = _run_threeway(
                name, probs, labels, args.alpha,
                args.cost_fn, args.cost_fp,
                args.rho_lo, args.rho_hi,
                output_dir, mask,
            )
            all_results[f"{name}_three_way"] = tw_dict

    # --- Persist -------------------------------------------------------
    payload = {
        "alpha": args.alpha,
        "three_way_enabled": args.three_way,
        "results": all_results,
    }
    if args.three_way:
        payload["three_way_params"] = {
            "cost_fn": args.cost_fn,
            "cost_fp": args.cost_fp,
            "rho_lo": args.rho_lo,
            "rho_hi": args.rho_hi,
        }

    out_path = output_dir / "crc_thresholds.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved CRC thresholds to {out_path.resolve()}")


if __name__ == "__main__":
    main()
