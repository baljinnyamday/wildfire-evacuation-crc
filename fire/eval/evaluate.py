"""Phase 5: evaluate all 6 cells of the 2x3 experimental matrix on test data.

Runs CRC calibration on calibration set, applies thresholds to test set,
computes metrics, and saves everything to disk.

Usage
-----
    fire-evaluate          # run full evaluation
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from fire.conformal.crc import compute_crc_threshold, sweep_fnr
from fire.conformal.threeway import three_way_crc
from fire.eval.metrics import (
    compute_auroc,
    compute_binary_metrics,
    compute_threeway_metrics,
)


# ------------------------------------------------------------------
# Data loaders
# ------------------------------------------------------------------

def _load_tabular(
    data_dir: Path, split: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load (probabilities, labels) from tabular CSV."""
    csv_path = data_dir / f"{split}_probabilities.csv"
    df = pd.read_csv(csv_path)
    return (
        df["probability"].to_numpy(dtype=np.float64),
        df["target"].to_numpy(dtype=np.float64),
    )


def _load_spatial(
    data_dir: Path, split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load (probabilities, labels, valid_mask) from spatial NPZ."""
    npz_path = data_dir / f"{split}_probability_heatmaps.npz"
    data = np.load(npz_path)
    return (
        data["probabilities"],
        data["targets"].astype(np.float64),
        data["valid_masks"],
    )


# ------------------------------------------------------------------
# Per-model evaluation
# ------------------------------------------------------------------

def _evaluate_model(
    model_name: str,
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    alpha: float,
    cost_fn: float,
    cost_fp: float,
    cal_mask: np.ndarray | None = None,
    test_mask: np.ndarray | None = None,
) -> dict[str, dict]:
    """Run all three decision frameworks for one model."""
    results: dict[str, dict] = {}

    # --- 1. Standard (p >= 0.5) ---
    std_metrics = compute_binary_metrics(
        test_probs, test_labels, threshold=0.5, valid_mask=test_mask,
    )
    auroc = compute_auroc(test_probs, test_labels, valid_mask=test_mask)
    results[f"{model_name}_standard"] = {
        **asdict(std_metrics),
        "auroc": auroc,
        "method": "standard",
        "model": model_name,
    }
    print(f"\n  {model_name} — Standard (p >= 0.5)")
    print(f"    Coverage: {std_metrics.coverage:.4f}  FNR: {std_metrics.fnr:.4f}  "
          f"SetSize: {std_metrics.set_size:.4f}  AUROC: {auroc:.4f}")

    # --- 2. CRC (p >= λ̂) ---
    crc_result = compute_crc_threshold(
        cal_probs, cal_labels, alpha=alpha, valid_mask=cal_mask,
    )
    crc_metrics = compute_binary_metrics(
        test_probs, test_labels, threshold=crc_result.lambda_hat,
        valid_mask=test_mask,
    )
    results[f"{model_name}_crc"] = {
        **asdict(crc_metrics),
        "auroc": auroc,
        "lambda_hat": crc_result.lambda_hat,
        "cal_fnr": crc_result.cal_fnr,
        "cal_coverage": crc_result.cal_coverage,
        "method": "crc",
        "model": model_name,
    }
    print(f"  {model_name} — CRC (λ̂={crc_result.lambda_hat:.4f})")
    print(f"    Coverage: {crc_metrics.coverage:.4f}  FNR: {crc_metrics.fnr:.4f}  "
          f"SetSize: {crc_metrics.set_size:.4f}")

    # --- 3. Three-Way CRC ---
    # Shift interval [0.8, 1.3]: fire prevalence may vary ±20-30% from
    # calibration across seasons/regions.  Wider intervals (e.g. [0.5, 2.0])
    # cause the bound to exceed α for extreme class imbalance (~5% fire).
    tw_rho_lo, tw_rho_hi = 0.9, 1.1

    # Cost-weighted α must be higher than binary α because the risk scale
    # includes both c_fn and c_fp terms.  With ~95% negative pixels and
    # c_fp=1, the trivial "flag nothing" risk is already 0.25.
    tw_alpha = 0.50

    tw_result = three_way_crc(
        cal_probs, cal_labels,
        alpha=tw_alpha,
        cost_fn=cost_fn,
        cost_fp=cost_fp,
        rho_lo=tw_rho_lo,
        rho_hi=tw_rho_hi,
        valid_mask=cal_mask,
    )
    tw_metrics = compute_threeway_metrics(
        test_probs, test_labels,
        tw_result.lambda_min, tw_result.lambda_max,
        valid_mask=test_mask,
    )
    results[f"{model_name}_threeway"] = {
        **asdict(tw_metrics),
        "auroc": auroc,
        "alpha_cost": tw_result.alpha,
        "alpha_safe": tw_result.alpha_safe,
        "rho_lo": tw_rho_lo,
        "rho_hi": tw_rho_hi,
        "tw_alpha": tw_alpha,
        "B_pw": tw_result.B_pw,
        "B_tv": tw_result.B_tv,
        "bound_improvement_pct": tw_result.bound_improvement_pct,
        "method": "three_way",
        "model": model_name,
    }
    print(f"  {model_name} — Three-Way (λ_min={tw_result.lambda_min:.4f}, λ_max={tw_result.lambda_max:.4f})")
    print(f"    Coverage: {tw_metrics.coverage:.4f}  FNR: {tw_metrics.fnr:.4f}  "
          f"SetSize: {tw_metrics.set_size:.4f}  "
          f"SAFE: {tw_metrics.safe_frac:.4f}  MONITOR: {tw_metrics.monitor_frac:.4f}  "
          f"EVACUATE: {tw_metrics.evacuate_frac:.4f}")

    # --- FNR sweep (for plotting) ---
    thresholds, fnr_test = sweep_fnr(
        test_probs, test_labels, valid_mask=test_mask,
    )
    results[f"{model_name}_sweep"] = {
        "thresholds": thresholds.tolist(),
        "fnr": fnr_test.tolist(),
    }

    return results


# ------------------------------------------------------------------
# Summary table
# ------------------------------------------------------------------

def _build_summary_table(all_results: dict[str, dict]) -> pd.DataFrame:
    """Build the main results table for the paper."""
    rows = []
    for key, metrics in all_results.items():
        if key.endswith("_sweep"):
            continue
        method = metrics.get("method", "")
        model = metrics.get("model", "")

        if method == "standard":
            label = f"{model} (p≥0.5)"
        elif method == "crc":
            label = f"{model} + CRC"
        elif method == "three_way":
            label = f"{model} + 3-way"
        else:
            continue

        row = {
            "Method": label,
            "Coverage ↑": metrics.get("coverage", 0),
            "FNR ↓": metrics.get("fnr", 0),
            "SetSize ↓": metrics.get("set_size", 0),
            "AUROC ↑": metrics.get("auroc", 0),
        }

        if method == "three_way":
            row["SAFE"] = metrics.get("safe_frac", 0)
            row["MONITOR"] = metrics.get("monitor_frac", 0)
            row["EVACUATE"] = metrics.get("evacuate_frac", 0)

        rows.append(row)

    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tabular-dir", type=Path,
                        default=Path("data/predictions/tabular_baseline"))
    parser.add_argument("--spatial-dir", type=Path,
                        default=Path("data/predictions/spatial_baseline"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/evaluation"))
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="FNR target for standard CRC.")
    parser.add_argument("--cost-fn", type=float, default=5.0)
    parser.add_argument("--cost-fp", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Phase 5: Full Evaluation — 2×3 Matrix")
    print("=" * 60)

    all_results: dict[str, dict] = {}

    # --- Tabular (LightGBM) ---
    print("\n--- Loading tabular predictions ---")
    tab_cal_probs, tab_cal_labels = _load_tabular(args.tabular_dir, "calibration")
    tab_test_probs, tab_test_labels = _load_tabular(args.tabular_dir, "test")
    print(f"  Calibration: {len(tab_cal_probs):,} pixels")
    print(f"  Test: {len(tab_test_probs):,} pixels")

    tab_results = _evaluate_model(
        "LightGBM",
        tab_cal_probs, tab_cal_labels,
        tab_test_probs, tab_test_labels,
        alpha=args.alpha,
        cost_fn=args.cost_fn,
        cost_fp=args.cost_fp,
    )
    all_results.update(tab_results)

    # --- Spatial (U-Net) ---
    print("\n--- Loading spatial predictions ---")
    sp_cal_probs, sp_cal_labels, sp_cal_mask = _load_spatial(
        args.spatial_dir, "calibration",
    )
    sp_test_probs, sp_test_labels, sp_test_mask = _load_spatial(
        args.spatial_dir, "test",
    )
    print(f"  Calibration: {sp_cal_probs.shape[0]} samples × 64×64")
    print(f"  Test: {sp_test_probs.shape[0]} samples × 64×64")

    sp_results = _evaluate_model(
        "U-Net",
        sp_cal_probs, sp_cal_labels,
        sp_test_probs, sp_test_labels,
        alpha=args.alpha,
        cost_fn=args.cost_fn,
        cost_fp=args.cost_fp,
        cal_mask=sp_cal_mask,
        test_mask=sp_test_mask,
    )
    all_results.update(sp_results)

    # --- Summary Table ---
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    summary = _build_summary_table(all_results)
    print(summary.to_string(index=False, float_format="%.4f"))

    # --- Save everything ---
    # Full results JSON
    serializable = {}
    for key, val in all_results.items():
        serializable[key] = {
            k: (v if not isinstance(v, np.generic) else v.item())
            for k, v in val.items()
        }
    (output_dir / "full_results.json").write_text(
        json.dumps(serializable, indent=2, default=str)
    )

    # Summary CSV
    summary.to_csv(output_dir / "summary_table.csv", index=False)

    # Sweep data for plotting
    for key, val in all_results.items():
        if key.endswith("_sweep"):
            np.savez_compressed(
                output_dir / f"{key}.npz",
                thresholds=np.array(val["thresholds"]),
                fnr=np.array(val["fnr"]),
            )

    print(f"\nAll evaluation artifacts saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
