"""Extra publication figures: Before/After CRC hero figure + 3-model comparison.

Usage
-----
    fire-plot-extra              # generate extra figures
    fire-plot-extra --output-dir figures/
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=Path("data/evaluation"))
    parser.add_argument("--spatial-dir", type=Path,
                        default=Path("data/predictions/spatial_baseline"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


# ------------------------------------------------------------------
# Figure: Before vs After CRC (Hero Figure)
# ------------------------------------------------------------------

def plot_before_after_crc(
    spatial_dir: Path,
    results: dict,
    output_dir: Path,
    dpi: int,
    n_samples: int = 3,
) -> None:
    """Multi-sample before/after CRC comparison with error overlays."""
    sp_data = np.load(spatial_dir / "test_probability_heatmaps.npz")
    sp_probs = sp_data["probabilities"]
    sp_targets = sp_data["targets"]
    sp_masks = sp_data["valid_masks"]

    lam_crc = results["U-Net_crc"]["lambda_hat"]

    # Find samples with meaningful fire content (diverse fire fractions)
    fire_fracs = []
    for i in range(sp_targets.shape[0]):
        valid = sp_masks[i] == 1
        if valid.sum() > 0:
            fire_fracs.append(float(sp_targets[i][valid].sum()) / valid.sum())
        else:
            fire_fracs.append(0.0)
    fire_fracs = np.array(fire_fracs)

    # Pick samples at different fire fraction percentiles (50th, 80th, 95th)
    positive_mask = fire_fracs > 0.01
    if positive_mask.sum() < n_samples:
        indices = np.argsort(fire_fracs)[-n_samples:]
    else:
        percentiles = [50, 80, 95]
        indices = []
        for pct in percentiles[:n_samples]:
            target_val = np.percentile(fire_fracs[positive_mask], pct)
            candidates = np.where(np.abs(fire_fracs - target_val) < 0.02)[0]
            if len(candidates) > 0:
                # Pick one not already selected
                for c in candidates:
                    if c not in indices:
                        indices.append(int(c))
                        break
        # Fill remaining if needed
        while len(indices) < n_samples:
            idx = int(np.argsort(fire_fracs)[-1 - len(indices)])
            if idx not in indices:
                indices.append(idx)

    # Create figure: n_samples rows × 4 columns
    # Columns: Ground Truth | Before CRC (p≥0.5) | After CRC (p≥λ̂) | Error Comparison
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4.2 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    legend_patches = [
        mpatches.Patch(color="#2ecc71", label="True Positive (caught fire)"),
        mpatches.Patch(color="#e74c3c", label="False Negative (MISSED fire)"),
        mpatches.Patch(color="#3498db", label="False Positive (over-alert)"),
        mpatches.Patch(color="#ecf0f1", label="True Negative"),
    ]

    for row, sample_idx in enumerate(indices):
        gt = sp_targets[sample_idx]
        probs = sp_probs[sample_idx]
        mask = sp_masks[sample_idx]
        valid = mask == 1

        fire_pct = fire_fracs[sample_idx] * 100

        # Column 0: Ground Truth
        gt_display = np.where(valid, gt, np.nan)
        axes[row, 0].imshow(gt_display, cmap="Reds", vmin=0, vmax=1,
                            interpolation="nearest")
        axes[row, 0].set_title(f"Ground Truth ({fire_pct:.1f}% fire)" if row == 0
                               else f"Ground Truth ({fire_pct:.1f}%)", fontsize=11)

        # Column 1: Before CRC (standard p≥0.5) — error overlay
        pred_std = probs >= 0.5
        overlay_before = np.full((*gt.shape, 3), 0.92)  # light gray
        tp_b = valid & (gt == 1) & pred_std
        fn_b = valid & (gt == 1) & ~pred_std
        fp_b = valid & (gt == 0) & pred_std
        tn_b = valid & (gt == 0) & ~pred_std
        overlay_before[tp_b] = [0.18, 0.8, 0.44]    # green — caught
        overlay_before[fn_b] = [0.91, 0.30, 0.24]    # red — MISSED
        overlay_before[fp_b] = [0.20, 0.60, 0.86]    # blue — over-alert
        overlay_before[tn_b] = [0.93, 0.94, 0.95]    # near-white
        overlay_before[~valid] = [0.5, 0.5, 0.5]

        fnr_before = fn_b.sum() / max((tp_b.sum() + fn_b.sum()), 1)
        axes[row, 1].imshow(overlay_before, interpolation="nearest")
        title_before = (f"Before CRC ($p \\geq 0.5$)\nMissed: {fnr_before:.0%} of fires"
                        if row == 0
                        else f"$p \\geq 0.5$ — Missed: {fnr_before:.0%}")
        axes[row, 1].set_title(title_before, fontsize=11, color="#c0392b")

        # Column 2: After CRC (p≥λ̂) — error overlay
        pred_crc = probs >= lam_crc
        overlay_after = np.full((*gt.shape, 3), 0.92)
        tp_a = valid & (gt == 1) & pred_crc
        fn_a = valid & (gt == 1) & ~pred_crc
        fp_a = valid & (gt == 0) & pred_crc
        tn_a = valid & (gt == 0) & ~pred_crc
        overlay_after[tp_a] = [0.18, 0.8, 0.44]
        overlay_after[fn_a] = [0.91, 0.30, 0.24]
        overlay_after[fp_a] = [0.20, 0.60, 0.86]
        overlay_after[tn_a] = [0.93, 0.94, 0.95]
        overlay_after[~valid] = [0.5, 0.5, 0.5]

        fnr_after = fn_a.sum() / max((tp_a.sum() + fn_a.sum()), 1)
        title_after = (f"After CRC ($\\hat{{\\lambda}}$={lam_crc:.4f})\n"
                       f"Missed: {fnr_after:.0%} of fires"
                       if row == 0
                       else f"CRC $\\hat{{\\lambda}}$={lam_crc:.4f} — Missed: {fnr_after:.0%}")
        axes[row, 2].set_title(title_after, fontsize=11, color="#27ae60")
        axes[row, 2].imshow(overlay_after, interpolation="nearest")

        # Column 3: Three-way CRC zones
        tw = results.get("U-Net_threeway", {})
        lam_min = tw.get("lambda_min", 0)
        lam_max = tw.get("lambda_max", 0.02)

        threeway_map = np.full_like(probs, np.nan)
        threeway_map[valid & (probs < lam_min)] = 0       # SAFE
        threeway_map[valid & (probs >= lam_min) & (probs < lam_max)] = 0.5  # MONITOR
        threeway_map[valid & (probs >= lam_max)] = 1.0     # EVACUATE

        cmap_tw = mcolors.ListedColormap(["#2ecc71", "#f39c12", "#e74c3c"])
        bounds = [-0.25, 0.25, 0.75, 1.25]
        norm_tw = mcolors.BoundaryNorm(bounds, cmap_tw.N)
        axes[row, 3].imshow(threeway_map, cmap=cmap_tw, norm=norm_tw,
                            interpolation="nearest")
        axes[row, 3].set_title(
            "Three-Way CRC\n(SAFE / MONITOR / EVACUATE)" if row == 0
            else "Three-Way CRC", fontsize=11)

    # Remove ticks
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    # Add legend
    fig.legend(handles=legend_patches, loc="lower center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.42, -0.02), frameon=True, edgecolor="gray")

    # Three-way legend
    tw_patches = [
        mpatches.Patch(color="#2ecc71", label="SAFE"),
        mpatches.Patch(color="#f39c12", label="MONITOR"),
        mpatches.Patch(color="#e74c3c", label="EVACUATE"),
    ]
    fig.legend(handles=tw_patches, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.88, -0.02), frameon=True, edgecolor="gray")

    fig.suptitle(
        "The Safety Gap: Standard Thresholds vs Conformal Risk Control",
        fontsize=15, fontweight="bold", y=1.01,
    )

    fig.tight_layout()
    fig.savefig(output_dir / "before_after_crc.pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / "before_after_crc.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved before_after_crc.pdf/png ({n_samples} samples)")


# ------------------------------------------------------------------
# Figure: 3-Model Comparison Bar Chart
# ------------------------------------------------------------------

def plot_three_model_comparison(results: dict, output_dir: Path, dpi: int) -> None:
    """Bar chart comparing all 3 models: LightGBM, U-Net, ResGNN-UNet."""
    model_names = ["LightGBM", "U-Net", "ResGNN-UNet"]
    colors_bar = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # Pull all values from results — no hardcoding
    aurocs = [results[f"{m}_standard"]["auroc"] for m in model_names]
    coverages_std = [results[f"{m}_standard"]["coverage"] for m in model_names]
    crc_coverages = [results[f"{m}_crc"]["coverage"] for m in model_names]
    crc_set_sizes = [results[f"{m}_crc"]["set_size"] for m in model_names]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # --- Panel 1: AUROC (model ranking quality) ---
    bars = axes[0].bar(model_names, aurocs, color=colors_bar, edgecolor="black",
                       linewidth=0.5, width=0.6)
    axes[0].set_ylabel("AUROC", fontsize=12)
    axes[0].set_title("Model Quality\n(ranking ability)", fontsize=13, fontweight="bold")
    axes[0].set_ylim(0.7, 1.02)
    axes[0].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, aurocs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # --- Panel 2: Coverage at standard threshold (UNSAFE!) ---
    bars2 = axes[1].bar(model_names, coverages_std, color=colors_bar, edgecolor="black",
                        linewidth=0.5, width=0.6)
    axes[1].axhline(0.95, color="red", linestyle="--", linewidth=2, label="95% safety target")
    axes[1].set_ylabel("Coverage (fire detection rate)", fontsize=12)
    axes[1].set_title("Without CRC\n(standard threshold)", fontsize=13, fontweight="bold",
                      color="#c0392b")
    axes[1].set_ylim(0, 1.15)
    axes[1].legend(fontsize=10, loc="upper left")
    axes[1].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, coverages_std):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{val:.1%}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # UNSAFE annotation
    for bar in bars2:
        axes[1].text(bar.get_x() + bar.get_width() / 2, 0.97,
                     "UNSAFE", ha="center", va="bottom", fontsize=8,
                     color="red", fontweight="bold")

    # --- Panel 3: With CRC — coverage + set size (all 3 models) ---
    crc_labels = [f"{m}\n+ CRC" for m in model_names]

    x = np.arange(len(crc_labels))
    width = 0.35

    bars3a = axes[2].bar(x - width / 2, crc_coverages, width, color=colors_bar,
                         edgecolor="black", linewidth=0.5, label="Coverage", alpha=0.9)
    bars3b = axes[2].bar(x + width / 2, crc_set_sizes, width, color=colors_bar,
                         edgecolor="black", linewidth=0.5, label="Set Size",
                         alpha=0.4, hatch="///")

    axes[2].axhline(0.95, color="red", linestyle="--", linewidth=2, label="95% target")
    axes[2].set_ylabel("Fraction", fontsize=12)
    axes[2].set_title("With CRC\n(guaranteed safe)", fontsize=13, fontweight="bold",
                      color="#27ae60")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(crc_labels, fontsize=10)
    axes[2].set_ylim(0, 1.15)
    axes[2].legend(fontsize=9, loc="upper right")
    axes[2].grid(axis="y", alpha=0.3)

    for bar, val in zip(bars3a, crc_coverages):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{val:.1%}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar, val in zip(bars3b, crc_set_sizes):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{val:.1%}", ha="center", va="bottom", fontsize=10)

    # Efficiency annotation: compare LightGBM vs best spatial model
    lgbm_set = crc_set_sizes[0]
    best_spatial_set = min(crc_set_sizes[1], crc_set_sizes[2])
    ratio = lgbm_set / best_spatial_set
    axes[2].annotate(f"{ratio:.1f}× smaller\nevacuation zone",
                     xy=(1 + width / 2, crc_set_sizes[1]), xytext=(2.2, 0.45),
                     fontsize=10, fontweight="bold", color="#27ae60",
                     arrowprops=dict(arrowstyle="->", color="#27ae60", lw=2),
                     ha="center")

    fig.tight_layout()
    fig.savefig(output_dir / "three_model_comparison.pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / "three_model_comparison.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved three_model_comparison.pdf/png")


# ------------------------------------------------------------------
# Figure: ResGNN-UNet Error Map (copy from Kaggle output)
# ------------------------------------------------------------------

def copy_resgnn_figures(output_dir: Path) -> None:
    """Copy ResGNN-UNet figures from Kaggle notebook output."""
    kaggle_dir = Path("/tmp/kaggle_notebook_output")
    copied = 0
    for name in ["error_analysis.png", "learning_curves.png", "prediction_sample.png"]:
        src = kaggle_dir / name
        if src.exists():
            dst = output_dir / f"resgnn_{name}"
            shutil.copy2(src, dst)
            copied += 1
            print(f"  Copied {name} → resgnn_{name}")
    if copied == 0:
        print("  No ResGNN-UNet figures found in /tmp/kaggle_notebook_output/")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results_path = args.eval_dir / "full_results.json"
    results = json.loads(results_path.read_text())

    print("Generating extra figures...")

    plot_before_after_crc(args.spatial_dir, results, args.output_dir, args.dpi)
    plot_three_model_comparison(results, args.output_dir, args.dpi)
    copy_resgnn_figures(args.output_dir)

    print(f"\nAll extra figures saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
