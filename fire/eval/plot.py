"""Phase 6: Generate publication-quality figures for the paper.

Produces:
  1. FNR sweep curves with threshold annotations
  2. Safety vs Efficiency bar chart
  3. Qualitative side-by-side maps (ground truth vs predictions)
  4. Training curves (from training_history.json)

Usage
-----
    fire-plot              # generate all figures
    fire-plot --output-dir figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


COLORS = {
    "LightGBM": "#1f77b4",
    "U-Net": "#ff7f0e",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=Path("data/evaluation"))
    parser.add_argument("--spatial-dir", type=Path,
                        default=Path("data/predictions/spatial_baseline"))
    parser.add_argument("--tabular-dir", type=Path,
                        default=Path("data/predictions/tabular_baseline"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


# ------------------------------------------------------------------
# Figure 1: FNR Sweep Curves
# ------------------------------------------------------------------

def plot_fnr_sweep(eval_dir: Path, results: dict, output_dir: Path, dpi: int) -> None:
    """FNR vs threshold with CRC and three-way annotations."""
    fig, ax = plt.subplots(figsize=(7, 4))

    for model_name in ("LightGBM", "U-Net"):
        sweep_path = eval_dir / f"{model_name}_sweep.npz"
        if not sweep_path.exists():
            continue
        data = np.load(sweep_path)
        ax.plot(data["thresholds"], data["fnr"],
                label=model_name, color=COLORS[model_name], linewidth=2)

    # Horizontal target line
    ax.axhline(0.05, color="red", linestyle="--", linewidth=1, label=r"$\alpha = 0.05$")

    # CRC threshold annotations
    for model_name, marker in [("LightGBM", "s"), ("U-Net", "o")]:
        key = f"{model_name}_crc"
        if key in results:
            lam = results[key].get("lambda_hat", results[key].get("threshold"))
            fnr = results[key]["fnr"]
            ax.plot(lam, fnr, marker=marker, color=COLORS[model_name],
                    markersize=10, zorder=5, markeredgecolor="black",
                    label=f"{model_name} CRC $\\hat{{\\lambda}}$={lam:.4f}")

    # Standard threshold
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.7,
               label="Standard ($p \\geq 0.5$)")

    ax.set_xlabel("Threshold $\\lambda$", fontsize=12)
    ax.set_ylabel("False Negative Rate (FNR)", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fnr_sweep.pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / "fnr_sweep.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fnr_sweep.pdf/png")


# ------------------------------------------------------------------
# Figure 2: Safety vs Efficiency Bar Chart
# ------------------------------------------------------------------

def plot_safety_efficiency(results: dict, output_dir: Path, dpi: int) -> None:
    """Grouped bar chart: Coverage and Set Size for all 6 methods."""
    methods = []
    coverages = []
    set_sizes = []
    colors = []

    order = [
        ("LightGBM_standard", "LGBM\n(p≥0.5)"),
        ("LightGBM_crc", "LGBM\n+ CRC"),
        ("LightGBM_threeway", "LGBM\n+ 3-way"),
        ("U-Net_standard", "U-Net\n(p≥0.5)"),
        ("U-Net_crc", "U-Net\n+ CRC"),
        ("U-Net_threeway", "U-Net\n+ 3-way"),
    ]

    for key, label in order:
        if key not in results:
            continue
        methods.append(label)
        coverages.append(results[key]["coverage"])
        set_sizes.append(results[key]["set_size"])
        model = "LightGBM" if "LightGBM" in key else "U-Net"
        colors.append(COLORS[model])

    x = np.arange(len(methods))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Coverage
    bars1 = ax1.bar(x, coverages, width, color=colors, edgecolor="black", linewidth=0.5)
    ax1.axhline(0.95, color="red", linestyle="--", linewidth=1.5, label="95% target")
    ax1.set_ylabel("Coverage (1 − FNR)", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=9)
    ax1.set_ylim(0, 1.1)
    ax1.legend(fontsize=10)
    ax1.set_title("Safety: Fire Detection Rate", fontsize=13, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars1, coverages):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.1%}", ha="center", va="bottom", fontsize=8)

    # Set Size
    bars2 = ax2.bar(x, set_sizes, width, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Set Size (fraction flagged)", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=9)
    ax2.set_title("Efficiency: Evacuation Zone Size", fontsize=13, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, set_sizes):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.1%}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "safety_efficiency.pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / "safety_efficiency.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved safety_efficiency.pdf/png")


# ------------------------------------------------------------------
# Figure 3: Qualitative Maps (Hero Figure)
# ------------------------------------------------------------------

def plot_qualitative_maps(
    spatial_dir: Path,
    tabular_dir: Path,
    results: dict,
    output_dir: Path,
    dpi: int,
    sample_idx: int = 0,
) -> None:
    """Side-by-side maps: ground truth, standard predictions, CRC predictions."""
    # Load spatial test data
    sp_data = np.load(spatial_dir / "test_probability_heatmaps.npz")
    sp_probs = sp_data["probabilities"]
    sp_targets = sp_data["targets"]
    sp_masks = sp_data["valid_masks"]

    # Find a sample with meaningful fire content
    fire_fracs = []
    for i in range(sp_targets.shape[0]):
        valid = sp_masks[i] == 1
        if valid.sum() > 0:
            fire_fracs.append(float(sp_targets[i][valid].sum()) / valid.sum())
        else:
            fire_fracs.append(0.0)
    fire_fracs = np.array(fire_fracs)

    # Pick sample near 90th percentile of fire fraction (visually interesting)
    target_pct = np.percentile(fire_fracs[fire_fracs > 0], 90) if (fire_fracs > 0).sum() > 0 else 0
    candidates = np.where(np.abs(fire_fracs - target_pct) < 0.02)[0]
    sample_idx = int(candidates[0]) if len(candidates) > 0 else int(np.argmax(fire_fracs))

    gt = sp_targets[sample_idx]
    probs_sp = sp_probs[sample_idx]
    mask = sp_masks[sample_idx]

    # Get CRC thresholds
    lam_sp = results.get("U-Net_crc", {}).get("lambda_hat",
             results.get("U-Net_crc", {}).get("threshold", 0.5))
    lam_lgbm = results.get("LightGBM_crc", {}).get("lambda_hat",
               results.get("LightGBM_crc", {}).get("threshold", 0.5))

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Row 1: Ground truth, U-Net probability, U-Net CRC
    # Ground truth
    gt_display = np.where(mask == 1, gt, np.nan)
    im0 = axes[0, 0].imshow(gt_display, cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
    axes[0, 0].set_title("Ground Truth", fontsize=12, fontweight="bold")

    # U-Net probability heatmap
    prob_display = np.where(mask == 1, probs_sp, np.nan)
    im1 = axes[0, 1].imshow(prob_display, cmap="hot", vmin=0, vmax=1, interpolation="nearest")
    axes[0, 1].set_title("U-Net Probabilities", fontsize=12)
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    # U-Net standard (p >= 0.5)
    pred_std = np.where(mask == 1, (probs_sp >= 0.5).astype(float), np.nan)
    axes[0, 2].imshow(pred_std, cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
    axes[0, 2].set_title("U-Net Standard ($p \\geq 0.5$)", fontsize=12)

    # Row 2: U-Net CRC, U-Net Three-Way, overlay
    # U-Net CRC
    pred_crc = np.where(mask == 1, (probs_sp >= lam_sp).astype(float), np.nan)
    axes[1, 0].imshow(pred_crc, cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
    axes[1, 0].set_title(f"U-Net + CRC ($\\hat{{\\lambda}}$={lam_sp:.4f})", fontsize=12)

    # U-Net Three-Way
    tw_data = results.get("U-Net_threeway", {})
    lam_min = tw_data.get("lambda_min", 0)
    lam_max = tw_data.get("lambda_max", 0.02)

    threeway_map = np.full_like(probs_sp, np.nan)
    valid = mask == 1
    threeway_map[valid & (probs_sp < lam_min)] = 0     # SAFE (green)
    threeway_map[valid & (probs_sp >= lam_min) & (probs_sp < lam_max)] = 0.5  # MONITOR (yellow)
    threeway_map[valid & (probs_sp >= lam_max)] = 1.0   # EVACUATE (red)

    cmap_tw = mcolors.ListedColormap(["#2ecc71", "#f39c12", "#e74c3c"])
    bounds = [-0.25, 0.25, 0.75, 1.25]
    norm_tw = mcolors.BoundaryNorm(bounds, cmap_tw.N)
    axes[1, 1].imshow(threeway_map, cmap=cmap_tw, norm=norm_tw, interpolation="nearest")
    axes[1, 1].set_title("U-Net + Three-Way CRC", fontsize=12)

    # Overlay: CRC prediction vs ground truth (FP=blue, TP=red, FN=yellow)
    overlay = np.full((*gt.shape, 3), 0.9)  # light gray background
    tp = valid & (gt == 1) & (probs_sp >= lam_sp)
    fp = valid & (gt == 0) & (probs_sp >= lam_sp)
    fn = valid & (gt == 1) & (probs_sp < lam_sp)
    tn = valid & (gt == 0) & (probs_sp < lam_sp)
    overlay[tp] = [0.8, 0.1, 0.1]   # red — true positive
    overlay[fp] = [0.2, 0.4, 0.8]   # blue — false positive
    overlay[fn] = [1.0, 0.8, 0.0]   # yellow — false negative (missed fire!)
    overlay[tn] = [0.95, 0.95, 0.95] # light gray — true negative
    overlay[~valid] = [0.5, 0.5, 0.5]  # dark gray — no data

    axes[1, 2].imshow(overlay, interpolation="nearest")
    axes[1, 2].set_title("CRC Error Map (TP/FP/FN/TN)", fontsize=12)

    # Remove tick labels
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Test Sample #{sample_idx}: Qualitative Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "qualitative_maps.pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / "qualitative_maps.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved qualitative_maps.pdf/png (sample #{sample_idx})")


# ------------------------------------------------------------------
# Figure 4: Training Curves
# ------------------------------------------------------------------

def plot_training_curves(spatial_dir: Path, output_dir: Path, dpi: int) -> None:
    """Plot training and validation loss from training_history.json."""
    history_path = spatial_dir / "training_history.json"
    if not history_path.exists():
        print("  Skipping training curves (no training_history.json)")
        return

    history = json.loads(history_path.read_text())
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss curves
    ax1.plot(epochs, history["train_loss"], label="Train BCE", color="#1f77b4", linewidth=2)
    ax1.plot(epochs, history["val_loss"], label="Val BCE", color="#ff7f0e", linewidth=2)
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    best_val = min(history["val_loss"])
    ax1.axvline(best_epoch, color="green", linestyle="--", alpha=0.7,
                label=f"Best (epoch {best_epoch}, {best_val:.4f})")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("BCE Loss", fontsize=12)
    ax1.set_title("Training & Validation Loss", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Learning rate
    ax2.plot(epochs, history["lr"], color="#2ca02c", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Learning Rate", fontsize=12)
    ax2.set_title("Cosine Annealing Schedule", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / "training_curves.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved training_curves.pdf/png")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results_path = args.eval_dir / "full_results.json"
    results = json.loads(results_path.read_text())

    print("Generating figures...")

    plot_fnr_sweep(args.eval_dir, results, args.output_dir, args.dpi)
    plot_safety_efficiency(results, args.output_dir, args.dpi)
    plot_qualitative_maps(
        args.spatial_dir, args.tabular_dir, results, args.output_dir, args.dpi,
    )
    plot_training_curves(args.spatial_dir, args.output_dir, args.dpi)

    print(f"\nAll figures saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
