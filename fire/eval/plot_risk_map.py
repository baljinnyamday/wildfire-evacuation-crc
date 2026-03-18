"""Risk gradient map: per-pixel fire probability with CRC threshold contours.

Shows the continuous probability landscape instead of binary yes/no,
with the CRC threshold drawn as a contour boundary.

Usage
-----
    fire-plot-risk              # generate risk gradient figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.lines import Line2D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=Path("data/evaluation"))
    parser.add_argument("--spatial-dir", type=Path,
                        default=Path("data/predictions/spatial_baseline"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--n-samples", type=int, default=3)
    return parser.parse_args()


def _pick_samples(
    targets: np.ndarray,
    masks: np.ndarray,
    n: int,
) -> list[int]:
    """Pick diverse samples with meaningful fire content."""
    fire_fracs = []
    for i in range(targets.shape[0]):
        valid = masks[i] == 1
        if valid.sum() > 0:
            fire_fracs.append(float(targets[i][valid].sum()) / valid.sum())
        else:
            fire_fracs.append(0.0)
    fire_fracs = np.array(fire_fracs)

    positive = fire_fracs > 0.02
    if positive.sum() < n:
        return list(np.argsort(fire_fracs)[-n:])

    percentiles = [60, 82, 96]
    indices: list[int] = []
    for pct in percentiles[:n]:
        target_val = np.percentile(fire_fracs[positive], pct)
        candidates = np.where(np.abs(fire_fracs - target_val) < 0.015)[0]
        for c in candidates:
            if int(c) not in indices:
                indices.append(int(c))
                break

    while len(indices) < n:
        idx = int(np.argsort(fire_fracs)[-1 - len(indices)])
        if idx not in indices:
            indices.append(idx)

    return indices


# ------------------------------------------------------------------
# Figure: Continuous Risk Gradient Map
# ------------------------------------------------------------------

def plot_risk_gradient(
    spatial_dir: Path,
    results: dict,
    output_dir: Path,
    dpi: int,
    n_samples: int = 3,
) -> None:
    """Per-pixel probability heatmap with CRC threshold contour."""
    sp_data = np.load(spatial_dir / "test_probability_heatmaps.npz")
    probs_all = sp_data["probabilities"]
    targets_all = sp_data["targets"]
    masks_all = sp_data["valid_masks"]

    lam_crc = results["U-Net_crc"]["lambda_hat"]
    lam_max_tw = results["U-Net_threeway"]["lambda_max"]

    indices = _pick_samples(targets_all, masks_all, n_samples)

    # Colormap: white → yellow → orange → red → dark red
    risk_colors = ["#ffffff", "#fff7bc", "#fec44f", "#fe9929",
                   "#ec7014", "#cc4c02", "#8c2d04"]
    risk_cmap = mcolors.LinearSegmentedColormap.from_list("fire_risk", risk_colors, N=256)

    fig, axes = plt.subplots(n_samples, 4, figsize=(18, 4.5 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for row, sample_idx in enumerate(indices):
        gt = targets_all[sample_idx]
        probs = probs_all[sample_idx]
        mask = masks_all[sample_idx]
        valid = mask == 1

        fire_pct = (gt[valid].sum() / valid.sum()) * 100 if valid.sum() > 0 else 0

        # --- Column 0: Ground Truth ---
        gt_display = np.where(valid, gt, np.nan)
        axes[row, 0].imshow(gt_display, cmap="Reds", vmin=0, vmax=1,
                            interpolation="nearest")
        axes[row, 0].set_title(
            f"Ground Truth\n({fire_pct:.1f}% fire)" if row == 0
            else f"Ground Truth ({fire_pct:.1f}%)", fontsize=12, fontweight="bold")

        # --- Column 1: Continuous probability heatmap ---
        prob_display = np.where(valid, probs, np.nan)
        im = axes[row, 1].imshow(prob_display, cmap=risk_cmap, vmin=0, vmax=1,
                                 interpolation="bilinear")
        if row == 0:
            axes[row, 1].set_title("Per-Pixel Fire Probability\n$p(\\mathrm{fire} \\mid x)$",
                                   fontsize=12, fontweight="bold")
        else:
            axes[row, 1].set_title("Fire Probability", fontsize=12)
        plt.colorbar(im, ax=axes[row, 1], fraction=0.046, pad=0.04,
                     label="$p(\\mathrm{fire})$" if row == n_samples - 1 else "")

        # --- Column 2: Risk zones with probability bands ---
        # Create banded risk zones: >80%, 50-80%, 20-50%, 5-20%, CRC-5%, <CRC threshold
        zone_colors = [
            "#f7f7f7",   # 0: below CRC threshold (safe-ish)
            "#fef0d9",   # 1: CRC threshold to 5%
            "#fdcc8a",   # 2: 5% to 20%
            "#fc8d59",   # 3: 20% to 50%
            "#e34a33",   # 4: 50% to 80%
            "#b30000",   # 5: >80%
        ]
        zone_cmap = mcolors.ListedColormap(zone_colors)
        bounds = [0, lam_crc, 0.05, 0.20, 0.50, 0.80, 1.01]
        zone_norm = mcolors.BoundaryNorm(bounds, zone_cmap.N)

        zone_map = np.where(valid, probs, np.nan)
        axes[row, 2].imshow(zone_map, cmap=zone_cmap, norm=zone_norm,
                            interpolation="nearest")

        # Add CRC threshold contour
        prob_contour = np.where(valid, probs, 0)
        try:
            axes[row, 2].contour(prob_contour, levels=[lam_crc],
                                 colors=["#2ecc71"], linewidths=2.5, linestyles="--")
            axes[row, 2].contour(prob_contour, levels=[lam_max_tw],
                                 colors=["#e74c3c"], linewidths=2, linestyles="-")
        except ValueError:
            pass

        if row == 0:
            axes[row, 2].set_title(
                "Risk Zones\n(probability bands + CRC boundary)",
                fontsize=12, fontweight="bold")
        else:
            axes[row, 2].set_title("Risk Zones", fontsize=12)

        # --- Column 3: Calibrated confidence map ---
        # For each pixel, compute: "this pixel is under fire with X% confidence"
        # Use the CRC guarantee: pixels above λ̂ are in the 95%-coverage set
        conf_map = np.full((*gt.shape, 3), 0.95)  # light gray bg
        conf_map[~valid] = [0.5, 0.5, 0.5]

        # Gradient from green (safe) through yellow (uncertain) to red (fire)
        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                if not valid[i, j]:
                    continue
                p = probs[i, j]
                if p < lam_crc:
                    # Below CRC threshold: green-ish, scaled by how far below
                    t = p / max(lam_crc, 1e-6)
                    conf_map[i, j] = [0.2 + 0.7 * t, 0.8 - 0.3 * t, 0.2]
                elif p < 0.2:
                    # Low-medium risk: yellow
                    t = (p - lam_crc) / (0.2 - lam_crc)
                    conf_map[i, j] = [1.0, 0.9 - 0.4 * t, 0.1]
                elif p < 0.5:
                    # Medium risk: orange
                    t = (p - 0.2) / 0.3
                    conf_map[i, j] = [1.0, 0.5 - 0.3 * t, 0.05]
                else:
                    # High risk: deep red
                    t = min((p - 0.5) / 0.5, 1.0)
                    conf_map[i, j] = [0.9 - 0.3 * t, 0.1, 0.05]

        axes[row, 3].imshow(conf_map, interpolation="bilinear")
        if row == 0:
            axes[row, 3].set_title(
                "Calibrated Risk Gradient\n(CRC-informed evacuation map)",
                fontsize=12, fontweight="bold")
        else:
            axes[row, 3].set_title("Calibrated Risk Gradient", fontsize=12)

    # Remove ticks
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    # Legends
    zone_legend = [
        mpatches.Patch(color="#f7f7f7", edgecolor="gray",
                       label=f"< CRC threshold ({lam_crc:.4f})"),
        mpatches.Patch(color="#fef0d9", edgecolor="gray", label="CRC–5%"),
        mpatches.Patch(color="#fdcc8a", edgecolor="gray", label="5–20%"),
        mpatches.Patch(color="#fc8d59", edgecolor="gray", label="20–50%"),
        mpatches.Patch(color="#e34a33", edgecolor="gray", label="50–80%"),
        mpatches.Patch(color="#b30000", edgecolor="gray", label="> 80%"),
        Line2D([0], [0], color="#2ecc71", linewidth=2.5, linestyle="--",
               label=f"CRC boundary ($\\hat{{\\lambda}}$={lam_crc:.4f})"),
        Line2D([0], [0], color="#e74c3c", linewidth=2, linestyle="-",
               label=f"EVACUATE boundary ({lam_max_tw:.4f})"),
    ]
    fig.legend(handles=zone_legend, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.04), frameon=True, edgecolor="gray")

    fig.suptitle(
        "Per-Pixel Fire Risk: From Raw Probability to Calibrated Evacuation Zones",
        fontsize=15, fontweight="bold", y=1.01,
    )

    fig.tight_layout()
    fig.savefig(output_dir / "risk_gradient_map.pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / "risk_gradient_map.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved risk_gradient_map.pdf/png ({n_samples} samples)")


# ------------------------------------------------------------------
# Figure: Single-sample deep dive with probability cross-section
# ------------------------------------------------------------------

def plot_probability_cross_section(
    spatial_dir: Path,
    results: dict,
    output_dir: Path,
    dpi: int,
) -> None:
    """Show a 1D cross-section through a fire region with CRC threshold."""
    sp_data = np.load(spatial_dir / "test_probability_heatmaps.npz")
    probs_all = sp_data["probabilities"]
    targets_all = sp_data["targets"]
    masks_all = sp_data["valid_masks"]

    lam_crc = results["U-Net_crc"]["lambda_hat"]

    # Find a good sample with a clear fire cluster
    indices = _pick_samples(targets_all, masks_all, 1)
    idx = indices[0]

    gt = targets_all[idx]
    probs = probs_all[idx]
    mask = masks_all[idx]
    valid = mask == 1

    # Find the row with maximum fire pixels for cross-section
    fire_per_row = []
    for r in range(gt.shape[0]):
        fire_per_row.append((gt[r] * (mask[r] == 1)).sum())
    best_row = int(np.argmax(fire_per_row))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                             gridspec_kw={"width_ratios": [1, 1.3]})

    # Left: probability heatmap with cross-section line
    risk_colors = ["#ffffff", "#fff7bc", "#fec44f", "#fe9929",
                   "#ec7014", "#cc4c02", "#8c2d04"]
    risk_cmap = mcolors.LinearSegmentedColormap.from_list("fire_risk", risk_colors, N=256)

    prob_display = np.where(valid, probs, np.nan)
    im = axes[0].imshow(prob_display, cmap=risk_cmap, vmin=0, vmax=1,
                        interpolation="bilinear")
    axes[0].axhline(best_row, color="cyan", linewidth=2, linestyle="--",
                    label=f"Cross-section (row {best_row})")

    # Draw CRC contour
    prob_contour = np.where(valid, probs, 0)
    try:
        axes[0].contour(prob_contour, levels=[lam_crc],
                        colors=["#2ecc71"], linewidths=2, linestyles="--")
    except ValueError:
        pass

    axes[0].set_title("U-Net Probability Map\n(fire risk per pixel)", fontsize=13,
                      fontweight="bold")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    plt.colorbar(im, ax=axes[0], fraction=0.046, label="$p(\\mathrm{fire})$")
    axes[0].legend(loc="lower left", fontsize=9)

    # Right: 1D cross-section plot
    row_probs = probs[best_row, :]
    row_gt = gt[best_row, :]
    row_mask = mask[best_row, :]
    x = np.arange(len(row_probs))

    # Plot probability curve
    masked_probs = np.where(row_mask == 1, row_probs, np.nan)
    axes[1].fill_between(x, 0, masked_probs, alpha=0.3, color="#e74c3c",
                         label="Fire probability")
    axes[1].plot(x, masked_probs, color="#c0392b", linewidth=2)

    # Highlight ground truth fire pixels
    fire_pixels = np.where((row_gt == 1) & (row_mask == 1))[0]
    if len(fire_pixels) > 0:
        axes[1].fill_between(x, 0, np.where((row_gt == 1) & (row_mask == 1), 1.05, 0),
                             alpha=0.15, color="red", label="Actual fire")

    # CRC threshold line
    axes[1].axhline(lam_crc, color="#2ecc71", linewidth=2.5, linestyle="--",
                    label=f"CRC threshold $\\hat{{\\lambda}}$={lam_crc:.4f}\n"
                          f"(guarantees ≥95% fire detection)")
    axes[1].axhline(0.5, color="gray", linewidth=1.5, linestyle=":",
                    label="Standard threshold ($p \\geq 0.5$)")

    # Shade the CRC-flagged region
    flagged = masked_probs >= lam_crc
    axes[1].fill_between(x, lam_crc, np.where(flagged, masked_probs, lam_crc),
                         alpha=0.2, color="#2ecc71", label="CRC evacuation zone")

    axes[1].set_xlabel("Pixel position along cross-section", fontsize=12)
    axes[1].set_ylabel("$p(\\mathrm{fire})$", fontsize=12)
    axes[1].set_title(f"Probability Cross-Section (row {best_row})\n"
                      f"Fire probability decays with distance from fire center",
                      fontsize=13, fontweight="bold")
    axes[1].set_ylim(-0.05, 1.1)
    axes[1].set_xlim(0, 63)
    axes[1].legend(fontsize=9, loc="upper right")
    axes[1].grid(True, alpha=0.3)

    # Annotate the key insight
    axes[1].annotate(
        "CRC catches fire pixels\nthat standard threshold misses",
        xy=(fire_pixels[0] - 3 if len(fire_pixels) > 0 else 20, lam_crc + 0.02),
        xytext=(5, 0.65),
        fontsize=10, fontweight="bold", color="#27ae60",
        arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.5),
    )

    fig.tight_layout()
    fig.savefig(output_dir / "probability_cross_section.pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / "probability_cross_section.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved probability_cross_section.pdf/png")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = json.loads((args.eval_dir / "full_results.json").read_text())

    print("Generating risk gradient figures...")
    plot_risk_gradient(args.spatial_dir, results, args.output_dir, args.dpi,
                       n_samples=args.n_samples)
    plot_probability_cross_section(args.spatial_dir, results, args.output_dir, args.dpi)

    print(f"\nAll risk figures saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
