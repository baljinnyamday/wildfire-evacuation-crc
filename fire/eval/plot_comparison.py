"""Before vs After CRC: side-by-side probability maps with threshold comparison.

Shows the same fire probability landscape under standard model vs CRC,
making the safety gap visually undeniable.

Usage
-----
    fire-plot-compare
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
    return parser.parse_args()


RISK_COLORS = ["#ffffff", "#fff7bc", "#fec44f", "#fe9929",
               "#ec7014", "#cc4c02", "#8c2d04"]
RISK_CMAP = mcolors.LinearSegmentedColormap.from_list("fire_risk", RISK_COLORS, N=256)


def _pick_best_samples(
    targets: np.ndarray,
    masks: np.ndarray,
    n: int = 4,
) -> list[int]:
    """Pick samples with diverse, visually interesting fire content."""
    fire_fracs = []
    for i in range(targets.shape[0]):
        valid = masks[i] == 1
        if valid.sum() > 0:
            fire_fracs.append(float(targets[i][valid].sum()) / valid.sum())
        else:
            fire_fracs.append(0.0)
    fire_fracs = np.array(fire_fracs)

    positive = fire_fracs > 0.01
    percentiles = [40, 65, 85, 96]
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
# Hero Figure: Side-by-side bare model vs CRC
# ------------------------------------------------------------------

def plot_bare_vs_crc(
    spatial_dir: Path,
    results: dict,
    output_dir: Path,
    dpi: int,
) -> None:
    """4 samples × 5 columns: GT, probability, bare model, CRC model, cross-section."""
    sp_data = np.load(spatial_dir / "test_probability_heatmaps.npz")
    probs_all = sp_data["probabilities"]
    targets_all = sp_data["targets"]
    masks_all = sp_data["valid_masks"]

    lam_crc = results["U-Net_crc"]["lambda_hat"]
    n_samples = 4
    indices = _pick_best_samples(targets_all, masks_all, n_samples)

    fig, axes = plt.subplots(n_samples, 5, figsize=(22, 4 * n_samples),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.4]})

    for row, sample_idx in enumerate(indices):
        gt = targets_all[sample_idx]
        probs = probs_all[sample_idx]
        mask = masks_all[sample_idx]
        valid = mask == 1
        fire_pct = (gt[valid].sum() / valid.sum()) * 100 if valid.sum() > 0 else 0

        # --- Col 0: Ground Truth ---
        gt_display = np.where(valid, gt, np.nan)
        axes[row, 0].imshow(gt_display, cmap="Reds", vmin=0, vmax=1,
                            interpolation="nearest")
        axes[row, 0].set_ylabel(f"Sample #{sample_idx}\n({fire_pct:.1f}% fire)",
                                fontsize=11, fontweight="bold")

        # --- Col 1: Probability heatmap ---
        prob_display = np.where(valid, probs, np.nan)
        im = axes[row, 1].imshow(prob_display, cmap=RISK_CMAP, vmin=0, vmax=1,
                                 interpolation="bilinear")
        if row == n_samples - 1:
            plt.colorbar(im, ax=axes[row, 1], fraction=0.046,
                         label="$p(\\mathrm{fire})$")

        # --- Col 2: Bare model (p≥0.5) error map ---
        pred_std = probs >= 0.5
        overlay_bare = np.full((*gt.shape, 3), 0.95)
        tp = valid & (gt == 1) & pred_std
        fn = valid & (gt == 1) & ~pred_std  # MISSED
        fp = valid & (gt == 0) & pred_std
        tn = valid & (gt == 0) & ~pred_std
        overlay_bare[tp] = [0.18, 0.80, 0.44]
        overlay_bare[fn] = [0.91, 0.30, 0.24]
        overlay_bare[fp] = [0.20, 0.60, 0.86]
        overlay_bare[tn] = [0.95, 0.95, 0.95]
        overlay_bare[~valid] = [0.6, 0.6, 0.6]

        n_fire = (valid & (gt == 1)).sum()
        fnr_bare = fn.sum() / max(n_fire, 1)
        caught_bare = tp.sum() / max(n_fire, 1)
        axes[row, 2].imshow(overlay_bare, interpolation="nearest")

        # Big missed fire percentage
        axes[row, 2].text(
            32, 58, f"Caught: {caught_bare:.0%}",
            fontsize=12, fontweight="bold", color="white",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#c0392b", alpha=0.9))

        # --- Col 3: CRC model (p≥λ̂) error map ---
        pred_crc = probs >= lam_crc
        overlay_crc = np.full((*gt.shape, 3), 0.95)
        tp_c = valid & (gt == 1) & pred_crc
        fn_c = valid & (gt == 1) & ~pred_crc
        fp_c = valid & (gt == 0) & pred_crc
        tn_c = valid & (gt == 0) & ~pred_crc
        overlay_crc[tp_c] = [0.18, 0.80, 0.44]
        overlay_crc[fn_c] = [0.91, 0.30, 0.24]
        overlay_crc[fp_c] = [0.20, 0.60, 0.86]
        overlay_crc[tn_c] = [0.95, 0.95, 0.95]
        overlay_crc[~valid] = [0.6, 0.6, 0.6]

        caught_crc = tp_c.sum() / max(n_fire, 1)
        set_size_crc = (pred_crc & valid).sum() / max(valid.sum(), 1)
        axes[row, 3].imshow(overlay_crc, interpolation="nearest")

        axes[row, 3].text(
            32, 58, f"Caught: {caught_crc:.0%}",
            fontsize=12, fontweight="bold", color="white",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#27ae60", alpha=0.9))

        # --- Col 4: Cross-section through densest fire row ---
        fire_per_row = [(gt[r] * (mask[r] == 1)).sum() for r in range(gt.shape[0])]
        best_row = int(np.argmax(fire_per_row))

        row_probs = probs[best_row, :]
        row_gt = gt[best_row, :]
        row_mask = mask[best_row, :]
        x = np.arange(64)

        masked_p = np.where(row_mask == 1, row_probs, np.nan)

        # Fire shading
        fire_cols = (row_gt == 1) & (row_mask == 1)
        axes[row, 4].fill_between(x, 0, np.where(fire_cols, 1.05, 0),
                                  alpha=0.12, color="#e74c3c")

        # Probability curve
        axes[row, 4].fill_between(x, 0, masked_p, alpha=0.25, color="#cc4c02")
        axes[row, 4].plot(x, masked_p, color="#8c2d04", linewidth=2,
                          label="$p(\\mathrm{fire})$")

        # Thresholds
        axes[row, 4].axhline(0.5, color="#7f8c8d", linewidth=2, linestyle=":",
                              label="Standard ($p \\geq 0.5$)" if row == 0 else "")
        axes[row, 4].axhline(lam_crc, color="#2ecc71", linewidth=2.5, linestyle="--",
                              label=f"CRC ($\\hat{{\\lambda}}$={lam_crc:.4f})"
                              if row == 0 else "")

        # Shade what CRC catches that standard misses
        between = (masked_p >= lam_crc) & (masked_p < 0.5)
        axes[row, 4].fill_between(
            x, lam_crc, np.where(between, masked_p, lam_crc),
            alpha=0.3, color="#2ecc71",
            label="Pixels CRC saves" if row == 0 else "")

        axes[row, 4].set_ylim(-0.05, 1.1)
        axes[row, 4].set_xlim(0, 63)
        axes[row, 4].grid(True, alpha=0.3)
        if row == 0:
            axes[row, 4].legend(fontsize=8, loc="upper right")
        if row == n_samples - 1:
            axes[row, 4].set_xlabel("Pixel position", fontsize=11)

        # Draw cross-section line on probability map
        axes[row, 1].axhline(best_row, color="cyan", linewidth=1.5, linestyle="--")

    # Column titles
    col_titles = [
        "Ground Truth",
        "Fire Probability\n$p(\\mathrm{fire} \\mid x)$",
        "Standard Model\n($p \\geq 0.5$, no CRC)",
        "With CRC\n($p \\geq \\hat{\\lambda}$, guaranteed safe)",
        "Probability Cross-Section\n(risk decay from fire center)",
    ]
    for col, title in enumerate(col_titles):
        color = "#c0392b" if col == 2 else "#27ae60" if col == 3 else "black"
        axes[0, col].set_title(title, fontsize=12, fontweight="bold", color=color)

    # Remove ticks from image columns
    for row_ax in axes:
        for col in range(4):
            row_ax[col].set_xticks([])
            row_ax[col].set_yticks([]) if col > 0 else None

    # Legend
    legend_patches = [
        mpatches.Patch(color="#2ecc71", label="True Positive (caught fire)"),
        mpatches.Patch(color="#e74c3c", label="False Negative (MISSED fire!)"),
        mpatches.Patch(color="#3498db", label="False Positive (over-alert)"),
        mpatches.Patch(color="#ecf0f1", label="True Negative"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4, fontsize=11,
               bbox_to_anchor=(0.42, -0.02), frameon=True, edgecolor="gray")

    fig.suptitle(
        "Standard Model vs CRC: Per-Pixel Fire Detection Comparison",
        fontsize=17, fontweight="bold", y=1.01,
    )

    fig.tight_layout()
    fig.savefig(output_dir / "bare_vs_crc_comparison.pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / "bare_vs_crc_comparison.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved bare_vs_crc_comparison.pdf/png ({n_samples} samples)")


# ------------------------------------------------------------------
# Compact 1-sample deep dive
# ------------------------------------------------------------------

def plot_single_sample_deep_dive(
    spatial_dir: Path,
    results: dict,
    output_dir: Path,
    dpi: int,
) -> None:
    """Single sample: 2×3 grid showing full before/after story."""
    sp_data = np.load(spatial_dir / "test_probability_heatmaps.npz")
    probs_all = sp_data["probabilities"]
    targets_all = sp_data["targets"]
    masks_all = sp_data["valid_masks"]

    lam_crc = results["U-Net_crc"]["lambda_hat"]
    lam_max_tw = results["U-Net_threeway"]["lambda_max"]

    # Pick a sample with ~5-8% fire (representative)
    indices = _pick_best_samples(targets_all, masks_all, 4)
    idx = indices[2]  # 85th percentile fire fraction

    gt = targets_all[idx]
    probs = probs_all[idx]
    mask = masks_all[idx]
    valid = mask == 1
    fire_pct = (gt[valid].sum() / valid.sum()) * 100

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)

    # --- Top row: Visual maps ---
    # (0,0) Ground truth
    ax00 = fig.add_subplot(gs[0, 0])
    gt_display = np.where(valid, gt, np.nan)
    ax00.imshow(gt_display, cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
    ax00.set_title(f"Ground Truth ({fire_pct:.1f}% fire)", fontsize=13, fontweight="bold")
    ax00.set_xticks([]); ax00.set_yticks([])

    # (0,1) Bare model
    ax01 = fig.add_subplot(gs[0, 1])
    pred_std = probs >= 0.5
    overlay = np.full((*gt.shape, 3), 0.95)
    tp = valid & (gt == 1) & pred_std
    fn = valid & (gt == 1) & ~pred_std
    fp = valid & (gt == 0) & pred_std
    overlay[tp] = [0.18, 0.80, 0.44]
    overlay[fn] = [0.91, 0.30, 0.24]
    overlay[fp] = [0.20, 0.60, 0.86]
    overlay[~valid] = [0.6, 0.6, 0.6]

    n_fire = (valid & (gt == 1)).sum()
    caught_std = tp.sum() / max(n_fire, 1)
    ax01.imshow(overlay, interpolation="nearest")
    ax01.set_title(f"Standard Model ($p \\geq 0.5$)\nCaught: {caught_std:.0%} of fires",
                   fontsize=13, fontweight="bold", color="#c0392b")
    ax01.set_xticks([]); ax01.set_yticks([])

    # (0,2) CRC model
    ax02 = fig.add_subplot(gs[0, 2])
    pred_crc = probs >= lam_crc
    overlay_c = np.full((*gt.shape, 3), 0.95)
    tp_c = valid & (gt == 1) & pred_crc
    fn_c = valid & (gt == 1) & ~pred_crc
    fp_c = valid & (gt == 0) & pred_crc
    overlay_c[tp_c] = [0.18, 0.80, 0.44]
    overlay_c[fn_c] = [0.91, 0.30, 0.24]
    overlay_c[fp_c] = [0.20, 0.60, 0.86]
    overlay_c[~valid] = [0.6, 0.6, 0.6]

    caught_crc = tp_c.sum() / max(n_fire, 1)
    set_size = (pred_crc & valid).sum() / max(valid.sum(), 1)
    ax02.imshow(overlay_c, interpolation="nearest")
    ax02.set_title(f"With CRC ($\\hat{{\\lambda}}$={lam_crc:.4f})\n"
                   f"Caught: {caught_crc:.0%} | Zone size: {set_size:.1%}",
                   fontsize=13, fontweight="bold", color="#27ae60")
    ax02.set_xticks([]); ax02.set_yticks([])

    # --- Bottom row: Analytical views ---
    # (1,0) Probability heatmap
    ax10 = fig.add_subplot(gs[1, 0])
    prob_display = np.where(valid, probs, np.nan)
    im = ax10.imshow(prob_display, cmap=RISK_CMAP, vmin=0, vmax=1,
                     interpolation="bilinear")
    # CRC contour
    prob_c = np.where(valid, probs, 0)
    try:
        ax10.contour(prob_c, levels=[lam_crc], colors=["#2ecc71"],
                     linewidths=2.5, linestyles="--")
        ax10.contour(prob_c, levels=[0.5], colors=["#7f8c8d"],
                     linewidths=2, linestyles=":")
    except ValueError:
        pass
    plt.colorbar(im, ax=ax10, fraction=0.046, label="$p(\\mathrm{fire})$")
    ax10.set_title("Probability Map\n(green = CRC, gray = standard)", fontsize=13,
                   fontweight="bold")
    ax10.set_xticks([]); ax10.set_yticks([])

    # Find best cross-section row
    fire_per_row = [(gt[r] * (mask[r] == 1)).sum() for r in range(64)]
    best_row = int(np.argmax(fire_per_row))
    ax10.axhline(best_row, color="cyan", linewidth=1.5, linestyle="--", alpha=0.8)

    # (1,1-2) Wide cross-section plot
    ax_cross = fig.add_subplot(gs[1, 1:])

    row_probs = probs[best_row, :]
    row_gt = gt[best_row, :]
    row_mask = mask[best_row, :]
    x = np.arange(64)
    masked_p = np.where(row_mask == 1, row_probs, np.nan)

    # Fire ground truth shading
    fire_cols = (row_gt == 1) & (row_mask == 1)
    ax_cross.fill_between(x, 0, np.where(fire_cols, 1.08, 0),
                          alpha=0.10, color="#e74c3c", label="Actual fire pixels")

    # Probability curve
    ax_cross.fill_between(x, 0, masked_p, alpha=0.20, color="#cc4c02")
    ax_cross.plot(x, masked_p, color="#8c2d04", linewidth=2.5,
                  label="Model probability $p(\\mathrm{fire})$")

    # Standard threshold
    ax_cross.axhline(0.5, color="#7f8c8d", linewidth=2.5, linestyle=":",
                     label="Standard threshold ($p \\geq 0.5$)")

    # CRC threshold
    ax_cross.axhline(lam_crc, color="#2ecc71", linewidth=3, linestyle="--",
                     label=f"CRC threshold $\\hat{{\\lambda}}$ = {lam_crc:.4f}"
                           f"\n(guarantees ≥95% fire detection)")

    # Shade what CRC catches that standard misses (the key visual!)
    between_mask = ~np.isnan(masked_p) & (masked_p >= lam_crc) & (masked_p < 0.5)
    ax_cross.fill_between(
        x, lam_crc,
        np.where(between_mask, masked_p, lam_crc),
        alpha=0.35, color="#2ecc71",
        label="Extra pixels CRC catches\n(standard model would miss these!)")

    # Shade what standard catches
    above_std = ~np.isnan(masked_p) & (masked_p >= 0.5)
    ax_cross.fill_between(
        x, 0.5,
        np.where(above_std, masked_p, 0.5),
        alpha=0.25, color="#7f8c8d",
        label="Pixels standard model catches")

    # Mark missed fire pixels under standard threshold
    missed_fire = fire_cols & (row_probs < 0.5) & (row_probs >= lam_crc)
    missed_x = np.where(missed_fire)[0]
    if len(missed_x) > 0:
        ax_cross.scatter(missed_x, row_probs[missed_x], color="#e74c3c",
                         s=40, zorder=5, marker="v",
                         label="Fire pixels saved by CRC")

    ax_cross.set_ylim(-0.05, 1.12)
    ax_cross.set_xlim(0, 63)
    ax_cross.set_xlabel("Pixel position along cross-section →", fontsize=12)
    ax_cross.set_ylabel("$p(\\mathrm{fire})$", fontsize=13)
    ax_cross.set_title(
        f"Cross-Section (row {best_row}): "
        f"Fire probability decays with distance — CRC captures the tail",
        fontsize=13, fontweight="bold")
    ax_cross.legend(fontsize=9, loc="upper right", ncol=2)
    ax_cross.grid(True, alpha=0.3)

    # Annotation arrow
    if len(missed_x) > 0:
        mid_missed = missed_x[len(missed_x) // 2]
        ax_cross.annotate(
            "These fire pixels have low\nprobability but ARE real fires.\n"
            "Standard model misses them.\nCRC catches them.",
            xy=(mid_missed, row_probs[mid_missed]),
            xytext=(max(mid_missed - 25, 2), 0.15),
            fontsize=10, color="#27ae60", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#27ae60", lw=2),
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#27ae60", alpha=0.95),
        )

    # Error map legend
    legend_patches = [
        mpatches.Patch(color="#2ecc71", label="True Positive"),
        mpatches.Patch(color="#e74c3c", label="False Negative (missed!)"),
        mpatches.Patch(color="#3498db", label="False Positive"),
    ]
    fig.legend(handles=legend_patches, loc="lower left", ncol=3, fontsize=10,
               bbox_to_anchor=(0.02, -0.02), frameon=True, edgecolor="gray",
               title="Error Map Colors", title_fontsize=10)

    fig.suptitle(
        "Why Conformal Risk Control Matters: The Same Model, Dramatically Different Safety",
        fontsize=16, fontweight="bold", y=1.02,
    )

    fig.savefig(output_dir / "crc_deep_dive.pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / "crc_deep_dive.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved crc_deep_dive.pdf/png (sample #{idx})")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = json.loads((args.eval_dir / "full_results.json").read_text())

    print("Generating comparison figures...")
    plot_bare_vs_crc(args.spatial_dir, results, args.output_dir, args.dpi)
    plot_single_sample_deep_dive(args.spatial_dir, results, args.output_dir, args.dpi)

    print(f"\nAll comparison figures saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
