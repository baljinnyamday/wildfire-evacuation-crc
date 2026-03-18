"""Conformal Risk Control (CRC) for False Negative Rate bounding.

This module is pure statistics — no ML models, no data loading.  It
implements the split-conformal calibration procedure from:

    Angelopoulos, Bates, et al. "Conformal Risk Control" (2022).

Given predicted probabilities and binary ground-truth labels from a
held-out *calibration* set, the function computes a threshold λ̂ such
that the False Negative Rate on future exchangeable data is controlled
at level α with finite-sample guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CRCResult:
    """Stores everything about a CRC calibration run."""

    lambda_hat: float
    alpha: float
    n_positive_pixels: int
    n_total_pixels: int
    cal_fnr: float
    cal_coverage: float
    cal_set_size_frac: float


def compute_crc_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.05,
    *,
    valid_mask: np.ndarray | None = None,
) -> CRCResult:
    """Find the CRC threshold λ̂ that guarantees FNR ≤ α.

    Parameters
    ----------
    probabilities : array of float
        Model-predicted fire probabilities (values in [0, 1]).
        Can be any shape; will be flattened internally.
    labels : array of int/float
        Binary ground-truth labels (1 = fire, 0 = no fire).
        Same shape as *probabilities*.
    alpha : float
        Target FNR bound.  Default 0.05 → ≥ 95 % coverage.
    valid_mask : array of bool/int, optional
        If provided, only pixels where ``valid_mask == 1`` are used.
        Same shape as *probabilities*.

    Returns
    -------
    CRCResult
        Threshold and diagnostics.
    """
    probs = probabilities.ravel().astype(np.float64)
    labs = labels.ravel().astype(np.float64)

    if valid_mask is not None:
        keep = valid_mask.ravel().astype(bool)
        probs = probs[keep]
        labs = labs[keep]

    n_total = len(probs)
    positive_mask = labs == 1
    positive_probs = probs[positive_mask]
    m = len(positive_probs)

    if m == 0:
        raise ValueError(
            "Zero positive (fire) pixels in the calibration data — "
            "cannot compute a meaningful FNR threshold."
        )

    # ------------------------------------------------------------------
    # Core CRC quantile with finite-sample correction.
    #
    # We want the *largest* λ such that the FNR on future data is ≤ α.
    # Among m positive calibration pixels sorted as p_(1) ≤ … ≤ p_(m):
    #
    #   k = ⌈α (m + 1)⌉          (finite-sample correction)
    #   λ̂ = p_(k)                 if 1 ≤ k ≤ m
    #
    # At threshold λ̂, at most k − 1 positive pixels fall below it,
    # giving empirical FNR ≤ (k − 1)/m  ≈  α.
    # ------------------------------------------------------------------
    sorted_probs = np.sort(positive_probs)
    k = int(np.ceil(alpha * (m + 1)))
    k = max(1, min(k, m))
    lambda_hat = float(sorted_probs[k - 1])

    # --- Calibration-set diagnostics (informational only) -------------
    predictions = (probs >= lambda_hat).astype(np.float64)
    fn = float(np.sum((labs == 1) & (predictions == 0)))
    tp = float(np.sum((labs == 1) & (predictions == 1)))
    cal_fnr = fn / m
    cal_coverage = 1.0 - cal_fnr
    cal_set_size_frac = float(predictions.sum()) / n_total

    return CRCResult(
        lambda_hat=lambda_hat,
        alpha=alpha,
        n_positive_pixels=m,
        n_total_pixels=n_total,
        cal_fnr=cal_fnr,
        cal_coverage=cal_coverage,
        cal_set_size_frac=cal_set_size_frac,
    )


def sweep_fnr(
    probabilities: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray | None = None,
    *,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute FNR at a sweep of thresholds (for plotting / inspection).

    Parameters
    ----------
    probabilities, labels, valid_mask
        Same semantics as :func:`compute_crc_threshold`.
    thresholds : 1-D array, optional
        Thresholds to evaluate.  Defaults to 200 evenly spaced values
        in [0, 1].

    Returns
    -------
    thresholds : 1-D array
    fnr_values : 1-D array
        FNR at each threshold.
    """
    probs = probabilities.ravel().astype(np.float64)
    labs = labels.ravel().astype(np.float64)

    if valid_mask is not None:
        keep = valid_mask.ravel().astype(bool)
        probs = probs[keep]
        labs = labs[keep]

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 200)

    positive_probs = probs[labs == 1]
    m = len(positive_probs)
    if m == 0:
        return thresholds, np.zeros_like(thresholds)

    fnr_values = np.array(
        [float(np.sum(positive_probs < t)) / m for t in thresholds]
    )
    return thresholds, fnr_values
