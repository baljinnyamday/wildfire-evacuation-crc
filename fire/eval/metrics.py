"""Pure metric computation for wildfire prediction evaluation.

All functions take flat numpy arrays of probabilities, labels, and
optional valid masks.  No I/O, no model loading.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinaryMetrics:
    """Metrics for a single binary threshold evaluation."""

    threshold: float
    coverage: float          # 1 - FNR (recall on fire pixels)
    fnr: float               # false negative rate
    set_size: float          # fraction of pixels flagged as fire
    precision: float         # TP / (TP + FP)
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int
    n_positive: int          # total fire pixels
    n_total: int             # total valid pixels


@dataclass(frozen=True)
class ThreeWayMetrics:
    """Metrics for three-way (SAFE / MONITOR / EVACUATE) evaluation."""

    lambda_min: float
    lambda_max: float
    coverage: float          # recall on decided fire pixels
    fnr: float
    set_size: float          # fraction flagged as EVACUATE
    safe_frac: float
    monitor_frac: float
    evacuate_frac: float
    deferral_rate: float     # = monitor_frac
    n_positive: int
    n_total: int


def compute_binary_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    valid_mask: np.ndarray | None = None,
) -> BinaryMetrics:
    """Compute binary classification metrics at a given threshold."""
    probs = probabilities.ravel().astype(np.float64)
    labs = labels.ravel().astype(np.float64)

    if valid_mask is not None:
        keep = valid_mask.ravel().astype(bool)
        probs = probs[keep]
        labs = labs[keep]

    preds = (probs >= threshold).astype(np.float64)

    tp = int(((preds == 1) & (labs == 1)).sum())
    fp = int(((preds == 1) & (labs == 0)).sum())
    fn = int(((preds == 0) & (labs == 1)).sum())
    tn = int(((preds == 0) & (labs == 0)).sum())

    n_positive = tp + fn
    n_total = len(probs)

    fnr = fn / n_positive if n_positive > 0 else 0.0
    coverage = 1.0 - fnr
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    set_size = float(preds.sum()) / n_total if n_total > 0 else 0.0

    return BinaryMetrics(
        threshold=threshold,
        coverage=coverage,
        fnr=fnr,
        set_size=set_size,
        precision=precision,
        f1=f1,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        n_positive=n_positive,
        n_total=n_total,
    )


def compute_threeway_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    lambda_min: float,
    lambda_max: float,
    valid_mask: np.ndarray | None = None,
) -> ThreeWayMetrics:
    """Compute metrics for three-way SAFE/MONITOR/EVACUATE decisions."""
    probs = probabilities.ravel().astype(np.float64)
    labs = labels.ravel().astype(np.float64)

    if valid_mask is not None:
        keep = valid_mask.ravel().astype(bool)
        probs = probs[keep]
        labs = labs[keep]

    n_total = len(probs)
    n_positive = int((labs == 1).sum())

    evacuate = probs >= lambda_max
    safe = probs < lambda_min
    monitor = ~evacuate & ~safe

    safe_frac = float(safe.sum()) / n_total
    monitor_frac = float(monitor.sum()) / n_total
    evacuate_frac = float(evacuate.sum()) / n_total

    # FNR: fire pixels wrongly classified as SAFE
    fire_missed = int(((labs == 1) & safe).sum())

    # FNR = fire pixels classified as SAFE / total fire pixels
    fnr = fire_missed / n_positive if n_positive > 0 else 0.0
    coverage = 1.0 - fnr

    return ThreeWayMetrics(
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        coverage=coverage,
        fnr=fnr,
        set_size=evacuate_frac,
        safe_frac=safe_frac,
        monitor_frac=monitor_frac,
        evacuate_frac=evacuate_frac,
        deferral_rate=monitor_frac,
        n_positive=n_positive,
        n_total=n_total,
    )


def compute_auroc(
    probabilities: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> float:
    """Compute AUROC without sklearn dependency (trapezoidal rule)."""
    probs = probabilities.ravel().astype(np.float64)
    labs = labels.ravel().astype(np.float64)

    if valid_mask is not None:
        keep = valid_mask.ravel().astype(bool)
        probs = probs[keep]
        labs = labs[keep]

    n_pos = int((labs == 1).sum())
    n_neg = int((labs == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0

    # Sort by descending probability
    order = np.argsort(-probs)
    sorted_labs = labs[order]

    # Accumulate TPR and FPR
    tps = np.cumsum(sorted_labs == 1).astype(np.float64)
    fps = np.cumsum(sorted_labs == 0).astype(np.float64)
    tpr = tps / n_pos
    fpr = fps / n_neg

    # Prepend origin
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])

    return float(np.trapezoid(tpr, fpr))
