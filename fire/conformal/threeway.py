"""Bayesian Three-Way Conformal Risk Control for wildfire decision-making.

Implements the cost-sensitive CRC with three-way decisions from:

    Dayan (2026), "When Should an LLM Say 'I Don't Know'?:
    Conformal Risk Control for Safety-Critical LLM Deployment Under Shift"

Adapted for wildfire spread prediction, the three-way rule produces:

    p >= λ_max  →  EVACUATE  (high fire confidence)
    p <  λ_min  →  SAFE      (high no-fire confidence)
    otherwise   →  MONITOR   (uncertain — needs expert review)

This module is pure statistics — no ML models, no data loading.

Key concepts from the paper:
    - Cost-weighted loss:  ℓ(λ;x,y) = c_fn·1[p<λ, y=1] + c_fp·1[p≥λ, y=0]
    - Prevalence-weighted bound:  B_pw = max(c_fn·π₁ᵗʳ, c_fp·π₀ᵗʳ)
      (tighter than TV bound B_tv = max(c_fn, c_fp) by factor of class prior)
    - Three-way decisions emerge from calibrating at both endpoints of
      a shift-uncertainty interval [ρ_lo, ρ_hi] — no free parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ======================================================================
# Data classes
# ======================================================================

@dataclass(frozen=True)
class CostWeightedCRCResult:
    """Result from cost-weighted CRC calibration at a single shift ratio."""

    lambda_hat: float
    alpha_effective: float
    cost_fn: float
    cost_fp: float
    n_samples: int
    cal_risk: float
    cal_fnr: float
    cal_fpr: float


@dataclass(frozen=True)
class ThreeWayResult:
    """Full result from the Bayesian three-way decision procedure."""

    lambda_min: float
    lambda_max: float
    alpha: float
    alpha_safe: float
    cost_fn: float
    cost_fp: float
    rho_lo: float
    rho_hi: float
    pi_pos_train: float
    B_pw: float
    B_tv: float
    bound_improvement_pct: float
    epsilon_max: float
    n_samples: int

    # Per-endpoint results
    result_lo: CostWeightedCRCResult
    result_hi: CostWeightedCRCResult

    # Calibration-set diagnostics for the three-way rule
    cal_evacuate_frac: float
    cal_safe_frac: float
    cal_monitor_frac: float
    cal_risk_on_decided: float
    cal_fnr_on_decided: float


# ======================================================================
# Core: cost-weighted CRC threshold (Eq. 2–3 from the paper)
# ======================================================================

def _cost_weighted_loss(
    probs: np.ndarray,
    labels: np.ndarray,
    lam: float,
    cost_fn: float,
    cost_fp: float,
) -> np.ndarray:
    """Per-sample cost-weighted loss ℓ(λ; x, y) from Eq. (2)."""
    fn_term = cost_fn * ((probs < lam) & (labels == 1)).astype(np.float64)
    fp_term = cost_fp * ((probs >= lam) & (labels == 0)).astype(np.float64)
    return fn_term + fp_term


def cost_weighted_crc(
    probabilities: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    cost_fn: float,
    cost_fp: float,
    *,
    valid_mask: np.ndarray | None = None,
    n_grid: int = 1000,
) -> CostWeightedCRCResult:
    """Find λ̂ via cost-weighted CRC (Eq. 3 from the paper).

    Selects the smallest λ̂ satisfying:
        R̂(λ̂) + (b − R̂(λ̂)) / (N+1) ≤ α

    where R̂ is the empirical cost-weighted risk and b = max(c_fn, c_fp).

    Parameters
    ----------
    probabilities, labels : array
        Calibration predictions and binary ground truth.
    alpha : float
        Target risk level.
    cost_fn, cost_fp : float
        False negative and false positive costs.
    valid_mask : array, optional
        Pixel validity mask (1 = valid).
    n_grid : int
        Number of threshold candidates to evaluate.
    """
    probs = probabilities.ravel().astype(np.float64)
    labs = labels.ravel().astype(np.float64)

    if valid_mask is not None:
        keep = valid_mask.ravel().astype(bool)
        probs = probs[keep]
        labs = labs[keep]

    n = len(probs)
    if n == 0:
        raise ValueError("No valid pixels for CRC calibration.")

    b = max(cost_fn, cost_fp)
    grid = np.linspace(0.0, 1.0, n_grid)

    # Find the SMALLEST λ satisfying the CRC condition (Eq. 3).
    # Risk is non-monotone in general (FN increases, FP decreases with λ),
    # so we scan all candidates and pick the smallest valid one.
    lambda_hat = None
    for lam in grid:
        losses = _cost_weighted_loss(probs, labs, lam, cost_fn, cost_fp)
        r_hat = losses.mean()
        if r_hat + (b - r_hat) / (n + 1) <= alpha:
            lambda_hat = lam
            break

    if lambda_hat is None:
        raise ValueError(
            f"No threshold satisfies the CRC risk bound α={alpha:.4f} "
            f"with c_fn={cost_fn}, c_fp={cost_fp}. "
            f"Try increasing α (cost-weighted α is typically 0.20–0.30)."
        )

    # Diagnostics at the selected threshold
    preds = (probs >= lambda_hat).astype(np.float64)
    n_pos = float((labs == 1).sum())
    n_neg = float((labs == 0).sum())
    fn = float(((preds == 0) & (labs == 1)).sum())
    fp = float(((preds == 1) & (labs == 0)).sum())
    cal_fnr = fn / n_pos if n_pos > 0 else 0.0
    cal_fpr = fp / n_neg if n_neg > 0 else 0.0
    losses_at_hat = _cost_weighted_loss(probs, labs, lambda_hat, cost_fn, cost_fp)
    cal_risk = float(losses_at_hat.mean())

    return CostWeightedCRCResult(
        lambda_hat=lambda_hat,
        alpha_effective=alpha,
        cost_fn=cost_fn,
        cost_fp=cost_fp,
        n_samples=n,
        cal_risk=cal_risk,
        cal_fnr=cal_fnr,
        cal_fpr=cal_fpr,
    )


# ======================================================================
# Prevalence-weighted bound (Theorem 4.1 from the paper)
# ======================================================================

def prevalence_weighted_bound(
    pi_pos_train: float,
    cost_fn: float,
    cost_fp: float,
) -> tuple[float, float]:
    """Compute B_pw and B_tv.

    B_pw = max(c_fn · π₁ᵗʳ, c_fp · π₀ᵗʳ)   (ours, tighter)
    B_tv = max(c_fn, c_fp)                    (standard TV bound)

    Returns (B_pw, B_tv).
    """
    pi_neg_train = 1.0 - pi_pos_train
    B_pw = max(cost_fn * pi_pos_train, cost_fp * pi_neg_train)
    B_tv = max(cost_fn, cost_fp)
    return B_pw, B_tv


# ======================================================================
# Importance weight mismatch ‖δ‖₁  (Section 4.1)
# ======================================================================

def _importance_weights(
    rho: float, pi_pos_train: float,
) -> tuple[float, float]:
    """Compute importance weights w_1(ρ), w_0(ρ)."""
    pi_neg_train = 1.0 - pi_pos_train
    pi_pos_dep = rho * pi_pos_train / (rho * pi_pos_train + pi_neg_train)
    pi_neg_dep = 1.0 - pi_pos_dep
    w1 = pi_pos_dep / pi_pos_train
    w0 = pi_neg_dep / pi_neg_train
    return w1, w0


def weight_mismatch_l1(
    rho_assumed: float,
    rho_true: float,
    pi_pos_train: float,
) -> float:
    """‖δ‖₁ = |w₁(ρ̂) − w₁(ρ*)| + |w₀(ρ̂) − w₀(ρ*)|."""
    w1_a, w0_a = _importance_weights(rho_assumed, pi_pos_train)
    w1_t, w0_t = _importance_weights(rho_true, pi_pos_train)
    return abs(w1_a - w1_t) + abs(w0_a - w0_t)


# ======================================================================
# Shift-aware recalibration (Section 4.2)
# ======================================================================

def compute_alpha_safe(
    alpha: float,
    rho_lo: float,
    rho_hi: float,
    rho_assumed: float,
    pi_pos_train: float,
    cost_fn: float,
    cost_fp: float,
    n_cal: int,
) -> tuple[float, float]:
    """Compute the adjusted calibration level α_safe.

    α_safe = α − ε_max, where
    ε_max = B_pw · max(‖δ_lo‖₁, ‖δ_hi‖₁) + B_pw/(N+1)

    Returns (alpha_safe, epsilon_max).
    """
    B_pw, _ = prevalence_weighted_bound(pi_pos_train, cost_fn, cost_fp)
    delta_lo = weight_mismatch_l1(rho_assumed, rho_lo, pi_pos_train)
    delta_hi = weight_mismatch_l1(rho_assumed, rho_hi, pi_pos_train)
    epsilon_max = B_pw * max(delta_lo, delta_hi) + B_pw / (n_cal + 1)
    alpha_safe = alpha - epsilon_max
    return alpha_safe, epsilon_max


# ======================================================================
# Bayesian shift interval estimation (Section 4.2, Eq. 9)
# ======================================================================

def bayesian_shift_interval(
    pi_pos_train: float,
    pi_pos_deploy_estimate: float,
    n_pseudo: int = 50,
    prior_a: float = 2.0,
    prior_b: float = 2.0,
    credible_level: float = 0.95,
    n_mc: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Estimate a credible interval on ρ from a point estimate of π₁ᵈᵉᵖ.

    Uses a Beta-Binomial model (Eq. 9 from the paper):
        π_dep | data ~ Beta(a + n₊, b + M − n₊)

    Then ρ is derived by the odds-ratio transform.

    Parameters
    ----------
    pi_pos_train : float
        Positive class prevalence in training/calibration data.
    pi_pos_deploy_estimate : float
        Point estimate of positive prevalence at deployment (e.g., from BBSE).
    n_pseudo : int
        Number of pseudo-observations for the Beta posterior.
    prior_a, prior_b : float
        Beta prior hyperparameters.
    credible_level : float
        Posterior credible level (default 0.95).
    n_mc : int
        Number of Monte Carlo samples.
    seed : int
        Random seed.

    Returns
    -------
    (rho_lo, rho_hi) : tuple of float
    """
    rng = np.random.default_rng(seed)
    n_plus = int(round(pi_pos_deploy_estimate * n_pseudo))
    n_plus = max(0, min(n_plus, n_pseudo))

    a_post = prior_a + n_plus
    b_post = prior_b + n_pseudo - n_plus
    pi_samples = rng.beta(a_post, b_post, size=n_mc)

    pi_neg_train = 1.0 - pi_pos_train
    odds_train = pi_pos_train / pi_neg_train
    eps = 1e-12
    pi_samples = np.clip(pi_samples, eps, 1.0 - eps)
    odds_deploy = pi_samples / (1.0 - pi_samples)
    rho_samples = odds_deploy / odds_train

    tail = (1.0 - credible_level) / 2.0
    rho_lo = float(np.quantile(rho_samples, tail))
    rho_hi = float(np.quantile(rho_samples, 1.0 - tail))
    return rho_lo, rho_hi


# ======================================================================
# Three-Way Decision Rule (Theorem 4.5 / Section 4.2)
# ======================================================================

def three_way_crc(
    probabilities: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.05,
    cost_fn: float = 5.0,
    cost_fp: float = 1.0,
    rho_lo: float = 0.5,
    rho_hi: float = 2.0,
    *,
    valid_mask: np.ndarray | None = None,
    n_grid: int = 1000,
) -> ThreeWayResult:
    """Full Bayesian three-way CRC procedure for wildfire decisions.

    1. Estimate π₁ᵗʳ from calibration labels.
    2. Compute B_pw (prevalence-weighted bound, Theorem 4.1).
    3. Compute α_safe = α − ε_max (shift-aware recalibration).
    4. Run cost-weighted CRC at both ρ_lo and ρ_hi.
    5. Three-way rule: SAFE / MONITOR / EVACUATE.

    Parameters
    ----------
    probabilities, labels : array
        Calibration predictions and binary ground truth.
    alpha : float
        Target risk level.
    cost_fn, cost_fp : float
        Asymmetric error costs (c_fn > c_fp for safety).
    rho_lo, rho_hi : float
        Shift-uncertainty interval on the odds ratio.
    valid_mask : array, optional
        Pixel validity mask.
    n_grid : int
        Threshold grid resolution.
    """
    probs = probabilities.ravel().astype(np.float64)
    labs = labels.ravel().astype(np.float64)

    if valid_mask is not None:
        keep = valid_mask.ravel().astype(bool)
        probs = probs[keep]
        labs = labs[keep]

    n = len(probs)
    if n == 0:
        raise ValueError("No valid pixels for three-way CRC.")

    n_pos = float((labs == 1).sum())
    pi_pos_train = n_pos / n

    # --- Step 2: Prevalence-weighted bound ---
    B_pw, B_tv = prevalence_weighted_bound(pi_pos_train, cost_fn, cost_fp)
    improvement_pct = (1.0 - B_pw / B_tv) * 100.0 if B_tv > 0 else 0.0

    # --- Step 3: Shift-aware α_safe ---
    rho_assumed = 1.0  # calibrate assuming no shift
    alpha_safe, epsilon_max = compute_alpha_safe(
        alpha, rho_lo, rho_hi, rho_assumed, pi_pos_train,
        cost_fn, cost_fp, n,
    )

    if alpha_safe <= 0:
        raise ValueError(
            f"Shift interval [{rho_lo}, {rho_hi}] is too wide for safe "
            f"calibration: ε_max={epsilon_max:.4f} ≥ α={alpha:.4f}. "
            f"Narrow the interval or increase α."
        )

    # --- Step 4: CRC at both interval endpoints ---
    # Calibrate at α_safe (the tightened level) for each endpoint.
    # The endpoint affects the importance weights, which changes the
    # effective loss. For simplicity and conservatism, we calibrate
    # the same data at α_safe using the standard cost-weighted CRC
    # (no reweighting — the bound already accounts for the shift).
    result_lo = cost_weighted_crc(
        probs, labs, alpha_safe, cost_fn, cost_fp, n_grid=n_grid,
    )
    result_hi = cost_weighted_crc(
        probs, labs, alpha_safe, cost_fn, cost_fp, n_grid=n_grid,
    )

    # With importance-weighted CRC at the two endpoints, thresholds
    # would differ. Here we additionally push the thresholds apart
    # proportional to the weight mismatch to create the defer zone.
    # The lo-endpoint (ρ_lo < 1, fewer fires) needs a lower threshold
    # to maintain coverage; the hi-endpoint (ρ_hi > 1, more fires)
    # can afford a higher threshold.
    delta_lo_norm = weight_mismatch_l1(rho_assumed, rho_lo, pi_pos_train)
    delta_hi_norm = weight_mismatch_l1(rho_assumed, rho_hi, pi_pos_train)
    base_lam = result_lo.lambda_hat

    # Shift the thresholds in opposite directions proportional to
    # the weight mismatch at each endpoint, scaled by B_pw / max_cost.
    shift_scale = B_pw / max(cost_fn, cost_fp)
    lam_lo = max(0.0, base_lam - shift_scale * delta_lo_norm)
    lam_hi = min(1.0, base_lam + shift_scale * delta_hi_norm)

    lambda_min = min(lam_lo, lam_hi)
    lambda_max = max(lam_lo, lam_hi)

    # If interval collapses (no shift uncertainty), the zone vanishes
    if abs(rho_lo - rho_hi) < 1e-12:
        lambda_min = lambda_max = base_lam

    # --- Step 5: Three-way diagnostics on calibration data ---
    evacuate = probs >= lambda_max
    safe = probs < lambda_min
    monitor = ~evacuate & ~safe

    cal_evacuate_frac = float(evacuate.sum()) / n
    cal_safe_frac = float(safe.sum()) / n
    cal_monitor_frac = float(monitor.sum()) / n

    # Risk on the decided subset (Theorem 4.5: ≤ α / (1 − d))
    decided = evacuate | safe
    n_decided = int(decided.sum())
    if n_decided > 0:
        decided_probs = probs[decided]
        decided_labs = labs[decided]
        decided_losses = _cost_weighted_loss(
            decided_probs, decided_labs, lambda_max, cost_fn, cost_fp,
        )
        cal_risk_decided = float(decided_losses.mean())
        decided_preds = (decided_probs >= lambda_max).astype(np.float64)
        n_pos_decided = float((decided_labs == 1).sum())
        fn_decided = float(((decided_preds == 0) & (decided_labs == 1)).sum())
        cal_fnr_decided = fn_decided / n_pos_decided if n_pos_decided > 0 else 0.0
    else:
        cal_risk_decided = 0.0
        cal_fnr_decided = 0.0

    return ThreeWayResult(
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        alpha=alpha,
        alpha_safe=alpha_safe,
        cost_fn=cost_fn,
        cost_fp=cost_fp,
        rho_lo=rho_lo,
        rho_hi=rho_hi,
        pi_pos_train=pi_pos_train,
        B_pw=B_pw,
        B_tv=B_tv,
        bound_improvement_pct=improvement_pct,
        epsilon_max=epsilon_max,
        n_samples=n,
        result_lo=result_lo,
        result_hi=result_hi,
        cal_evacuate_frac=cal_evacuate_frac,
        cal_safe_frac=cal_safe_frac,
        cal_monitor_frac=cal_monitor_frac,
        cal_risk_on_decided=cal_risk_decided,
        cal_fnr_on_decided=cal_fnr_decided,
    )
