"""
Confidence intervals for standardized effect sizes.

Academic publication conventions now expect point estimates of effect sizes
to be reported with confidence intervals (Cumming & Calin-Jageman 2017,
Lakens 2013). Parametric CIs for standardized effect sizes are non-trivial
because the sampling distributions are noncentral (t for d, F for eta²).

This module centralizes those calculations so every group-comparison plugin
in the framework reports CIs the same way.

The math:

Cohen's d (independent samples, equal n or general):
  d = (x̄1 - x̄2) / s_pooled
  Observed t = d * sqrt(n1*n2 / (n1+n2)) on df = n1+n2-2
  Invert the noncentral t-distribution to find the noncentrality parameter
  range [L, U] such that
      P(T(df, L) > t_obs) = alpha/2     (upper tail at lower ncp)
      P(T(df, U) <= t_obs) = alpha/2    (lower tail at upper ncp)
  Then d in [L / sqrt(n1*n2/(n1+n2)), U / sqrt(n1*n2/(n1+n2))].

Hedges' g uses the same CI as Cohen's d, then multiplies endpoints by the
Hedges small-sample correction factor J = 1 - 3/(4(n1+n2)-9).

Cohen's d_z (paired samples):
  Already implemented in paired_comparison.py; the formula is t_obs = d_z*sqrt(n)
  on df = n-1; CI for ncp / sqrt(n).

Eta squared (one-way ANOVA, equal-variance F):
  Observed F on (df1, df2)
  ncp range from noncentral F:
      P(F(df1, df2, L) > F_obs) = alpha/2
      P(F(df1, df2, U) <= F_obs) = alpha/2
  eta² = ncp / (ncp + df1 + df2 + 1)
  This is the Smithson (2003) procedure.

Omega squared uses the same noncentral-F bracket and a different point
transform; we leave omega² CI for a follow-up.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from scipy import stats
from scipy import optimize


# ============================================================
# Cohen's d / Hedges' g for independent two-sample designs
# ============================================================

def _independent_d_to_t(d: float, n1: int, n2: int) -> Optional[float]:
    """
    For Cohen's d on independent samples, t_obs = d * sqrt(n1*n2/(n1+n2)).
    """
    if n1 < 2 or n2 < 2:
        return None
    return d * math.sqrt((n1 * n2) / float(n1 + n2))


def _independent_t_to_d(t: float, n1: int, n2: int) -> Optional[float]:
    if n1 < 2 or n2 < 2:
        return None
    return t / math.sqrt((n1 * n2) / float(n1 + n2))


def _hedges_correction(n1: int, n2: int) -> float:
    """Hedges' small-sample correction factor J. Approaches 1 as n grows."""
    if n1 + n2 <= 2:
        return 1.0
    return 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)


def _solve_ncp_bracket(
    t_obs: float,
    df: float,
    alpha: float,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Solve the noncentral-t inversion for the lower and upper bounds of the
    noncentrality parameter at confidence level (1 - alpha).

    Returns (ncp_lower, ncp_upper). Either entry may be None if brentq fails
    to converge (e.g. extremely small or extremely large t).

    Implementation note: scipy.stats.nct.cdf/sf can return NaN for very large
    |ncp - t_obs|. We therefore find a *finite* bracket by walking outward
    until both endpoints produce finite target values with opposite signs.
    """
    if not math.isfinite(t_obs) or df <= 0:
        return None, None

    def lower_target(ncp: float) -> float:
        # P(T(df, ncp) > t_obs) - alpha/2
        val = float(stats.nct.sf(t_obs, df, ncp))
        return val - alpha / 2.0

    def upper_target(ncp: float) -> float:
        # P(T(df, ncp) <= t_obs) - alpha/2
        val = float(stats.nct.cdf(t_obs, df, ncp))
        return val - alpha / 2.0

    def _find_finite_bracket(
        target_fn,
        anchor: float,
        direction: int,
        max_extension: float,
    ) -> Optional[float]:
        """
        Walk away from `anchor` in `direction` (+1 or -1), starting small and
        doubling, until target_fn produces a finite value whose sign differs
        from target_fn(anchor). Returns the far endpoint, or None.
        """
        anchor_val = target_fn(anchor)
        if not math.isfinite(anchor_val):
            return None

        step = 1.0
        far = anchor + direction * step
        last_finite_far = None

        while abs(far - anchor) <= max_extension:
            try:
                far_val = target_fn(far)
            except Exception:
                far_val = float("nan")

            if math.isfinite(far_val):
                last_finite_far = far
                if (anchor_val > 0) != (far_val > 0):
                    return far
            else:
                # Hit a NaN region; back off and stop
                return last_finite_far

            step *= 2.0
            far = anchor + direction * step

        return last_finite_far

    # ---- lower endpoint: walk leftward from t_obs ----
    ncp_lower: Optional[float] = None
    max_ext = max(50.0, 10.0 * math.sqrt(df))

    far_lo = _find_finite_bracket(lower_target, t_obs, direction=-1, max_extension=max_ext)
    if far_lo is not None:
        try:
            v_anchor = lower_target(t_obs)
            v_far = lower_target(far_lo)
            if math.isfinite(v_anchor) and math.isfinite(v_far) and (v_anchor > 0) != (v_far > 0):
                ncp_lower = optimize.brentq(
                    lower_target, far_lo, t_obs, maxiter=200,
                )
        except Exception:
            ncp_lower = None

    # ---- upper endpoint: walk rightward from t_obs ----
    ncp_upper: Optional[float] = None
    far_hi = _find_finite_bracket(upper_target, t_obs, direction=+1, max_extension=max_ext)
    if far_hi is not None:
        try:
            v_anchor = upper_target(t_obs)
            v_far = upper_target(far_hi)
            if math.isfinite(v_anchor) and math.isfinite(v_far) and (v_anchor > 0) != (v_far > 0):
                ncp_upper = optimize.brentq(
                    upper_target, t_obs, far_hi, maxiter=200,
                )
        except Exception:
            ncp_upper = None

    return ncp_lower, ncp_upper


def cohens_d_independent_ci(
    d: float,
    n1: int,
    n2: int,
    alpha: float = 0.05,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Confidence interval for Cohen's d in an independent two-sample design.

    Returns (d_lower, d_upper). Either may be None if the bracket fails.
    """
    if not math.isfinite(d) or n1 < 2 or n2 < 2:
        return None, None

    t_obs = _independent_d_to_t(d, n1, n2)
    if t_obs is None:
        return None, None

    df = n1 + n2 - 2

    ncp_lower, ncp_upper = _solve_ncp_bracket(t_obs, df, alpha)

    d_lower = _independent_t_to_d(ncp_lower, n1, n2) if ncp_lower is not None else None
    d_upper = _independent_t_to_d(ncp_upper, n1, n2) if ncp_upper is not None else None

    return d_lower, d_upper


def hedges_g_independent_ci(
    g: float,
    n1: int,
    n2: int,
    alpha: float = 0.05,
) -> Tuple[Optional[float], Optional[float]]:
    """
    CI for Hedges' g via the Hedges-corrected endpoints of the Cohen's d CI.

    g = J * d, so the CI is also rescaled by the same factor J.
    """
    if not math.isfinite(g) or n1 < 2 or n2 < 2:
        return None, None

    J = _hedges_correction(n1, n2)

    # Re-derive Cohen's d from g (g = J * d -> d = g / J)
    if J == 0:
        return None, None

    d = g / J

    d_lower, d_upper = cohens_d_independent_ci(d, n1, n2, alpha=alpha)

    g_lower = J * d_lower if d_lower is not None else None
    g_upper = J * d_upper if d_upper is not None else None

    return g_lower, g_upper


# ============================================================
# Cohen's d_z for paired designs (kept here so all CIs live together;
# paired_comparison.py can import from this module if desired)
# ============================================================

def cohens_d_z_ci(
    d_z: float,
    n: int,
    alpha: float = 0.05,
) -> Tuple[Optional[float], Optional[float]]:
    """
    CI for Cohen's d_z (paired-samples Cohen's d). Uses noncentral t with
    df = n - 1, ncp = d_z * sqrt(n).

    This duplicates the implementation in paired_comparison.py; we keep both
    so the modules remain self-contained and the test surface stays small.
    Refactor paired_comparison.py to import this if you want a single source.
    """
    if not math.isfinite(d_z) or n < 2:
        return None, None

    df = n - 1
    t_obs = d_z * math.sqrt(n)

    ncp_lower, ncp_upper = _solve_ncp_bracket(t_obs, df, alpha)

    d_lower = ncp_lower / math.sqrt(n) if ncp_lower is not None else None
    d_upper = ncp_upper / math.sqrt(n) if ncp_upper is not None else None

    return d_lower, d_upper


# ============================================================
# Eta-squared for one-way ANOVA (Smithson 2003 procedure)
# ============================================================

def _solve_ncp_bracket_f(
    f_obs: float,
    df1: float,
    df2: float,
    alpha: float,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Invert the noncentral F distribution for the (1 - alpha) bracket on the
    noncentrality parameter.

    For small F (near or below F_alpha), the lower endpoint is clamped to 0
    because eta² is bounded below by 0.
    """
    if not math.isfinite(f_obs) or df1 <= 0 or df2 <= 0:
        return None, None

    # Search range: F-distribution ncp can be very large for big F. Use a
    # generous starting upper bound and let brentq expand if needed.
    upper_search = max(1000.0, f_obs * df1 * 20.0)

    def lower_target(ncp: float) -> float:
        # P(F(df1, df2, ncp) > F_obs) - alpha/2
        return float(stats.ncf.sf(f_obs, df1, df2, ncp)) - alpha / 2.0

    def upper_target(ncp: float) -> float:
        # P(F(df1, df2, ncp) <= F_obs) - alpha/2
        return float(stats.ncf.cdf(f_obs, df1, df2, ncp)) - alpha / 2.0

    # --- lower endpoint ---
    # lower_target is monotone increasing in ncp:
    #   at ncp = 0 (and significant F), sf(F_obs) ≈ 0 so target ≈ -alpha/2 (negative)
    #   at the true lower endpoint, sf = alpha/2 so target = 0
    #   at large ncp, sf -> 1 so target -> 1 - alpha/2 (positive)
    #
    # If lower_target(0) >= 0 already, the F is small enough that the lower
    # CI bound is at or below 0 (effectively, the data are compatible with
    # zero effect); clamp to 0 because eta² >= 0.
    try:
        if lower_target(0.0) >= 0:
            ncp_lower = 0.0
        else:
            ncp_lower = optimize.brentq(
                lower_target,
                0.0,
                upper_search,
                maxiter=200,
            )
    except Exception:
        ncp_lower = None

    # --- upper endpoint ---
    # Expand the bracket if needed.
    try:
        hi = upper_search
        ncp_upper = None
        for _ in range(8):
            try:
                ncp_upper = optimize.brentq(
                    upper_target,
                    0.0,
                    hi,
                    maxiter=200,
                )
                break
            except ValueError:
                hi *= 4.0
    except Exception:
        ncp_upper = None

    return ncp_lower, ncp_upper


def _ncp_to_eta_squared(ncp: float, df1: float, df2: float) -> Optional[float]:
    """
    Convert F-noncentrality parameter to eta²:
        eta² = ncp / (ncp + df1 + df2 + 1)

    This is the Smithson (2003) transformation. Returns a value clamped to
    [0, 1].
    """
    if ncp is None or not math.isfinite(ncp) or ncp < 0:
        return None

    total = ncp + df1 + df2 + 1.0
    if total <= 0:
        return None

    value = ncp / total
    return max(0.0, min(1.0, value))


def eta_squared_ci(
    f_obs: float,
    df_between: float,
    df_within: float,
    alpha: float = 0.05,
) -> Tuple[Optional[float], Optional[float]]:
    """
    CI for eta² in a one-way ANOVA, computed via the noncentral F bracket
    of the noncentrality parameter and the Smithson (2003) transformation.

    Inputs:
      f_obs:       observed F statistic
      df_between:  k - 1
      df_within:   N - k
      alpha:       two-sided alpha (default 0.05 for a 95% CI)
    """
    if not math.isfinite(f_obs) or df_between <= 0 or df_within <= 0:
        return None, None

    ncp_lower, ncp_upper = _solve_ncp_bracket_f(f_obs, df_between, df_within, alpha)

    eta2_lower = _ncp_to_eta_squared(ncp_lower, df_between, df_within) if ncp_lower is not None else None
    eta2_upper = _ncp_to_eta_squared(ncp_upper, df_between, df_within) if ncp_upper is not None else None

    return eta2_lower, eta2_upper


# ============================================================
# Convenience: omega² CI via the same noncentral F bracket
# ============================================================

def _ncp_to_omega_squared(
    ncp: float,
    df_between: float,
    df_within: float,
    n_total: int,
) -> Optional[float]:
    """
    Omega² as a function of F-noncentrality:
        omega² = (ncp - df_between) / (ncp + N)   (clamped to [0, 1])

    Derived by translating Cohen's f² = ncp / N into omega² = f² / (1 + f²).
    """
    if ncp is None or not math.isfinite(ncp) or n_total <= 0:
        return None

    numerator = ncp - df_between
    denominator = ncp + n_total

    if denominator <= 0:
        return None

    value = numerator / denominator
    return max(0.0, min(1.0, value))


def omega_squared_ci(
    f_obs: float,
    df_between: float,
    df_within: float,
    n_total: int,
    alpha: float = 0.05,
) -> Tuple[Optional[float], Optional[float]]:
    """
    CI for omega² via the same noncentral-F bracket as eta². The transform
    differs slightly because omega² is an unbiased estimator.
    """
    if (
        not math.isfinite(f_obs)
        or df_between <= 0
        or df_within <= 0
        or n_total <= 0
    ):
        return None, None

    ncp_lower, ncp_upper = _solve_ncp_bracket_f(f_obs, df_between, df_within, alpha)

    omega_lower = _ncp_to_omega_squared(ncp_lower, df_between, df_within, n_total)
    omega_upper = _ncp_to_omega_squared(ncp_upper, df_between, df_within, n_total)

    return omega_lower, omega_upper