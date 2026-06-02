"""
Bootstrap resampling utilities used by `bootstrap_inference` and reusable by
other plugins.

Provides:

  - `classical_bootstrap_indices`: standard multinomial bootstrap.
  - `sequential_bootstrap_indices`: Sequential Bootstrap (Rao-Pathak-Koltchinskii
    1997; cross-seed properties characterised in Peng 2025).
  - `bootstrap_ci`: three CI methods (percentile, basic, BCa).
  - `bootstrap_with_stability`: end-to-end driver that returns the primary CI
    plus a cross-seed CI stability diagnostic. The diagnostic quantifies how
    much the CI endpoints move when only the bootstrap RNG seed is changed,
    and is the in-plugin operationalisation of the variance decomposition in
    Section 3.4 of Peng (2025).

Design notes
------------
* The samplers, CI estimators and statistic functions are kept as pure
  callables with explicit inputs. This makes them easy to cross-validate
  against `scipy.stats.bootstrap` in `benchmark/carpet/`.
* Sequential Bootstrap is implemented exactly as in Peng (2025): draw
  indices uniformly with replacement until exactly `k_n = floor(rho * n)`
  distinct indices have been collected. Only the resampler differs from
  classical bootstrap; the CI machinery downstream is shared, so the
  numerical correctness of the CI math itself is validated only once.
* `bootstrap_with_stability` runs `n_seeds` independent sub-bootstraps,
  each of size `B_per_seed`, then pools them for the primary CI and
  computes a cross-seed coefficient of variation (CV) on the sub-CI
  endpoints. Total replicates: `n_seeds * B_per_seed`. This adds the
  stability diagnostic at no extra replicate cost.

References
----------
Peng, C. (2025). Sequential Bootstrap for Out-of-Bag Error Estimation:
A 100-Seed Replication Study and Variance-Structure Analysis.
arXiv:2511.18065.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as _scipy_stats


# ==========================================================
# Resamplers
# ==========================================================

def classical_bootstrap_indices(
    n: int,
    B: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Classical multinomial bootstrap.

    Returns
    -------
    np.ndarray of shape (B, n), dtype int64.
        Each row is one bootstrap replicate's indices into the data of size n.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if B <= 0:
        raise ValueError(f"B must be positive, got {B}")

    return rng.integers(0, n, size=(B, n), dtype=np.int64)


def sequential_bootstrap_indices(
    n: int,
    k_n: int,
    B: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """
    Sequential Bootstrap (Rao, Pathak & Koltchinskii 1997).

    For each replicate, draw indices uniformly with replacement until
    exactly `k_n` distinct indices have been collected; return the full
    sequence of draws (length T_b >= k_n, random stopping time).

    The number of distinct indices is exactly `k_n` in every replicate,
    by construction. The total length T_b is random and depends on n and
    k_n via the coupon-collector-like stopping time.

    Returns
    -------
    list of np.ndarray (length B). Each element has length T_b >= k_n
    and contains exactly k_n distinct values in {0, ..., n-1}.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if k_n <= 0 or k_n > n:
        raise ValueError(f"k_n must be in (0, n]; got k_n={k_n}, n={n}")
    if B <= 0:
        raise ValueError(f"B must be positive, got {B}")

    out: List[np.ndarray] = []

    # Heuristic upper bound on T_b: coupon-collector E[T] ~= n * H_{n} -
    # H_{n-k_n} ~ n * log(n / (n - k_n + 1)). We oversample in blocks to
    # avoid Python-level loops on large n; trim each block once k_n
    # distinct indices have been seen.
    # For k_n = floor(0.632 n), expected T_b is ~= n.
    block_size = max(2 * k_n, 64)

    for _ in range(B):
        seen = np.zeros(n, dtype=bool)
        n_seen = 0
        draws: List[int] = []

        while n_seen < k_n:
            block = rng.integers(0, n, size=block_size, dtype=np.int64)

            for j in block:
                draws.append(int(j))
                if not seen[j]:
                    seen[j] = True
                    n_seen += 1
                    if n_seen == k_n:
                        break

        out.append(np.asarray(draws, dtype=np.int64))

    return out


def expected_kn(n: int, rho: float = 0.632) -> int:
    """Target distinct-count for Sequential Bootstrap at proportion rho."""
    return int(np.floor(rho * n))


# ==========================================================
# CI estimators
# ==========================================================

def _percentile_ci(boot_stats: np.ndarray, alpha: float) -> Tuple[float, float]:
    lo, hi = np.quantile(boot_stats, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def _basic_ci(
    boot_stats: np.ndarray,
    observed: float,
    alpha: float,
) -> Tuple[float, float]:
    # Hall's basic bootstrap: 2 * theta_hat - q_{1 - alpha/2}, 2 * theta_hat - q_{alpha/2}
    q_lo, q_hi = np.quantile(boot_stats, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(2.0 * observed - q_hi), float(2.0 * observed - q_lo)


def _bca_ci(
    boot_stats: np.ndarray,
    observed: float,
    data: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float],
    alpha: float,
) -> Tuple[float, float]:
    # Bias correction
    n_boot = len(boot_stats)
    p0 = float(np.sum(boot_stats < observed)) / n_boot
    # Guard against edge cases where ppf would blow up.
    p0 = min(max(p0, 1.0 / (n_boot + 1)), 1.0 - 1.0 / (n_boot + 1))
    z0 = _scipy_stats.norm.ppf(p0)

    # Acceleration via jackknife
    n = len(data)

    if n < 2:
        # Cannot do jackknife; fall back to percentile.
        return _percentile_ci(boot_stats, alpha)

    jack = np.empty(n, dtype=float)

    for i in range(n):
        # Leave-one-out
        if i == 0:
            sub = data[1:]
        elif i == n - 1:
            sub = data[:-1]
        else:
            sub = np.concatenate([data[:i], data[i + 1:]])
        jack[i] = float(statistic_fn(sub))

    jack_mean = jack.mean()
    num = np.sum((jack_mean - jack) ** 3)
    den = 6.0 * (np.sum((jack_mean - jack) ** 2)) ** 1.5

    if den < 1e-12:
        a = 0.0
    else:
        a = float(num / den)

    z_alpha_lo = _scipy_stats.norm.ppf(alpha / 2.0)
    z_alpha_hi = _scipy_stats.norm.ppf(1.0 - alpha / 2.0)

    def _adjust(z_target: float) -> float:
        denom = 1.0 - a * (z0 + z_target)
        if abs(denom) < 1e-12:
            return float(_scipy_stats.norm.cdf(z0 + z_target))
        return float(_scipy_stats.norm.cdf(z0 + (z0 + z_target) / denom))

    alpha_lo = _adjust(z_alpha_lo)
    alpha_hi = _adjust(z_alpha_hi)

    # Clamp into [0, 1] (numerical safety)
    alpha_lo = min(max(alpha_lo, 1e-6), 1.0 - 1e-6)
    alpha_hi = min(max(alpha_hi, 1e-6), 1.0 - 1e-6)

    lo, hi = np.quantile(boot_stats, [alpha_lo, alpha_hi])

    return float(lo), float(hi)


def bootstrap_ci(
    boot_stats: np.ndarray,
    observed: float,
    method: str = "BCa",
    alpha: float = 0.05,
    *,
    data: Optional[np.ndarray] = None,
    statistic_fn: Optional[Callable[[np.ndarray], float]] = None,
) -> Tuple[float, float]:
    """
    Compute a bootstrap confidence interval from a bootstrap distribution.

    Parameters
    ----------
    boot_stats : np.ndarray
        Statistic values across bootstrap replicates.
    observed : float
        Statistic value on the full sample.
    method : {"percentile", "basic", "BCa"}, default "BCa"
        CI method. BCa requires `data` and `statistic_fn` for the jackknife.
    alpha : float, default 0.05
        Two-sided significance level. CI is (1 - alpha) coverage.
    data : np.ndarray, optional
        Original sample. Required for BCa (used in the jackknife step).
    statistic_fn : callable, optional
        Statistic. Required for BCa.

    Returns
    -------
    (lower, upper) : tuple of float
    """
    method = method.lower()

    if method == "percentile":
        return _percentile_ci(boot_stats, alpha)

    if method == "basic":
        return _basic_ci(boot_stats, observed, alpha)

    if method == "bca":
        if data is None or statistic_fn is None:
            # Caller-side bug; fall back to percentile rather than crash.
            return _percentile_ci(boot_stats, alpha)
        return _bca_ci(boot_stats, observed, data, statistic_fn, alpha)

    raise ValueError(
        f"Unknown CI method '{method}'. Use one of: percentile, basic, BCa."
    )


# ==========================================================
# Statistic dispatch (paired-difference statistics)
# ==========================================================

def _mean(d: np.ndarray) -> float:
    return float(np.mean(d))


def _median(d: np.ndarray) -> float:
    return float(np.median(d))


def _trimmed_mean(d: np.ndarray, proportiontocut: float = 0.1) -> float:
    return float(_scipy_stats.trim_mean(d, proportiontocut=proportiontocut))


def _cohens_dz(d: np.ndarray) -> float:
    n = len(d)

    if n < 2:
        return float("nan")

    sd = float(np.std(d, ddof=1))

    if sd < 1e-12:
        return float("nan")

    return float(np.mean(d) / sd)


PAIRED_STATISTICS: Dict[str, Callable[[np.ndarray], float]] = {
    "mean_diff": _mean,
    "median_diff": _median,
    "trimmed_mean_diff": _trimmed_mean,
    "cohens_dz": _cohens_dz,
}


def get_paired_statistic(name: str) -> Callable[[np.ndarray], float]:
    """Resolve a paired-difference statistic name into a callable."""
    if name not in PAIRED_STATISTICS:
        raise ValueError(
            f"Unknown paired statistic '{name}'. "
            f"Choose one of: {sorted(PAIRED_STATISTICS)}."
        )
    return PAIRED_STATISTICS[name]


# ==========================================================
# Stability-aware driver
# ==========================================================

def _classify_endpoint_drift(drift: float) -> Tuple[str, Optional[str]]:
    """
    Map a width-normalised CI endpoint drift to (interpretation, recommendation).

    `drift` is `max(sd(sub_lowers), sd(sub_uppers)) / ci_width`, i.e. the
    largest cross-seed sd of either endpoint expressed as a fraction of the
    primary CI width. This is scale-invariant and well-defined when an
    endpoint is near zero (which the classical CV is not).
    """
    if not np.isfinite(drift):
        return "undefined", None

    if drift < 0.02:
        return "low", None

    if drift < 0.05:
        return "moderate", None

    return "high", (
        f"CI endpoints vary substantially across bootstrap seeds "
        f"(endpoint sd is {drift:.1%} of the CI width). "
        f"To stabilise the CI, either increase B (the most common remedy) "
        f"or, if reproducibility of the CI itself is required for "
        f"regulatory or clinical reporting, re-run with use_sequential=True. "
        f"See Peng (2025), arXiv:2511.18065."
    )


def bootstrap_with_stability(
    data: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float],
    *,
    B: int = 2000,
    n_seeds: int = 5,
    alpha: float = 0.05,
    method: str = "BCa",
    use_sequential: bool = False,
    rho: float = 0.632,
    seed: int = 0,
) -> Dict[str, object]:
    """
    Run `n_seeds` independent sub-bootstraps and return both the pooled
    primary CI and a cross-seed CI stability diagnostic.

    Compute budget
    --------------
    Total replicates drawn: `n_seeds * (B // n_seeds)` (each sub-bootstrap
    uses an equal share of the budget). The diagnostic therefore adds no
    replicates beyond `B`.

    Returns
    -------
    dict with keys:
      - observed_statistic : float
      - ci_lower, ci_upper : float (primary CI, pooled across sub-seeds)
      - resampler : "classical" | "sequential"
      - B_total, n_seeds_for_diagnostic, B_per_seed
      - stability_diagnostic : dict with cv values and interpretation
    """
    data = np.asarray(data, dtype=float)
    n = len(data)

    if n < 3:
        raise ValueError(
            f"bootstrap_with_stability requires at least 3 observations; "
            f"got n={n}."
        )

    B_per_seed = max(B // n_seeds, 1)

    if use_sequential:
        k_n = expected_kn(n, rho=rho)
        if k_n < 1:
            raise ValueError(
                f"Sequential Bootstrap target k_n=floor({rho}*n)={k_n} is too "
                f"small for n={n}. Increase n or rho."
            )

    pooled_boot_stats: List[np.ndarray] = []
    sub_ci_lowers: List[float] = []
    sub_ci_uppers: List[float] = []

    observed = float(statistic_fn(data))

    for s in range(n_seeds):
        rng = np.random.default_rng(seed * 1_000_003 + s + 1)

        if use_sequential:
            idx_list = sequential_bootstrap_indices(n, k_n, B_per_seed, rng)
            boot_stats_s = np.fromiter(
                (statistic_fn(data[idx]) for idx in idx_list),
                dtype=float,
                count=B_per_seed,
            )
        else:
            idx_mat = classical_bootstrap_indices(n, B_per_seed, rng)
            # Vectorised over replicates: take rows of idx_mat
            boot_stats_s = np.fromiter(
                (statistic_fn(data[idx_mat[i]]) for i in range(B_per_seed)),
                dtype=float,
                count=B_per_seed,
            )

        pooled_boot_stats.append(boot_stats_s)

        lo_s, hi_s = bootstrap_ci(
            boot_stats_s,
            observed,
            method=method,
            alpha=alpha,
            data=data,
            statistic_fn=statistic_fn,
        )

        sub_ci_lowers.append(lo_s)
        sub_ci_uppers.append(hi_s)

    # Primary CI: pool all sub-bootstrap replicates.
    pooled = np.concatenate(pooled_boot_stats)

    primary_lo, primary_hi = bootstrap_ci(
        pooled,
        observed,
        method=method,
        alpha=alpha,
        data=data,
        statistic_fn=statistic_fn,
    )

    # Stability diagnostic.
    mean_lo = float(np.mean(sub_ci_lowers))
    mean_hi = float(np.mean(sub_ci_uppers))
    sd_lo = float(np.std(sub_ci_lowers, ddof=1)) if n_seeds > 1 else 0.0
    sd_hi = float(np.std(sub_ci_uppers, ddof=1)) if n_seeds > 1 else 0.0

    ci_width = primary_hi - primary_lo

    # Primary diagnostic: width-normalised endpoint drift. Scale-invariant and
    # well-defined when an endpoint is near zero.
    if ci_width > 1e-12:
        endpoint_drift = float(max(sd_lo, sd_hi) / ci_width)
    else:
        endpoint_drift = float("nan")

    # Secondary: classical CV (kept for back-compat / diagnostic richness, but
    # *not* used for the interpretation ladder because it blows up near zero).
    def _safe_cv(sd_v: float, mean_v: float) -> float:
        if abs(mean_v) < 1e-12:
            return float("nan")
        return float(sd_v / abs(mean_v))

    cv_lo = _safe_cv(sd_lo, mean_lo)
    cv_hi = _safe_cv(sd_hi, mean_hi)

    interpretation, recommendation = _classify_endpoint_drift(endpoint_drift)

    return {
        "observed_statistic": observed,
        "ci_lower": primary_lo,
        "ci_upper": primary_hi,
        "resampler": "sequential" if use_sequential else "classical",
        "B_total": B_per_seed * n_seeds,
        "n_seeds_for_diagnostic": n_seeds,
        "B_per_seed": B_per_seed,
        "stability_diagnostic": {
            "endpoint_drift": endpoint_drift,
            "ci_lower_sd": sd_lo,
            "ci_upper_sd": sd_hi,
            "ci_lower_cv": cv_lo,
            "ci_upper_cv": cv_hi,
            "interpretation": interpretation,
            "recommendation": recommendation,
            "sub_ci_lowers": sub_ci_lowers,
            "sub_ci_uppers": sub_ci_uppers,
        },
    }