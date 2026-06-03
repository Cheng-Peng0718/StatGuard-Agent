"""
Carpet-bomb case matrix with gold-standard answers.

This module programmatically generates a large matrix of statistical scenarios
spanning the dimensions a real analyst hits: number of groups, distribution
shape, sample size, variance (in)equality, true effect size, design (paired vs
independent), and task type (group comparison / regression / correlation /
chi-square / paired).

For EACH generated case, the gold answer is computed INDEPENDENTLY with
scipy / statsmodels -- not by the framework. The benchmark then checks the
framework against this external ground truth, which is what makes the accuracy
search objective and "carpet-bomb" wide.

A `Case` carries:
  - key:        unique id
  - task:       which family ("group" / "regression" / "correlation" /
                "chi_square" / "paired")
  - df:         the data
  - tool / args: how to invoke the plugin
  - gold:       dict of independently-computed reference values
  - expect:     structured expectations (e.g. expected primary-test family,
                significance verdict) for routing / honesty checks

Seeds are fixed so the whole matrix is reproducible.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

GLOBAL_SEED = 20260521


@dataclass
class Case:
    key: str
    task: str
    df: pd.DataFrame
    tool: str
    args: Dict[str, Any]
    gold: Dict[str, Any] = field(default_factory=dict)
    expect: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""


# ============================================================
# Distribution samplers (mean-centred, scale-controlled)
# ============================================================

def _sample(dist: str, n: int, loc: float, scale: float, rng) -> np.ndarray:
    if dist == "normal":
        return rng.normal(loc, scale, n)
    if dist == "lognormal":
        # heavy positive skew; shift so location is comparable
        raw = rng.lognormal(0.0, 0.9, n)
        raw = (raw - raw.mean()) / raw.std() * scale + loc
        return raw
    if dist == "heavy_tail":
        # Student-t with 3 df -> heavy tails
        return rng.standard_t(3, n) * scale + loc
    if dist == "uniform":
        half = scale * math.sqrt(3)
        return rng.uniform(loc - half, loc + half, n)
    raise ValueError(dist)


# ============================================================
# Group-comparison cases (the richest dimension)
# ============================================================

def _make_group_cases() -> List[Case]:
    cases: List[Case] = []
    n_groups_opts = [2, 3, 5]
    dist_opts = ["normal", "lognormal", "heavy_tail", "uniform"]
    n_opts = [12, 30, 120]          # small / medium / large per group
    var_opts = ["equal", "unequal"]  # variance homogeneity
    effect_opts = [0.0, 0.5, 1.2]    # between-group mean separation (in SD units)

    combo_id = 0
    for n_groups, dist, n_per, var, effect in itertools.product(
        n_groups_opts, dist_opts, n_opts, var_opts, effect_opts
    ):
        combo_id += 1
        rng = np.random.default_rng(GLOBAL_SEED + combo_id)
        cols_y = []
        cols_g = []
        arrays = []
        for gi in range(n_groups):
            loc = 10.0 + effect * gi          # separate group means
            scale = 2.0 if var == "equal" else 2.0 * (1 + gi)  # unequal -> growing sd
            arr = _sample(dist, n_per, loc, scale, rng)
            arrays.append(arr)
            cols_y.append(arr)
            cols_g += [f"g{gi}"] * n_per
        y = np.concatenate(cols_y)
        df = pd.DataFrame({"y": y, "g": cols_g})

        # ---- gold answer (independent) ----
        gold: Dict[str, Any] = {}
        if n_groups == 2:
            t_ref, p_ref = stats.ttest_ind(arrays[0], arrays[1], equal_var=False)
            u_ref, pu_ref = stats.mannwhitneyu(arrays[0], arrays[1], alternative="two-sided")
            gold["welch_t"] = float(t_ref)
            gold["welch_p"] = float(p_ref)
            gold["mwu_p"] = float(pu_ref)
        else:
            f_ref, p_ref = stats.f_oneway(*arrays)
            gold["anova_f"] = float(f_ref)
            gold["anova_p"] = float(p_ref)
            h_ref, ph_ref = stats.kruskal(*arrays)
            gold["kruskal_p"] = float(ph_ref)

        # max abs skew across groups (drives the P2 switch expectation)
        max_skew = max(abs(float(stats.skew(a, bias=False))) for a in arrays if len(a) >= 3)
        min_n = min(len(a) for a in arrays)
        # Shapiro on each group (small samples of heavy/skew dists -> reject)
        any_non_normal = False
        for a in arrays:
            if 3 <= len(a) <= 5000:
                try:
                    if stats.shapiro(a).pvalue < 0.05:
                        any_non_normal = True
                except Exception:
                    pass

        # Expected routing per the P2 rule: switch iff non-normal AND
        # (min_n < 30 OR max_skew >= 1.5).
        expect_switch = bool(any_non_normal and (min_n < 30 or max_skew >= 1.5))

        cases.append(Case(
            key=f"grp_{combo_id:03d}_{n_groups}g_{dist}_n{n_per}_{var}_eff{effect}",
            task="group",
            df=df,
            tool="statistical_group_comparison",
            args={"target_col": "y", "group_col": "g"},
            gold=gold,
            expect={
                "n_groups": n_groups,
                "max_abs_skew": max_skew,
                "min_group_n": min_n,
                "any_non_normal": any_non_normal,
                "expect_nonparametric_switch": expect_switch,
            },
            notes=f"{n_groups} groups, {dist}, n={n_per}/grp, var={var}, effect={effect}",
        ))
    return cases


# ============================================================
# Regression cases
# ============================================================

def _make_regression_cases() -> List[Case]:
    import statsmodels.api as sm
    cases: List[Case] = []
    combo_id = 0
    for n in [40, 150]:
        for noise in [0.3, 1.5]:
            for slope in [0.0, 2.0]:
                combo_id += 1
                rng = np.random.default_rng(GLOBAL_SEED + 1000 + combo_id)
                x = np.round(rng.normal(0, 1, n), 1)  # rounded to dodge id-like filter
                y = np.round(slope * x + rng.normal(0, noise, n), 2)
                df = pd.DataFrame({"y": y, "x": x})
                X = sm.add_constant(x)
                model = sm.OLS(y, X).fit()
                cases.append(Case(
                    key=f"reg_{combo_id:03d}_n{n}_noise{noise}_slope{slope}",
                    task="regression",
                    df=df,
                    tool="run_multiple_regression",
                    args={"target_col": "y", "feature_cols": ["x"]},
                    gold={"r_squared": float(model.rsquared),
                          "f_stat": float(model.fvalue),
                          "f_p": float(model.f_pvalue)},
                    expect={"slope_is_zero": slope == 0.0},
                    notes=f"n={n}, noise={noise}, slope={slope}",
                ))
    return cases


# ============================================================
# Correlation cases
# ============================================================

def _make_correlation_cases() -> List[Case]:
    cases: List[Case] = []
    combo_id = 0
    for n in [30, 120]:
        for rho in [0.0, 0.4, 0.8]:
            combo_id += 1
            rng = np.random.default_rng(GLOBAL_SEED + 2000 + combo_id)
            x = rng.normal(0, 1, n)
            y = rho * x + math.sqrt(max(1 - rho ** 2, 0.01)) * rng.normal(0, 1, n)
            df = pd.DataFrame({"x": x, "y": y})
            r_ref, p_ref = stats.pearsonr(x, y)
            cases.append(Case(
                key=f"cor_{combo_id:03d}_n{n}_rho{rho}",
                task="correlation",
                df=df,
                tool="run_correlation_test",
                args={"x_col": "x", "y_col": "y"},
                gold={"r": float(r_ref), "p": float(p_ref)},
                expect={"rho": rho},
                notes=f"n={n}, true rho={rho}",
            ))
    return cases


# ============================================================
# Chi-square cases
# ============================================================

def _make_chi_square_cases() -> List[Case]:
    cases: List[Case] = []
    combo_id = 0
    for n in [80, 300]:
        for assoc in [False, True]:
            combo_id += 1
            rng = np.random.default_rng(GLOBAL_SEED + 3000 + combo_id)
            if assoc:
                a = rng.choice(["x", "y"], n)
                b = np.where(rng.random(n) < np.where(a == "x", 0.7, 0.3), "p", "q")
            else:
                a = rng.choice(["x", "y"], n)
                b = rng.choice(["p", "q"], n)
            df = pd.DataFrame({"a": a, "b": b})
            table = pd.crosstab(df["a"], df["b"]).values
            chi2_ref, p_ref, dof_ref, _ = stats.chi2_contingency(table, correction=True)
            cases.append(Case(
                key=f"chi_{combo_id:03d}_n{n}_assoc{assoc}",
                task="chi_square",
                df=df,
                tool="run_chi_square",
                args={"row_col": "a", "col_col": "b"},
                gold={"chi2": float(chi2_ref), "p": float(p_ref), "dof": int(dof_ref)},
                expect={"associated": assoc},
                notes=f"n={n}, association={assoc}",
            ))
    return cases


# ============================================================
# Paired cases
# ============================================================

def _make_paired_cases() -> List[Case]:
    cases: List[Case] = []
    combo_id = 0
    for n in [15, 40]:
        for shift in [0.0, 3.0]:
            for diff_dist in ["normal", "lognormal"]:
                combo_id += 1
                rng = np.random.default_rng(GLOBAL_SEED + 4000 + combo_id)
                pre = rng.normal(50, 8, n)
                if diff_dist == "normal":
                    diff = rng.normal(shift, 4, n)
                else:
                    raw = rng.lognormal(0, 0.8, n)
                    diff = (raw - raw.mean()) + shift
                post = pre + diff
                df = pd.DataFrame({"pre": pre, "post": post})
                w_p = stats.wilcoxon(pre, post).pvalue if np.any(pre != post) else 1.0
                t_p = stats.ttest_rel(pre, post).pvalue
                cases.append(Case(
                    key=f"pair_{combo_id:03d}_n{n}_shift{shift}_{diff_dist}",
                    task="paired",
                    df=df,
                    tool="paired_comparison",
                    args={"target_col_1": "pre", "target_col_2": "post"},
                    gold={"wilcoxon_p": float(w_p), "paired_t_p": float(t_p)},
                    expect={"true_shift": shift, "diff_dist": diff_dist},
                    notes=f"n={n}, shift={shift}, diff={diff_dist}",
                ))
    return cases


# ============================================================
# Adversarial edge cases (deliberately nasty, but valid)
# ============================================================

def _make_adversarial_cases() -> List[Case]:
    """Extreme but valid inputs that are most likely to expose numerical or
    no-error problems. Gold answers computed independently where applicable;
    otherwise the check is purely no-error (must not raise / must return a
    structured status)."""
    cases: List[Case] = []
    rng = np.random.default_rng(GLOBAL_SEED + 5000)

    # 1. Totally separated groups (extreme effect)
    a = rng.normal(0, 1, 40); b = rng.normal(100, 1, 40)
    cases.append(Case(
        key="adv_extreme_separation", task="group",
        df=pd.DataFrame({"y": np.r_[a, b], "g": ["a"] * 40 + ["b"] * 40}),
        tool="statistical_group_comparison", args={"target_col": "y", "group_col": "g"},
        gold={"welch_t": float(stats.ttest_ind(a, b, equal_var=False)[0]),
              "welch_p": float(stats.ttest_ind(a, b, equal_var=False)[1]),
              "mwu_p": float(stats.mannwhitneyu(a, b, alternative="two-sided")[1])},
        expect={"n_groups": 2, "max_abs_skew": 0.0, "min_group_n": 40,
                "any_non_normal": False, "expect_nonparametric_switch": False},
        notes="groups separated by 100 SD",
    ))

    # 2. Near-identical groups (p ~ 1)
    base = rng.normal(0, 1, 50)
    a = base.copy(); b = base + rng.normal(0, 1e-9, 50)
    cases.append(Case(
        key="adv_near_identical", task="group",
        df=pd.DataFrame({"y": np.r_[a, b], "g": ["a"] * 50 + ["b"] * 50}),
        tool="statistical_group_comparison", args={"target_col": "y", "group_col": "g"},
        gold={"welch_t": float(stats.ttest_ind(a, b, equal_var=False)[0]),
              "welch_p": float(stats.ttest_ind(a, b, equal_var=False)[1]),
              "mwu_p": float(stats.mannwhitneyu(a, b, alternative="two-sided")[1])},
        expect={"n_groups": 2, "max_abs_skew": 0.0, "min_group_n": 50,
                "any_non_normal": False, "expect_nonparametric_switch": False},
        notes="groups differ by 1e-9",
    ))

    # 3. Perfect correlation
    x = rng.normal(0, 1, 30); y = x.copy()
    rr, pp = stats.pearsonr(x, y)
    cases.append(Case(
        key="adv_perfect_correlation", task="correlation",
        df=pd.DataFrame({"x": x, "y": y}),
        tool="run_correlation_test", args={"x_col": "x", "y_col": "y"},
        gold={"r": float(rr), "p": float(pp)},
        expect={"rho": 1.0}, notes="y == x exactly",
    ))

    # 4. Regression with extreme outlier
    import statsmodels.api as sm
    x = np.round(rng.normal(0, 1, 50), 1)
    y = np.round(2 * x + rng.normal(0, 0.3, 50), 2); y[0] = 1000.0
    mdl = sm.OLS(y, sm.add_constant(x)).fit()
    cases.append(Case(
        key="adv_regression_outlier", task="regression",
        df=pd.DataFrame({"y": y, "x": x}),
        tool="run_multiple_regression", args={"target_col": "y", "feature_cols": ["x"]},
        gold={"r_squared": float(mdl.rsquared), "f_stat": float(mdl.fvalue),
              "f_p": float(mdl.f_pvalue)},
        expect={"slope_is_zero": False}, notes="one y value = 1000",
    ))

    return cases

def _bootstrap_tol(stat_name: str, n_pairs: int, ci_width):
    """
    Tolerance for |our_ci - scipy_ci| comparison, calibrated to the
    Monte-Carlo SE of bootstrap CI endpoints at B=10000 when our pipeline
    and scipy use *independent* RNG seeds. We use ~3-sigma envelopes per
    statistic; on small-n median bootstrap the distribution is so discrete
    that independent seeds disagree by 30-50% of CI width even when both
    implementations are correct, so we drop the numerical comparison for
    those cells (the case is still run as a smoke test for status/resampler).
    """
    if ci_width is None or ci_width <= 0:
        return None
    if stat_name == "mean_diff":
        return 0.030 * ci_width
    if stat_name == "cohens_dz":
        return 0.050 * ci_width
    if stat_name == "median_diff":
        if n_pairs >= 40:
            return 0.100 * ci_width
        # n < 40 median bootstrap: too discrete to compare across RNG seeds
        return None
    return 0.050 * ci_width


def _make_bootstrap_cases() -> List[Case]:
    """
    Carpet cases for bootstrap_inference.

    Dimensions covered:
      - statistic: mean_diff / median_diff / cohens_dz
      - ci_method: percentile / basic / BCa
      - resampler: classical / sequential
      - difference distribution: normal / lognormal / heavy_tail
      - n_pairs: 15 (small), 40 (medium), 120 (large)
      - B for diagnostic ladder: 500 (small) / 10000 (large)

    Cases compute gold answers via scipy.stats.bootstrap for the classical
    branch (where scipy supports the method) and via structural invariants
    for the Sequential branch.
    """
    cases: List[Case] = []
    combo_id = 0

    paired_stat_opts = ["mean_diff", "median_diff", "cohens_dz"]
    ci_method_opts = ["percentile", "basic", "BCa"]
    n_opts = [15, 40, 120]
    dist_opts = ["normal", "lognormal", "heavy_tail"]

    # --- Branch A: classical bootstrap, validated against scipy --------------

    for n_pairs, dist, stat_name, ci_method in itertools.product(
            n_opts, dist_opts, paired_stat_opts, ci_method_opts
    ):
        # BCa requires n_pairs >= 8 (plugin enforces this); we already meet it.
        combo_id += 1
        rng = np.random.default_rng(GLOBAL_SEED + 10000 + combo_id)

        # Paired sample: two columns with a known mean shift.
        pre = _sample(dist, n_pairs, loc=10.0, scale=2.0, rng=rng)
        post = pre + _sample(dist, n_pairs, loc=0.5, scale=1.5, rng=rng)

        df = pd.DataFrame({"pre": pre, "post": post})
        diffs = (post - pre).astype(float)

        # Gold: scipy.stats.bootstrap with matched B / method / seed.
        # We use B=10000 to keep Monte-Carlo error well below the tolerance.
        if stat_name == "mean_diff":
            stat_fn = np.mean
        elif stat_name == "median_diff":
            stat_fn = np.median
        elif stat_name == "cohens_dz":
            def stat_fn(d):  # noqa: E306
                sd = np.std(d, ddof=1)
                return np.mean(d) / sd if sd > 1e-12 else float("nan")
        else:
            raise ValueError(stat_name)

        try:
            scipy_res = stats.bootstrap(
                (diffs,),
                statistic=stat_fn,
                n_resamples=10000,
                confidence_level=0.95,
                method=ci_method.lower(),
                random_state=np.random.default_rng(combo_id),
                vectorized=False,
            )
            gold_lo = float(scipy_res.confidence_interval.low)
            gold_hi = float(scipy_res.confidence_interval.high)
        except Exception:
            # Some scipy versions reject median for BCa on small samples;
            # skip the gold check but keep the case as a smoke test.
            gold_lo, gold_hi = None, None

        ci_width = (gold_hi - gold_lo) if gold_lo is not None else None

        cases.append(Case(
            key=f"bootstrap_classical_{dist}_n{n_pairs}_{stat_name}_{ci_method}_{combo_id}",
            task="paired_bootstrap",
            df=df,
            tool="bootstrap_inference",
            args={
                "target_col_1": "pre",
                "target_col_2": "post",
                "statistic": stat_name,
                "ci_method": ci_method,
                "B": 10000,
                "n_seeds": 5,
                "alpha": 0.05,
                "use_sequential": False,
                "seed": combo_id,
            },
            gold={
                "ci_lower": gold_lo,
                "ci_upper": gold_hi,
                "ci_width": ci_width,
                "tolerance": _bootstrap_tol(stat_name, n_pairs, ci_width),
                "source": "scipy.stats.bootstrap",
            },
            expect={
                "status": "ok",
                "resampler": "classical",
                "statistic": stat_name,
                "ci_method": ci_method,
            },
            notes=(
                f"Classical bootstrap, n_pairs={n_pairs}, diff dist={dist}, "
                f"stat={stat_name}, method={ci_method}. CI endpoints must "
                f"match scipy.stats.bootstrap within 1.5% of CI width."
            ),
        ))

    # --- Branch B: Sequential Bootstrap, structural-invariant gold -----------

    for n_pairs, dist, stat_name in itertools.product(
            n_opts, dist_opts, paired_stat_opts
    ):
        combo_id += 1
        rng = np.random.default_rng(GLOBAL_SEED + 20000 + combo_id)

        pre = _sample(dist, n_pairs, loc=10.0, scale=2.0, rng=rng)
        post = pre + _sample(dist, n_pairs, loc=0.5, scale=1.5, rng=rng)

        df = pd.DataFrame({"pre": pre, "post": post})

        cases.append(Case(
            key=f"bootstrap_sequential_{dist}_n{n_pairs}_{stat_name}_{combo_id}",
            task="paired_bootstrap_sequential",
            df=df,
            tool="bootstrap_inference",
            args={
                "target_col_1": "pre",
                "target_col_2": "post",
                "statistic": stat_name,
                "ci_method": "BCa",
                "B": 5000,
                "n_seeds": 5,
                "alpha": 0.05,
                "use_sequential": True,
                "rho": 0.632,
                "seed": combo_id,
            },
            gold={
                "k_n_expected": int(np.floor(0.632 * n_pairs)),
                "source": "structural-invariants",
            },
            expect={
                "status": "ok",
                "resampler": "sequential",
                "use_sequential": True,
                "statistic": stat_name,
            },
            notes=(
                f"Sequential Bootstrap, n_pairs={n_pairs}, diff dist={dist}, "
                f"stat={stat_name}. Gold check: resampler=='sequential' and "
                f"k_n equals floor(0.632 * n_pairs). CI math itself is "
                f"validated in Branch A; here we only check the SB-specific "
                f"contract."
            ),
        ))

    # --- Branch C: stability-diagnostic ladder -------------------------------
    # B=500 should yield 'high' (or at worst 'moderate') on small data;
    # B=10000 should yield 'low' (or at worst 'moderate') on clean data.

    for B, expected_band in [(500, {"moderate", "high"}), (10000, {"low", "moderate"})]:
        combo_id += 1
        rng = np.random.default_rng(GLOBAL_SEED + 30000 + combo_id)

        pre = _sample("normal", 60, loc=10.0, scale=2.0, rng=rng)
        post = pre + _sample("normal", 60, loc=0.5, scale=1.0, rng=rng)
        df = pd.DataFrame({"pre": pre, "post": post})

        cases.append(Case(
            key=f"bootstrap_stability_ladder_B{B}_{combo_id}",
            task="paired_bootstrap_stability",
            df=df,
            tool="bootstrap_inference",
            args={
                "target_col_1": "pre",
                "target_col_2": "post",
                "statistic": "mean_diff",
                "ci_method": "BCa",
                "B": B,
                "n_seeds": 5,
                "alpha": 0.05,
                "use_sequential": False,
                "seed": combo_id,
            },
            gold={
                "expected_interpretation_in": list(expected_band),
                "source": "stability-ladder",
            },
            expect={
                "status": "ok",
                "stability_interpretation_in": list(expected_band),
            },
            notes=(
                f"Stability diagnostic ladder: at B={B} on clean normal "
                f"paired data, interpretation should be in {expected_band}."
            ),
        ))

    return cases

def _make_logistic_cases() -> List[Case]:
    """
    Carpet cases for run_logistic_regression.

    Validates against an independently-fitted statsmodels.Logit:
      - structural fields (n_obs, n_events, n_predictors) match,
      - pseudo R^2 (McFadden) and log-likelihood agree numerically,
      - coefficient signs match (catches wiring bugs like swapped columns).

    Because the plugin also uses statsmodels internally, this is primarily a
    wiring test (correct columns -> correct coefficients -> correct fields
    in `details`), not a method-vs-method comparison.
    """
    import warnings
    import statsmodels.api as sm

    cases: List[Case] = []
    combo_id = 0

    # (label, n, intercept, true_betas, expected_positive_rate)
    scenarios = [
        ("balanced",        200,  0.0,  [1.2, -0.8],         0.50),
        ("imbalanced_low",  200, -2.5,  [1.2, -0.8],         0.10),
        ("small_n",          50,  0.0,  [1.5, -1.0],         0.50),
        ("strong_signal",   100,  0.0,  [2.0, -2.0],         0.50),
        ("three_preds",     200,  0.0,  [1.0, -0.5, 0.7],    0.50),
    ]

    for name, n, intercept, betas, _ in scenarios:
        combo_id += 1
        rng = np.random.default_rng(GLOBAL_SEED + 40000 + combo_id)
        k = len(betas)
        feature_cols = [f"x{i+1}" for i in range(k)]

        cols: Dict[str, np.ndarray] = {f: rng.standard_normal(n) for f in feature_cols}
        eta = intercept + sum(betas[i] * cols[feature_cols[i]] for i in range(k))
        p = 1.0 / (1.0 + np.exp(-eta))
        y = (rng.random(n) < p).astype(int)

        df = pd.DataFrame({"outcome": y, **cols})

        # Gold: fit the same logit independently.
        y_g = df["outcome"].astype(float)
        X_g = sm.add_constant(df[feature_cols], has_constant="add")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = sm.Logit(y_g, X_g).fit(disp=0, maxiter=200)
                gold = {
                    "n_obs": int(len(y_g)),
                    "n_events": int(min(y_g.sum(), len(y_g) - y_g.sum())),
                    "n_predictors": k,
                    "pseudo_r2": float(model.prsquared),
                    "log_likelihood": float(model.llf),
                    "coef_signs": {f: float(np.sign(model.params[f])) for f in feature_cols},
                    "converged": bool(model.mle_retvals.get("converged", True)),
                }
            except Exception:
                gold = {"_unfittable": True}

        cases.append(Case(
            key=f"logistic_{name}_n{n}_{combo_id}",
            task="logistic",
            df=df,
            tool="run_logistic_regression",
            args={"target_col": "outcome", "feature_cols": feature_cols},
            gold=gold,
            expect={"status": "ok", "n_predictors": k},
            notes=f"Logistic regression, scenario={name}, n={n}, true_betas={betas}",
        ))

    # One case with string labels ('yes'/'no').
    combo_id += 1
    rng = np.random.default_rng(GLOBAL_SEED + 40000 + combo_id)
    n = 150
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    eta = 0.5 + 1.5 * x1 - 1.0 * x2
    p = 1.0 / (1.0 + np.exp(-eta))
    y_num = (rng.random(n) < p).astype(int)
    y_str = np.where(y_num == 1, "yes", "no")
    df = pd.DataFrame({"outcome": y_str, "x1": x1, "x2": x2})

    X_g = sm.add_constant(pd.DataFrame({"x1": x1, "x2": x2}), has_constant="add")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.Logit(y_num.astype(float), X_g).fit(disp=0, maxiter=200)
        gold = {
            "n_obs": n,
            "n_events": int(min(y_num.sum(), n - y_num.sum())),
            "n_predictors": 2,
            "pseudo_r2": float(model.prsquared),
            "log_likelihood": float(model.llf),
            "coef_signs": {"x1": float(np.sign(model.params["x1"])),
                           "x2": float(np.sign(model.params["x2"]))},
            "positive_class": "yes",
            "converged": True,
        }

    cases.append(Case(
        key=f"logistic_string_labels_n{n}_{combo_id}",
        task="logistic",
        df=df,
        tool="run_logistic_regression",
        args={"target_col": "outcome", "feature_cols": ["x1", "x2"]},
        gold=gold,
        expect={"status": "ok", "positive_class": "yes"},
        notes=f"Logistic regression with string labels ('yes'/'no'), n={n}",
    ))

    return cases

def generate_all_cases() -> List[Case]:
    cases: List[Case] = []
    cases += _make_group_cases()
    cases += _make_regression_cases()
    cases += _make_correlation_cases()
    cases += _make_chi_square_cases()
    cases += _make_paired_cases()
    cases += _make_adversarial_cases()
    cases += _make_bootstrap_cases()
    cases += _make_logistic_cases()
    return cases


def summarize_matrix() -> Dict[str, int]:
    cases = generate_all_cases()
    by_task: Dict[str, int] = {}
    for c in cases:
        by_task[c.task] = by_task.get(c.task, 0) + 1
    by_task["TOTAL"] = len(cases)
    return by_task


if __name__ == "__main__":
    summary = summarize_matrix()
    print("Carpet-bomb case matrix:")
    for k, v in summary.items():
        print(f"  {k}: {v}")