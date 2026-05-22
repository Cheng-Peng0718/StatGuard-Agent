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

def generate_all_cases() -> List[Case]:
    cases: List[Case] = []
    cases += _make_group_cases()
    cases += _make_regression_cases()
    cases += _make_correlation_cases()
    cases += _make_chi_square_cases()
    cases += _make_paired_cases()
    cases += _make_adversarial_cases()
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