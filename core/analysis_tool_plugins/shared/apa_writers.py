"""
APA 7th-edition Methods paragraph generators, one per inferential tool family.

Each writer accepts a finished `analysis_run` dict and returns a Methods-style
paragraph (Markdown formatting) suitable for inclusion in a research paper.

Style is anchored by `core.analysis_tool_plugins.shared.apa_formatting` so
every paragraph in the assembled Methods section reads consistently.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .apa_formatting import (
    fmt_p,
    fmt_general,
    fmt_signed,
    fmt_int,
    fmt_ci,
    fmt_bounded_unit,
)


def _alpha_of(run: Dict[str, Any]) -> float:
    args = run.get("arguments", {}) or {}
    try:
        return float(args.get("alpha", 0.05))
    except Exception:
        return 0.05


def _arg(run: Dict[str, Any], name: str, default: str) -> str:
    args = run.get("arguments", {}) or {}
    val = args.get(name)
    return str(val) if val is not None else default


# ============================================================
# statistical_group_comparison: 3 routing paths
#   - 2 groups        -> Welch t-test  (+/- post-hoc N/A)
#   - 3+, eq vars     -> classic ANOVA + Tukey HSD
#   - 3+, unequal vars-> Welch ANOVA  + Games-Howell
# ============================================================

def write_apa_statistical_group_comparison(run: Dict[str, Any]) -> Optional[str]:
    metrics = run.get("metrics", {}) or {}
    method = (metrics.get("method") or "").lower()
    p = metrics.get("p_value")
    alpha = metrics.get("alpha") or _alpha_of(run)
    target = _arg(run, "target_col", "the outcome")
    group_col = _arg(run, "group_col", "group")

    # If P2 routed this comparison to a rank-based primary test, reuse the
    # nonparametric writer rather than duplicating its logic here.
    if "mann-whitney" in method or "mann whitney" in method or "kruskal" in method:
        return write_apa_nonparametric_group_comparison(run)

    # -------- Welch t-test path --------
    if "t-test" in method or "t test" in method:
        t = metrics.get("t_statistic")
        df = metrics.get("degrees_of_freedom")
        g = metrics.get("effect_size")  # Hedges' g
        g_lo, g_hi = metrics.get("effect_size_ci_low"), metrics.get("effect_size_ci_high")
        d = metrics.get("cohens_d")
        md = (
            metrics.get("mean_difference_group1_minus_group2")
            or metrics.get("mean_difference")
        )
        md_lo, md_hi = metrics.get("mean_difference_ci_low"), metrics.get("mean_difference_ci_high")

        parts = [
            f"A Welch's two-sample *t* test was conducted to compare {target} between the two levels of {group_col}."
        ]
        if md is not None:
            ci = fmt_ci(md_lo, md_hi)
            ci_phrase = f", {ci}" if ci else ""
            parts.append(f"The mean difference was {fmt_general(md)}{ci_phrase}.")
        if t is not None and df is not None and p is not None:
            parts.append(f"The test was *t*({fmt_general(df, 2)}) = {fmt_general(t)}, {fmt_p(p)}.")
        if g is not None:
            ci = fmt_ci(g_lo, g_hi)
            ci_phrase = f", {ci}" if ci else ""
            parts.append(f"Hedges' *g* = {fmt_signed(g)}{ci_phrase}.")
        elif d is not None:
            parts.append(f"Cohen's *d* = {fmt_signed(d)}.")
        return " ".join(parts)

    # -------- ANOVA path (classic or Welch) --------
    if "anova" in method:
        is_welch = "welch" in method
        f_stat = metrics.get("F_statistic")
        df_b = metrics.get("degrees_of_freedom_between")
        df_w = metrics.get("degrees_of_freedom_within")
        eta2 = metrics.get("eta_squared")
        eta_lo, eta_hi = metrics.get("eta_squared_ci_low"), metrics.get("eta_squared_ci_high")
        omega2 = metrics.get("omega_squared")

        opener = (
            "Because Levene's test indicated heterogeneity of variance across groups, "
            f"Welch's one-way ANOVA was conducted to compare {target} across levels of {group_col}."
            if is_welch
            else f"A one-way ANOVA was conducted to compare {target} across levels of {group_col}."
        )
        parts = [opener]

        if f_stat is not None and df_b is not None and df_w is not None and p is not None:
            parts.append(
                f"The test was *F*({fmt_int(df_b)}, {fmt_general(df_w, 2)}) = "
                f"{fmt_general(f_stat)}, {fmt_p(p)}."
            )
        if eta2 is not None:
            ci = fmt_ci(eta_lo, eta_hi, bounded_unit=True, digits=3)
            ci_phrase = f", {ci}" if ci else ""
            parts.append(f"η² = {fmt_bounded_unit(eta2)}{ci_phrase}.")
        if omega2 is not None:
            parts.append(f"ω² = {fmt_bounded_unit(omega2)}.")

        # Post-hoc summary
        post_hoc = (run.get("tables", {}) or {}).get("post_hoc_pairwise") or []
        if metrics.get("significant_at_alpha") and post_hoc:
            sig_pairs = [r for r in post_hoc if r.get("significant_at_alpha")]
            adj = post_hoc[0].get("adjustment_method") or post_hoc[0].get("method") or "post-hoc"
            adj_lower = str(adj).lower()
            label = (
                "Tukey's HSD"
                if "tukey" in adj_lower
                else "Games-Howell" if "games" in adj_lower
                else str(adj)
            )
            parts.append(
                f"Post-hoc pairwise comparisons were conducted using {label}; "
                f"{len(sig_pairs)} of {len(post_hoc)} contrasts were statistically significant at α = {alpha}."
            )

        return " ".join(parts)

    return None


# ============================================================
# independent_t_test: Welch t-test (always)
# ============================================================

def write_apa_independent_t_test(run: Dict[str, Any]) -> Optional[str]:
    metrics = run.get("metrics", {}) or {}
    t = metrics.get("t_statistic")
    df = metrics.get("degrees_of_freedom")
    p = metrics.get("p_value")
    g = metrics.get("effect_size")  # Hedges' g
    g_lo, g_hi = metrics.get("effect_size_ci_low"), metrics.get("effect_size_ci_high")
    d = metrics.get("cohens_d")
    md = (
        metrics.get("mean_difference_group1_minus_group2")
        or metrics.get("mean_difference")
    )
    md_lo, md_hi = metrics.get("mean_difference_ci_low"), metrics.get("mean_difference_ci_high")

    args = run.get("arguments", {}) or {}
    target = args.get("target_col", "the outcome")
    g1 = args.get("group1_val") or args.get("group1") or "group 1"
    g2 = args.get("group2_val") or args.get("group2") or "group 2"

    parts = [
        f"A Welch's two-sample *t* test was conducted to compare {target} between {g1} and {g2}."
    ]
    if md is not None:
        ci = fmt_ci(md_lo, md_hi)
        ci_phrase = f", {ci}" if ci else ""
        parts.append(f"The mean difference was {fmt_general(md)}{ci_phrase}.")
    if t is not None and df is not None and p is not None:
        parts.append(f"The test was *t*({fmt_general(df, 2)}) = {fmt_general(t)}, {fmt_p(p)}.")
    if g is not None:
        ci = fmt_ci(g_lo, g_hi)
        ci_phrase = f", {ci}" if ci else ""
        parts.append(f"Hedges' *g* = {fmt_signed(g)}{ci_phrase}.")
    elif d is not None:
        parts.append(f"Cohen's *d* = {fmt_signed(d)}.")

    return " ".join(parts)


# ============================================================
# paired_comparison: paired t recommended OR Wilcoxon recommended
# ============================================================

def write_apa_paired_comparison(run: Dict[str, Any]) -> Optional[str]:
    metrics = run.get("metrics", {}) or {}
    recommended = metrics.get("recommended_test")
    n = metrics.get("n_complete_pairs")

    args = run.get("arguments", {}) or {}
    c1 = args.get("target_col_1", "measurement 1")
    c2 = args.get("target_col_2", "measurement 2")

    # -------- Paired t-test as primary --------
    if recommended == "paired_t_test":
        t = metrics.get("t_statistic")
        df = metrics.get("degrees_of_freedom")
        p = metrics.get("paired_t_p_value") or metrics.get("p_value")
        d_z = metrics.get("cohens_d_z")
        d_lo, d_hi = metrics.get("cohens_d_z_ci_low"), metrics.get("cohens_d_z_ci_high")
        md = metrics.get("mean_difference")
        md_lo, md_hi = metrics.get("mean_difference_ci_low"), metrics.get("mean_difference_ci_high")

        parts = [
            f"A paired-samples *t* test was conducted to compare {c1} and {c2} "
            f"(n = {n} complete pairs)."
        ]
        if md is not None:
            ci = fmt_ci(md_lo, md_hi)
            ci_phrase = f", {ci}" if ci else ""
            parts.append(f"The mean difference was {fmt_general(md)}{ci_phrase}.")
        if t is not None and df is not None and p is not None:
            parts.append(f"The test was *t*({fmt_int(df)}) = {fmt_general(t)}, {fmt_p(p)}.")
        if d_z is not None:
            ci = fmt_ci(d_lo, d_hi)
            ci_phrase = f", {ci}" if ci else ""
            parts.append(f"Cohen's *d*_z = {fmt_signed(d_z)}{ci_phrase}.")
        return " ".join(parts)

    # -------- Wilcoxon signed-rank as primary --------
    w = metrics.get("W_statistic")
    p = metrics.get("wilcoxon_p_value") or metrics.get("p_value")
    rb = metrics.get("rank_biserial_correlation")
    hl = metrics.get("hodges_lehmann_pseudomedian")
    hl_lo, hl_hi = metrics.get("hodges_lehmann_ci_low"), metrics.get("hodges_lehmann_ci_high")

    parts = [
        "Because Shapiro-Wilk indicated the paired differences departed from normality, "
        f"a Wilcoxon signed-rank test was conducted to compare {c1} and {c2} "
        f"(n = {n} complete pairs)."
    ]
    if w is not None and p is not None:
        parts.append(f"The test was *W* = {fmt_general(w)}, {fmt_p(p)}.")
    if rb is not None:
        parts.append(f"The matched-pairs rank-biserial *r* = {fmt_signed(rb)}.")
    if hl is not None:
        ci = fmt_ci(hl_lo, hl_hi)
        ci_phrase = f", {ci}" if ci else ""
        parts.append(f"The Hodges-Lehmann pseudomedian shift was {fmt_general(hl)}{ci_phrase}.")
    return " ".join(parts)


# ============================================================
# nonparametric_group_comparison: Mann-Whitney OR Kruskal-Wallis (+ Dunn's)
# ============================================================

def write_apa_nonparametric_group_comparison(run: Dict[str, Any]) -> Optional[str]:
    metrics = run.get("metrics", {}) or {}
    method = (metrics.get("method") or "").lower()
    p = metrics.get("p_value")
    alpha = metrics.get("alpha") or _alpha_of(run)
    target = _arg(run, "target_col", "the outcome")
    group_col = _arg(run, "group_col", "group")

    # -------- Mann-Whitney U --------
    if "mann-whitney" in method or "mann whitney" in method:
        u = metrics.get("U_statistic")
        rb = metrics.get("effect_size")  # rank-biserial
        hl = metrics.get("hodges_lehmann_location_shift")
        hl_lo, hl_hi = metrics.get("hodges_lehmann_ci_low"), metrics.get("hodges_lehmann_ci_high")

        parts = [
            f"A Mann-Whitney *U* test was conducted to compare {target} between the two levels of {group_col}."
        ]
        if u is not None and p is not None:
            parts.append(f"The test was *U* = {fmt_general(u)}, {fmt_p(p)}.")
        if rb is not None:
            parts.append(f"The rank-biserial correlation was *r* = {fmt_signed(rb)}.")
        if hl is not None:
            ci = fmt_ci(hl_lo, hl_hi)
            ci_phrase = f", {ci}" if ci else ""
            parts.append(f"The Hodges-Lehmann location shift was {fmt_general(hl)}{ci_phrase}.")
        return " ".join(parts)

    # -------- Kruskal-Wallis H + (if sig) Dunn's --------
    if "kruskal" in method:
        h = metrics.get("H_statistic")
        df = metrics.get("degrees_of_freedom_between")
        eps2 = metrics.get("epsilon_squared")

        parts = [
            f"A Kruskal-Wallis *H* test was conducted to compare {target} across levels of {group_col}."
        ]
        if h is not None and df is not None and p is not None:
            parts.append(f"The test was *H*({fmt_int(df)}) = {fmt_general(h)}, {fmt_p(p)}.")
        if eps2 is not None:
            parts.append(f"ε² = {fmt_bounded_unit(eps2)}.")

        post_hoc = (run.get("tables", {}) or {}).get("post_hoc_pairwise") or []
        if metrics.get("significant_at_alpha") and post_hoc:
            sig_pairs = [r for r in post_hoc if r.get("significant_at_alpha")]
            adj = post_hoc[0].get("adjustment_method") or "Dunn's test"
            parts.append(
                f"Post-hoc pairwise comparisons were conducted using {adj}; "
                f"{len(sig_pairs)} of {len(post_hoc)} contrasts were significant at α = {alpha}."
            )

        return " ".join(parts)

    return None


# ============================================================
# Dispatch lookup
# ============================================================

APA_WRITERS_BY_TOOL_NAME: Dict[str, Any] = {
    "statistical_group_comparison": write_apa_statistical_group_comparison,
    "run_independent_t_test": write_apa_independent_t_test,
    "independent_t_test": write_apa_independent_t_test,
    "paired_comparison": write_apa_paired_comparison,
    "nonparametric_group_comparison": write_apa_nonparametric_group_comparison,
}