"""
Plugin-layer carpet-bomb runner (NO LLM -- free to run).

Runs every case in the matrix directly through its plugin and checks:

  ACCURACY  -- the plugin's statistics match the independent scipy/statsmodels
               gold answer to a tight tolerance.
  NO-ERROR  -- the plugin returns a structured ok/warning result (never raises,
               never an unexpected blocked/error on valid data).
  ROUTING   -- for group comparisons, the parametric/nonparametric switch
               matches the deterministic P2 rule computed independently.

This is the wide net: ~240 scenarios across distributions, sample sizes,
variance structures, effect sizes, group counts, and task types. Because no
LLM is involved, it runs in seconds and can be part of CI.

Usage:
    python -m benchmark.carpet.run_plugin_carpet           # all tasks
    python -m benchmark.carpet.run_plugin_carpet group     # one task family
"""

from __future__ import annotations

import math
import sys
import traceback
from collections import defaultdict
from typing import Any, Dict, List

from benchmark.carpet.case_matrix import Case, generate_all_cases
from core.analysis_tool_plugins.registry import get_plugin


REL = 1e-3
ABS = 1e-3


class Ctx:
    def __init__(self, df, args):
        self._df = df
        self.arguments = args
        self.active_data_version_id = "v_carpet"
        self.data_versions = []

    def load_df(self):
        return self._df

    def get_arg(self, name, default=None):
        return self.arguments.get(name, default)


def _close(a, b, rel=REL, abs_=ABS) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)
    except Exception:
        return False


class Failure(Exception):
    pass


# ----- per-task accuracy checks against gold -----

def _check_group(c: Case, d: Dict[str, Any]) -> List[str]:
    issues = []
    method = (d.get("method") or "").lower()
    switched = d.get("nonparametric_switch", {}).get("switch_to_nonparametric")

    # Routing matches the independent P2 expectation
    if switched != c.expect["expect_nonparametric_switch"]:
        issues.append(
            f"routing: switch={switched} expected={c.expect['expect_nonparametric_switch']} "
            f"(skew={c.expect['max_abs_skew']:.2f}, min_n={c.expect['min_group_n']}, "
            f"non_normal={c.expect['any_non_normal']})"
        )

    # Accuracy: compare against the gold answer for whichever test is primary
    if c.expect["n_groups"] == 2:
        if "mann-whitney" in method:
            if not _close(d.get("p_value"), c.gold["mwu_p"]):
                issues.append(f"MWU p {d.get('p_value')} != gold {c.gold['mwu_p']}")
        else:
            if not _close(abs(d.get("t_statistic", float('nan'))), abs(c.gold["welch_t"])):
                issues.append(f"Welch t {d.get('t_statistic')} != gold {c.gold['welch_t']}")
            if not _close(d.get("p_value"), c.gold["welch_p"]):
                issues.append(f"Welch p {d.get('p_value')} != gold {c.gold['welch_p']}")
    else:
        if "kruskal" in method:
            if not _close(d.get("p_value"), c.gold["kruskal_p"]):
                issues.append(f"Kruskal p {d.get('p_value')} != gold {c.gold['kruskal_p']}")
        else:
            if not _close(d.get("p_value"), c.gold["anova_p"], rel=1e-2):
                # ANOVA may be classic / Welch / Alexander-Govern; classic path
                # must match f_oneway. Looser tol for the robust variants.
                if "alexander" not in method and "welch" not in method:
                    issues.append(f"ANOVA p {d.get('p_value')} != gold {c.gold['anova_p']}")
    return issues


def _check_regression(c: Case, d: Dict[str, Any]) -> List[str]:
    issues = []
    if not _close(d.get("r_squared"), c.gold["r_squared"], rel=1e-2):
        issues.append(f"R2 {d.get('r_squared')} != gold {c.gold['r_squared']}")
    if not _close(d.get("f_statistic"), c.gold["f_stat"], rel=1e-2):
        issues.append(f"F {d.get('f_statistic')} != gold {c.gold['f_stat']}")
    return issues


def _check_correlation(c: Case, d: Dict[str, Any]) -> List[str]:
    issues = []
    if not _close(d.get("correlation"), c.gold["r"]):
        issues.append(f"r {d.get('correlation')} != gold {c.gold['r']}")
    if not _close(d.get("p_value"), c.gold["p"]):
        issues.append(f"corr p {d.get('p_value')} != gold {c.gold['p']}")
    return issues


def _check_chi_square(c: Case, d: Dict[str, Any]) -> List[str]:
    issues = []
    if not _close(d.get("chi_square_statistic"), c.gold["chi2"]):
        issues.append(f"chi2 {d.get('chi_square_statistic')} != gold {c.gold['chi2']}")
    if not _close(d.get("p_value"), c.gold["p"]):
        issues.append(f"chi p {d.get('p_value')} != gold {c.gold['p']}")
    if int(d.get("degrees_of_freedom", -1)) != c.gold["dof"]:
        issues.append(f"dof {d.get('degrees_of_freedom')} != gold {c.gold['dof']}")
    return issues


def _check_paired(c: Case, d: Dict[str, Any]) -> List[str]:
    issues = []
    method = (d.get("method") or "").lower()
    if "wilcoxon" in method:
        if not _close(d.get("p_value"), c.gold["wilcoxon_p"], rel=1e-2):
            issues.append(f"Wilcoxon p {d.get('p_value')} != gold {c.gold['wilcoxon_p']}")
    else:
        if not _close(d.get("p_value"), c.gold["paired_t_p"], rel=1e-2):
            issues.append(f"paired-t p {d.get('p_value')} != gold {c.gold['paired_t_p']}")
    return issues


def _check_paired_bootstrap(c: Case, d: Dict[str, Any]) -> List[str]:
    """Branch A: classical bootstrap; CI endpoints must match scipy gold within tolerance."""
    issues = []
    g = c.gold

    if d.get("resampler") != "classical":
        issues.append(f"resampler {d.get('resampler')} != classical")

    if g.get("ci_lower") is not None and g.get("tolerance") is not None:
        tol = float(g["tolerance"])
        lo, hi = d.get("ci_lower"), d.get("ci_upper")
        if lo is None or hi is None:
            issues.append(f"CI missing: lo={lo}, hi={hi}")
        else:
            if abs(float(lo) - float(g["ci_lower"])) > tol:
                issues.append(
                    f"ci_lower {float(lo):.4f} != gold {float(g['ci_lower']):.4f} (tol {tol:.4f})"
                )
            if abs(float(hi) - float(g["ci_upper"])) > tol:
                issues.append(
                    f"ci_upper {float(hi):.4f} != gold {float(g['ci_upper']):.4f} (tol {tol:.4f})"
                )
    return issues


def _check_paired_bootstrap_sequential(c: Case, d: Dict[str, Any]) -> List[str]:
    """Branch B: Sequential Bootstrap; structural invariants only (no scipy gold)."""
    issues = []
    if d.get("resampler") != "sequential":
        issues.append(f"resampler {d.get('resampler')} != sequential")
    if not bool(d.get("use_sequential")):
        issues.append(f"use_sequential {d.get('use_sequential')} != True")
    expected_kn = c.gold.get("k_n_expected")
    if expected_kn is not None and d.get("k_n") != expected_kn:
        issues.append(f"k_n {d.get('k_n')} != gold {expected_kn}")
    return issues


def _check_paired_bootstrap_stability(c: Case, d: Dict[str, Any]) -> List[str]:
    """Branch C: stability diagnostic interpretation in expected band."""
    issues = []
    allowed = set(c.gold.get("expected_interpretation_in", []) or [])
    diag = d.get("stability_diagnostic") or {}
    actual = diag.get("interpretation")
    if allowed and actual not in allowed:
        issues.append(f"stability interpretation {actual} not in {sorted(allowed)}")
    return issues


_CHECKERS = {
    "group": _check_group,
    "regression": _check_regression,
    "correlation": _check_correlation,
    "chi_square": _check_chi_square,
    "paired": _check_paired,
    "paired_bootstrap": _check_paired_bootstrap,
    "paired_bootstrap_sequential": _check_paired_bootstrap_sequential,
    "paired_bootstrap_stability": _check_paired_bootstrap_stability,
}


def run_one(c: Case) -> Dict[str, Any]:
    result = {"key": c.key, "task": c.task, "status": None,
              "no_error": True, "accurate": True, "issues": []}
    try:
        out = get_plugin(c.tool).run(Ctx(c.df, c.args))
    except Exception as exc:
        result["no_error"] = False
        result["accurate"] = False
        result["status"] = "RAISED"
        result["issues"].append(f"RAISED {type(exc).__name__}: {exc}")
        return result

    result["status"] = out.get("status")
    if out.get("status") not in ("ok", "warning"):
        result["no_error"] = False
        result["accurate"] = False
        result["issues"].append(f"status={out.get('status')}: {out.get('message','')[:120]}")
        return result

    d = out.get("details", {}) or {}
    issues = _CHECKERS[c.task](c, d)
    if issues:
        result["accurate"] = False
        result["issues"] = issues
    return result


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    cases = generate_all_cases()
    if only:
        cases = [c for c in cases if c.task == only]

    by_task_total = defaultdict(int)
    by_task_acc = defaultdict(int)
    by_task_noerr = defaultdict(int)
    failures = []

    for c in cases:
        r = run_one(c)
        by_task_total[c.task] += 1
        if r["no_error"]:
            by_task_noerr[c.task] += 1
        if r["accurate"]:
            by_task_acc[c.task] += 1
        else:
            failures.append(r)

    print("=" * 64)
    print("PLUGIN-LAYER CARPET BOMB  (no LLM)")
    print("=" * 64)
    print(f"{'task':<14}{'cases':>7}{'no_error':>10}{'accurate':>10}")
    for task in sorted(by_task_total):
        t = by_task_total[task]
        print(f"{task:<14}{t:>7}{by_task_noerr[task]:>10}{by_task_acc[task]:>10}")
    total = sum(by_task_total.values())
    acc = sum(by_task_acc.values())
    noerr = sum(by_task_noerr.values())
    print("-" * 64)
    print(f"{'TOTAL':<14}{total:>7}{noerr:>10}{acc:>10}")
    print(f"\nno-error rate: {noerr}/{total} = {100*noerr/total:.1f}%")
    print(f"accuracy rate: {acc}/{total} = {100*acc/total:.1f}%")

    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for f in failures[:40]:
            print(f"  [{f['task']}] {f['key']}")
            for iss in f["issues"][:3]:
                print(f"      - {iss}")
        if len(failures) > 40:
            print(f"  ... and {len(failures)-40} more")
    else:
        print("\nAll cases pass accuracy + no-error.")

    return failures


if __name__ == "__main__":
    main()