from __future__ import annotations

from typing import Any, Dict, List, Tuple, Callable
import math

import numpy as np
from scipy import stats
from scipy import optimize

from core.analysis_tool_plugins.base import (
    AnalysisToolPlugin,
    ArgumentSchema,
    DisplayConfig,
    MetricDisplayConfig,
    TableDisplayConfig,
    compact_dict,
    format_bool_yes_no,
    format_number,
    format_p_value,
)
from core.analysis_tool_plugins.registry import register_plugin
from core.guardrails import _new_finding


# ==========================================================
# Output helpers (no data-loading needed for power analysis)
# ==========================================================

def _ok(message: str, details: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": "ok",
        "message": message,
        "recoverable": False,
        "details": details or {},
        "artifacts": [],
    }


def _warning(message: str, details: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": "warning",
        "message": message,
        "recoverable": False,
        "details": details or {},
        "artifacts": [],
    }


def _blocked(error_code: str, message: str, details=None, suggested_next_actions=None):
    result = {
        "status": "blocked",
        "error_code": error_code,
        "message": message,
        "recoverable": True,
        "details": details or {},
        "artifacts": [],
    }

    if suggested_next_actions:
        result["suggested_next_actions"] = suggested_next_actions

    return result


def _failed(error_code: str, message: str, exc: Exception) -> Dict[str, Any]:
    return {
        "status": "failed",
        "error_code": error_code,
        "message": message,
        "recoverable": True,
        "details": {
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        },
        "artifacts": [],
    }


def _round_or_none(x: Any, digits: int = 6):
    try:
        v = float(x)
        if not math.isfinite(v):
            return None
        return round(v, digits)
    except Exception:
        return None


def _get_arguments(context) -> dict[str, Any]:
    return getattr(context, "arguments", None) or getattr(context, "args", None) or {}


# ==========================================================
# Effect-size magnitude (Cohen 1988 conventions per test family)
# ==========================================================

def _interpret_cohens_d(d: float | None) -> str | None:
    if d is None:
        return None
    a = abs(float(d))
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"


def _interpret_cohens_f(f: float | None) -> str | None:
    if f is None:
        return None
    a = abs(float(f))
    if a < 0.10:
        return "negligible"
    if a < 0.25:
        return "small"
    if a < 0.40:
        return "medium"
    return "large"


def _interpret_cohens_f2(f2: float | None) -> str | None:
    if f2 is None:
        return None
    a = abs(float(f2))
    if a < 0.02:
        return "negligible"
    if a < 0.15:
        return "small"
    if a < 0.35:
        return "medium"
    return "large"


def _interpret_pearson_r(r: float | None) -> str | None:
    if r is None:
        return None
    a = abs(float(r))
    if a < 0.10:
        return "negligible"
    if a < 0.30:
        return "small"
    if a < 0.50:
        return "moderate"
    if a < 0.70:
        return "large"
    return "very large"


def _interpret_cohens_w(w: float | None) -> str | None:
    if w is None:
        return None
    a = abs(float(w))
    if a < 0.10:
        return "negligible"
    if a < 0.30:
        return "small"
    if a < 0.50:
        return "medium"
    return "large"


# Cohen 1988 conventions catalogue. Surfaced to the user with each result
# so the magnitude interpretation is transparent.
_COHEN_CONVENTIONS: Dict[str, Dict[str, Any]] = {
    "two_sample_t": {
        "effect_size_name": "Cohen's d",
        "small": 0.2,
        "medium": 0.5,
        "large": 0.8,
        "magnitude_fn": _interpret_cohens_d,
    },
    "one_way_anova": {
        "effect_size_name": "Cohen's f",
        "small": 0.10,
        "medium": 0.25,
        "large": 0.40,
        "magnitude_fn": _interpret_cohens_f,
    },
    "linear_regression": {
        "effect_size_name": "Cohen's f^2",
        "small": 0.02,
        "medium": 0.15,
        "large": 0.35,
        "magnitude_fn": _interpret_cohens_f2,
    },
    "correlation": {
        "effect_size_name": "Pearson r",
        "small": 0.10,
        "medium": 0.30,
        "large": 0.50,
        "magnitude_fn": _interpret_pearson_r,
    },
    "chi_square": {
        "effect_size_name": "Cohen's w",
        "small": 0.10,
        "medium": 0.30,
        "large": 0.50,
        "magnitude_fn": _interpret_cohens_w,
    },
}


# ==========================================================
# Power functions per test family
# ==========================================================
#
# Each function returns power in [0, 1]. Each accepts a one-sided/two-sided
# selector when applicable.
#
# Sample-size semantics:
#   - two_sample_t:    n is per-group (equal-n)
#   - one_way_anova:   n is per-group (equal-n), n_groups is k
#   - linear_regression: n is total sample size, n_predictors is p
#   - correlation:     n is total sample size
#   - chi_square:      n is total sample size, df is the chi-square degrees
#                      of freedom (e.g. (r-1)(c-1) for independence)
# ==========================================================


def _power_two_sample_t(
    d: float,
    n_per_group: float,
    alpha: float,
    alternative: str = "two-sided",
) -> float:
    """Independent two-sample t-test, equal n per group."""
    if n_per_group < 2 or d == 0:
        # With zero effect, power equals alpha for two-sided, alpha for
        # one-sided.
        if d == 0:
            return alpha
        return 0.0

    df = 2.0 * (n_per_group - 1.0)
    ncp = d * math.sqrt(n_per_group / 2.0)

    if alternative == "one-sided":
        t_crit = float(stats.t.ppf(1.0 - alpha, df))
        # Power = P(T > t_crit | ncp)  (upper tail; assumes d > 0 direction)
        power = float(stats.nct.sf(t_crit, df, ncp))
    else:
        # Two-sided: P(|T| > t_crit | ncp)
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
        power = (
            float(stats.nct.sf(t_crit, df, ncp))
            + float(stats.nct.cdf(-t_crit, df, ncp))
        )

    return max(0.0, min(1.0, power))


def _power_one_way_anova(
    f: float,
    n_per_group: float,
    n_groups: int,
    alpha: float,
) -> float:
    """Fisher one-way ANOVA, equal n per group."""
    if n_per_group < 2 or n_groups < 2:
        return 0.0

    total_n = float(n_per_group * n_groups)
    df1 = float(n_groups - 1)
    df2 = float(total_n - n_groups)

    if df2 <= 0:
        return 0.0

    ncp = (f ** 2) * total_n

    f_crit = float(stats.f.ppf(1.0 - alpha, df1, df2))
    power = float(stats.ncf.sf(f_crit, df1, df2, ncp))

    return max(0.0, min(1.0, power))


def _power_linear_regression(
    f2: float,
    n_total: float,
    n_predictors: int,
    alpha: float,
) -> float:
    """OLS multiple linear regression - overall F test for R^2."""
    if n_predictors < 1 or n_total <= n_predictors + 1:
        return 0.0

    df1 = float(n_predictors)
    df2 = float(n_total - n_predictors - 1)

    if df2 <= 0:
        return 0.0

    # Cohen / G*Power convention: ncp = f^2 * (u + v + 1) = f^2 * n
    ncp = float(f2) * float(n_total)

    f_crit = float(stats.f.ppf(1.0 - alpha, df1, df2))
    power = float(stats.ncf.sf(f_crit, df1, df2, ncp))

    return max(0.0, min(1.0, power))


def _power_correlation(
    r: float,
    n_total: float,
    alpha: float,
    alternative: str = "two-sided",
) -> float:
    """Pearson correlation, H0: rho = 0. Fisher z-transform approximation."""
    if n_total < 4 or abs(r) >= 1.0:
        if abs(r) >= 1.0 and n_total >= 4:
            return 1.0
        return 0.0

    z_r = math.atanh(float(r))
    shift = z_r * math.sqrt(n_total - 3)

    if alternative == "one-sided":
        z_crit = float(stats.norm.ppf(1.0 - alpha))
        power = float(stats.norm.sf(z_crit - shift))
    else:
        z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
        power = (
            float(stats.norm.sf(z_crit - shift))
            + float(stats.norm.cdf(-z_crit - shift))
        )

    return max(0.0, min(1.0, power))


def _power_chi_square(
    w: float,
    n_total: float,
    df: int,
    alpha: float,
) -> float:
    """Chi-square test (independence or goodness of fit). df supplied by caller."""
    if n_total < 2 or df < 1:
        return 0.0

    ncp = (float(w) ** 2) * float(n_total)
    chi2_crit = float(stats.chi2.ppf(1.0 - alpha, df))
    power = float(stats.ncx2.sf(chi2_crit, df, ncp))

    return max(0.0, min(1.0, power))


# ==========================================================
# Generic solver - inverts any of the power functions
# ==========================================================

def _solve_monotone(
    target: float,
    fn: Callable[[float], float],
    low: float,
    high: float,
    *,
    max_high_doubles: int = 25,
    xtol: float = 1e-6,
) -> float | None:
    """
    Solve fn(x) = target on [low, high] assuming fn is monotone-nondecreasing.

    If fn(high) < target, doubles `high` up to `max_high_doubles` times before
    giving up. Returns None if the target cannot be bracketed.
    """
    try:
        f_low = fn(low)
        f_high = fn(high)
    except Exception:
        return None

    if not (math.isfinite(f_low) and math.isfinite(f_high)):
        return None

    if f_low >= target:
        return low

    doubles = 0
    while f_high < target and doubles < max_high_doubles:
        high = high * 2.0
        try:
            f_high = fn(high)
        except Exception:
            return None
        doubles += 1

    if f_high < target:
        return None

    try:
        root = optimize.brentq(
            lambda x: fn(x) - target,
            low,
            high,
            xtol=xtol,
            maxiter=300,
        )
        return float(root)
    except Exception:
        return None


# ==========================================================
# Dispatch helpers
# ==========================================================

_TEST_TYPES = {
    "two_sample_t", "one_way_anova", "linear_regression",
    "correlation", "chi_square",
}

_MODES = {"sample_size", "power", "effect_size"}


def _validate_alpha(alpha: Any, default: float = 0.05) -> float:
    try:
        a = float(alpha)
    except Exception:
        return default
    if not (0 < a < 1):
        return default
    return a


def _validate_power(p: Any, default: float = 0.80) -> float:
    try:
        v = float(p)
    except Exception:
        return default
    if not (0 < v < 1):
        return default
    return v


def _power_curve(
    fn_of_n: Callable[[float], float],
    n_center: float,
    multipliers: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
    integer_per_group: bool = False,
) -> list[dict[str, Any]]:
    """
    Build a small table of (n, power) points around n_center. Useful in
    sample-size mode for showing the curve a researcher would see in G*Power.
    """
    rows: list[dict[str, Any]] = []
    seen = set()

    for m in multipliers:
        n_eval = n_center * m

        if integer_per_group:
            n_eval = max(2, int(math.ceil(n_eval)))
        else:
            n_eval = max(2.0, n_eval)

        key = (round(float(n_eval), 6),)
        if key in seen:
            continue
        seen.add(key)

        try:
            power_at_n = float(fn_of_n(n_eval))
        except Exception:
            power_at_n = float("nan")

        rows.append({
            "n": int(n_eval) if integer_per_group else _round_or_none(n_eval, 2),
            "multiplier_of_target": _round_or_none(m, 3),
            "power_at_n": _round_or_none(power_at_n),
        })

    return rows


# ==========================================================
# Per-test wrappers that return one power value plus a curve fn
# ==========================================================

def _build_power_callable(
    test_type: str,
    arguments: Dict[str, Any],
    alpha: float,
) -> Tuple[Callable[..., float], Dict[str, Any]]:
    """
    Returns a callable(effect, n) -> power and a dict of test-specific
    static parameters used by the test. The caller passes effect / n into
    the callable for forward power, and into solvers for inverse modes.
    """
    if test_type == "two_sample_t":
        alternative = str(arguments.get("alternative", "two-sided")).lower().strip()
        if alternative not in {"two-sided", "one-sided"}:
            alternative = "two-sided"

        params = {"alternative": alternative}

        def power_fn(d: float, n_per_group: float) -> float:
            return _power_two_sample_t(d, n_per_group, alpha, alternative)

        return power_fn, params

    if test_type == "one_way_anova":
        try:
            n_groups = int(arguments.get("n_groups", 0))
        except Exception:
            n_groups = 0
        if n_groups < 2:
            raise ValueError("n_groups must be at least 2 for one_way_anova.")

        params = {"n_groups": n_groups}

        def power_fn(f: float, n_per_group: float) -> float:
            return _power_one_way_anova(f, n_per_group, n_groups, alpha)

        return power_fn, params

    if test_type == "linear_regression":
        try:
            n_predictors = int(arguments.get("n_predictors", 0))
        except Exception:
            n_predictors = 0
        if n_predictors < 1:
            raise ValueError("n_predictors must be at least 1 for linear_regression.")

        params = {"n_predictors": n_predictors}

        def power_fn(f2: float, n_total: float) -> float:
            return _power_linear_regression(f2, n_total, n_predictors, alpha)

        return power_fn, params

    if test_type == "correlation":
        alternative = str(arguments.get("alternative", "two-sided")).lower().strip()
        if alternative not in {"two-sided", "one-sided"}:
            alternative = "two-sided"

        params = {"alternative": alternative}

        def power_fn(r: float, n_total: float) -> float:
            return _power_correlation(r, n_total, alpha, alternative)

        return power_fn, params

    if test_type == "chi_square":
        try:
            df = int(arguments.get("df", 0))
        except Exception:
            df = 0
        if df < 1:
            raise ValueError(
                "df must be at least 1 for chi_square. "
                "Use (rows-1)*(cols-1) for independence, or (k-1) for goodness of fit."
            )

        params = {"df": df}

        def power_fn(w: float, n_total: float) -> float:
            return _power_chi_square(w, n_total, df, alpha)

        return power_fn, params

    raise ValueError(f"Unknown test_type: {test_type}")


def _n_semantics(test_type: str) -> str:
    if test_type in {"two_sample_t", "one_way_anova"}:
        return "per-group"
    return "total"


def _integer_n(test_type: str) -> bool:
    return True  # all sample sizes are integers in practice


def _effect_search_upper_bound(test_type: str) -> float:
    """Search upper bound when solving for the minimum detectable effect."""
    if test_type == "correlation":
        return 0.999
    if test_type == "two_sample_t":
        return 5.0
    if test_type == "one_way_anova":
        return 3.0
    if test_type == "linear_regression":
        return 5.0  # Cohen's f^2; very large
    if test_type == "chi_square":
        return 3.0
    return 5.0


# ==========================================================
# Main execute
# ==========================================================

def execute_power_analysis(context) -> Dict[str, Any]:
    arguments = _get_arguments(context)

    test_type = str(arguments.get("test_type", "")).lower().strip()
    mode = str(arguments.get("mode", "")).lower().strip()

    alpha = _validate_alpha(arguments.get("alpha"), default=0.05)
    target_power = _validate_power(arguments.get("power"), default=0.80)

    if test_type not in _TEST_TYPES:
        return _blocked(
            "UNSUPPORTED_TEST_TYPE",
            f"test_type must be one of {sorted(_TEST_TYPES)}",
            details={"test_type": test_type},
        )

    if mode not in _MODES:
        return _blocked(
            "UNSUPPORTED_MODE",
            f"mode must be one of {sorted(_MODES)}",
            details={"mode": mode},
        )

    # Build the power callable plus static params for this test family
    try:
        power_fn, test_params = _build_power_callable(test_type, arguments, alpha)
    except ValueError as e:
        return _blocked(
            "MISSING_TEST_PARAMETERS",
            str(e),
            details={"test_type": test_type, "received_arguments": arguments},
        )

    convention = _COHEN_CONVENTIONS[test_type]
    n_semantics = _n_semantics(test_type)
    integer_per_group = _integer_n(test_type)

    # Effect-size parameter name in the user's arguments
    effect_arg_aliases = (
        "effect_size", "d", "f", "f2", "f_squared", "r", "w",
    )
    effect_provided: float | None = None
    for k in effect_arg_aliases:
        if k in arguments and arguments[k] is not None:
            try:
                effect_provided = float(arguments[k])
                break
            except Exception:
                pass

    # n parameter
    n_provided: float | None = None
    if "n" in arguments and arguments["n"] is not None:
        try:
            n_provided = float(arguments["n"])
        except Exception:
            pass

    try:
        if mode == "power":
            # Post-hoc power: need effect_size and n
            if effect_provided is None:
                return _blocked(
                    "MISSING_EFFECT_SIZE",
                    f"mode=power requires an effect size ({convention['effect_size_name']}).",
                    details={"test_type": test_type},
                )
            if n_provided is None:
                return _blocked(
                    "MISSING_SAMPLE_SIZE",
                    f"mode=power requires `n` ({n_semantics} for this test).",
                    details={"test_type": test_type},
                )

            achieved_power = float(power_fn(effect_provided, n_provided))

            # Build a curve around the provided n for context
            curve = _power_curve(
                lambda nn: power_fn(effect_provided, nn),
                n_provided,
                integer_per_group=integer_per_group,
            )

            details = _build_result_details(
                test_type=test_type,
                mode=mode,
                alpha=alpha,
                target_power=target_power,
                achieved_power=achieved_power,
                effect_size=effect_provided,
                n_value=n_provided,
                test_params=test_params,
                convention=convention,
                n_semantics=n_semantics,
                power_curve=curve,
                power_curve_axis="n",
            )

            status = "ok"
            message = (
                f"Achieved power computed for {test_type}: "
                f"power = {round(achieved_power, 4)} at "
                f"{convention['effect_size_name']} = {effect_provided} and n = {n_provided} "
                f"({n_semantics})."
            )

            if achieved_power < target_power:
                status = "warning"
                message += (
                    f" Achieved power is below the target ({target_power}); "
                    f"the study appears underpowered."
                )

            return (_warning if status == "warning" else _ok)(message, details)

        if mode == "sample_size":
            # A priori: need effect_size and target power
            if effect_provided is None:
                return _blocked(
                    "MISSING_EFFECT_SIZE",
                    f"mode=sample_size requires an effect size ({convention['effect_size_name']}).",
                    details={"test_type": test_type},
                )

            # Solve power_fn(effect, n) = target_power for n
            def power_at_n(nn: float) -> float:
                return power_fn(effect_provided, nn)

            n_solution = _solve_monotone(
                target_power,
                power_at_n,
                low=2.0,
                high=200.0,
            )

            if n_solution is None:
                return _blocked(
                    "SAMPLE_SIZE_NOT_FOUND",
                    "Could not find a sample size that meets the target power within reasonable bounds.",
                    details={
                        "test_type": test_type,
                        "effect_size": effect_provided,
                        "target_power": target_power,
                        "max_attempted_n": 200.0 * (2 ** 25),
                    },
                    suggested_next_actions=[
                        "Check that the effect size and target power are realistic for this test.",
                    ],
                )

            n_required = int(math.ceil(n_solution))
            achieved_power = float(power_fn(effect_provided, n_required))

            curve = _power_curve(
                power_at_n,
                n_required,
                integer_per_group=integer_per_group,
            )

            details = _build_result_details(
                test_type=test_type,
                mode=mode,
                alpha=alpha,
                target_power=target_power,
                achieved_power=achieved_power,
                effect_size=effect_provided,
                n_value=n_required,
                n_value_continuous=n_solution,
                test_params=test_params,
                convention=convention,
                n_semantics=n_semantics,
                power_curve=curve,
                power_curve_axis="n",
            )

            total_n_note = ""
            if n_semantics == "per-group" and "n_groups" in test_params:
                total = n_required * int(test_params["n_groups"])
                details["total_sample_size"] = total
                total_n_note = f" (total N = {total} across {test_params['n_groups']} groups)"
            elif n_semantics == "per-group":
                # two_sample_t -> 2 groups
                details["total_sample_size"] = n_required * 2
                total_n_note = f" (total N = {n_required * 2} across 2 groups)"

            message = (
                f"Required sample size: {n_required} {n_semantics}{total_n_note} "
                f"to achieve power = {target_power} at "
                f"{convention['effect_size_name']} = {effect_provided}, alpha = {alpha}."
            )

            return _ok(message, details)

        if mode == "effect_size":
            # MDE: need n and target power
            if n_provided is None:
                return _blocked(
                    "MISSING_SAMPLE_SIZE",
                    f"mode=effect_size requires `n` ({n_semantics} for this test).",
                    details={"test_type": test_type},
                )

            upper = _effect_search_upper_bound(test_type)

            def power_at_effect(eff: float) -> float:
                return power_fn(eff, n_provided)

            effect_solution = _solve_monotone(
                target_power,
                power_at_effect,
                low=1e-4,
                high=upper,
                max_high_doubles=3,
            )

            if effect_solution is None:
                return _blocked(
                    "EFFECT_SIZE_NOT_FOUND",
                    "Could not find an effect size that meets the target power at the supplied n.",
                    details={
                        "test_type": test_type,
                        "n": n_provided,
                        "target_power": target_power,
                        "search_upper_bound": upper,
                    },
                    suggested_next_actions=[
                        "Increase the sample size, or lower the target power.",
                    ],
                )

            achieved_power = float(power_fn(effect_solution, n_provided))

            # Curve over a small effect-size grid around the solution
            grid = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
            curve_rows: list[dict[str, Any]] = []
            for m in grid:
                eff_eval = effect_solution * m
                try:
                    p_eval = float(power_fn(eff_eval, n_provided))
                except Exception:
                    p_eval = float("nan")
                curve_rows.append({
                    "effect_size": _round_or_none(eff_eval),
                    "multiplier_of_target": _round_or_none(m, 3),
                    "power_at_effect": _round_or_none(p_eval),
                })

            details = _build_result_details(
                test_type=test_type,
                mode=mode,
                alpha=alpha,
                target_power=target_power,
                achieved_power=achieved_power,
                effect_size=effect_solution,
                n_value=n_provided,
                test_params=test_params,
                convention=convention,
                n_semantics=n_semantics,
                power_curve=curve_rows,
                power_curve_axis="effect_size",
            )

            message = (
                f"Minimum detectable {convention['effect_size_name']}: "
                f"{round(effect_solution, 4)} at n = {n_provided} ({n_semantics}), "
                f"power = {target_power}, alpha = {alpha}."
            )

            return _ok(message, details)

        return _blocked("UNREACHABLE_MODE", f"Unhandled mode: {mode}")

    except Exception as exc:
        return _failed(
            "POWER_ANALYSIS_EXCEPTION",
            "power_analysis failed.",
            exc,
        )


def _build_result_details(
    *,
    test_type: str,
    mode: str,
    alpha: float,
    target_power: float,
    achieved_power: float,
    effect_size: float,
    n_value: float,
    test_params: Dict[str, Any],
    convention: Dict[str, Any],
    n_semantics: str,
    power_curve: list[dict[str, Any]],
    power_curve_axis: str,
    n_value_continuous: float | None = None,
) -> Dict[str, Any]:
    magnitude_fn = convention.get("magnitude_fn")
    magnitude = magnitude_fn(effect_size) if magnitude_fn else None

    assumptions = [
        f"Power calculation uses Cohen's standardized effect size ({convention['effect_size_name']}); "
        "results assume the analytic model in Cohen (1988).",
        f"Test family: {test_type}; sample-size argument is interpreted as {n_semantics}.",
        f"Type I error rate alpha = {alpha}; target power = {target_power}.",
    ]

    if test_type in {"two_sample_t", "one_way_anova"}:
        assumptions.append(
            "Equal sample size across groups is assumed. Unequal-n designs require "
            "harmonic-mean corrections not currently implemented."
        )

    if test_type == "one_way_anova":
        assumptions.append(
            "Classical (equal-variance) ANOVA. For unequal variances, the "
            "power of Welch's ANOVA differs and is not modeled here."
        )

    if test_type == "correlation":
        assumptions.append(
            "Fisher z-transform approximation for the Pearson correlation. Adequate when n >= 10."
        )

    if test_type == "chi_square":
        assumptions.append(
            "df must be supplied by the caller: (rows-1)*(cols-1) for independence, or (k-1) for goodness of fit."
        )

    details: Dict[str, Any] = {
        "test_type": test_type,
        "mode": mode,
        "alpha": alpha,
        "target_power": target_power,
        "achieved_power": _round_or_none(achieved_power),
        "effect_size_name": convention["effect_size_name"],
        "effect_size": _round_or_none(effect_size),
        "effect_size_magnitude": magnitude,
        "cohens_small": convention["small"],
        "cohens_medium": convention["medium"],
        "cohens_large": convention["large"],
        "n_value": int(n_value) if float(n_value).is_integer() else _round_or_none(n_value, 4),
        "n_semantics": n_semantics,
        "test_parameters": test_params,
        "power_curve": power_curve,
        "power_curve_axis": power_curve_axis,
        "assumptions_and_limitations": assumptions,
    }

    if n_value_continuous is not None:
        details["n_value_continuous"] = _round_or_none(n_value_continuous, 4)

    return details


# ==========================================================
# Plugin-specific guardrail (inline)
# ==========================================================

def evaluate_power_analysis_guardrails(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []

    metrics = run.get("metrics", {}) or {}
    metadata = run.get("metadata", {}) or {}

    test_type = metadata.get("test_type") or metrics.get("test_type")
    mode = metadata.get("mode") or metrics.get("mode")
    achieved_power = metrics.get("achieved_power")
    target_power = metrics.get("target_power")
    effect_size = metrics.get("effect_size")
    effect_size_name = metrics.get("effect_size_name")
    effect_size_magnitude = metrics.get("effect_size_magnitude")
    n_value = metrics.get("n_value")
    n_semantics = metrics.get("n_semantics")
    total_n = metrics.get("total_sample_size")

    # 1) Post-hoc power < target
    try:
        if (
            mode == "power"
            and achieved_power is not None
            and target_power is not None
            and float(achieved_power) < float(target_power)
        ):
            findings.append(_new_finding(
                category="study_design",
                severity="warning",
                title=f"Underpowered: power = {round(float(achieved_power), 3)} < target {target_power}",
                message=(
                    "The achieved power is below the target. A non-significant result in such a "
                    "study cannot be interpreted as evidence of absence; a significant result "
                    "may still be unreliable (winner's curse / inflated effect sizes)."
                ),
                evidence={
                    "achieved_power": achieved_power,
                    "target_power": target_power,
                    "effect_size_name": effect_size_name,
                    "effect_size": effect_size,
                    "n_value": n_value,
                    "n_semantics": n_semantics,
                },
                recommendation=(
                    "Consider increasing the sample size, focusing on a larger effect of "
                    "practical interest, or framing the study as exploratory rather than confirmatory."
                ),
            ))
    except Exception:
        pass

    # 2) A priori sample-size feasibility
    try:
        if mode == "sample_size":
            if n_value is not None and int(n_value) > 1000:
                findings.append(_new_finding(
                    category="study_design",
                    severity="warning",
                    title=f"Very large sample size required (n = {n_value} {n_semantics})",
                    message=(
                        "The required sample size is large. This usually indicates a small "
                        "expected effect, a stringent target power, or both. Confirm that the "
                        "effect size used is realistic for the question."
                    ),
                    evidence={
                        "n_value": n_value,
                        "total_sample_size": total_n,
                        "effect_size": effect_size,
                        "target_power": target_power,
                    },
                    recommendation=(
                        "Reconsider whether the effect size is plausible. For exploratory work "
                        "with limited resources, raise the minimum effect of interest; for "
                        "confirmatory work, accept the cost or split into stages."
                    ),
                ))

            # Tiny effect-size magnitude flag
            if effect_size_magnitude in {"negligible"}:
                findings.append(_new_finding(
                    category="effect_size",
                    severity="info",
                    title=f"Effect size used is negligible by Cohen's conventions",
                    message=(
                        f"The {effect_size_name} value used ({effect_size}) is below the small-effect "
                        "threshold. Power calculations for tiny effects often require sample sizes "
                        "that are scientifically uninteresting."
                    ),
                    evidence={
                        "effect_size": effect_size,
                        "effect_size_name": effect_size_name,
                        "effect_size_magnitude": effect_size_magnitude,
                    },
                ))
    except Exception:
        pass

    # 3) MDE interpretation
    try:
        if mode == "effect_size" and effect_size_magnitude in {"large", "very large"}:
            findings.append(_new_finding(
                category="study_design",
                severity="warning",
                title="Minimum detectable effect is large",
                message=(
                    f"At the supplied sample size, only a {effect_size_magnitude} "
                    f"{effect_size_name} (>= {effect_size}) is detectable with the target power. "
                    "Smaller real effects would likely be missed."
                ),
                evidence={
                    "effect_size": effect_size,
                    "effect_size_name": effect_size_name,
                    "effect_size_magnitude": effect_size_magnitude,
                    "n_value": n_value,
                    "n_semantics": n_semantics,
                },
                recommendation=(
                    "If a small effect would be scientifically meaningful, increase the sample size."
                ),
            ))
    except Exception:
        pass

    # 4) Post-hoc power 反对意见（标准学术 warning）
    if mode == "power":
        findings.append(_new_finding(
            category="study_design",
            severity="info",
            title="Post-hoc power computed; consider its limitations",
            message=(
                "Post-hoc power computed from observed data is informative for design review, "
                "but it should not be used to justify or defend a non-significant finding. "
                "Achieved power is a monotone transformation of the observed p-value and adds "
                "no independent evidence once the p-value is reported."
            ),
            evidence={
                "achieved_power": achieved_power,
                "test_type": test_type,
            },
            recommendation=(
                "Use a priori power analysis (mode = sample_size or effect_size) for study planning."
            ),
        ))

    return findings


# ==========================================================
# Extractor / Display
# ==========================================================

def extract_power_analysis(
    *,
    payload: Dict[str, Any],
    arguments: Dict[str, Any],
    default_title: str,
    default_summary: str,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    test_type = payload.get("test_type") or arguments.get("test_type")
    mode = payload.get("mode") or arguments.get("mode")

    title_map = {
        "sample_size": "Power Analysis: Required Sample Size",
        "power": "Power Analysis: Achieved Power",
        "effect_size": "Power Analysis: Minimum Detectable Effect",
    }
    title = title_map.get(mode, "Power Analysis")

    if test_type:
        title = f"{title} ({test_type})"

    summary_pieces = [
        f"Test: {test_type}.",
        f"Mode: {mode}.",
    ]

    effect_size = payload.get("effect_size")
    effect_name = payload.get("effect_size_name")
    n_value = payload.get("n_value")
    n_semantics = payload.get("n_semantics")
    achieved_power = payload.get("achieved_power")
    target_power = payload.get("target_power")
    total_n = payload.get("total_sample_size")

    if mode == "sample_size":
        summary_pieces.append(
            f"Required n = {n_value} ({n_semantics}) to achieve power {target_power} "
            f"at {effect_name} = {effect_size}."
        )
        if total_n is not None:
            summary_pieces.append(f"Total sample size N = {total_n}.")
    elif mode == "power":
        summary_pieces.append(
            f"Achieved power = {achieved_power} at {effect_name} = {effect_size} and n = {n_value} ({n_semantics})."
        )
    elif mode == "effect_size":
        summary_pieces.append(
            f"Minimum detectable {effect_name} = {effect_size} at n = {n_value} ({n_semantics}) "
            f"for target power {target_power}."
        )

    summary = " ".join(summary_pieces)

    metrics = compact_dict({
        "test_type": test_type,
        "mode": mode,
        "alpha": payload.get("alpha"),
        "target_power": target_power,
        "achieved_power": achieved_power,
        "effect_size_name": effect_name,
        "effect_size": effect_size,
        "effect_size_magnitude": payload.get("effect_size_magnitude"),
        "cohens_small": payload.get("cohens_small"),
        "cohens_medium": payload.get("cohens_medium"),
        "cohens_large": payload.get("cohens_large"),
        "n_value": n_value,
        "n_semantics": n_semantics,
        "total_sample_size": total_n,
    })

    tables: Dict[str, Any] = {}

    power_curve = payload.get("power_curve") or []
    if power_curve:
        tables["power_curve"] = power_curve

    if payload.get("assumptions_and_limitations"):
        tables["assumptions_and_limitations"] = [
            {"item": item}
            for item in payload.get("assumptions_and_limitations", [])
        ]

    metadata = compact_dict({
        "test_type": test_type,
        "mode": mode,
        "test_parameters": payload.get("test_parameters"),
        "power_curve_axis": payload.get("power_curve_axis"),
        "n_value_continuous": payload.get("n_value_continuous"),
    })

    return title, summary, metrics, tables, metadata


POWER_ANALYSIS_DISPLAY = DisplayConfig(
    metrics=MetricDisplayConfig(
        labels={
            "test_type": "Test type",
            "mode": "Mode",
            "alpha": "Alpha",
            "target_power": "Target power",
            "achieved_power": "Achieved power",
            "effect_size_name": "Effect size",
            "effect_size": "Effect size value",
            "effect_size_magnitude": "Effect size magnitude",
            "cohens_small": "Cohen small threshold",
            "cohens_medium": "Cohen medium threshold",
            "cohens_large": "Cohen large threshold",
            "n_value": "Sample size",
            "n_semantics": "Sample size meaning",
            "total_sample_size": "Total sample size",
        },
        formatters={
            "alpha": format_number,
            "target_power": format_number,
            "achieved_power": format_number,
            "effect_size": format_number,
            "cohens_small": format_number,
            "cohens_medium": format_number,
            "cohens_large": format_number,
        },
        order=[
            "test_type",
            "mode",
            "alpha",
            "target_power",
            "achieved_power",
            "effect_size_name",
            "effect_size",
            "effect_size_magnitude",
            "cohens_small",
            "cohens_medium",
            "cohens_large",
            "n_value",
            "n_semantics",
            "total_sample_size",
        ],
    ),
    tables={
        "power_curve": TableDisplayConfig(
            column_labels={
                "n": "n",
                "effect_size": "Effect size",
                "multiplier_of_target": "Multiplier of solution",
                "power_at_n": "Power at n",
                "power_at_effect": "Power at effect",
            },
            column_order=[
                "n",
                "effect_size",
                "multiplier_of_target",
                "power_at_n",
                "power_at_effect",
            ],
            column_formatters={
                "effect_size": format_number,
                "multiplier_of_target": format_number,
                "power_at_n": format_number,
                "power_at_effect": format_number,
            },
        ),
        "assumptions_and_limitations": TableDisplayConfig(
            column_labels={
                "item": "Assumption / limitation",
            },
            column_order=["item"],
        ),
    },
)


PLUGIN = register_plugin(AnalysisToolPlugin(
    tool_name="power_analysis",
    display_name="Power Analysis",
    is_inferential=False,
    evidence_categories=["study_design", "power_analysis"],
    description=(
        "A priori sample size, achieved power, and minimum detectable effect calculations for "
        "five test families: two-sample t-test, one-way ANOVA, multiple linear regression, "
        "Pearson correlation, and chi-square. Effect sizes follow Cohen (1988) conventions "
        "(d, f, f^2, r, w). Each result includes a small power-curve table around the solution."
    ),
    usage_guidance=(
        "This is a pure statistical calculation; it does not require an active dataset.\n"
        "Three modes:\n"
        "  - mode='sample_size': supply effect_size + target power; solves for n.\n"
        "  - mode='power': supply effect_size + n; reports achieved power.\n"
        "  - mode='effect_size': supply n + target power; reports the minimum detectable effect.\n"
        "Sample-size semantics by test:\n"
        "  - two_sample_t: n is per-group (equal-n assumed).\n"
        "  - one_way_anova: n is per-group; also supply n_groups (k).\n"
        "  - linear_regression: n is total; also supply n_predictors (p).\n"
        "  - correlation: n is total.\n"
        "  - chi_square: n is total; also supply df = (rows-1)*(cols-1) or (k-1)."
    ),
    use_when=[
        "The user is planning a study and asks how many subjects they need.",
        "The user asks for the achieved or post-hoc power of a finished study.",
        "The user wants to know the smallest effect they can reliably detect at their current sample size.",
        "The user mentions 'power', 'sample size justification', 'a priori power', or 'effect size'.",
    ],
    do_not_use_when=[
        "The user wants to actually run a test on data; use the relevant inferential plugin instead.",
        "The user has unequal group sizes; results may differ from equal-n assumptions.",
    ],
    requires_data_source=None,
    produces_active_dataset=False,
    requires_confirmation=False,
    argument_schema=ArgumentSchema(
        required={
            "test_type": str,
            "mode": str,
        },
        optional={
            "effect_size": float,
            "d": float,
            "f": float,
            "f2": float,
            "f_squared": float,
            "r": float,
            "w": float,
            "n": float,
            "alpha": float,
            "power": float,
            "n_groups": int,
            "n_predictors": int,
            "df": int,
            "alternative": str,
        },
        column_args=[],
        column_list_args=[],
        allow_all_columns=False,
    ),
    execute=execute_power_analysis,
    extractor=extract_power_analysis,
    guardrail_evaluators=[
        evaluate_power_analysis_guardrails,
    ],
    display_config=POWER_ANALYSIS_DISPLAY,
    examples=[
        {
            "user_request": "How many subjects per group do I need to detect a medium effect (d=0.5) with 80% power at alpha=0.05 in a two-sample t-test?",
            "arguments": {
                "test_type": "two_sample_t",
                "mode": "sample_size",
                "effect_size": 0.5,
                "power": 0.80,
                "alpha": 0.05,
            },
        },
        {
            "user_request": "I have 30 subjects per group and Cohen's d = 0.3. What's my power?",
            "arguments": {
                "test_type": "two_sample_t",
                "mode": "power",
                "effect_size": 0.3,
                "n": 30,
            },
        },
        {
            "user_request": "Required sample size for a one-way ANOVA with 4 groups, f=0.25, power 0.80.",
            "arguments": {
                "test_type": "one_way_anova",
                "mode": "sample_size",
                "effect_size": 0.25,
                "n_groups": 4,
                "power": 0.80,
            },
        },
        {
            "user_request": "Sample size for a multiple regression with 5 predictors, medium effect f^2=0.15.",
            "arguments": {
                "test_type": "linear_regression",
                "mode": "sample_size",
                "f2": 0.15,
                "n_predictors": 5,
                "power": 0.80,
            },
        },
        {
            "user_request": "Minimum detectable correlation with n=100 at 80% power, two-sided alpha=0.05.",
            "arguments": {
                "test_type": "correlation",
                "mode": "effect_size",
                "n": 100,
                "power": 0.80,
            },
        },
        {
            "user_request": "Chi-square test of independence on a 3x4 table; sample size for a small effect (w=0.2) at 80% power.",
            "arguments": {
                "test_type": "chi_square",
                "mode": "sample_size",
                "w": 0.2,
                "df": 6,
                "power": 0.80,
            },
        },
    ],
))