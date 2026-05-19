"""
APA 7th-edition formatting helpers.

These are the conventions every per-plugin `apa_methods_writer` should use,
so the assembled Methods section reads as a single document instead of a
patchwork of styles.

Conventions implemented (APA 7th):
  - Statistics are italicized in the rendered output (Markdown uses `*x*`).
  - p-values: `.027`, `p < .001`, leading zero dropped (per APA).
  - Other probability-like quantities (e.g. R^2, eta^2) also drop the leading zero.
  - Test statistics that can exceed 1 (t, F, M, SD) keep their leading zero.
  - Effect sizes always reported with a 95% CI when available.
  - Degrees of freedom shown in parentheses immediately after the symbol.
  - English commas, oxford comma optional.
"""

from __future__ import annotations

import math
from typing import Any, Optional


# ============================================================
# Number formatters
# ============================================================

def _is_finite_number(x: Any) -> bool:
    try:
        v = float(x)
        return math.isfinite(v)
    except Exception:
        return False


def fmt_p(p: Any) -> str:
    """
    APA p-value formatting.

    Examples:
        0.0003   -> "p < .001"
        0.0142   -> "p = .014"
        0.05     -> "p = .050"
        0.5      -> "p = .500"
        None     -> "p = n/a"
    """
    if not _is_finite_number(p):
        return "p = n/a"

    v = float(p)

    if v < 0.001:
        return "p < .001"

    # 3 decimal places, drop the leading 0
    s = f"{v:.3f}"
    if s.startswith("0."):
        s = s[1:]
    return f"p = {s}"


def fmt_bounded_unit(value: Any, digits: int = 3) -> str:
    """
    Format a quantity bounded in [0, 1] (R^2, eta^2, omega^2, Cramer's V) per
    APA: drop the leading zero.

    Example:
        0.142  -> ".142"
        0.005  -> ".005"
        None   -> "n/a"
    """
    if not _is_finite_number(value):
        return "n/a"

    v = float(value)
    fmt = f"{{:.{digits}f}}"
    s = fmt.format(v)
    if s.startswith("0."):
        s = s[1:]
    elif s.startswith("-0."):
        s = "-" + s[2:]
    return s


def fmt_general(value: Any, digits: int = 2) -> str:
    """General-purpose 2-decimal formatter for stats that can exceed 1
    (t, F, M, SD, mean differences)."""
    if not _is_finite_number(value):
        return "n/a"

    return f"{float(value):.{digits}f}"


def fmt_signed(value: Any, digits: int = 2) -> str:
    """Always show a sign (useful for Cohen's d so the direction is visible)."""
    if not _is_finite_number(value):
        return "n/a"

    v = float(value)
    return f"{v:+.{digits}f}"


def fmt_int(value: Any) -> str:
    if not _is_finite_number(value):
        return "n/a"
    return f"{int(round(float(value)))}"


def fmt_ci(
    lower: Any,
    upper: Any,
    *,
    digits: int = 2,
    bounded_unit: bool = False,
) -> Optional[str]:
    """
    Format a 95% confidence interval as `95% CI [a, b]`.

    Returns None when either endpoint is missing; callers can then choose to
    omit the CI rather than render `[n/a, n/a]`.
    """
    if not _is_finite_number(lower) or not _is_finite_number(upper):
        return None

    if bounded_unit:
        a = fmt_bounded_unit(lower, digits=digits)
        b = fmt_bounded_unit(upper, digits=digits)
    else:
        a = fmt_general(lower, digits=digits)
        b = fmt_general(upper, digits=digits)

    return f"95% CI [{a}, {b}]"


# ============================================================
# Markdown italics helper
# ============================================================

def md_italic(text: str) -> str:
    """Wrap text in Markdown italics. Used for stats symbols."""
    return f"*{text}*"