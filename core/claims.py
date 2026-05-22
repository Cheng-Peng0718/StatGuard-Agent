"""
Statistical claims ledger.

The problem this solves: in an LLM-routed analyst, the final answer is authored
by the language model. Even when every number is computed by a deterministic
plugin, the model can still STATE a wrong conclusion in prose -- e.g. calling a
p = .147 result "statistically significant" (observed in benchmark case4).

The fix: statistical assertions are not free text. Each inferential plugin emits
a set of structured `Claim` objects, computed mechanically from its results. The
language model may write the surrounding narrative freely, but every statistical
assertion must be a reference of the form `[CLAIM:<id>]`. At render time, those
references are replaced by wording the Claim generates from its own structured
fields -- wording the model never controls.

Result: the model can compose the story, but it cannot author (or rewrite) the
numbers, the significance verdict, or the direction of an effect.

This module has no dependency on the plugin layer or the graph, so it stays
unit-testable in isolation.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# Number formatting (APA-ish; mirrors shared/apa_formatting but kept local
# so this module has zero plugin-layer dependencies)
# ============================================================

def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _fmt_p(p: Any) -> str:
    if not _finite(p):
        return "p = n/a"
    v = float(p)
    if v < 0.001:
        return "p < .001"
    s = f"{v:.3f}"
    if s.startswith("0."):
        s = s[1:]
    return f"p = {s}"


def _fmt_num(x: Any, digits: int = 2) -> str:
    if not _finite(x):
        return "n/a"
    return f"{float(x):.{digits}f}"


def _fmt_unit(x: Any, digits: int = 3) -> str:
    """For quantities in [0,1] (eta², R², etc.): drop the leading zero."""
    if not _finite(x):
        return "n/a"
    s = f"{float(x):.{digits}f}"
    if s.startswith("0."):
        return s[1:]
    if s.startswith("-0."):
        return "-" + s[2:]
    return s


def _fmt_ci(lo: Any, hi: Any, digits: int = 2, unit: bool = False) -> Optional[str]:
    if not _finite(lo) or not _finite(hi):
        return None
    a = _fmt_unit(lo, digits) if unit else _fmt_num(lo, digits)
    b = _fmt_unit(hi, digits) if unit else _fmt_num(hi, digits)
    return f"95% CI [{a}, {b}]"


# ============================================================
# Claim kinds
# ============================================================

CLAIM_SIGNIFICANCE = "significance"
CLAIM_EFFECT_SIZE = "effect_size"
CLAIM_DIRECTION = "direction"
CLAIM_TEST_STATISTIC = "test_statistic"
CLAIM_SESSION_WARNING = "session_warning"

VALID_KINDS = {
    CLAIM_SIGNIFICANCE,
    CLAIM_EFFECT_SIZE,
    CLAIM_DIRECTION,
    CLAIM_TEST_STATISTIC,
    CLAIM_SESSION_WARNING,
}


# ============================================================
# Claim
# ============================================================

@dataclass
class Claim:
    """
    A single statistical assertion, generated mechanically from a plugin's
    structured results. Its `render()` method is the ONLY authority on how the
    assertion is worded; the language model can reference it by `claim_id` but
    cannot change what it says.
    """
    claim_id: str
    kind: str
    # Free-form structured payload; the keys used depend on `kind`.
    data: Dict[str, Any] = field(default_factory=dict)
    # Human-readable subject, e.g. "response_time between placebo and drug".
    subject: Optional[str] = None
    # Provenance: which run produced this claim.
    source_run_id: Optional[str] = None
    source_tool_name: Optional[str] = None

    def __post_init__(self):
        if self.kind not in VALID_KINDS:
            raise ValueError(f"Unknown claim kind: {self.kind!r}")

    # ----- rendering -----

    def render(self) -> str:
        """Authoritative wording, generated from structured fields only."""
        if self.kind == CLAIM_SIGNIFICANCE:
            return self._render_significance()
        if self.kind == CLAIM_EFFECT_SIZE:
            return self._render_effect_size()
        if self.kind == CLAIM_DIRECTION:
            return self._render_direction()
        if self.kind == CLAIM_TEST_STATISTIC:
            return self._render_test_statistic()
        if self.kind == CLAIM_SESSION_WARNING:
            return self._render_session_warning()
        return "(unrenderable claim)"

    def _render_significance(self) -> str:
        # Render ONLY the numeric support, as a parenthetical. The verdict
        # phrasing ("significant" / "not significant") is left to the model's
        # narrative; the claim supplies the exact, tamper-proof numbers. This
        # avoids duplicating the verdict the model already wrote.
        p = self.data.get("p_value")
        alpha = self.data.get("alpha", 0.05)
        alpha_str = _fmt_unit(alpha) if _finite(alpha) else "n/a"
        return f"({_fmt_p(p)}, \u03b1 = {alpha_str})"

    def _render_effect_size(self) -> str:
        # Parenthetical numeric support: name, value, magnitude, CI.
        name = self.data.get("name", "effect size")
        value = self.data.get("value")
        magnitude = self.data.get("magnitude")
        ci = _fmt_ci(
            self.data.get("ci_low"),
            self.data.get("ci_high"),
            unit=bool(self.data.get("bounded_unit", False)),
            digits=3 if self.data.get("bounded_unit") else 2,
        )
        val_str = (
            _fmt_unit(value) if self.data.get("bounded_unit") else _fmt_num(value)
        )
        inner = f"{name} = {val_str}"
        if magnitude:
            inner += f", {magnitude}"
        if ci:
            inner += f", {ci}"
        return f"({inner})"

    def _render_direction(self) -> str:
        higher = self.data.get("higher_group")
        lower = self.data.get("lower_group")
        if higher and lower:
            return f"{higher} scored higher than {lower}"
        return self.data.get("text", "the direction of the effect")

    def _render_test_statistic(self) -> str:
        label = self.data.get("label", "test statistic")
        value = self.data.get("value")
        df = self.data.get("df")
        if df is not None and _finite(df):
            df_f = float(df)
            df_str = f"{int(df_f)}" if df_f.is_integer() else _fmt_num(df_f)
            return f"({label}({df_str}) = {_fmt_num(value)})"
        return f"({label} = {_fmt_num(value)})"

    def _render_session_warning(self) -> str:
        return self.data.get("text", "(session warning)")

    # ----- serialization (for state/checkpoint) -----

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "kind": self.kind,
            "data": self.data,
            "subject": self.subject,
            "source_run_id": self.source_run_id,
            "source_tool_name": self.source_tool_name,
            "rendered": self.render(),  # cached wording for convenience
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Claim":
        return Claim(
            claim_id=d["claim_id"],
            kind=d["kind"],
            data=d.get("data", {}) or {},
            subject=d.get("subject"),
            source_run_id=d.get("source_run_id"),
            source_tool_name=d.get("source_tool_name"),
        )


# ============================================================
# ClaimSet: collection + substitution + validation
# ============================================================

# Matches [CLAIM:some_id]
_CLAIM_REF_RE = re.compile(r"\[CLAIM:([A-Za-z0-9_\-]+)\]")

# Heuristic patterns for STATISTICAL assertions written as bare prose (i.e.
# not via a [CLAIM:...] reference). These are used only to WARN, never to parse
# meaning -- we are detecting "the model wrote a naked statistical claim",
# which is a policy violation, not trying to interpret it.
_BARE_P_VALUE_RE = re.compile(r"\bp\s*[<>=]\s*\.?\d", re.IGNORECASE)
_BARE_SIGNIFICANCE_RE = re.compile(
    r"\b(statistically\s+significant|not\s+significant|significant\s+at)\b",
    re.IGNORECASE,
)


@dataclass
class ClaimSet:
    claims: Dict[str, Claim] = field(default_factory=dict)

    def add(self, claim: Claim) -> None:
        self.claims[claim.claim_id] = claim

    def add_many(self, claims: List[Claim]) -> None:
        for c in claims:
            self.add(c)

    def get(self, claim_id: str) -> Optional[Claim]:
        return self.claims.get(claim_id)

    def is_empty(self) -> bool:
        return not self.claims

    # ----- the catalogue shown to the LLM -----

    def catalogue_text(self) -> str:
        """
        A human/LLM-readable list of available claims with their stable IDs and
        authoritative wording. This is injected into the supervisor context so
        the model can reference claims by ID.
        """
        if not self.claims:
            return "(no statistical claims available yet)"
        lines = []
        for cid, claim in self.claims.items():
            subj = f" — {claim.subject}" if claim.subject else ""
            lines.append(f"[CLAIM:{cid}]{subj}: {claim.render()}")
        return "\n".join(lines)

    # ----- substitution -----

    def substitute(self, text: str) -> Tuple[str, List[str]]:
        """
        Replace every [CLAIM:id] in `text` with the claim's authoritative
        wording (a parenthetical numeric support, e.g. "(p = .147, α = .05)").
        Returns (rendered_text, unresolved_ids).

        Claims render as inline parentheticals, so they slot into the model's
        own sentence without duplicating its verdict. Unresolved references are
        left verbatim and reported.
        """
        unresolved: List[str] = []

        def _repl(m: "re.Match") -> str:
            cid = m.group(1)
            claim = self.claims.get(cid)
            if claim is None:
                unresolved.append(cid)
                return m.group(0)
            return claim.render()

        rendered = _CLAIM_REF_RE.sub(_repl, text)
        # Tidy spacing: collapse runs of spaces, and remove a stray space
        # before sentence punctuation. Use a negative lookahead so we do NOT
        # touch a period that begins a decimal number (e.g. "= .147").
        rendered = re.sub(r"[ \t]{2,}", " ", rendered)
        rendered = re.sub(r"\s+([,;:])", r"\1", rendered)
        rendered = re.sub(r"\s+\.(?!\d)", ".", rendered)
        return rendered, unresolved

    # ----- validation -----

    def validate(self, text: str) -> Dict[str, Any]:
        """
        Inspect the LLM's final-answer text (BEFORE substitution) for policy
        compliance. This is a soft AUDIT signal, not a hard gate.

          - referenced_ids: claim ids the text references
          - unresolved_ids: referenced ids with no matching claim (real problem)
          - bare_numeric_assertions: naked numeric statistical values (e.g.
            "p = .03", a hand-typed effect size) that were NOT inside a
            [CLAIM:...] reference. A naked NUMBER is the real fabrication risk.
          - restated_significance: the prose contains a significance verdict in
            words ("not significant") outside a claim. This is NOT a violation
            on its own — when the model has the correct claim catalogue, a
            verbal restatement is consistent by construction. We surface it only
            as an informational flag.

        We deliberately do NOT treat a verbal significance restatement as a
        violation: forbidding the words makes the prose stilted, and the
        substitution mechanism already guarantees the authoritative wording.
        Only naked NUMBERS and unresolved references move `is_clean` to False.
        """
        referenced = _CLAIM_REF_RE.findall(text)
        unresolved = [cid for cid in referenced if cid not in self.claims]

        # Strip claim references before scanning, so wording we inject later is
        # not what we scan (we scan the model's own prose).
        stripped = _CLAIM_REF_RE.sub("", text)

        bare_numeric = bool(_BARE_P_VALUE_RE.search(stripped))
        restated_significance = bool(_BARE_SIGNIFICANCE_RE.search(stripped))

        bare_numeric_assertions = ["bare_p_value"] if bare_numeric else []

        return {
            "referenced_ids": referenced,
            "unresolved_ids": unresolved,
            "bare_numeric_assertions": bare_numeric_assertions,
            "restated_significance": restated_significance,
            # Only naked numbers and unresolved refs are real violations.
            "is_clean": (not unresolved) and (not bare_numeric_assertions),
        }

    # ----- serialization -----

    def to_list(self) -> List[Dict[str, Any]]:
        return [c.to_dict() for c in self.claims.values()]

    @staticmethod
    def from_list(items: List[Dict[str, Any]]) -> "ClaimSet":
        cs = ClaimSet()
        for it in items or []:
            try:
                cs.add(Claim.from_dict(it))
            except Exception:
                continue
        return cs