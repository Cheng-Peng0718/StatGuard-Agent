"""
Pytest wrapper for the plugin-layer carpet bomb.

Folds the ~246-case accuracy/no-error/routing matrix into the normal test
suite. No LLM is involved, so this runs in seconds. Each case becomes one
parametrized test, so a regression points at the exact scenario that broke.
"""

from __future__ import annotations

import warnings

import pytest

from benchmark.carpet.case_matrix import generate_all_cases
from benchmark.carpet.run_plugin_carpet import run_one

warnings.filterwarnings("ignore")

_CASES = generate_all_cases()


@pytest.mark.parametrize("case", _CASES, ids=[c.key for c in _CASES])
def test_carpet_case(case):
    r = run_one(case)
    assert r["no_error"], f"{case.key} no-error failed: {r['issues']}"
    assert r["accurate"], f"{case.key} accuracy failed: {r['issues']}"