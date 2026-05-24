"""
Standalone smoke test: invoke the deterministic statistical engine directly,
WITHOUT the Streamlit UI or any LLM / API key.

This demonstrates that the analysis engine is a library that can be called
programmatically -- the UI is just one optional front-end. It builds a small
in-memory dataset, runs a group comparison through the plugin layer, prints a
human-readable summary, and asserts the core statistics are present and sane.

Run:  python smoke_test.py
Exit code 0 = engine works headless.
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

from core.analysis_tool_plugins.registry import get_plugin


class _Ctx:
    """Minimal analysis context: just enough for a plugin to run headless."""

    def __init__(self, df: pd.DataFrame, args: dict):
        self._df = df
        self.arguments = args
        self.active_data_version_id = "smoke_v1"
        self.data_versions = []

    def load_df(self) -> pd.DataFrame:
        return self._df

    def get_arg(self, name, default=None):
        return self.arguments.get(name, default)


def main() -> int:
    # 1. Build a tiny, fixed dataset (two groups with a real mean difference).
    rng = np.random.default_rng(20260523)
    df = pd.DataFrame(
        {
            "test_score": np.concatenate(
                [rng.normal(100.0, 15.0, 45), rng.normal(108.0, 15.0, 45)]
            ),
            "cohort": (["2024"] * 45) + (["2025"] * 45),
        }
    )

    # 2. Invoke the statistical engine directly (no UI, no LLM).
    plugin = get_plugin("statistical_group_comparison")
    ctx = _Ctx(df, {"target_col": "test_score", "group_col": "cohort"})
    result = plugin.run(ctx)
    d = result.get("details", {}) or {}

    # 3. Show what the deterministic engine produced.
    print("=" * 60)
    print("HEADLESS ENGINE SMOKE TEST")
    print("=" * 60)
    print(f"status        : {result.get('status')}")
    print(f"message       : {result.get('message')}")
    print(f"method        : {d.get('method')}")
    print(f"t_statistic   : {d.get('t_statistic')}")
    print(f"p_value       : {d.get('p_value')}")
    print(f"effect_size   : {d.get('effect_size_name')} = {d.get('effect_size')}")
    print(f"significant   : {d.get('significant_at_0_05')}")
    print("=" * 60)

    # 4. Assert the engine behaved (fail loudly with non-zero exit otherwise).
    assert result.get("status") in {"ok", "warning"}, "engine did not return ok/warning"
    assert d.get("p_value") is not None, "no p-value produced"
    assert d.get("method"), "no test method reported"
    assert d.get("effect_size") is not None, "no effect size produced"
    print("SMOKE TEST PASSED: engine runs headless and returns valid statistics.")
    return 0


if __name__ == "__main__":
    sys.exit(main())