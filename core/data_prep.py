"""Data-preparation entry points (deterministic, context-free).

A thin, stable surface the orchestrator (DecisionGuard) calls to FORCE a
data-quality diagnosis before analysis -- no agent graph, no LLM, runs once
per dataset. This is the seed of the future shared data-prep layer.
"""
from __future__ import annotations

import pandas as pd


def diagnose_dataframe(df: pd.DataFrame, columns="all") -> dict:
    """Run the deterministic data-quality diagnosis on a DataFrame and return the
    details payload (n_issues, severity_counts, issues, column_diagnostics)."""
    from core.analysis_tool_plugins.plugins.data_quality_report import diagnose
    return diagnose(df, columns)