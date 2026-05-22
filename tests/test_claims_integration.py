"""
Integration test for the deliverable gate's P0 wiring (core/graph.py).

This is the seam where the claims ledger meets the live pipeline:
  - the multiple-comparison session guardrail is invoked correctly (the bug
    fixed in P0 was a positional-arg call that silently produced no findings),
  - statistical claims are built from the session's analysis_runs,
  - [CLAIM:id] tokens in the final answer are substituted with verified wording
    and written back onto the action the UI reads,
  - the family-wise-error note is appended when multiple inferential tests ran.

We exercise deliverable_gate_node directly with a constructed state rather than
driving the whole graph, so the test is fast and deterministic (no LLM).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import pandas as pd
import pytest

from core.graph import deliverable_gate_node
from core.schema import ActionProposal
from core.analysis_tool_plugins.registry import get_plugin


DATASETS = os.path.join(os.path.dirname(__file__), "..", "benchmark", "datasets")


class DummyContext:
    def __init__(self, df, args=None):
        self.df = df
        self.arguments = args or {}
        self.active_data_version_id = "data_v_test"
        self.data_versions = []

    def load_df(self):
        return self.df

    def get_arg(self, key, default=None):
        return self.arguments.get(key, default)


def _run(tool: str, df: pd.DataFrame, args: Dict[str, Any], rid: str) -> Dict[str, Any]:
    out = get_plugin(tool).run(DummyContext(df, args))
    return {
        "run_id": rid,
        "tool_name": tool,
        "status": out.get("status", "ok"),
        "metrics": out.get("details", {}),
        "arguments": args,
        "is_inferential": True,
        "evidence_categories": ["group_comparison", "statistical_inference"],
    }


def _final_action(text: str) -> ActionProposal:
    return ActionProposal(
        action_id="act_final",
        action_type="final_answer",
        tool_name="none",
        arguments={},
        reasoning_summary=text,
    )


def _base_state(runs: List[Dict[str, Any]], action: ActionProposal) -> Dict[str, Any]:
    return {
        "user_request": "test",
        "current_action": action,
        "analysis_runs": runs,
        "observations": [],
        "deliverable_gate_attempts": 0,
        "workspace_dir": "./",
    }


datasets_present = os.path.exists(os.path.join(DATASETS, "case4_nonnormal_two_group.parquet"))


@pytest.mark.skipif(not datasets_present, reason="benchmark datasets not present")
class TestDeliverableGateClaims:

    def test_claim_tokens_substituted_in_final_answer(self):
        df = pd.read_parquet(os.path.join(DATASETS, "case4_nonnormal_two_group.parquet"))
        run = _run("statistical_group_comparison", df,
                   {"target_col": "response_time", "group_col": "arm"}, "run_x")
        # case4 is strongly skewed, so P2 routes the primary test to
        # Mann-Whitney U; the claims reflect that (rank-biserial effect size).
        action = _final_action(
            "No significant difference [CLAIM:run_x_sig], small effect [CLAIM:run_x_es]."
        )
        state = _base_state([run], action)

        deliverable_gate_node(state)

        # The action the UI reads must now contain verified wording, no tokens.
        rendered = state["current_action"].reasoning_summary
        assert "[CLAIM:" not in rendered
        assert "p = " in rendered          # a tamper-proof p-value is present
        assert "rank-biserial" in rendered  # rank-based effect size from Mann-Whitney

    def test_validation_recorded_and_clean(self):
        df = pd.read_parquet(os.path.join(DATASETS, "case4_nonnormal_two_group.parquet"))
        run = _run("statistical_group_comparison", df,
                   {"target_col": "response_time", "group_col": "arm"}, "run_x")
        action = _final_action(
            "No statistically significant difference [CLAIM:run_x_sig] "
            "with a small effect [CLAIM:run_x_es]."
        )
        state = _base_state([run], action)

        out = deliverable_gate_node(state)

        v = out.get("claims_validation")
        assert v is not None
        # Referencing claims + stating the verdict in words is clean.
        assert v["is_clean"]

    def test_multiple_comparison_note_appended(self):
        # Five inferential runs in one session must trigger the family-wise
        # warning and append a statistical note to the final answer.
        df = pd.read_parquet(os.path.join(DATASETS, "case3_multiple_comparisons.parquet"))
        runs = [
            _run("statistical_group_comparison", df,
                 {"target_col": f"metric_{i}", "group_col": "group"}, f"run_{i}")
            for i in range(1, 6)
        ]
        action = _final_action(
            "Across the metrics there were no significant differences "
            "[CLAIM:run_1_sig]."
        )
        state = _base_state(runs, action)

        deliverable_gate_node(state)

        rendered = state["current_action"].reasoning_summary
        # The appended note flags multiple comparisons.
        assert "Statistical note" in rendered or "multiple" in rendered.lower()

    def test_no_claims_no_crash_when_runs_empty(self):
        action = _final_action("Nothing was computed yet.")
        state = _base_state([], action)
        # Must not raise and must leave the answer untouched.
        deliverable_gate_node(state)
        assert state["current_action"].reasoning_summary == "Nothing was computed yet."