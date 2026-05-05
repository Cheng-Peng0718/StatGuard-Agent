from core.analysis_tool_plugins.validation import validate_plugin_action
from core.schema import ActionProposal


def test_clean_data_strategy_drop_is_canonicalized_to_rows():
    action = ActionProposal(
        action_id="act_test",
        action_type="tool_call",
        tool_name="clean_data",
        arguments={
            "action_type": "drop",
            "strategy": "drop",
            "columns": ["GPA", "SATM"],
        },
        reasoning_summary="Drop rows with missing GPA and SATM.",
    )

    result = validate_plugin_action(action, profile=None)

    assert result.status == "needs_review"
    assert action.arguments["strategy"] == "rows"
    assert result.details["canonical_arguments"]["strategy"] == "rows"


from core.analysis_tool_plugins.validation import validate_plugin_action
from core.schema import ActionProposal


def test_clean_data_invalid_impute_rows_is_rejected_before_review():
    action = ActionProposal(
        action_id="act_test",
        action_type="tool_call",
        tool_name="clean_data",
        arguments={
            "action_type": "impute",
            "strategy": "rows",
            "columns": ["GPA"],
        },
        reasoning_summary="Invalid imputation request.",
    )

    result = validate_plugin_action(action, profile=None)

    assert result.status == "rejected_recoverable"
    assert result.error_code == "INVALID_TOOL_ARGUMENTS"
    assert result.details["conditional_violations"]


import os
import tempfile

import pandas as pd

from core.data_versions import create_initial_data_version
from core.schema import AgentContext
from core.analysis_tool_plugins import get_plugin


def test_clean_data_execution_returns_valid_data_version_update():
    workspace = tempfile.mkdtemp(prefix="agent_clean_data_regression_")

    df = pd.DataFrame({
        "GPA": [3.2, None, 4.0],
        "SATM": [700, 650, None],
    })

    raw_version = create_initial_data_version(
        df=df,
        workspace_dir=workspace,
        created_by="test",
        description="test raw data",
    )

    context = AgentContext(
        workspace_dir=workspace,
        arguments={
            "action_type": "drop",
            "strategy": "rows",
            "columns": ["GPA", "SATM"],
        },
        data_versions=[raw_version],
        active_data_version_id=raw_version["version_id"],
        data_audit_log=[],
    )

    plugin = get_plugin("clean_data")
    result = plugin.run(context)

    update = result.get("data_version_update")

    assert result.get("status") in {"ok", "warning"}
    assert isinstance(update, dict)
    assert "new_version" in update
    assert "active_data_version_id" in update
    assert "audit_event" in update
    assert update["new_version"]["version_id"] == update["active_data_version_id"]
    assert update["active_data_version_id"] is not None
    assert os.path.exists(update["new_version"]["path"])