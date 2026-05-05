import tempfile
from pathlib import Path

import pandas as pd

from core.schema import ActionProposal, DatasetProfile, ColumnProfile, AgentContext
from core.analysis_tool_plugins import get_plugin
from core.analysis_tool_plugins.validation import validate_plugin_action


def make_profile():
    return DatasetProfile(
        dataset_name="test",
        n_rows=3,
        n_cols=2,
        columns={
            "GPA": ColumnProfile(
                name="GPA",
                dtype="float64",
                n_missing=1,
                missing_rate=1 / 3,
                n_unique=2,
                semantic_type="numeric",
                is_numeric_like=True,
                is_id_like=False,
            ),
            "SATM": ColumnProfile(
                name="SATM",
                dtype="float64",
                n_missing=1,
                missing_rate=1 / 3,
                n_unique=2,
                semantic_type="numeric",
                is_numeric_like=True,
                is_id_like=False,
            ),
        },
    )


def test_clean_data_strategy_drop_is_canonicalized_before_review():
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

    result = validate_plugin_action(action, make_profile())

    assert result.status == "needs_review"
    assert action.arguments["strategy"] == "rows"
    assert result.details["canonical_arguments"]["strategy"] == "rows"


def test_clean_data_illegal_impute_strategy_rejected_before_review():
    action = ActionProposal(
        action_id="act_test",
        action_type="tool_call",
        tool_name="clean_data",
        arguments={
            "action_type": "impute",
            "strategy": "rows",
            "columns": ["GPA"],
        },
        reasoning_summary="Bad imputation request.",
    )

    result = validate_plugin_action(action, make_profile())

    assert result.status == "rejected_recoverable"
    assert result.error_code == "INVALID_TOOL_ARGUMENTS"
    assert result.details["conditional_violations"]


def test_clean_data_valid_drop_rows_needs_review():
    action = ActionProposal(
        action_id="act_test",
        action_type="tool_call",
        tool_name="clean_data",
        arguments={
            "action_type": "drop",
            "strategy": "rows",
            "columns": ["GPA", "SATM"],
        },
        reasoning_summary="Drop missing rows.",
    )

    result = validate_plugin_action(action, make_profile())

    assert result.status == "needs_review"
    assert result.error_code is None


def test_clean_data_plugin_returns_new_data_version_contract(tmp_path):
    df = pd.DataFrame({
        "GPA": [3.2, None, 4.0],
        "SATM": [700, 650, None],
    })

    versions_dir = tmp_path / "data_versions"
    versions_dir.mkdir(exist_ok=True)

    raw_path = versions_dir / "raw_v1.parquet"
    df.to_parquet(raw_path, index=False)

    raw_version = {
        "version_id": "raw_v1",
        "parent_version_id": None,
        "path": str(raw_path),
        "n_rows": 3,
        "n_cols": 2,
        "created_by": "test",
        "operation": "initial_load",
        "metadata": {},
    }

    context = AgentContext(
        workspace_dir=str(tmp_path),
        arguments={
            "action_type": "drop",
            "strategy": "rows",
            "columns": ["GPA", "SATM"],
        },
        data_versions=[raw_version],
        active_data_version_id="raw_v1",
        data_audit_log=[],
    )

    plugin = get_plugin("clean_data")
    result = plugin.run(context)

    assert result["status"] in {"ok", "warning"}
    assert "data_version_update" in result

    update = result["data_version_update"]

    assert "new_version" in update
    assert "active_data_version_id" in update
    assert "audit_event" in update
    assert update["new_version"]["version_id"] == update["active_data_version_id"]
    assert Path(update["new_version"]["path"]).exists()