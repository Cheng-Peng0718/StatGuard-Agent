

def test_dataframe_tool_allowed_with_active_data_version_even_if_profile_none(tmp_path):
    import pandas as pd
    from core.schema import ActionProposal
    from verifiers.validators import verify

    df = pd.DataFrame({
        "region": ["East", "West"],
        "total_revenue": [100.0, 200.0],
    })

    path = tmp_path / "active.parquet"
    df.to_parquet(path, index=False)

    state = {
        "active_data_version_id": "data_v_test",
        "data_versions": [
            {
                "version_id": "data_v_test",
                "path": str(path),
                "n_rows": 2,
                "n_cols": 2,
            }
        ],
    }

    action = ActionProposal(
        action_id="act_test",
        action_type="tool_call",
        tool_name="groupby_summary",
        arguments={
            "group_cols": ["region"],
            "value_col": "total_revenue",
        },
        reasoning_summary="Compare revenue by region.",
    )

    status, feedback = verify(action, profile=None, state=state)

    assert status in {"allowed", "needs_review"}

def test_dataframe_tool_rejected_without_profile_or_active_data_version():
    from core.schema import ActionProposal
    from verifiers.validators import verify

    state = {
        "active_data_version_id": None,
        "data_versions": [],
    }

    action = ActionProposal(
        action_id="act_test",
        action_type="tool_call",
        tool_name="groupby_summary",
        arguments={
            "group_cols": ["region"],
            "value_col": "total_revenue",
        },
        reasoning_summary="Compare revenue by region.",
    )

    status, feedback = verify(action, profile=None, state=state)

    assert status == "rejected_recoverable"