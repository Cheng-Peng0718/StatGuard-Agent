"""
Integration test: run LangGraph with data/class_survey.xls and a GPA modeling query.

Notes:
- Test code only; does not change application logic.
- Requires OPENAI_API_KEY; skipped when unset.
"""
from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_XLS = PROJECT_ROOT / "data" / "class_survey.xls"

USER_QUERY = (
    "Help me build a GPA prediction model. Before and after modeling, please run "
    "multicollinearity diagnostics, use the code interpreter to plot a histogram of "
    "residuals, and deliver a professional report."
)


@pytest.fixture()
def class_survey_workspace(tmp_path: Path) -> str:
    """Sandbox: working_data.xls + working_data.parquet (same as Streamlit / AgentContext)."""
    ws = tmp_path / "sandbox"
    ws.mkdir(parents=True)
    shutil.copy(DATA_XLS, ws / "working_data.xls")
    df = pd.read_excel(ws / "working_data.xls")
    df.to_parquet(ws / "working_data.parquet", index=False)
    return str(ws)


def test_gpa_prediction_query_end_to_end(class_survey_workspace: str) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY required for Supervisor")

    os.environ.setdefault("LANGGRAPH_ALLOWED_MSGPACK_MODULES", "core.schema")

    import tools.methods  # noqa: F401 — register tools

    from core.context_builder import generate_profile
    from core.graph import app
    from core.schema import VerificationResult

    workspace = class_survey_workspace
    profile = generate_profile(str(Path(workspace) / "working_data.xls"))

    thread_id = f"pytest_gpa_{uuid.uuid4().hex[:10]}"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "user_request": USER_QUERY,
        "dataset_profile": profile,
        "workspace_dir": workspace,
        "observations": [],
        "current_step": 0,
        "max_steps": 14,
        "current_context_text": "",
        "current_action": None,
        "current_verification": None,
        "current_execution": None,
    }

    state_input: dict | None = initial_state
    outer_loops = 0
    max_outer = 40
    node_trace: list[tuple[str, object]] = []

    while outer_loops < max_outer:
        outer_loops += 1
        for event in app.stream(state_input, config):
            for node_name, state_updates in event.items():
                node_trace.append((node_name, state_updates))
        state_input = None

        snap = app.get_state(config)
        if not snap.next:
            break

        if snap.next[0] == "human_review":
            action = snap.values.get("current_action")
            action_id = getattr(action, "action_id", None) or "unknown"
            app.update_state(
                config,
                {"current_verification": VerificationResult(action_id=action_id, status="allowed")},
            )
            continue

        break

    final_snap = app.get_state(config)
    values = final_snap.values
    observations = values.get("observations") or []
    last_action = values.get("current_action")

    png_files = [f for f in os.listdir(workspace) if f.endswith(".png")]
    obs_blob = "\n".join(str(o) for o in observations)

    obs_summaries: list[str] = []
    for o in observations:
        if isinstance(o, dict):
            obs_summaries.append(str(o.get("summary", o))[:800])
        else:
            obs_summaries.append(str(o)[:800])

    final_summary_preview = ""
    if last_action is not None and getattr(last_action, "reasoning_summary", None):
        final_summary_preview = str(last_action.reasoning_summary)[:2500]

    diagnostics = {
        "workspace": workspace,
        "written_report": str(Path(workspace) / "pytest_gpa_diagnostics.json"),
        "outer_loops": outer_loops,
        "graph_pending_next": repr(final_snap.next),
        "final_action_type": getattr(last_action, "action_type", None),
        "final_tool_name": getattr(last_action, "tool_name", None),
        "final_answer_preview": final_summary_preview,
        "observation_count": len(observations),
        "png_count": len(png_files),
        "png_files": png_files,
        "vif_mentioned_in_observations": "VIF" in obs_blob or "vif" in obs_blob.lower(),
        "residual_mentioned": "residual" in obs_blob.lower(),
        "execute_error_signature": "unexpected keyword argument 'action_args'" in obs_blob,
        "agent_context_missing_args": "'AgentContext' object has no attribute 'args'" in obs_blob,
        "observation_summaries": obs_summaries,
    }

    print("\n=== class_survey GPA pipeline diagnostics (test output) ===")
    for k, v in diagnostics.items():
        print(f"  {k}: {v}")
    if observations:
        print("\n--- Latest observation snippet ---")
        last = observations[-1]
        if isinstance(last, dict):
            print(last.get("summary", last)[:1200])
        else:
            print(str(last)[:1200])

    diag_path = Path(workspace) / "pytest_gpa_diagnostics.json"
    diag_path.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    assert outer_loops < max_outer, "Too many outer loops; graph may not converge"
    # LangGraph often ends with next=() rather than None
    assert not final_snap.next, f"Graph should have no pending tasks; next={final_snap.next!r}"
