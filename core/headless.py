"""
Headless entry point for the StatGuard analysis graph.

Programmatic equivalent of app.py, minus Streamlit. Lets another process
(DecisionGuard) hand StatGuard a question + a DataFrame and drive the SAME
compiled LangGraph the UI drives. The UI and this module are two consumers of one
graph; the graph is unchanged.

Human-in-the-loop across the boundary
-------------------------------------
When a high-risk tool needs review, the graph interrupts before `human_review`.
Three policies:
  * "approve"  -- auto-approve and continue (fully automated)
  * "skip"     -- continue without approving (high-risk tool is not executed)
  * "escalate" -- PAUSE and return the pending review to the caller (DecisionGuard),
                  which confirms with the user and calls resume_headless().

Escalate relies on LangGraph's checkpointer persisting state by thread_id. The
compiled graph is a module-level singleton, so an in-process caller can pause,
ask the user, and resume on the same thread_id. (In-memory checkpointer lives for
the process lifetime; swap StatGuard's checkpointer to SqliteSaver for durability.)

Requires OPENAI_API_KEY for the supervisor's tool routing. Statistics are still
computed by the deterministic, cross-validated plugins.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from typing import Any, Dict, Optional

import pandas as pd

from core.context_builder import generate_profile
from core.data_versions import create_initial_data_version


# --------------------------------------------------------------------------
# State construction (no LLM; fully testable offline)
# --------------------------------------------------------------------------

def _prepare_state_input(question, df, workspace_dir, max_steps=20) -> Dict[str, Any]:
    os.makedirs(workspace_dir, exist_ok=True)
    dv = create_initial_data_version(df, workspace_dir, created_by="decisionguard")
    profile = generate_profile(dv["path"])
    profile_dict = profile.model_dump() if hasattr(profile, "model_dump") else dict(profile)
    return {
        "user_request": question,
        "max_steps": max_steps,
        "workspace_dir": workspace_dir,
        "dataset_profile": profile_dict,
        "deliverable_gate_attempts": 0,
        "deliverable_check": None,
        "task_contract": None,
        "data_versions": [dv],
        "active_data_version_id": dv["version_id"],
        "data_audit_log": [],
        "analysis_runs": [],
    }


# --------------------------------------------------------------------------
# Interrupt helpers
# --------------------------------------------------------------------------

def _interrupted(graph_app, config) -> bool:
    snap = graph_app.get_state(config)
    return bool(snap and snap.next and "human_review" in snap.next)


def _pending_review(graph_app, config) -> Dict[str, Any]:
    """What the graph is paused on: the tool + args + why it needs review."""
    snap = graph_app.get_state(config)
    v = snap.values or {}
    action = v.get("current_action")
    vr = v.get("current_verification")
    return {
        "tool_name": getattr(action, "tool_name", None),
        "arguments": getattr(action, "arguments", {}) or {},
        "reason": getattr(vr, "feedback", None),
        "status": getattr(vr, "status", None),
    }


def _approve_pending(graph_app, config) -> None:
    """Flip the pending verification to 'allowed' so the tool runs on resume."""
    snap = graph_app.get_state(config)
    vr = (snap.values or {}).get("current_verification")
    if vr is None:
        return
    try:
        approved = vr.model_copy(update={"status": "allowed"})
    except Exception:
        approved = dict(vr) if isinstance(vr, dict) else vr
        if isinstance(approved, dict):
            approved["status"] = "allowed"
    graph_app.update_state(config, {"current_verification": approved})


# --------------------------------------------------------------------------
# Core driver loop
# --------------------------------------------------------------------------

def _drive(graph_app, config, stream_input, on_human_review, max_steps,
           workspace_dir, on_event=None) -> Dict[str, Any]:
    """Drain the stream; on interrupt, act per policy. Returns a result dict
    whose 'status' is 'done' or 'needs_review'.

    on_event (optional): called with a small dict for each meaningful step --
    {"node", "type":"tool", "tool", "reasoning"} when the supervisor selects a
    tool, and {"node", "type":"final", "reasoning"} at the final answer. Lets a
    caller surface the agent's tool selection / reasoning as a live trace. A
    failing callback never interrupts the run.
    """
    final_answer = {"text": None}
    _seen_actions = set()

    def _emit(node, action):
        if on_event is None or action is None:
            return
        aid = getattr(action, "action_id", None)
        atype = getattr(action, "action_type", None)
        tool = getattr(action, "tool_name", None)
        reasoning = getattr(action, "reasoning_summary", None)
        key = aid or (atype, tool, reasoning)
        if key in _seen_actions:
            return
        _seen_actions.add(key)
        try:
            if atype == "final_answer":
                on_event({"node": node, "type": "final", "reasoning": reasoning})
            elif tool:
                on_event({"node": node, "type": "tool", "tool": tool,
                          "reasoning": reasoning})
        except Exception:
            pass

    def _drain(stream):
        for event in stream:
            for _node, state_data in event.items():
                if not isinstance(state_data, dict):
                    continue
                action = state_data.get("current_action")
                _emit(_node, action)
                if getattr(action, "action_type", None) == "final_answer":
                    final_answer["text"] = getattr(action, "reasoning_summary", None)

    _drain(graph_app.stream(stream_input, config))

    guard = 0
    while _interrupted(graph_app, config) and guard < max_steps:
        guard += 1
        if on_human_review == "escalate":
            review = _pending_review(graph_app, config)
            return {
                "status": "needs_review",
                "thread_id": config["configurable"]["thread_id"],
                "review": review,
                "workspace_dir": workspace_dir,
            }
        if on_human_review == "approve":
            _approve_pending(graph_app, config)
        # "skip": resume without approving -> tool not executed
        _drain(graph_app.stream(None, config))

    final = graph_app.get_state(config)
    values = final.values if final else {}
    return {
        "status": "done",
        "analysis_runs": values.get("analysis_runs", []) or [],
        "final_answer": final_answer["text"],
        "data_versions": values.get("data_versions", []) or [],
        "active_data_version_id": values.get("active_data_version_id"),
        "steps": values.get("current_step"),
        "workspace_dir": workspace_dir,
    }


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def run_headless(question, df, workspace_dir=None, max_steps=20,
                 on_human_review="approve", thread_id=None, on_event=None) -> Dict[str, Any]:
    """Start a headless analysis. Returns status 'done' (with analysis_runs) or,
    under on_human_review='escalate', status 'needs_review' (resume via
    resume_headless).

    on_event (optional): per-step callback for surfacing the agent's tool
    selection / reasoning -- see _drive.
    """
    from core.graph import app as graph_app
    if workspace_dir is None:
        workspace_dir = tempfile.mkdtemp(prefix="statguard_")
    state_input = _prepare_state_input(question, df, workspace_dir, max_steps)
    config = {"configurable": {"thread_id": thread_id or f"dg_{uuid.uuid4().hex[:8]}"}}
    return _drive(graph_app, config, state_input, on_human_review, max_steps,
                  workspace_dir, on_event=on_event)


def resume_headless(thread_id, approved, workspace_dir, max_steps=20,
                    on_human_review="escalate") -> Dict[str, Any]:
    """Resume a paused analysis after the user decided in DecisionGuard.
    approved=True  -> the pending tool runs; approved=False -> it is skipped.
    May return another 'needs_review' if a later step also needs confirmation."""
    from core.graph import app as graph_app
    config = {"configurable": {"thread_id": thread_id}}
    if approved:
        _approve_pending(graph_app, config)
    # If not approved, leave verification as needs_review; resuming lets the
    # human_review node record the skip and the supervisor moves on.
    final_answer = {"text": None}

    def _drain(stream):
        for event in stream:
            for _node, state_data in event.items():
                if not isinstance(state_data, dict):
                    continue
                action = state_data.get("current_action")
                if getattr(action, "action_type", None) == "final_answer":
                    final_answer["text"] = getattr(action, "reasoning_summary", None)

    _drain(graph_app.stream(None, config))

    guard = 0
    while _interrupted(graph_app, config) and guard < max_steps:
        guard += 1
        if on_human_review == "escalate":
            return {
                "status": "needs_review",
                "thread_id": thread_id,
                "review": _pending_review(graph_app, config),
                "workspace_dir": workspace_dir,
            }
        if on_human_review == "approve":
            _approve_pending(graph_app, config)
        _drain(graph_app.stream(None, config))

    final = graph_app.get_state(config)
    values = final.values if final else {}
    return {
        "status": "done",
        "analysis_runs": values.get("analysis_runs", []) or [],
        "final_answer": final_answer["text"],
        "data_versions": values.get("data_versions", []) or [],
        "active_data_version_id": values.get("active_data_version_id"),
        "steps": values.get("current_step"),
        "workspace_dir": workspace_dir,
    }