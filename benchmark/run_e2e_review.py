"""
End-to-end REVIEW generator -- FULL AGENT (LLM routing included).

Unlike run_our_agent_e2e.py (which auto-scores via keyword detectors and is
therefore noisy on free-text output), this script does NOT try to score
automatically. It runs the full LangGraph pipeline for each case and dumps a
clean, human-readable review sheet:

  - the natural-language prompt that was sent
  - which tools the LLM supervisor actually chose (and in what order)
  - the key statistical results each tool produced
  - the final report text
  - the rubric checklist for that case, left UNCHECKED for you to fill in

Why human scoring here: the same rubric will be applied to Julius / ChatGPT /
Auto-Analyst, which only produce free-text reports. To compare fairly, our
agent must be judged by the SAME human reading of its final output -- not by a
detector that has privileged access to our structured `details` dict. Two
products, one yardstick, applied the same way.

Usage:
    python -m benchmark.run_e2e_review                       # all cases
    python -m benchmark.run_e2e_review case4_nonnormal_two_group   # one case

Requires OPENAI_API_KEY (the supervisor uses gpt-4o).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import uuid
from typing import Any, Dict, List

from benchmark.rubric import RUBRICS
from benchmark.run_our_agent_e2e import CASE_PROMPTS, _make_workspace, MAX_STEPS


REVIEW_DIR = os.path.join(os.path.dirname(__file__), "review")
os.makedirs(REVIEW_DIR, exist_ok=True)


def _require_api_key() -> bool:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. This script drives gpt-4o; set the key and re-run.")
        return False
    return True


# ============================================================
# Drive the graph (same resume/auto-approve logic as the scorer)
# ============================================================

def _run_full_agent(case_key: str, prompt: str) -> Dict[str, Any]:
    from core.graph import app as graph_app

    workspace = _make_workspace(case_key)
    thread_id = f"review_{case_key}_{uuid.uuid4().hex[:6]}"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "user_request": prompt,
        "workspace_dir": workspace,
        "max_steps": MAX_STEPS,
        "current_step": 0,
        "observations": [],
        "analysis_runs": [],
        "data_versions": [],
        "active_data_version_id": None,
        "data_audit_log": [],
        "deliverable_gate_attempts": 0,
        "deliverable_check": None,
        "task_contract": None,
        "dataset_profile": None,
    }

    final_state: Dict[str, Any] = {}
    guard = 0
    stream = graph_app.stream(initial_state, config=config)

    while True:
        guard += 1
        if guard > 50:
            print(f"  [WARN] {case_key}: exceeded resume guard.")
            break
        try:
            for event in stream:
                for node_name, node_state in event.items():
                    if isinstance(node_state, dict):
                        final_state.update(node_state)
        except Exception as exc:
            print(f"  [ERROR] {case_key}: {type(exc).__name__}: {exc}")
            break

        snapshot = graph_app.get_state(config)
        if not snapshot.next:
            break

        if "human_review" in snapshot.next:
            vr = snapshot.values.get("current_verification")
            if vr is None:
                break
            if hasattr(vr, "model_copy"):
                vr = vr.model_copy(update={"status": "allowed"})
            elif hasattr(vr, "copy"):
                vr = vr.copy(update={"status": "allowed"})
            elif isinstance(vr, dict):
                vr = {**vr, "status": "allowed"}
            else:
                vr.status = "allowed"
            graph_app.update_state(config, {
                "current_verification": vr,
                "human_review_required": False,
            })
            stream = graph_app.stream(None, config=config)
            continue

        stream = graph_app.stream(None, config=config)

    snapshot = graph_app.get_state(config)
    if snapshot and snapshot.values:
        final_state = dict(snapshot.values)
    final_state["_workspace"] = workspace
    return final_state


# ============================================================
# Extract a readable summary of what happened
# ============================================================

# Fields worth showing in the review sheet (across all tool types).
_INTERESTING_KEYS = [
    "method", "test_family",
    "p_value", "significant_at_alpha", "alpha",
    "t_statistic", "F_statistic", "U_statistic", "H_statistic", "W_statistic",
    "degrees_of_freedom", "degrees_of_freedom_between", "degrees_of_freedom_within",
    "effect_size", "effect_size_name", "effect_size_magnitude",
    "effect_size_ci_low", "effect_size_ci_high",
    "cohens_d", "cohens_d_ci_low", "cohens_d_ci_high",
    "cohens_d_z", "hedges_g",
    "eta_squared", "eta_squared_ci_low", "eta_squared_ci_high", "omega_squared",
    "epsilon_squared", "rank_biserial_correlation",
    "hodges_lehmann_location_shift", "hodges_lehmann_pseudomedian",
    "mean_difference_ci_low", "mean_difference_ci_high",
    "r_squared", "adj_r_squared",
    "max_vif", "breusch_pagan_lm_p_value", "durbin_watson_statistic",
    "n_high_cooks_distance", "residuals_appear_normal_at_0_05",
    "recommended_test", "post_hoc_method",
]


def _extract_runs(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    runs = state.get("analysis_runs", []) or []
    out = []
    for r in runs:
        if not isinstance(r, dict):
            continue
        metrics = r.get("metrics", {}) or {}
        key_results = {k: metrics[k] for k in _INTERESTING_KEYS if k in metrics and metrics[k] is not None}
        # Guardrail finding titles
        gr = r.get("guardrails", []) or []
        finding_titles = [g.get("title") for g in gr if isinstance(g, dict) and g.get("title")]
        out.append({
            "tool_name": r.get("tool_name"),
            "title": r.get("title"),
            "status": r.get("status"),
            "key_results": key_results,
            "guardrail_findings": finding_titles,
        })
    return out


def _final_report_text(state: Dict[str, Any]) -> str:
    # The real final answer lives on the supervisor's final_answer action,
    # in reasoning_summary. By the time the graph ends, the deliverable gate
    # has substituted any [CLAIM:id] tokens with verified wording, so this is
    # the exact text the user sees.
    action = state.get("current_action")
    if action is not None:
        atype = getattr(action, "action_type", None)
        summary = getattr(action, "reasoning_summary", None)
        if atype == "final_answer" and isinstance(summary, str) and summary.strip():
            return summary.strip()

    # Fallbacks for other state shapes.
    for key in ["final_answer", "final_report", "report", "answer"]:
        val = state.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    obs = state.get("observations", []) or []
    for o in reversed(obs):
        if isinstance(o, dict):
            s = o.get("summary") or o.get("message")
            if s:
                return str(s).strip()
    return "(no final report text captured)"


# ============================================================
# Build the markdown review sheet
# ============================================================

def _build_review_md(case_key: str, prompt: str, state: Dict[str, Any]) -> str:
    rubric = RUBRICS[case_key]
    runs = _extract_runs(state)
    report = _final_report_text(state)

    lines: List[str] = []
    lines.append(f"# Review: {rubric['title']}")
    lines.append("")
    lines.append(f"**Case:** `{case_key}`  ")
    lines.append(f"**Dataset:** `{rubric['dataset']}`  ")
    lines.append(f"**The trap:** {rubric['trap']}")
    lines.append("")
    lines.append("## Prompt sent to the agent")
    lines.append("")
    lines.append(f"> {prompt}")
    lines.append("")

    lines.append("## Tools the LLM supervisor chose (in order)")
    lines.append("")
    if runs:
        for i, r in enumerate(runs, 1):
            lines.append(f"{i}. `{r['tool_name']}` — status: {r['status']}")
    else:
        lines.append("_(no analysis runs recorded)_")
    lines.append("")

    lines.append("## Key statistical results produced")
    lines.append("")
    for i, r in enumerate(runs, 1):
        lines.append(f"### {i}. {r['tool_name']}")
        if r["key_results"]:
            for k, v in r["key_results"].items():
                lines.append(f"- **{k}**: {v}")
        else:
            lines.append("- _(no key numeric results captured)_")
        if r["guardrail_findings"]:
            lines.append("")
            lines.append("  Guardrail findings:")
            for t in r["guardrail_findings"]:
                lines.append(f"  - {t}")
        lines.append("")

    lines.append("## Final report text")
    lines.append("")
    lines.append("```")
    lines.append(report[:4000])
    lines.append("```")
    lines.append("")

    lines.append("## Rubric — score by hand")
    lines.append("")
    lines.append("Mark each item PASS / MISS based ONLY on what a reader sees in the")
    lines.append("tools used + results + final report above. Apply the SAME standard")
    lines.append("to the competitor products.")
    lines.append("")
    for item in rubric["rigor_items"]:
        lines.append(f"- [ ] **{item.label}**")
        lines.append(f"      _Why it matters:_ {item.rationale}")
    lines.append("")
    lines.append(f"**Score: ___ / {len(rubric['rigor_items'])}**")
    lines.append("")

    return "\n".join(lines)


def main():
    if not _require_api_key():
        sys.exit(1)

    only = sys.argv[1] if len(sys.argv) > 1 else None
    cases = [only] if only else list(CASE_PROMPTS.keys())

    for case_key in cases:
        if case_key not in CASE_PROMPTS:
            print(f"Unknown case: {case_key}")
            continue
        prompt = CASE_PROMPTS[case_key]
        print(f"\n[{case_key}] running full agent...")
        state = _run_full_agent(case_key, prompt)

        md = _build_review_md(case_key, prompt, state)
        out_path = os.path.join(REVIEW_DIR, f"{case_key}_review.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"  tools used: {[r['tool_name'] for r in _extract_runs(state)]}")
        print(f"  review sheet -> {out_path}")

        ws = state.get("_workspace")
        if ws and os.path.isdir(ws):
            shutil.rmtree(ws, ignore_errors=True)

    print(f"\nAll review sheets written to: {REVIEW_DIR}")
    print("Fill them in by hand, then run the same prompts through the competitors.")


if __name__ == "__main__":
    main()