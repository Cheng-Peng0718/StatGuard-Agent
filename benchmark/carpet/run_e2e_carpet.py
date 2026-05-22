"""
E2E carpet-bomb runner (USES gpt-4o -- costs money; run on your own machine).

The plugin-layer carpet (run_plugin_carpet.py) already proves the statistics are
correct across ~246 scenarios for free. This e2e layer does NOT re-verify the
numbers; instead it verifies the parts that only exist end-to-end:

  NO-ERROR  -- the full graph runs to a final_answer without raising.
  ROUTING   -- the LLM selected the right tool family for the task.
  HONESTY   -- the final answer cites claims and the claims validation is clean
               (no fabricated significance, no bare numbers, no unresolved ids).
  ACCURACY  -- the analysis_run actually recorded matches the scipy gold answer
               (this catches a wrong tool / wrong column silently producing the
               wrong number, which the plugin layer cannot see).

Cost control: a representative SUBSET of the matrix is selected (not all 246),
the per-case LLM-call count is bounded by MAX_STEPS, and a running cost estimate
is printed with a hard ceiling (--max-usd). It stops before exceeding the budget.

Usage:
    export OPENAI_API_KEY=...
    python -m benchmark.carpet.run_e2e_carpet                 # default subset
    python -m benchmark.carpet.run_e2e_carpet --max-usd 5     # set a ceiling
    python -m benchmark.carpet.run_e2e_carpet --list          # show subset only
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import tempfile
import uuid
from typing import Any, Dict, List, Optional

from benchmark.carpet.case_matrix import Case, generate_all_cases


class _Tee:
    """Mirror everything written to stdout/stderr into a transcript file, so
    the full PowerShell output (every supervisor decision, every gate dump,
    every attempt N/10 loop) is captured to disk and never lost to the
    terminal scrollback buffer."""

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, data):
        self._stream.write(data)
        try:
            self._fh.write(data)
        except Exception:
            pass

    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass
        try:
            self._fh.flush()
        except Exception:
            pass


# ---- cost model (gpt-4o, 2026 rates: $2.50/M in, $10/M out) ----
USD_PER_M_INPUT = 2.50
USD_PER_M_OUTPUT = 10.0
# Conservative per-call token estimates (input dominated by prompt + context).
EST_INPUT_TOKENS_PER_CALL = 3500
EST_OUTPUT_TOKENS_PER_CALL = 300
EST_CALLS_PER_CASE = 5  # coverage brief + ~4 supervisor decisions

def _est_case_usd() -> float:
    inp = EST_INPUT_TOKENS_PER_CALL * EST_CALLS_PER_CASE
    out = EST_OUTPUT_TOKENS_PER_CALL * EST_CALLS_PER_CASE
    return inp / 1e6 * USD_PER_M_INPUT + out / 1e6 * USD_PER_M_OUTPUT


# ---- task -> natural-language prompt ----
def _prompt_for(case: Case) -> str:
    cols = list(case.df.columns)
    if case.task == "group":
        return (f"Compare `{case.args['target_col']}` across the groups in "
                f"`{case.args['group_col']}` and tell me whether the difference "
                f"is statistically significant.")
    if case.task == "regression":
        return (f"Fit a regression predicting `{case.args['target_col']}` from "
                f"`{case.args['feature_cols'][0]}` and describe the relationship.")
    if case.task == "correlation":
        return (f"Is there a correlation between `{case.args['x_col']}` and "
                f"`{case.args['y_col']}`?")
    if case.task == "chi_square":
        return (f"Are `{case.args['row_col']}` and `{case.args['col_col']}` "
                f"associated?")
    if case.task == "paired":
        return (f"These are the same subjects measured as `{case.args['target_col_1']}` "
                f"and `{case.args['target_col_2']}`. Did the value change significantly?")
    return "Analyze this dataset."


# ---- representative subset selection ----
def select_subset() -> List[Case]:
    """Pick a small, diverse subset that exercises every task family and the
    key traps, instead of all 246 cases."""
    allc = {c.key: c for c in generate_all_cases()}
    picks: List[Case] = []

    def first_matching(pred, limit):
        out = []
        for c in allc.values():
            if pred(c):
                out.append(c)
            if len(out) >= limit:
                break
        return out

    # group: cover switch / no-switch, 2 & 3 groups, normal & skewed
    picks += first_matching(
        lambda c: c.task == "group" and c.expect.get("expect_nonparametric_switch") is True, 4)
    picks += first_matching(
        lambda c: c.task == "group" and c.expect.get("expect_nonparametric_switch") is False
                  and c.expect.get("n_groups") == 2, 3)
    picks += first_matching(
        lambda c: c.task == "group" and c.expect.get("n_groups") == 3, 3)
    picks += first_matching(
        lambda c: c.task == "group" and c.expect.get("n_groups") == 5, 2)
    # other tasks
    picks += first_matching(lambda c: c.task == "regression", 4)
    picks += first_matching(lambda c: c.task == "correlation", 3)
    picks += first_matching(lambda c: c.task == "chi_square", 3)
    picks += first_matching(lambda c: c.task == "paired", 4)
    # adversarial
    picks += first_matching(lambda c: c.key.startswith("adv_"), 4)

    # de-dup preserving order
    seen = set()
    uniq = []
    for c in picks:
        if c.key not in seen:
            seen.add(c.key)
            uniq.append(c)
    return uniq


# ---- workspace from a dynamic dataframe ----
def _make_workspace_from_df(case: Case) -> str:
    ws = tempfile.mkdtemp(prefix=f"carpet_{case.task}_")
    case.df.to_csv(os.path.join(ws, "working_data.csv"), index=False)
    case.df.to_parquet(os.path.join(ws, "working_data.parquet"), index=False)
    return ws


# ---- run the full graph (mirrors run_e2e_review._run_full_agent) ----
def _run_full_agent(case: Case, prompt: str) -> Dict[str, Any]:
    from core.graph import app as graph_app
    from benchmark.run_our_agent_e2e import MAX_STEPS

    workspace = _make_workspace_from_df(case)
    config = {"configurable": {"thread_id": f"carpet_{case.key}_{uuid.uuid4().hex[:6]}"}}
    initial_state = {
        "user_request": prompt, "workspace_dir": workspace,
        "max_steps": MAX_STEPS, "current_step": 0, "observations": [],
        "analysis_runs": [], "data_versions": [], "active_data_version_id": None,
        "data_audit_log": [], "deliverable_gate_attempts": 0,
        "deliverable_check": None, "task_contract": None, "dataset_profile": None,
    }
    final_state: Dict[str, Any] = {}
    guard = 0
    stream = graph_app.stream(initial_state, config=config)
    while True:
        guard += 1
        if guard > 50:
            break
        try:
            for event in stream:
                for _node, node_state in event.items():
                    if isinstance(node_state, dict):
                        final_state.update(node_state)
        except Exception as exc:
            final_state["_error"] = f"{type(exc).__name__}: {exc}"
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
            elif isinstance(vr, dict):
                vr = {**vr, "status": "allowed"}
            else:
                vr.status = "allowed"
            graph_app.update_state(config, {"current_verification": vr,
                                            "human_review_required": False})
            stream = graph_app.stream(None, config=config)
            continue
        stream = graph_app.stream(None, config=config)

    snapshot = graph_app.get_state(config)
    if snapshot and snapshot.values:
        final_state = dict(snapshot.values)
    final_state["_workspace"] = workspace
    return final_state


# ---- judge a finished run against gold + structural expectations ----
EXPECTED_TOOL_FAMILY = {
    "group": {"statistical_group_comparison", "nonparametric_group_comparison",
              "run_independent_t_test", "run_anova"},
    "regression": {"run_multiple_regression", "regression_diagnostics"},
    "correlation": {"run_correlation_test", "get_correlation_matrix"},
    "chi_square": {"run_chi_square"},
    "paired": {"paired_comparison"},
}


def _close(a, b, rel=1e-2, abs_=1e-2) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)
    except Exception:
        return False


def judge(case: Case, state: Dict[str, Any]) -> Dict[str, Any]:
    res = {"key": case.key, "task": case.task,
           "no_error": True, "routing": True, "honesty": True, "accuracy": True,
           "issues": []}

    if state.get("_error"):
        res.update(no_error=False, routing=False, honesty=False, accuracy=False)
        res["issues"].append(f"graph raised: {state['_error']}")
        return res

    runs = state.get("analysis_runs", []) or []
    tools_used = {r.get("tool_name") for r in runs if isinstance(r, dict)}

    # routing: did at least one expected-family tool run?
    if not (tools_used & EXPECTED_TOOL_FAMILY.get(case.task, set())):
        res["routing"] = False
        res["issues"].append(f"no expected tool used; got {tools_used}")

    # honesty: claims validation clean (if present)
    cv = state.get("claims_validation")
    if isinstance(cv, dict) and not cv.get("is_clean", True):
        res["honesty"] = False
        res["issues"].append(f"claims not clean: {cv}")

    # accuracy: the recorded run's primary p-value matches the gold answer for
    # whichever method was chosen.
    primary = None
    for r in runs:
        if isinstance(r, dict) and r.get("tool_name") in EXPECTED_TOOL_FAMILY.get(case.task, set()):
            primary = r
            break
    if primary is None:
        res["accuracy"] = False
        res["issues"].append("no primary analysis run to verify")
        return res

    m = primary.get("metrics", {}) or {}
    method = (m.get("method") or "").lower()
    p = m.get("p_value")
    g = case.gold

    if case.task == "group":
        if case.expect["n_groups"] == 2:
            target = g["mwu_p"] if "mann-whitney" in method else g["welch_p"]
        else:
            target = g["kruskal_p"] if "kruskal" in method else g["anova_p"]
        if not _close(p, target, rel=5e-2):
            res["accuracy"] = False
            res["issues"].append(f"p={p} != gold≈{target} (method={method})")
    elif case.task == "regression":
        if not _close(m.get("r_squared"), g["r_squared"], rel=5e-2):
            res["accuracy"] = False
            res["issues"].append(f"R2={m.get('r_squared')} != gold {g['r_squared']}")
    elif case.task == "correlation":
        if not _close(m.get("correlation"), g["r"], rel=5e-2):
            res["accuracy"] = False
            res["issues"].append(f"r={m.get('correlation')} != gold {g['r']}")
    elif case.task == "chi_square":
        if not _close(p, g["p"], rel=5e-2):
            res["accuracy"] = False
            res["issues"].append(f"chi p={p} != gold {g['p']}")
    elif case.task == "paired":
        target = g["wilcoxon_p"] if "wilcoxon" in method else g["paired_t_p"]
        if not _close(p, target, rel=5e-2):
            res["accuracy"] = False
            res["issues"].append(f"paired p={p} != gold≈{target} (method={method})")

    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-usd", type=float, default=8.0,
                    help="hard ceiling on estimated spend; stops before exceeding")
    ap.add_argument("--list", action="store_true", help="print subset and cost estimate, then exit")
    args = ap.parse_args()

    subset = select_subset()
    per_case = _est_case_usd()
    est_total = per_case * len(subset)

    print(f"Representative subset: {len(subset)} cases")
    print(f"Estimated cost: ~${per_case:.3f}/case  ->  ~${est_total:.2f} total "
          f"(ceiling ${args.max_usd:.2f})")
    by_task = {}
    for c in subset:
        by_task[c.task] = by_task.get(c.task, 0) + 1
    print("By task:", by_task)

    if args.list:
        for c in subset:
            print(f"  {c.key}")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        print("\nOPENAI_API_KEY not set. This runner drives gpt-4o; set the key and re-run.")
        sys.exit(1)

    # ---- mirror ALL terminal output to a transcript file from here on ----
    import datetime
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    run_ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    transcript_path = os.path.join(results_dir, f"e2e_carpet_{run_ts}_transcript.log")
    transcript_fh = open(transcript_path, "w", encoding="utf-8")
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(_orig_stdout, transcript_fh)
    sys.stderr = _Tee(_orig_stderr, transcript_fh)
    print(f"[transcript] mirroring all output to {transcript_path}")

    results = []
    spent = 0.0
    for i, case in enumerate(subset, 1):
        if spent + per_case > args.max_usd:
            print(f"\n[BUDGET] stopping before case {i}: would exceed ${args.max_usd:.2f}")
            break
        print(f"\n[{i}/{len(subset)}] {case.key} ...")
        prompt = _prompt_for(case)
        try:
            state = _run_full_agent(case, prompt)
            r = judge(case, state)
            # Capture context for the written report.
            r["prompt"] = prompt
            r["notes"] = case.notes
            action = state.get("current_action")
            r["final_answer"] = (getattr(action, "reasoning_summary", None)
                                 if action is not None else None)
            r["tools_used"] = sorted({
                run.get("tool_name") for run in (state.get("analysis_runs") or [])
                if isinstance(run, dict)
            })
            ws = state.get("_workspace")
            if ws and os.path.isdir(ws):
                shutil.rmtree(ws, ignore_errors=True)
        except Exception as exc:
            r = {"key": case.key, "task": case.task, "no_error": False,
                 "routing": False, "honesty": False, "accuracy": False,
                 "issues": [f"runner raised: {type(exc).__name__}: {exc}"],
                 "prompt": prompt, "notes": case.notes,
                 "final_answer": None, "tools_used": []}
        results.append(r)
        spent += per_case
        flags = [k for k in ("no_error", "routing", "honesty", "accuracy") if not r[k]]
        print(f"    {'PASS' if not flags else 'FAIL: ' + ','.join(flags)}")
        for iss in r["issues"][:3]:
            print(f"      - {iss}")

    # summary
    print("\n" + "=" * 60)
    print("E2E CARPET SUMMARY")
    print("=" * 60)
    n = len(results)
    for dim in ("no_error", "routing", "honesty", "accuracy"):
        ok = sum(1 for r in results if r[dim])
        print(f"  {dim:<10}: {ok}/{n}")
    print(f"\nEstimated spend: ~${spent:.2f}")
    fails = [r for r in results if any(not r[k] for k in ("no_error","routing","honesty","accuracy"))]
    if fails:
        print(f"\n{len(fails)} case(s) with issues:")
        for r in fails:
            print(f"  {r['key']}: {r['issues']}")

    # ---- persist results so they survive the terminal buffer ----
    import json
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    ts = run_ts

    summary = {dim: sum(1 for r in results if r[dim]) for dim in
               ("no_error", "routing", "honesty", "accuracy")}
    payload = {
        "timestamp": ts,
        "model": "gpt-4o",
        "n_cases": len(results),
        "estimated_spend_usd": round(spent, 2),
        "summary": summary,
        "results": results,
    }
    json_path = os.path.join(results_dir, f"e2e_carpet_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    md_path = os.path.join(results_dir, f"e2e_carpet_{ts}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# E2E Carpet Benchmark — {ts}\n\n")
        f.write(f"Model: gpt-4o  ·  Cases: {len(results)}  ·  "
                f"Est. spend: ${spent:.2f}\n\n")
        f.write("| dimension | pass |\n|---|---|\n")
        for dim, ok in summary.items():
            f.write(f"| {dim} | {ok}/{len(results)} |\n")
        f.write("\n## Per-case results\n\n")
        for r in results:
            flags = [k for k in ("no_error", "routing", "honesty", "accuracy") if not r[k]]
            status = "PASS" if not flags else "FAIL: " + ", ".join(flags)
            f.write(f"### {r['key']}  —  {status}\n\n")
            f.write(f"- task: {r['task']}  ·  {r.get('notes','')}\n")
            f.write(f"- prompt: {r.get('prompt','')}\n")
            f.write(f"- tools used: {r.get('tools_used', [])}\n")
            if r.get("issues"):
                f.write(f"- issues:\n")
                for iss in r["issues"]:
                    f.write(f"    - {iss}\n")
            fa = r.get("final_answer")
            if fa:
                f.write(f"- final answer:\n\n  > {fa}\n")
            f.write("\n")

    print(f"\nResults written:\n  {json_path}\n  {md_path}")

    # restore streams and close the transcript
    print(f"  {transcript_path}  (full terminal transcript)")
    sys.stdout = _orig_stdout
    sys.stderr = _orig_stderr
    try:
        transcript_fh.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()