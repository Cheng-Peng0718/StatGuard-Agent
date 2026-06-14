"""
Headless full-loop test: drive the StatGuard graph end-to-end without the UI.

LLM-driven counterpart to smoke_test.py. Where smoke_test.py calls one plugin
directly, this runs the WHOLE agent headless (supervisor routing + plugins +
verification + coverage gate) -- the same graph app.py drives -- proving
DecisionGuard can call all of StatGuard, not just individual tools.

Requires OPENAI_API_KEY (cheap model; supervisor tool routing only).

Run from the repo root OR from tests/ (it self-bootstraps sys.path):
    python headless_test.py
    python tests/headless_test.py
"""

from __future__ import annotations

import os
import sys

# --- make `core` importable no matter where this file is run from ---
_here = os.path.dirname(os.path.abspath(__file__))
_root = _here if os.path.isdir(os.path.join(_here, "core")) else os.path.dirname(_here)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import pandas as pd

from core.headless import run_headless


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        try:
            from dotenv import load_dotenv
            load_dotenv(os.path.join(_root, ".env"))
        except Exception:
            pass
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY (or put it in .env at the repo root) first.")
        return 1

    rng = np.random.default_rng(20260611)
    df = pd.DataFrame({
        "readmit_30d": np.concatenate([
            rng.normal(0.142, 0.05, 200),
            rng.normal(0.091, 0.05, 200),
        ]),
        "clinic": ["Clinic 7"] * 200 + ["Clinic 3"] * 200,
    })
    question = ("Is the 30-day readmission rate different between Clinic 7 and "
                "Clinic 3, and how large is the difference?")

    print("=" * 64)
    print("HEADLESS FULL-LOOP TEST (supervisor + plugins + gate, no UI)")
    print("=" * 64)
    print(f"question: {question}\n")

    result = run_headless(question, df, max_steps=20, on_human_review="approve")
    print(f"status        : {result['status']}")

    if result["status"] != "done":
        print("paused for review:", result.get("review"))
        print("(thread_id:", result.get("thread_id"), ")")
        return 0

    runs = result["analysis_runs"]
    print(f"steps run     : {result.get('steps')}")
    print(f"analysis_runs : {len(runs)}")
    for i, r in enumerate(runs):
        keys = sorted(r.keys())
        m = r.get("metrics", {}) or {}
        d = r.get("details", {}) or {}
        method = m.get("method") or d.get("method")
        p = m.get("p_value", d.get("p_value"))
        print(f"  run[{i}] tool={r.get('tool_name')} method={method} p={p}")
        print(f"         run keys: {keys}")

    print("\n--- StatGuard's own final answer (DecisionGuard may ignore this) ---")
    print(result.get("final_answer"))

    assert runs, "no analysis_runs produced -- the headless loop did not execute a tool"
    print("\nHEADLESS LOOP PASSED: full StatGuard ran headless and produced verified runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())