# App V3 Result Interpretation Contract

App V3 displays analysis results through deterministic insight cards.

## Purpose

Each analysis run should be shown to the user as:

```text
What was computed
Key findings
Caveats
Recommended next steps
```

## Boundary

Insight cards are built by:

```python
core.ui_adapter.insight_cards
```

The UI may render insight cards, but it must not create statistical interpretation logic directly inside ui/app_v3.py.

## Current scope

The current insight layer is deterministic.

It does not call an LLM.

It uses:

```text
analysis_run.summary
analysis_run.message
analysis_run.metrics
analysis_run.guardrails
analysis_run.error_code
analysis_run.success
```

## Failed runs

Failed runs should be archived and explained.

A failed run should say:

```text
the tool did not complete successfully
the result should not be interpreted as a valid statistical result
what the user can check next
```

## Future scope

A later Insight Synthesizer may add LLM-based interpretation, but it must remain backend/controller-driven and should not be embedded in the Streamlit UI.