# Adding a New Analysis Method

This project uses a plugin-based analysis architecture.

A new statistical method should be added as a **tool + plugin pair**:

```text
Tool      = executes the analysis
Plugin    = converts tool output into AnalysisRun / report / guardrails
```

The generic reporting and dispacthing layers must stay method-agnostic

## 1. Files You Are Allowed to Change

For a normal new method, you should only need to change or add:

```text
tools/methods.py
core/analysis_plugins/plugins/<method_name>.py
tests/plugins/test_<method_name>.py
```

In most cases, do **NOT** modify:

```text
tools/methods.py
core/analysis_plugins/plugins/<method_name>.py
tests/plugins/test_<method_name>.py
```

## 2. Architecture Boundary

### Generic layer

The generic layer only handles dispatching and rendering.

These files should not know about any specific method:

```text
core/report_builder.py
core/analysis_runs.py
core/analysis_plugins/base.py
core/analysis_plugins/registry.py
```

They should not contain method-specific terms such as:

```text
regression
ANOVA
t-test
VIF
Breusch-Pagan
coef_table
diagnostic_flags
GPA
SATM
```

### Plugin layer

Method-specific logic belongings inside:

```text
core/analysis_plugins/plugins/<method_name>.py
```

A plugin us allowed to know:

```text
regression
ANOVA
t-test
VIF
Breusch-Pagan
coef_table
diagnostic_flags
GPA
SATM
```

## 3. Required Tool Output Shape

Each method in `tools/methods.py` should return a result shaped like this:

```text
{
    "success": True,
    "status": "ok",
    "message": "Analysis completed.",
    "payload": {
        # method-specific results
    },
    "artifacts": [],
}
```

The `payload` may contain raw method-specific output.

The plugin decides which parts of `payload` become:

```text
metrics      user-facing scalar results
tables       user-facing tabular results
metadata     internal values for guardrails/debugging
artifacts    files such as plots
```

## 4. Plugin File Template

Create one file:

```text
core/analysis_plugins/plugins/<method_name>.py
```

Use this structure:

```text
from typing import Any, Dict, Tuple

from core.analysis_plugins.base import (
    AnalysisPlugin,
    DisplayConfig,
    MetricDisplayConfig,
    TableDisplayConfig,
    compact_dict,
    format_number,
    format_p_value,
)
from core.analysis_plugins.registry import register_plugin


def extract_method_result(
    *,
    payload: Dict[str, Any],
    arguments: Dict[str, Any],
    default_title: str,
    default_summary: str,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Convert raw tool output into AnalysisRun components.

    Return:
        title, summary, metrics, tables, metadata
    """

    title = "Method Display Name"

    # User-facing scalar values.
    metrics = compact_dict({
        "statistic": payload.get("statistic"),
        "p_value": payload.get("p_value"),
    })

    # User-facing tables.
    tables: Dict[str, Any] = {}

    # Internal values used by guardrails/debugging.
    # These do not appear in the report by default.
    metadata = compact_dict({
        "raw_options": payload.get("raw_options"),
    })

    summary = "Completed method analysis."

    return title, summary, metrics, tables, metadata


METHOD_DISPLAY = DisplayConfig(
    metrics=MetricDisplayConfig(
        labels={
            "statistic": "Test statistic",
            "p_value": "p-value",
        },
        formatters={
            "statistic": lambda x: format_number(x, digits=4),
            "p_value": format_p_value,
        },
        order=[
            "statistic",
            "p_value",
        ],
    ),
)


PLUGIN = register_plugin(AnalysisPlugin(
    tool_name="run_method_name",
    display_name="Method Display Name",
    extractor=extract_method_result,
    guardrail_evaluators=[],
    display_config=METHOD_DISPLAY,
))
```

## 5. Metrics vs Metadata

### Metrics

Use `metrics` for values the user should see in the UI and report.

Examples:

```text
metrics = {
    "statistic": 2.31,
    "p_value": 0.018,
    "effect_size": 0.42,
}
```

Do not put internal/debug fields in `metrics`.

Bad:

```text
metrics = {
    "raw_model_object": ...,
    "debug_trace": ...,
    "internal_df": ...,
}
```

### Metadata

Use `metadata` for values needed by the system but not shown in the report by default.

Examples:

```text
metadata = {
    "n_eff": 216,
    "p_eff": 2,
    "method_options": {...},
}
```

Guardrails may read from `metadata`.

## 6. Display Formatting

Formatting belongs in the plugin, not in `report_builder.py`.

Use `DisplayConfig` to control labels, ordering, and formatting.

Example:

```text
METHOD_DISPLAY = DisplayConfig(
    metrics=MetricDisplayConfig(
        labels={
            "p_value": "p-value",
            "effect_size": "Effect size",
        },
        formatters={
            "p_value": format_p_value,
            "effect_size": lambda x: format_number(x, digits=3),
        },
        order=[
            "effect_size",
            "p_value",
        ],
    ),
)
```

For tables:

```text
METHOD_DISPLAY = DisplayConfig(
    tables={
        "result_table": TableDisplayConfig(
            column_labels={
                "term": "Term",
                "estimate": "Estimate",
                "p_value": "p-value",
            },
            column_formatters={
                "estimate": lambda x: format_number(x, digits=4),
                "p_value": format_p_value,
            },
            column_order=[
                "term",
                "estimate",
                "p_value",
            ],
        ),
    },
)
```

## 7. Guardrails

Guardrail functions should accept an `AnalysisRun` dictionary and return a list of findings.

Example:

```text
def check_small_sample(run: Dict[str, Any]) -> list[Dict[str, Any]]:
    metrics = run.get("metrics", {})
    metadata = run.get("metadata", {})

    findings = []

    n = metrics.get("n") or metadata.get("n")
    if n is not None and n < 30:
        findings.append({
            "finding_id": "small_sample",
            "category": "sample_size",
            "severity": "warning",
            "title": "Small sample size",
            "message": "The sample size is small, so inference may be unstable.",
            "evidence": {"n": n},
            "recommendation": "Interpret results cautiously or collect more data.",
        })

    return findings
```

Then attach it in the plugin:

```text
PLUGIN = register_plugin(AnalysisPlugin(
    tool_name="run_method_name",
    display_name="Method Display Name",
    extractor=extract_method_result,
    guardrail_evaluators=[
        check_small_sample,
    ],
    display_config=METHOD_DISPLAY,
))
```

## 8. Minimum Plugin Test

Create:

```text
tests/plugins/test_<method_name>.py
```

Example:

```text
from core.analysis_plugins import get_plugin


def test_method_plugin_builds_analysis_run():
    plugin = get_plugin("run_method_name")

    run = plugin.build_analysis_run(
        action_id="act_test",
        arguments={},
        data_version_id="raw_v1",
        status="ok",
        success=True,
        message="Test complete.",
        payload={
            "statistic": 2.34567,
            "p_value": 0.00001,
        },
        artifacts=[],
        observation_id="obs_test",
    )

    assert run["tool_name"] == "run_method_name"
    assert run["title"] == "Method Display Name"

    assert run["metrics"]["statistic"] == 2.34567
    assert run["metrics"]["p_value"] == 0.00001

    assert run["report_blocks"]

    metric_block = next(
        block for block in run["report_blocks"]
        if block["type"] == "metric_table"
    )

    labels = [row["label"] for row in metric_block["rows"]]

    assert "Test statistic" in labels
    assert "p-value" in labels
```

## 9. Validation Checklist

After adding a method, run:

```text
pytest tests/plugins/test_<method_name>.py -q
pytest tests/architecture -q
```

Then verify plugin discovery:

```text
python -c "from core.analysis_plugins import PLUGIN_REGISTRY; print(sorted(PLUGIN_REGISTRY.keys()))"
```

## 10. Final Rule

A new method should not require changing the generic report or dispatch system.

If you need to modify:

```text
core/report_builder.py
core/analysis_runs.py
core/analysis_plugins/base.py
core/analysis_plugins/registry.py
```

then the correct next step is not to patch those files directly.

Instead, ask:

```text
Is this a generic capability that all plugins may need?
```

If yes, add it carefully to the plugin framework.

If no, keep it inside the method plugin.