# App V3 Report Export Contract

App V3 restores report download through a backend UI adapter.

## Backend adapter

Report export must go through:

```python
from core.ui_adapter.report_export import build_report_package_from_state
```

The adapter returns:

```text
markdown
html
plain_text_summary
metadata
```

# Existing report builder

The project already has:

```python
build_markdown_report
build_html_report_from_state
```

App V3 should not call these directly.

# UI boundary

App V3 may render:

```text
Download Markdown
Download HTML
Plain-text summary
```

App V3 must not:

```
construct markdown manually
construct HTML manually
call graph nodes
execute tools
alter analysis_runs
alter observations
```

# Assumption checks

Model assumption/diagnostic features are separate from report export.

Existing diagnostic components include:

```text
regression_diagnostics
residual_histogram
diagnostic guardrails
residual guardrails
```

These should be surfaced through the normal analysis_run/report_blocks/guardrails pipeline.