# Plugin Method Template

This project uses a plugin-based analysis architecture.

Adding a new statistical method should not require modifying the generic report builder, analysis dispatcher, or plugin framework.

## Allowed files to change when adding a new method

For a normal new method, only these files should be changed:

1. `tools/methods.py`
2. `core/analysis_plugins/plugins/<method_name>.py`
3. `tests/plugins/test_<method_name>.py`

## Forbidden files for normal method additions

Do not modify these files just to add a new method:

- `core/report_builder.py`
- `core/analysis_runs.py`
- `core/analysis_plugins/base.py`
- `core/analysis_plugins/registry.py`

If a new method seems to require changing one of those files, stop and ask whether the plugin abstraction is missing a generic capability.

---

# Method Addition Checklist

## 1. Add the executable tool

Add the real execution function in:

```text
tools/methods.py