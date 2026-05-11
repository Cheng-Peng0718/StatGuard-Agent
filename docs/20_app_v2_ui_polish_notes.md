# App V2 UI Polish Notes

This document records the first App V2 UI cleanup pass.

## Goal

The goal of this pass is presentation polish only.

This pass must not move backend business logic into `ui/app_v2.py`.

## Changes

The UI now emphasizes:

```text
System status
Plan status
Execution status
Audit status
Current data version
Analysis run summaries
Human review prompts
```

Debug information remains available but should stay behind expandable panels:

```text
Repair / Debug
Audits
Raw UI snapshot
```

# Boundary

App V2 must continue to use:

```python
apply_ui_event_to_state
run_backend_turn
build_ui_snapshot
prepare_uploaded_dataset_state
```

App V2 must not directly call graph nodes such as:

```text
verify_node
execute_node
summarize_node
execute_pending_plan_node
plan_only_node
```

# Design principle

UI should show state clearly, not decide backend behavior.

If a problem appears in:

```text
planning
verification
execution
summarization
data versions
audit
```

the fix belongs in backend/controller/planning code, not in UI presentation code.