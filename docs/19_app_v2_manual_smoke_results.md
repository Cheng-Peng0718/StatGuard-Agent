# App V2 Manual Smoke Results

This document records the first successful manual smoke run for `ui/app_v2.py`.

## Date

Manual smoke completed after App V2 was wired through:

```python
apply_ui_event_to_state
run_backend_turn
build_ui_snapshot
prepare_uploaded_dataset_state
```

# Confirmed working flow

The following App V2 flow has been manually verified:

```text
1. Start App V2 with streamlit run ui/app_v2.py
2. Upload CSV dataset
3. Dataset creates raw_v1 data version
4. Advisory response uses real dataset summary
5. Plan-only request creates a pending plan
6. Run plan executes safe EDA steps
7. User-choice steps expose variable selectors
8. clean_data requires user choices
9. clean_data triggers human review
10. Approving clean_data executes it
11. clean_data creates a new active data version
12. Execution audit status is ok
13. State serialization audit status is ok
```

# Verified safe EDA steps

The following steps were manually verified as executable without unnecessary user confirmation:

```text
get_summary_stats
missingness_report
get_correlation_matrix
```

Expected behavior:

```text
- verify_node returns allowed
- execute_node runs the plugin
- summarize_node archives the result
- pending_plan step is marked completed
- runtime fields are cleared after summarize
- execution_audit.status remains ok
```

# Verified user-choice behavior

The following tools require user-selected variables before execution:

```text
run_multiple_regression
run_anova
clean_data
```

Expected behavior:

```text
- UI shows required_user_choices
- UI renders choice widgets
- Save choices sends update_plan_step_choices event
- Backend updates the pending_plan step
- UI does not directly mutate pending_plan
```

# Verified clean_data behavior

The clean_data path was manually verified:

```text
1. User selects action_type, strategy, and columns.
2. Step becomes ready.
3. User clicks Run plan.
4. verify_node returns needs_review.
5. Human Review panel appears.
6. User approves.
7. clean_data executes.
8. active_data_version_id changes from raw_v1 to a child data version.
9. data_versions length increases.
10. execution_audit.status remains ok.
```

# Important design decisions confirmed

## Regression plugin should not silently clean data

Modeling plugins should not silently mutate or clean the active dataset.

If missing data must be handled by modifying the dataset, the plan should route through clean_data so that:

```text
- user choices are explicit
- human review happens
- a new data version is created
- data_audit_log records provenance
```

## UI must remain thin

App V2 should continue to follow this pattern:

```text
UIEvent -> apply_ui_event_to_state -> run_backend_turn -> UISnapshot
```

The UI must not directly implement:

```text
tool execution
verification
planning rules
data-version logic
human-review policy
deliverable logic
repair logic
```

# Current known non-blocking behavior

## Regression may fail on tiny test data

In the small manual smoke CSV, regression may fail due to insufficient sample size after valid-row requirements.

This is acceptable for the smoke test as long as:

```text
- the tool is called with correct arguments
- the failure is archived as an analysis run
- the plan step is marked failed
- the system continues to later steps
- execution_audit remains structurally ok
```

## deliverable_check may be null

`deliverable_check = null` is expected when the user has not requested a final report or deliverable.

Deliverable gate should run only when a task contract or final deliverable request exists.

# Pass criteria

The App V2 smoke flow is considered passing when:

```text
1. App starts without import errors.
2. Dataset upload creates raw_v1.
3. Advisory uses real row/column counts.
4. Plan-only creates pending_plan without executing tools.
5. Safe EDA steps execute.
6. User-choice steps expose controls.
7. clean_data requires human review.
8. Approval executes clean_data and creates a child data version.
9. execution_audit.status is ok.
10. state_serialization_audit.status is ok.
```

# Failure protocol

If this flow breaks later:

```text
1. Do not patch ui/app_v2.py first.
2. Identify whether the failure is in:
   - UI rendering
   - UI event adapter
   - backend controller
   - planning verifier
   - execution queue
   - summarize_node
   - plugin execution
   - audit/state serialization
3. Add a regression or integration test before patching production logic.
```