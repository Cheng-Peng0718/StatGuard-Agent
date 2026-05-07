# App V2 Manual Smoke Test Checklist

This checklist defines the first manual smoke test for `ui/app_v2.py`.

The goal is not to verify every UI detail. The goal is to verify that the new UI skeleton can talk to the backend through the approved adapter boundary:

```python
apply_ui_event_to_state
run_backend_turn
build_ui_snapshot
prepare_uploaded_dataset_state
```

The UI must not rely on legacy `app.py`.

# 0. Pre-flight

Before starting the manual UI test, run:

```PowerShell
pytest -q
```

Expected result:

```text
all tests passed
```

Also confirm the new UI compiles:

```PowerShell
python -m py_compile .\ui\app_v2.py
```

Expected result:

```text
no output / no error
```

# 1. Start App V2

Run:

```Powershell
streamlit run ui/app_v2.py
```

Expected:

```text
The app opens in the browser.
The page title is Analysis Agent V2.
No Python traceback appears.
No Streamlit runtime error appears.
```

Failure signs:

```text
ImportError
ModuleNotFoundError
KeyError in st.session_state
TypeError while building snapshot
Streamlit page does not load
```

# 2. Initial screen smoke

On first load, expected visible sections:

```text
Chat
Assistant Response
Plan
Human Review
Analysis Results
Dataset Upload
Data Versions
Repair / Debug
Audits
Raw UI snapshot
```

Expected initial state:

```text
No assistant response yet.
No pending plan.
No human review required.
No analysis results yet.
No dataset uploaded yet.
No data versions yet.
```

Failure signs:

```text
Any section crashes while rendering.
Raw UI snapshot is not JSON-renderable.
Human review incorrectly appears before any action.
Plan panel shows malformed data.
```

# 3. Dataset upload smoke

Prepare a small CSV file, for example:

```CSV
GPA,SATM,Sex
3.0,600,F
3.2,,M
,650,F
3.8,680,M
4.0,700,F
```

Upload it through the Dataset Upload panel.

Click:

```text
Load dataset
```

Expected:

```text
Assistant Response says the dataset loaded successfully.
Data Versions shows active version raw_v1.
Data Versions shows one data version.
uploaded_dataset_info shows filename, rows, columns.
Analysis Results remains empty.
No current_action/current_execution/current_verification is active.
```

Failure signs:

```text
Upload crashes.
active_data_version_id is None.
data_versions is empty.
dataset_profile is not created.
assistant_response is missing.
```

# 4. Advisory chat smoke

In chat input, enter:

```text
I want to do analysis to this dataset, what can I do?
```

Expected:

```text
Assistant gives an advisory response.
No tool executes.
Analysis Results remains empty.
No human review is required.
Runtime has no current action after response.
```

Failure signs:

```text
UI asks for unnecessary confirmation.
Tool execution starts unexpectedly.
Analysis runs appear even though user only asked for advice.
Assistant response is empty.
```

# 5. Plan request smoke

In chat input, enter:

```text
Could you make up a plan and tell me?
```

Expected:

```text
Plan panel shows a pending plan.
Plan has one or more steps.
Analysis Results remains empty.
No tool executes yet.
No human review is required unless a step is actually executed.
```

Failure signs:

```text
Plan is missing.
Tool executes immediately during plan_only request.
current_action remains stuck.
assistant_response is empty.
```

# 6. Run safe plan step smoke

Click:

```text
Run plan
```

Expected if the first ready step is a safe non-mutating tool such as `get_summary_stats`:

```text
The step executes.
Analysis Results shows one successful run.
The plan step is marked completed.
Execution audit status is ok.
Runtime current_action/current_execution/current_verification are cleared.
No human review required.
```

Failure signs:

```
Run plan does nothing.
Step remains not_started.
Analysis Results stays empty after execution.
Execution audit reports error.
Runtime fields remain stuck after summarize.
```

# 7. Human review smoke for clean_data

Use a dataset with missing values.

Ask for or run a plan step that uses:

```text
clean_data
```

Expected:

```text
Human Review panel appears.
It shows clean_data action.
It shows arguments such as action_type, strategy, columns.
Approve and Reject buttons appear.
Tool does not execute before approval.
```

Click:

```text
Approve
```

Expected after approval:

```text
clean_data executes.
Analysis Results shows clean_data success.
Data Versions now has a new active version, not raw_v1.
data_versions length increases.
Data audit log records a data version event.
Human Review disappears.
Runtime fields are cleared.
```

Failure signs:

```text
clean_data executes before approval.
Approve causes fingerprint gate block.
Human Review stays stuck after approval.
active_data_version_id becomes None.
data_versions does not append a new version.
```

# 8. Reject human review smoke

Trigger a clean_data human review again.

Click:

```text
Reject
```

Expected:

```text
Tool does not execute.
No new data version is created.
Human review does not become allowed.
The UI remains stable.
```

Failure signs:

```text
Reject still executes the tool.
Reject creates a new data version.
UI crashes after rejection.
```
# 9. Debug panel smoke

Open:

```text
Repair / Debug
Audits
Raw UI snapshot
```

Expected:

```text
All panels render JSON.
No SimpleNamespace or Python object repr appears.
No unserializable object error appears.
```

Failure signs:

```text
Object of type X is not JSON serializable.
SimpleNamespace(...) appears in UI snapshot.
ActionProposal(...) appears in UI snapshot.
Pydantic object leaks into snapshot.
```

# 10. Browser refresh smoke

Refresh the browser page.

Expected:

```text
Streamlit session state remains valid.
The app does not crash.
Current UI snapshot can still render.
```

Failure signs:

```text
KeyError from missing session_state key.
backend_state disappears unexpectedly.
ui_snapshot cannot rebuild.
```

# 11. Stop app

Stop Streamlit with:

```text
Ctrl + C
```

Expected:

```text
No cleanup error.
No background process remains stuck.
```

# Manual smoke pass criteria

This manual smoke test passes if:

```text
1. App V2 starts.
2. Dataset upload works.
3. Advisory message works.
4. Plan request works.
5. Safe plan execution works.
6. clean_data requires approval.
7. approval lets clean_data execute.
8. rejection does not execute clean_data.
9. UI snapshot/debug panels are JSON-safe.
10. Runtime fields do not get stuck.
```

# Manual smoke failure protocol

If a failure occurs:

```text
1. Do not patch UI immediately.
2. Copy the exact traceback or backend log.
3. Identify whether the failure is:
   - UI rendering issue
   - UI adapter issue
   - backend controller issue
   - graph node issue
   - plugin issue
   - state serialization issue
4. Add or update a backend/unit/integration test before patching production code.
```

Do not add business logic to `ui/app_v2.py` to hide backend bugs.

