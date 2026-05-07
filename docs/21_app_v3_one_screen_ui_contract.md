# App V3 One-Screen UI Contract

App V3 is a full UI rewrite of App V2.

The goal is to keep the entire core workflow visible within one screen.

App V3 must not add backend business logic. It must continue to use:

```python
apply_ui_event_to_state
run_backend_turn
build_ui_snapshot
prepare_uploaded_dataset_state
```

# Layout

App V3 uses a fixed one-screen layout:

```text
Header / System Status
Left: Chat Panel
Center: Active Workspace
Right: Plan Timeline
Bottom: Action Bar
```

The page should avoid vertical scrolling for normal use.

# Chat Panel

The chat panel must have fixed height and internal scrolling.

The chat panel must not make the whole page longer.

The latest assistant response may use a typewriter-style display effect, but this is only a UI presentation effect.

# Active Workspace

The center panel displays exactly one active focus at a time.

Focus priority:

```text
1. Human Review
2. Required user choices
3. Failed latest result
4. Latest successful result
5. Pending plan overview
6. Dataset upload prompt
```

# Plan Timeline

The plan is displayed as a compact timeline, not as stacked full cards.

Each step should show:

```text
status icon
step title
tool name
current/active highlight
```

Status icons:

```text
completed -> solid circle
ready -> half circle
needs_user_choice -> warning / hollow circle
failed -> cross
not_started -> hollow circle
```

Only the active step should be expanded in the center panel.

# Bottom Action Bar

The bottom action bar is always visible.

It may show:

```text
Run next step
Save choices
Approve
Reject
Cancel
```

Human review actions must be visible both in the Active Workspace and the Bottom Action Bar.

# Human Review Priority

If human_review.required is true, App V3 enters Review Mode.

Review Mode must show:

```text
tool name
arguments
risk message
Approve button
Reject button
```

The user must not need to scroll to find approval controls.

# Analysis Result Display

App V3 should show:

```text
Latest result
Result history
Raw result/debug
```

The default view should be user-facing, not raw JSON.

Raw JSON should stay behind an expandable debug panel.

# Debug Information

Debug panels are allowed but must not dominate the main UI.

Debug sections:

```text
Raw UI snapshot
Audit state
Repair state
Backend state excerpt
```

These must be collapsed by default.

# Planner Behavior

The current planner may be deterministic for the same dataset and same user request.

Future planner work should separate:

```text
deterministic baseline planning
LLM-guided adaptive planning
Interpretation Layer
```

App V3 does not by itself create statistical insights.

A separate Result Interpreter / Insight Synthesizer layer is required.

Expected future interpretation output:

```text
What was computed
Key findings
Interpretation
Caveats
Recommended next step
```

# Forbidden Behavior

App V3 must not:

```text
directly call graph nodes
directly execute tools
directly mutate current_action/current_execution/current_verification
directly implement planning rules
directly implement data-version rules
directly implement human-review policy
hide backend failures with UI workarounds
```

# Implementation Plan

```text
S18A: document App V3 one-screen UI contract
S18B: create ui/app_v3.py skeleton
S18C: implement fixed layout shell
S18D: implement plan timeline
S18E: implement active workspace focus routing
S18F: implement bounded chat panel
S18G: implement human review priority mode
S18H: implement result history / latest result panel
S18I: move debug views behind collapsed panels
S19A: design result interpretation layer
S20A: design adaptive planner
```