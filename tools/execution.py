from core.analysis_tool_plugins.execution import execute_analysis_tool


def execute_tool(action, context_pkg):
    """
    Legacy adapter.

    New execution logic lives in:
        core.analysis_tool_plugins.execution.execute_analysis_tool

    This wrapper keeps existing graph imports working during migration.
    """
    return execute_analysis_tool(action, context_pkg)