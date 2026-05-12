# Resume Bullets

## Main Project Entry

**LLM Business Analytics Agent** | Python, Streamlit, LangGraph, DuckDB, OpenAI API, pandas, pytest

- Built a supervisor-driven LLM analytics agent that selects plugin-based tools using dataset context, observation history, and tool cards.
- Integrated DuckDB SQL tools for schema inspection, safe read-only querying, and SQL result materialization into workspace data versions.
- Implemented DataFrame-based business analysis tools, including groupby summaries for comparing metrics across regions, segments, and customer groups.
- Designed structured `ToolExecutionResult` and `Observation` contracts to support reliable tool execution, result reuse, and report generation.
- Added data provenance tracking and HTML report generation with SQL query transparency, active data version metadata, and business summary tables.
- Used pytest to validate SQL plugins, result contracts, data-source guardrails, and core analysis workflows.

## Shorter Version

- Developed an LLM-powered business analytics agent that connects to SQL databases, materializes selected query results, and performs DataFrame-based analysis through plugin tools.
- Added safe SQL execution, data versioning, structured observations, and HTML reporting with SQL provenance using Python, Streamlit, LangGraph, DuckDB, and pytest.
