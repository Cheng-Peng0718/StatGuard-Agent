# LLM Business Analytics Agent

A supervisor-driven **AI data analyst** that connects to CSV or SQL data sources, selects analysis tools dynamically, and generates structured analytical reports with data provenance.

This project is an MVP / research prototype focused on the intersection of **data analysis, SQL analytics, statistical tooling, and LLM agents**.

## Why this project exists

Most LLM data-analysis demos either generate ad hoc code or rely on rigid predefined workflows. This project explores a different architecture:

> An LLM acts as a data analyst sitting at a workbench.  
> It sees the current data context, previous observations, and available analysis tool cards, then chooses the next best tool.

The goal is not to replace analysts, but to build a safer and more transparent AI assistant for exploratory analytics and business reporting.

## Key Features

### Data sources

- Upload CSV / tabular datasets for DataFrame-based analysis
- Connect to local DuckDB databases
- Inspect SQL schema before querying
- Run safe read-only SQL queries
- Materialize selected SQL query results into workspace data versions

### Analysis tools

- Dataset inspection
- Summary statistics
- Missingness reporting
- Correlation matrix
- Linear regression
- Regression diagnostics
- Residual histogram
- Scatterplot generation
- Data cleaning with human review
- SQL schema inspection
- Safe SQL query execution
- SQL query result materialization
- Groupby business summaries such as revenue by region or segment

### Agent architecture

- Supervisor-driven tool selection
- Plugin-based analysis tools
- Tool cards describing tool usage and data-source requirements
- Verifier for schema and data-source checks
- Human-in-the-loop safeguards for mutating operations
- Structured ToolExecutionResult and Observation contracts
- Data version tracking after transformations or SQL materialization
- HTML report generation with SQL query transparency and data provenance

## Example Demo Flow

This demo uses a synthetic ecommerce DuckDB database generated locally.

### 1. Create demo database

```bash
python scripts/create_demo_ecommerce_db.py
```

This creates:

```text
demo_data/ecommerce_demo.duckdb
```

### 2. Start the app

```bash
streamlit run app.py
```

### 3. Try these prompts

```text
Inspect the SQL schema for demo_data/ecommerce_demo.duckdb
```

```text
Using demo_data/ecommerce_demo.duckdb, materialize a customer-level revenue dataset with customer_id, region, segment, number of orders, and total revenue.
```

```text
Compare total_revenue by region in the active dataset.
```

Expected behavior:

1. The agent inspects the SQL schema.
2. The agent writes a SQL query joining customers, orders, and order_items.
3. The SQL result is materialized as an active workspace dataset.
4. The agent uses a DataFrame analysis tool to summarize total revenue by region.
5. The report includes SQL provenance, data version information, and a groupby summary table.

## Architecture

```text
User request
  ↓
Context builder
  ↓
LLM Supervisor
  ↓
Tool-card-aware tool selection
  ↓
Verifier
  ↓
Human review if needed
  ↓
Analysis tool plugin execution
  ↓
Structured ToolExecutionResult
  ↓
Observation history
  ↓
Report generation / next Supervisor step
```

## Design Principles

- One LLM analyst brain, not a rigid planner/executor queue
- Tools define their own schemas and usage guidance
- The Supervisor uses tool cards instead of hard-coded tool rules
- SQL data is not copied wholesale into the workspace
- Large SQL sources are handled through query pushdown and selected materialization
- Mutating actions require human confirmation
- Analysis results are tied to data versions
- Reports show provenance, query transparency, and limitations

## SQL Analytics Design

SQL databases are treated as external data sources.

The agent does **not** copy the full database into the workspace. Instead:

1. `inspect_sql_schema` reads table and column metadata.
2. `run_sql_query` executes safe read-only SQL for previews and KPI summaries.
3. `materialize_sql_query_result` saves a selected SQL result as a workspace DataFrame data version when downstream analysis is needed.
4. DataFrame tools such as `groupby_summary` operate on the materialized active dataset.

## Tech Stack

- Python
- Streamlit
- LangGraph
- OpenAI API / LangChain
- DuckDB
- pandas
- statsmodels / scipy
- pytest

## Installation

```bash
git clone <your-repo-url>
cd <your-repo-name>

python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

Create a `.env` file for local development if needed:

```text
OPENAI_API_KEY=your_key_here
```

Do not commit `.env`.

## Run Tests

```bash
python -m pytest -q
```

## Project Status

This is an MVP / portfolio project, not a production system.

Current strengths:

- SQL-backed business analytics workflow
- Plugin-based tool architecture
- Data versioning and provenance
- Structured result contracts
- HTML reporting
- Test coverage for core workflows

Current limitations:

- The report currently emphasizes tool results more than executive business narrative
- Dashboard-style visualizations are planned but not fully developed
- Multi-table semantic relationship inference is basic
- Deployment and authentication are not production-ready
- The agent still requires guardrails for broader real-world SQL environments

## Roadmap

- Add request timeline to multi-turn reports
- Add stronger executive business summary generation
- Add KPI cards and dashboard-style report sections
- Add more business analytics tools
- Add optional chart generation for grouped summaries
- Improve SQL relationship inference
- Add deployment-ready configuration
- Add evaluation cases for tool selection quality

## Portfolio Positioning

This project demonstrates AI agent workflow design, LLM tool use, SQL analytics integration, data analysis automation, plugin architecture, data provenance, statistical/business reporting, and engineering discipline through tests and structured contracts.
