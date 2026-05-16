<img width="1448" height="1086" alt="Structure" src="https://github.com/user-attachments/assets/27b80f08-30a1-42bb-98c9-3615d8b21c3c" />

# SQL-Connected AI Data Analyst Agent

An AI data analyst agent that turns natural-language analysis requests into end-to-end statistical analysis reports.

This project is not just an LLM-to-SQL chatbot.

The goal is to build an analyst-style workbench where an LLM supervisor can inspect the user request, understand the available data, choose appropriate analysis tools, run statistical methods, track evidence coverage, and generate a report grounded in actual computations.

## Demo Concept

Imagine a data analyst sitting at a desk.

A supervisor is standing beside them.

On the desk, there are many tools: SQL inspection, data materialization, KPI summaries, statistical tests, regression models, diagnostics, guardrails, and report generation.

The analyst does not blindly follow a fixed checklist.

Instead, the supervisor looks at the question, checks the data, decides what evidence is needed, and chooses the right tool at the right time.

That is the core idea behind this project.

## What This Agent Can Do

Given a request such as:

> Inspect the SQL data in `demo_data/ecommerce_demo.duckdb`, then analyze the ecommerce database end-to-end. Build an analysis-ready customer-level dataset, summarize business KPIs, compare whether revenue differs across customer segments and regions, model the drivers of customer revenue, diagnose the model, and generate an executive report with key findings, limitations, and recommended next steps.

The agent can automatically:

1. Inspect the SQL database schema.
2. Materialize an analysis-ready customer-level dataset from SQL.
3. Track the active data version and SQL provenance.
4. Summarize business KPIs.
5. Compare revenue across customer segments using ANOVA.
6. Compare revenue across regions using ANOVA.
7. Fit a multiple regression model for revenue drivers.
8. Interpret numeric and categorical regression coefficients.
9. Run regression diagnostics, including VIF and Breusch-Pagan tests.
10. Surface statistical guardrails and limitations.
11. Generate an executive HTML report with findings, limitations, and recommended next steps.

## Example Output

The ecommerce showcase produces a report with sections such as:

- Executive Summary
- Data Provenance
- SQL Schema Summary
- SQL Query Materialization
- KPI Summary
- Statistical Group Comparison by Segment
- Statistical Group Comparison by Region
- Multiple Linear Regression
- Coefficient Interpretations
- Regression Diagnostics
- Statistical Guardrails
- Notes and Limitations

Example insights from the demo:

- Customer-level total revenue was analyzed using a materialized dataset of 98 customers.
- Revenue differences across customer segments were tested using one-way ANOVA.
- Revenue differences across regions were tested using one-way ANOVA.
- A multiple regression model explained around 61.6% of the variation in customer revenue.
- `number_of_orders` was identified as a statistically significant predictor of `total_revenue`.
- Regression diagnostics detected possible heteroscedasticity, which was surfaced as a statistical guardrail.

## Why This Project Is Different

Many LLM data tools focus on answering a single SQL question.

This project focuses on a broader problem:

> How can an AI agent behave more like a real data analyst?

Real data analysis is adaptive.  
You inspect the data, build an analysis-ready dataset, choose methods, check assumptions, revise based on results, and communicate limitations.

To support that, this project uses an evidence-driven agent architecture rather than a rigid workflow pipeline.

## Core Architecture

### 1. LLM Supervisor

The supervisor is responsible for choosing one next action at a time.

It reads:

- user request
- current data context
- previous observations
- active data version
- available tool cards
- analysis coverage feedback

Then it chooses either:

- one tool call, or
- a final answer

The supervisor does not create rigid task queues or executable workflow plans.

### 2. Plugin-Based Analysis Tools

Each analysis tool is implemented as a plugin with metadata such as:

- tool name
- description
- argument schema
- usage guidance
- data source requirements
- whether it produces an active dataset
- evidence categories
- reporting configuration

This makes the system easier to extend. New tools can be added without rewriting the core agent loop.

Current tools include:

- SQL schema inspection
- SQL query materialization
- KPI summary
- missingness and data quality checks
- group-by summaries
- statistical group comparison
- multiple regression
- regression diagnostics
- summary statistics
- dataset inspection

### 3. Evidence Coverage Instead of Rigid Workflows

For broad analysis requests, the agent generates an analysis coverage brief.

The brief describes what kinds of evidence are needed before the final answer is complete, such as:

- KPI summary
- group comparison
- regression model
- regression diagnostics

Each tool declares what evidence it can produce.

The answer quality gate compares:

```text
required evidence
vs.
actual evidence produced by tool runs
```

This allows the agent to continue analysis when important evidence is missing, without forcing every request through a fixed workflow.

### 4. Data Version Tracking

When the agent materializes a dataset from SQL, it creates an active data version.

Analysis results are tied to that data version.

This helps prevent stale results from being reused after the data changes.

The report includes:

- active data version
- parent version
- number of rows
- number of columns
- operation that produced the dataset
- SQL query used for materialization

### 5. SQL Provenance

The agent records the SQL query used to create the analysis-ready dataset.

This makes the final report auditable.

Instead of only showing final statistics, the report shows how the analysis dataset was constructed from the source database.

### 6. Model-Spec Handoff

The regression tool stores a model specification after fitting a model.

Regression diagnostics can then consume the previous regression model contract instead of relying on the LLM to manually reconstruct feature columns.

This avoids common mistakes such as passing encoded coefficient terms as raw dataset columns.

### 7. Statistical Guardrails

The report surfaces statistical warnings and limitations, such as:

- observational analysis does not imply causation
- ANOVA assumptions and follow-up testing requirements
- regression assumptions
- multicollinearity diagnostics
- heteroscedasticity warnings
- data version dependency

The goal is to make the agent's output more statistically honest and auditable.

## Tech Stack

- Python
- Streamlit
- LangGraph
- OpenAI API
- DuckDB
- pandas
- statsmodels
- pytest

## Example Showcase Flow

```text
The ecommerce demo follows this sequence:

User request
    ↓
SQL schema inspection
    ↓
Customer-level dataset materialization
    ↓
KPI summary
    ↓
Revenue comparison by customer segment
    ↓
Revenue comparison by region
    ↓
Multiple regression model
    ↓
Regression diagnostics
    ↓
Evidence-based executive report
```

## Project Structure

```text
analysis_agent_mvp/
│
├── app.py
├── agents/
│   ├── supervisor.py
│   └── coverage_brief.py
│
├── core/
│   ├── graph.py
│   ├── schema.py
│   ├── context_builder.py
│   ├── deliverables.py
│   ├── analysis_coverage.py
│   ├── report_builder.py
│   └── analysis_tool_plugins/
│       ├── base.py
│       ├── registry.py
│       ├── execution.py
│       └── plugins/
│           ├── inspect_sql_schema.py
│           ├── materialize_sql_query_result.py
│           ├── kpi_summary.py
│           ├── statistical_group_comparison.py
│           ├── linear_model.py
│           ├── regression_diagnostics.py
│           ├── missingness_report.py
│           └── ...
│
├── demo_data/
│   └── ecommerce_demo.duckdb
│
├── tests/
└── README.md
```

## Quick Start

### 1. Clone the repository

```Bash
git clone https://github.com/Cheng-Peng0718/Analysis_Agent_MVP.git
cd Analysis_Agent_MVP
```

### 2. Create a virtual environment

```Bash
python -m venv venv
```

On Windows:

```Bash
venv\Scripts\activate
```

On macOS/Linux:

```Bash
source venv/bin/activate
```

### 3. Install dependencies

```Bash
pip install -r requirements.txt
```

### 4. Set environment variables

Create a `.env` file or set your environment variables manually:

```Bash
OPENAI_API_KEY=your_api_key_here
```

Optional:

```Bash
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_TRACING_V2=true
```

### 5. Run the app

```Bash
streamlit run app.py
```

## Demo Prompt

Use this prompt in the app:

```text
Inspect the SQL data in demo_data/ecommerce_demo.duckdb, then analyze the ecommerce database end-to-end. Build an analysis-ready customer-level dataset, summarize business KPIs, compare whether revenue differs across customer segments and regions, model the drivers of customer revenue, diagnose the model, and generate an executive report with key findings, limitations, and recommended next steps.
```

## Testing

Run the main test suite:

```Bash
python -m pytest -q
```

Run core showcase-related tests:

```Bash
python -m pytest tests/test_analysis_coverage.py tests/test_answer_quality_gate.py tests/test_coverage_brief_agent.py tests/plugins/test_evidence_categories.py tests/plugins/test_regression_diagnostics_unified.py tests/plugins/test_linear_model_unified.py -q
```

## Current Status

This is an active MVP.

The current version demonstrates:

- SQL-connected analysis
- plugin-based tool execution
- evidence coverage checking
- data version tracking
- regression model-spec handoff
- statistical diagnostics
- HTML report generation

The project is still evolving. Planned improvements include:

- improved report UI
- richer visualization support
- logistic regression
- model comparison
- time-series analysis
- more robust data quality checks
- better automatic variable-role detection
- stronger support for user-provided datasets

## Design Philosophy

This project is built around a simple idea:

```text
A data analysis agent should not be a rigid workflow engine.
It should behave more like an analyst at a workbench.
```

The agent should understand the user's goal, inspect the available data, select the right tools, verify what evidence has been produced, and communicate results with statistical honesty.

## Author

Cheng Peng
M.S. Statistics, Michigan State University
Interested in statistics, machine learning, data science, and AI agents.
