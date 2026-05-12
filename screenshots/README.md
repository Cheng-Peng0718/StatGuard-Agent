# LLM Business Analytics Agent

A supervisor-driven **AI business analytics agent** that connects to CSV or SQL data sources, selects analysis tools dynamically, materializes SQL query results into workspace data versions, and generates transparent analytical reports.

This project explores the intersection of:

- LLM agents
- SQL analytics
- data analysis automation
- statistical tooling
- business intelligence workflows
- human-in-the-loop analytical systems

> Instead of using a rigid predefined pipeline, the agent behaves like an analyst at a workbench: it reads the current data context, reviews previous observations, checks available tool cards, and chooses the next best analysis tool.

---

## Demo Preview

### SQL schema inspection

![SQL schema inspection](screenshots/02_sql_schema_inspection.png)

### SQL result materialization

![SQL materialization](screenshots/03_sql_materialization.png)

### Business groupby summary

![Groupby summary](screenshots/04_groupby_summary.png)

### HTML report with SQL provenance

![HTML report](screenshots/05_html_report.png)

---

## What This Agent Can Do

The current MVP supports a complete SQL-backed business analytics workflow:

```text
SQL database
→ schema inspection
→ safe SQL query generation
→ selected SQL result materialization
→ active workspace data version
→ DataFrame-based business analysis
→ HTML analytical report
```

Example user flow:

```text
Inspect the SQL schema for demo_data/ecommerce_demo.duckdb
```

```text
Using demo_data/ecommerce_demo.duckdb, materialize a customer-level revenue dataset with customer_id, region, segment, number of orders, and total revenue.
```

```text
Compare total_revenue by region in the active dataset.
```

The agent can inspect the database schema, join SQL tables, create a customer-level analytical dataset, register it as the active data version, and then run a groupby business summary on the materialized dataset.

---

## Key Features

### SQL Analytics

- Inspect DuckDB database schemas
- List tables, columns, data types, and row counts
- Run safe read-only SQL queries
- Block unsafe SQL operations such as `DROP`, `DELETE`, `INSERT`, `UPDATE`, and `ALTER`
- Materialize selected SQL query results into workspace data versions
- Preserve SQL query provenance in reports

### DataFrame Analysis

- Dataset inspection
- Summary statistics
- Missingness reporting
- Correlation matrix
- Correlation testing
- Linear regression
- Regression diagnostics
- Residual histogram
- Scatterplot generation
- Groupby business summaries
- Data cleaning with human review

### Agent Architecture

- Supervisor-driven LLM workflow
- Plugin-based analysis tools
- Tool cards for tool usage guidance
- Verifier for argument validation and data-source checks
- Structured `ToolExecutionResult` and `Observation` contracts
- Data version tracking
- Human-in-the-loop safeguards for mutating operations
- HTML report generation with data provenance

---

## Why This Project Is Different

Many LLM data-analysis demos either generate arbitrary code or follow a fixed workflow. This project uses a different design:

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
Tool plugin execution
  ↓
Structured observation
  ↓
Report generation / next step
```

The LLM does not directly execute arbitrary Python code. Instead, it chooses from validated analysis plugins. Each plugin defines:

- what it does
- when it should be used
- when it should not be used
- what arguments it requires
- what data source type it needs
- whether it produces a new active dataset

This keeps the system flexible while avoiding a brittle hard-coded pipeline.

---

## Design Principles

### 1. One analyst brain, many tools

The Supervisor acts as the only decision-making agent. It does not create a rigid executable plan queue. Instead, it chooses one next action at a time based on current context and tool cards.

### 2. Tools own their contracts

Analysis tools are implemented as plugins. Each tool carries its own schema, execution logic, and usage guidance.

### 3. SQL data is not copied blindly

For SQL databases, the agent does not copy full tables into the workspace. It uses SQL pushdown:

```text
filter
join
aggregate
select columns
materialize only the needed query result
```

This is closer to how real analytics systems handle large databases.

### 4. Results are tied to data versions

When a SQL query result or cleaned dataset becomes the active dataset, it receives a data version ID. Reports show which data version was used.

### 5. Reports include provenance

Generated reports include:

- active data version
- SQL source database
- SQL query used
- materialized dataset shape
- analysis tool outputs
- limitations and notes

---

## Example SQL Demo

### Step 1: Create the demo database

```bash
python scripts/create_demo_ecommerce_db.py
```

This creates a local DuckDB database:

```text
demo_data/ecommerce_demo.duckdb
```

The synthetic ecommerce database contains:

- `customers`
- `orders`
- `order_items`
- `products`

### Step 2: Start the app

```bash
streamlit run app.py
```

### Step 3: Run the demo prompts

#### Inspect schema

```text
Inspect the SQL schema for demo_data/ecommerce_demo.duckdb
```

Expected behavior:

```text
Tool: inspect_sql_schema
```

The agent should return tables and columns such as:

```text
customers(customer_id, region, segment, signup_month)
orders(order_id, customer_id, order_date, channel)
order_items(order_id, product_id, quantity, discount, net_revenue)
products(product_id, category, price)
```

#### Materialize customer-level revenue dataset

```text
Using demo_data/ecommerce_demo.duckdb, materialize a customer-level revenue dataset with customer_id, region, segment, number of orders, and total revenue.
```

Expected behavior:

```text
Tool: materialize_sql_query_result
```

The agent should generate a SQL query similar to:

```sql
SELECT
    c.customer_id,
    c.region,
    c.segment,
    COUNT(DISTINCT o.order_id) AS number_of_orders,
    SUM(oi.net_revenue) AS total_revenue
FROM customers c
JOIN orders o
    ON c.customer_id = o.customer_id
JOIN order_items oi
    ON o.order_id = oi.order_id
GROUP BY
    c.customer_id,
    c.region,
    c.segment
```

The result becomes the active workspace dataset.

#### Compare revenue by region

```text
Compare total_revenue by region in the active dataset.
```

Expected behavior:

```text
Tool: groupby_summary
```

The agent summarizes `total_revenue` by `region`, including count, sum, mean, and median.

---

## Example Report Output

The generated HTML report includes:

- analysis request
- active data version
- data audit trail
- SQL schema summary
- SQL query materialization details
- SQL query used
- groupby summary table
- notes and limitations

Example report sections:

```text
Data Provenance
- Active data version: data_v_xxxx
- Rows: 98
- Columns: 5
- Operation: materialize_sql_query_result

Analysis Results
1. Inspect SQL Database Schema
2. Materialize SQL Query Result
3. Groupby Summary
```

---

## Tech Stack

- Python
- Streamlit
- LangGraph
- LangChain / OpenAI API
- DuckDB
- SQLGlot
- pandas
- NumPy
- SciPy
- statsmodels
- scikit-learn
- matplotlib
- pytest

---

## Installation

Clone the repository:

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a local `.env` file if needed:

```text
OPENAI_API_KEY=your_openai_key_here
```

Do not commit `.env`.

---

## Running the App

```bash
streamlit run app.py
```

Then open the Streamlit URL in your browser.

---

## Running Tests

```bash
python -m pytest -q
```

The test suite covers core plugin behavior, SQL tool execution, result contracts, and data-source guardrails.

---

## Repository Structure

```text
analysis_agent_mvp/
├── app.py
├── agents/
│   └── supervisor.py
├── core/
│   ├── analysis_tool_plugins/
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── execution.py
│   │   ├── validation.py
│   │   ├── plugins/
│   │   │   ├── inspect_sql_schema.py
│   │   │   ├── run_sql_query.py
│   │   │   ├── materialize_sql_query_result.py
│   │   │   ├── groupby_summary.py
│   │   │   └── ...
│   │   └── shared/
│   │       └── sql_utils.py
│   ├── graph.py
│   ├── context_builder.py
│   ├── data_versions.py
│   ├── report_builder.py
│   └── schema.py
├── verifiers/
│   └── validators.py
├── scripts/
│   └── create_demo_ecommerce_db.py
├── demo_data/
│   └── .gitkeep
├── screenshots/
├── tests/
├── requirements.txt
└── README.md
```

---

## Core Components

### Supervisor

The Supervisor is the LLM decision-making layer. It receives:

- user request
- current data context
- recent observations
- available tool cards

It returns one next action:

- call a tool
- answer from existing observations
- explain a blocker or ask for missing information

### Tool Plugins

Each analysis tool is implemented as an `AnalysisToolPlugin`.

A plugin defines:

- tool name
- display name
- description
- usage guidance
- argument schema
- data-source requirement
- execution function
- examples

### Verifier

The verifier checks whether a proposed tool call is valid before execution.

It validates:

- tool existence
- required arguments
- argument types
- column references
- data-source availability
- human-review requirements for mutating tools

### Data Versions

When data is uploaded, cleaned, or materialized from SQL, it is tracked as a data version.

This prevents the agent from mixing results computed on different datasets.

### Reports

The report builder generates standalone HTML reports with:

- data provenance
- analysis results
- SQL query transparency
- summary tables
- generated artifacts
- limitations

---

## Current Status

This project is an MVP / portfolio project, not a production system.

It is currently strong enough to demonstrate:

- LLM tool-use workflow design
- SQL analytics integration
- business analysis automation
- plugin architecture
- data provenance
- structured result contracts
- Streamlit UI development
- pytest-based engineering discipline

---

## Limitations

Current limitations include:

- The executive summary is still relatively template-based.
- Multi-turn reports currently emphasize tool outputs more than narrative storytelling.
- Dashboard-style KPI cards and visual report sections are planned but not fully developed.
- SQL relationship inference is basic and depends on schema visibility.
- The app is designed for local/demo usage, not secure production deployment.
- Authentication, multi-user isolation, and production database credentials are not implemented.

---

## Roadmap

Planned improvements:

- Add multi-turn request timeline to reports
- Add stronger executive business summary generation
- Add KPI cards and dashboard-style report sections
- Add chart generation for grouped business summaries
- Add richer SQL relationship inference
- Add more business analytics tools
- Add deployment-ready configuration
- Add evaluation tests for tool-selection quality
- Improve report narrative and business recommendations

---

## Portfolio Use Case

This project is designed as a showcase for roles involving:

- data analysis + AI
- analytics engineering
- AI data tooling
- LLM applications
- business intelligence automation
- junior AI engineer / applied AI roles
- data science tooling

It demonstrates the ability to combine statistical/data analysis knowledge with modern LLM application engineering.

---

## Resume Summary

```text
Built a supervisor-driven LLM business analytics agent using Python, Streamlit, LangGraph, DuckDB, and OpenAI API. Integrated plugin-based tools for SQL schema inspection, safe SQL querying, SQL result materialization, DataFrame-based groupby analysis, data version tracking, and HTML report generation with SQL provenance.
```

---

## License

This project is intended for educational and portfolio use. Add a license file before public release if you plan to distribute or reuse it broadly.


## Screenshots

![App home](screenshots/01_app_home.png)
![SQL schema inspection](screenshots/02_sql_schema.png)
![SQL materialization](screenshots/03_materialized_dataset.png)
![Groupby summary](screenshots/04_groupby_summary.png)
![HTML report](screenshots/05_html_report.html)