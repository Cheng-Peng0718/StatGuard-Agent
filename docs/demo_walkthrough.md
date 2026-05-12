# Demo Walkthrough

This walkthrough demonstrates the SQL-backed business analytics workflow.

## Goal

Show that the agent can:

1. Inspect a SQL database schema.
2. Create a customer-level analytical dataset from SQL.
3. Register the SQL query result as an active workspace data version.
4. Analyze the active dataset with a DataFrame-based business analysis tool.
5. Export an HTML report with data provenance and SQL transparency.

## Setup

```bash
python scripts/create_demo_ecommerce_db.py
streamlit run app.py
```

## Prompt 1: SQL Schema Inspection

```text
Inspect the SQL schema for demo_data/ecommerce_demo.duckdb
```

Expected behavior:

- Tool: `inspect_sql_schema`
- Output: tables, columns, row counts

## Prompt 2: SQL Materialization

```text
Using demo_data/ecommerce_demo.duckdb, materialize a customer-level revenue dataset with customer_id, region, segment, number of orders, and total revenue.
```

Expected behavior:

- Tool: `materialize_sql_query_result`
- The SQL result becomes the active workspace dataset.

Expected output columns:

- `customer_id`
- `region`
- `segment`
- `number_of_orders`
- `total_revenue`

## Prompt 3: Business Groupby Analysis

```text
Compare total_revenue by region in the active dataset.
```

Expected behavior:

- Tool: `groupby_summary`
- Grouping column: `region`
- Value column: `total_revenue`
- The report should show count, sum, mean, and median revenue by region.

## What this demo proves

```text
DuckDB database
→ schema inspection
→ selected SQL query
→ workspace data version
→ groupby business summary
→ HTML report
```

## Screenshots to Capture

1. Main Streamlit interface
2. SQL schema inspection result
3. Materialized data version
4. Groupby summary output
5. Final HTML report
