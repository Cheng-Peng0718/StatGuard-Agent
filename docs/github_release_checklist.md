# GitHub Showcase Release Checklist

## 1. Tests

```bash
python -m pytest -q
```

## 2. Secrets Check

PowerShell:

```powershell
Select-String -Path .\**\*.py,.\**\*.txt,.\**\*.md,.\**\*.json,.\**\*.toml -Pattern "sk-|OPENAI_API_KEY|LANGSMITH_API_KEY|password|secret|token"
```

Confirm no real keys are committed.

## 3. Files Not to Commit

- `.env`
- `venv/`
- `workspaces/`
- generated `.parquet`
- generated `.duckdb`
- API keys
- local logs

## 4. Demo Data

```bash
python scripts/create_demo_ecommerce_db.py
```

Do not commit generated `.duckdb` unless intentionally small and safe.

## 5. Screenshots

Recommended screenshots:

- App home / chat interface
- SQL schema result
- SQL materialization result
- Groupby summary result
- HTML report

Place them in:

```text
screenshots/
```

## 6. Git Commit and Tag

```bash
git status
git add -A
git commit -m "docs: prepare GitHub showcase release"
git tag showcase-v0.1
```

## Suggested GitHub Description

```text
SQL-backed LLM business analytics agent with plugin tools, data versioning, safe SQL execution, and HTML reports.
```

## Suggested Repo Topics

```text
llm-agent data-analysis business-analytics streamlit langgraph duckdb sql python openai analytics-engineering
```
