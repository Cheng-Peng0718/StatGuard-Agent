# requirements.txt Notes

Keep your actual `requirements.txt` generated from the working environment if possible.

Recommended command for a full lock file:

```bash
pip freeze > requirements.lock.txt
```

For a cleaner showcase `requirements.txt`, include only top-level dependencies used directly by the app, such as:

```text
streamlit
langgraph
langchain
langchain-openai
langsmith
openai
pandas
numpy
scipy
statsmodels
scikit-learn
matplotlib
duckdb
sqlglot
pyarrow
pydantic
pytest
python-dotenv
```

Then verify from a fresh environment:

```bash
python -m venv .venv_test
.venv_test\Scripts\activate
pip install -r requirements.txt
python -m pytest -q
streamlit run app.py
```
