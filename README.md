[![DOI](https://zenodo.org/badge/1228703550.svg)](https://doi.org/10.5281/zenodo.20369635)

# StatGuard Agent

**An auditable statistical analysis framework that pairs LLM orchestration with a deterministic, cross-validated statistics engine.**

StatGuard Agent turns a natural-language analysis request into an end-to-end, reproducible statistical report. It is built on a deliberate separation of concerns:

- **The LLM orchestrates** — it reads the request, inspects the data, and decides *which* analysis to run next.
- **The deterministic engine computes** — every statistic is produced by hardcoded, plugin-based methods that are cross-validated against `scipy` / `statsmodels`, never by the LLM itself.

This division is the core design principle. A general-purpose LLM asked to "compare these groups" may silently pick the wrong test, skip an assumption check, or report a number it did not actually compute — and may do so *differently every time it is run*. A traditional tool like SPSS is reproducible but cannot interpret an open-ended request. StatGuard Agent aims for both: **as adaptable as an LLM, as reproducible as a fixed statistical routine.**

---

## Why this project exists

Large language models are increasingly used as data analysts, but their statistical output is **stochastic and hard to reproduce**: the same data and the same question can yield different methods and different conclusions across runs, depending on whether the model happens to check an assumption that time. For low-stakes exploration this is tolerable; for research, clinical, or regulatory analysis, it is not.

StatGuard Agent addresses this by **removing statistical computation from the LLM entirely**. The LLM's role is reduced to routing — selecting an appropriate, pre-validated analysis tool — while the numbers come from deterministic plugins whose correctness is verified against independent reference implementations.

The result is a system where:

- the **same input always produces the same statistics** (deterministic engine),
- every reported number is **traceable to a specific tool run** (anti-fabrication claims ledger), and
- the **choice of method follows explicit, inspectable rules** (e.g. assumption checks that route between parametric and non-parametric tests).

---

## Key properties

- **Deterministic statistics.** All 27 analysis tools are implemented as plugins with hardcoded methods. The same data and arguments always produce identical results.
- **Independently cross-validated.** A 362-case "carpet" benchmark checks every plugin's output against `scipy` / `statsmodels` ground truth — currently **362 / 362 passing**.
- **Assumption-aware routing.** Tools encode statistical decision rules rather than leaving them to the LLM — for example, normality and variance checks that route between Welch's t-test, Alexander-Govern / Welch ANOVA, and rank-based non-parametric alternatives.
- **Replication-aware bootstrap inference.** A dedicated `bootstrap_inference` plugin provides confidence intervals for paired-difference statistics (mean / median / trimmed mean / Cohen's d_z) under three CI methods (percentile, basic, BCa) cross-validated against `scipy.stats.bootstrap`. An optional Sequential Bootstrap mode ([Peng 2025](https://arxiv.org/abs/2511.18065)) stabilises the resampler so the CI itself is reproducible across bootstrap RNG seeds — appropriate for regulatory / clinical / audit-grade contexts. Every run emits a cross-seed CI endpoint-stability diagnostic.
- **Anti-fabrication claims ledger.** Plugins emit structured, verified claims; the LLM may only *reference* them by ID, and a render layer substitutes the verified wording. The LLM cannot invent a statistic.
- **Reproducibility and provenance.** Analyses are tied to an explicit data version; the report records how the analysis-ready dataset was constructed (including SQL provenance when applicable).
- **Tested.** 764 deterministic unit/integration tests, runnable with no API key and no network access.

---

## Architecture in brief

```
natural-language request
        │
        ▼
   LLM Supervisor ──────────► chooses ONE next action at a time
        │                     (a tool call, or a final answer)
        ▼
 Deterministic plugin  ─────► runs a hardcoded statistical method,
        │                     cross-validated against scipy/statsmodels
        ▼
 Claims ledger + gate ──────► verifies evidence coverage and that every
        │                     reported number came from an actual run
        ▼
   Auditable report
```

The LLM never computes a statistic. It selects tools; the plugins compute and self-verify; a quality gate checks that the evidence required by the request has actually been produced before a final answer is allowed.

### Analysis tools (27 plugins)

Statistical inference and comparison: `statistical_group_comparison`, `run_independent_t_test`, `run_anova`, `nonparametric_group_comparison`, `paired_comparison`, `bootstrap_inference`, `run_correlation_test`, `run_chi_square`, `power_analysis`.

Modeling and diagnostics: `run_multiple_regression`, `regression_diagnostics`, `run_logistic_regression`.

---

## Installation

Requires Python 3.12 (3.12 is the tested reference; the deterministic engine has no exotic dependencies).

```bash
git clone https://github.com/<your-username>/statguard-agent.git
cd statguard-agent
pip install -r requirements.txt
```

A pinned `requirements.lock.txt` is also provided for fully reproducible environments.

---

## Verifying the installation

The deterministic core requires **no API key and no network access**. Two checks confirm a working install.

**1. Run the test suite (764 deterministic tests):**

```
python -m pytest -q
```

Expected: `764 passed`.

**2. Run the headless engine smoke test:**

```bash
python smoke_test.py
```

This invokes the statistics engine directly — without the UI and without an LLM — on a small in-memory dataset, and prints a Welch t-test result. Expected: `SMOKE TEST PASSED`.

### Reproducing with Docker

A `Dockerfile` is provided. To verify the full install and test suite in a clean environment:

```bash
docker build -t statguard-agent .
docker run --rm statguard-agent python -m pytest -q
docker run --rm statguard-agent python smoke_test.py
```

---

## Using the deterministic engine as a library

The statistics engine is callable directly, independent of the LLM and the UI. `smoke_test.py` is the minimal example; in short:

```python
import numpy as np
import pandas as pd
from core.analysis_tool_plugins.registry import get_plugin

df = pd.DataFrame({
    "score": np.concatenate([np.random.normal(100, 15, 45),
                             np.random.normal(108, 15, 45)]),
    "cohort": ["2024"] * 45 + ["2025"] * 45,
})

class Ctx:
    def __init__(self, df, args): self._df, self.arguments = df, args
    def load_df(self): return self._df
    def get_arg(self, name, default=None): return self.arguments.get(name, default)

result = get_plugin("statistical_group_comparison").run(
    Ctx(df, {"target_col": "score", "group_col": "cohort"})
)
print(result["details"]["method"], result["details"]["p_value"])
```

---

## Running the full agent (with an LLM)

The complete experience — natural-language requests, automatic tool selection, and report generation — requires an LLM backend. Set an OpenAI API key and launch the interface:

```bash
cp .env.example .env      # then add your OPENAI_API_KEY to .env
streamlit run app.py
```

Note: the LLM is used **only for orchestration** (deciding which tool to run). All statistics are still computed by the deterministic, cross-validated engine.

---

## The 362-case validation benchmark

The `benchmark/carpet/` suite generates 362 statistical scenarios with known ground truth and runs each through its plugin, checking the plugin's output against an independent `scipy` / `statsmodels` computation:

```
python -m benchmark.carpet.run_plugin_carpet
```

This is the primary evidence for the engine's correctness: every statistic the framework reports has been checked, case by case, against a reference implementation.

An end-to-end variant, `benchmark/carpet/run_e2e_carpet.py`, drives the full agent (with `gpt-4o`) on a representative subset of the same matrix and checks **routing** (the LLM selects the right plugin), **honesty** (claims ledger is clean), and **no-error** (the agent reaches a final answer). It also verifies plugin-specific routing decisions — for example, whether regulatory or audit-grade prompt framing causes `bootstrap_inference` to select Sequential Bootstrap. Current pass rate: **42 / 42 cases** pass all four dimensions. Requires an `OPENAI_API_KEY`.

---

## Relationship to prior and concurrent work

The idea that LLM-driven analysis should be constrained for reproducibility and auditability is an active research direction. Recent work includes frameworks for reproducibility-constrained execution of LLM/agent workflows, and empirical studies documenting that LLM data analysis is not reproducible across runs. StatGuard Agent is complementary and distinct in its **focus on statistical inference specifically**, and in **validating statistical correctness** (not merely execution traceability) against reference implementations across a 362-case benchmark. Related work is discussed in the accompanying paper.

---

## Project status and scope

StatGuard Agent is complementary and distinct in its **focus on statistical inference specifically**, and in **validating statistical correctness** (not merely execution traceability) against reference implementations across a 356-case benchmark.

**What it covers.** The framework targets standard univariate statistical inference and ordinary least squares regression: group comparisons (parametric and non-parametric, with post-hoc tests and multiple-comparison correction), correlation, chi-square association, paired comparisons, multiple linear regression with diagnostics and robust standard errors, power analysis, and the supporting description, data-preparation, and reporting tools. This range covers the large majority of everyday applied statistical analysis. It is a deliberate design choice — reproducible scientific analysis should use standard, validated methods rather than code improvised per request — and the trade-off is that the framework does not attempt to handle arbitrary or non-standard analyses.

**Graceful handling of edge cases.** Because tool arguments are supplied by the LLM orchestrator, plugins are written defensively: missing or non-existent columns, non-numeric data, degenerate inputs (empty groups, single rows, all-missing data, constant columns, infinities), and invalid parameters return a structured, explanatory result rather than crashing. This behaviour is verified by a dedicated robustness test suite (`tests/test_plugin_robustness.py`).

**Roadmap.** Planned additions include two-way / factorial ANOVA with interaction effects, and broadened effect-size and diagnostic reporting. Contributions, issues, and independent validation are welcome via the issue tracker.

---

## Related work by the author

- Peng, C. (2025). *Sequential Bootstrap for Out-of-Bag Error Estimation: A 100-Seed Replication Study and Variance-Structure Analysis.* arXiv:2511.18065. <https://arxiv.org/abs/2511.18065>

  This methodological study motivates StatGuard's Sequential Bootstrap mode for `bootstrap_inference`. It establishes empirically that the resampler-side variance term in the bootstrap variance decomposition — $\mathrm{Var}(\mathbb{E}[\widehat{\theta}_b \mid U_b])$ — is a non-trivial contributor to cross-seed CI variability on real-world datasets, and that holding the distinct-sample count $U_b$ at a fixed target produces CIs that are more reproducible across bootstrap RNG seeds.

## Author

Cheng Peng, Independent Researcher.

## License

Released under the MIT License. See [LICENSE](LICENSE).
