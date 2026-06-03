---
title: 'StatGuard Agent: An auditable statistical analysis framework pairing LLM orchestration with a deterministic, cross-validated statistics engine'
tags:
  - Python
  - statistics
  - reproducible research
  - large language models
  - statistical inference
  - data analysis
authors:
  - name: Cheng Peng
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 2 June 2026
bibliography: paper.bib
---

# Summary

`StatGuard Agent` is a Python framework for conducting statistical analysis from
natural-language requests while guaranteeing that every reported statistic is
deterministic, reproducible, and independently verified. It is built on a strict
separation of responsibilities: a large language model (LLM) acts only as an
*orchestrator* that interprets the request and selects which analysis to run,
while all statistical computation is performed by a fixed library of plugins
whose numerical output is cross-validated against `scipy` and `statsmodels`
[@virtanen2020scipy; @seabold2010statsmodels]. The LLM never computes a
statistic and cannot alter a numerical result.

This design targets a specific failure mode of using LLMs as data analysts: the
same request, on the same data, can produce different statistical methods and
different conclusions on different runs. By confining the LLM to routing and
delegating all computation to deterministic, pre-validated methods, `StatGuard
Agent` produces analyses that are identical across runs and auditable down to the
individual reported number.

# Statement of need

LLMs are increasingly used to perform data analysis by generating and executing
code. Their statistical output, however, is stochastic and difficult to
reproduce: recent empirical work shows that the same prompt and data can yield
materially different analytical pathways and conclusions across independent runs
[@cui2026same], and that reproducibility of LLM-generated data-science workflows
is generally low [@zeng2025airepr]. For exploratory work this variability may be
acceptable; for research, clinical, or regulatory analysis — settings where a
reported p-value or effect size must be defensible and reproducible — it is not.

A parallel line of work argues that LLM-driven scientific workflows must be
constrained for reproducibility and auditability, and proposes execution-level
frameworks that enforce deterministic, traceable execution of agent actions
[@sureshkumar2026rlam]. These frameworks establish that an analysis can be
*replayed*; they do not, however, verify that the statistics produced are
*correct* — that the appropriate test was chosen, that its assumptions were
checked, and that the resulting numbers match an independent reference
implementation.

`StatGuard Agent` addresses this gap for the specific domain of statistical
inference. Its contribution is not the general idea of constraining an LLM, but
the combination of (1) deterministic, assumption-aware statistical methods that
encode the kind of decisions a statistician makes, and (2) a validation regime
that checks those methods' numerical output, case by case, against reference
implementations. The framework is intended for researchers, graduate students,
and analysts who need LLM-level convenience in expressing an analysis but
require statistical results that are reproducible and independently verifiable —
a combination not offered by general-purpose LLM tools (which compute statistics
stochastically) nor by traditional statistical software such as SPSS or base
`scipy`/`statsmodels` (which are reproducible but require the user to specify the
method and check assumptions themselves).

# Design and key features

**LLM orchestration, deterministic computation.** A supervisor LLM selects one
analysis action at a time from a registry of 27 tool plugins. Each plugin
implements a hardcoded statistical method; the same data and arguments always
produce identical output. The LLM's role is limited to choosing which plugin to
invoke.

**Assumption-aware routing.** Method selection is governed by explicit,
inspectable rules rather than left to the LLM. For example, the group-comparison
plugin checks variance homogeneity (Levene's test) to route between Welch's
t-test and classical procedures, and switches from a parametric test to a
rank-based non-parametric alternative when normality is violated together with a
small sample (group *n* < 30) or strong skew (|skewness| $\ge$ 1.5).

**Independent cross-validation.** A 362-case benchmark (`benchmark/carpet/`)
generates statistical scenarios with fixed random seeds and compares each
plugin's output against an independent `scipy`/`statsmodels` computation of the
same quantity. All 362 cases currently pass, providing case-by-case evidence
that the framework's reported statistics are numerically correct. A complementary
end-to-end benchmark drives the full agent (with `gpt-4o`) on a representative
42-case subset and checks four dimensions — routing (the LLM picks the right
plugin), no-error, honesty (the claims ledger is clean), and numerical accuracy
— currently 42 / 42 pass.

**Replication-aware bootstrap inference.** A `bootstrap_inference` plugin
provides confidence intervals for paired-difference statistics (mean, median,
trimmed mean, Cohen's $d_z$) under percentile, basic, and BCa methods,
cross-validated against `scipy.stats.bootstrap`. An optional Sequential
Bootstrap mode [@peng2025sboob] constrains the resampler so that the number of
distinct in-bag samples per replicate is held at a fixed target, eliminating
the resampler-side variance component $\mathrm{Var}\big(\mathbb{E}[\widehat{\theta}_b
\mid U_b]\big)$ in the bootstrap's variance decomposition. The resulting CI
endpoints are reproducible across bootstrap RNG seeds — a property required
in regulatory, clinical, or audit-grade settings where the *same* CI, not
merely the same point estimate, must be reproducible on independent re-run.
Each call emits a cross-seed CI endpoint-stability diagnostic.

**Anti-fabrication claims ledger.** Plugins emit structured, verified claims;
the LLM may only reference these claims by identifier, and a render layer
substitutes the verified wording into the final report. This prevents the LLM
from reporting a statistic that was not actually computed.

**Reproducibility and provenance.** Analyses are tied to an explicit data
version, and reports record how the analysis-ready dataset was constructed,
including SQL provenance where applicable. The framework also exports a
reproducibility manifest and APA-style methods text.

**Testing.** The framework includes 764 deterministic unit and integration tests
that run without an API key or network access, in addition to the 362-case
numerical validation benchmark and the 42-case end-to-end agent benchmark.

# Acknowledgements

The author thanks the maintainers of `scipy`, `statsmodels`, and `LangGraph`,
on which this framework builds.

# References
