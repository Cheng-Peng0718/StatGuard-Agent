import numpy as np
import pandas as pd

from core.analysis_tool_plugins import get_plugin, PLUGIN_REGISTRY


class DummyContext:
    def __init__(self, df, args=None):
        self.df = df
        self.args = args or {}

    def load_df(self):
        return self.df

    def get_arg(self, key, default=None):
        return self.args.get(key, default)


def _fit(df, args):
    plugin = get_plugin("run_logistic_regression")
    raw = plugin.run(DummyContext(df, args))
    return plugin, raw


def _findings(plugin, raw, args):
    run = plugin.build_analysis_run(
        action_id="act_test",
        arguments=args,
        data_version_id="raw_v1",
        status=raw["status"],
        success=raw["status"] == "ok",
        message=raw.get("message", ""),
        payload=raw.get("details", {}),
        artifacts=raw.get("artifacts", []),
        observation_id="obs_test",
    )
    return run["guardrails"]


def _binary_dataset(n=200, seed=42):
    rng = np.random.RandomState(seed)
    x1 = rng.randn(n)
    x2 = rng.randn(n)
    logit = -0.5 + 1.2 * x1 - 0.8 * x2
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.rand(n) < p).astype(int)
    return pd.DataFrame({"outcome": y, "x1": x1, "x2": x2})


# ----------------------------------------------------------------------
# Registration / discovery
# ----------------------------------------------------------------------

def test_logistic_plugin_is_registered_and_inferential():
    assert "run_logistic_regression" in PLUGIN_REGISTRY
    plugin = get_plugin("run_logistic_regression")
    assert plugin is not None
    assert plugin.execute is not None
    assert plugin.is_inferential is True
    # Guardrails must be wired, not just defined.
    assert plugin.guardrail_evaluators, "logistic plugin has no guardrail evaluators wired"


# ----------------------------------------------------------------------
# Happy path
# ----------------------------------------------------------------------

def test_logistic_runs_and_reports_odds_ratios():
    df = _binary_dataset()
    args = {"target_col": "outcome", "feature_cols": ["x1", "x2"]}
    plugin, raw = _fit(df, args)

    assert raw["status"] == "ok", raw
    metrics = raw["details"]["metrics"]
    assert metrics["converged"] is True
    assert metrics["separation_detected"] is False
    assert metrics["n_obs"] == len(df)
    assert metrics["events_per_variable"] is not None

    coefs = {r["term"]: r for r in raw["details"]["tables"]["coefficients"]}
    # Known generative signs: x1 increases odds (OR > 1), x2 decreases (OR < 1).
    assert coefs["x1"]["odds_ratio"] > 1.0
    assert coefs["x2"]["odds_ratio"] < 1.0
    # CIs present.
    assert coefs["x1"]["or_ci_low"] is not None
    assert coefs["x1"]["or_ci_high"] is not None


def test_logistic_accepts_string_labels_and_picks_positive_class():
    df = _binary_dataset()
    df["outcome"] = np.where(df["outcome"] == 1, "yes", "no")
    args = {"target_col": "outcome", "feature_cols": ["x1", "x2"]}
    plugin, raw = _fit(df, args)

    assert raw["status"] == "ok", raw
    assert raw["details"]["metadata"]["positive_class"] == "yes"


# ----------------------------------------------------------------------
# Binary outcome gate (the thing linear regression does NOT enforce)
# ----------------------------------------------------------------------

def test_logistic_blocks_non_binary_outcome():
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"outcome": rng.randint(0, 5, 80), "x": rng.randn(80)})
    plugin, raw = _fit(df, {"target_col": "outcome", "feature_cols": ["x"]})

    assert raw["status"] == "blocked"
    assert raw["error_code"] == "OUTCOME_NOT_BINARY"
    assert raw["details"]["n_classes"] == 5


def test_logistic_blocks_constant_outcome():
    df = pd.DataFrame({"outcome": [1] * 50, "x": np.linspace(0, 1, 50)})
    plugin, raw = _fit(df, {"target_col": "outcome", "feature_cols": ["x"]})
    assert raw["status"] == "blocked"
    assert raw["error_code"] in {"OUTCOME_NOT_VARIABLE", "OUTCOME_NOT_VARIABLE_AFTER_FILTERING"}


# ----------------------------------------------------------------------
# Guardrails: low EPV
# ----------------------------------------------------------------------

def test_logistic_low_epv_emits_warning():
    rng = np.random.RandomState(7)
    n = 60
    feats = {f"f{i}": rng.randn(n) for i in range(8)}
    df = pd.DataFrame(feats)
    df["outcome"] = np.array([0] * 56 + [1] * 4)  # 4 events, 8 predictors -> EPV 0.5
    args = {"target_col": "outcome", "feature_cols": list(feats.keys())}
    plugin, raw = _fit(df, args)

    assert raw["status"] == "ok", raw
    assert raw["details"]["metrics"]["events_per_variable"] < 10
    titles = [f["title"] for f in _findings(plugin, raw, args)]
    assert any("events-per-variable" in t.lower() for t in titles)


# ----------------------------------------------------------------------
# Guardrails: (quasi-)complete separation
# ----------------------------------------------------------------------

def test_logistic_perfect_separation_is_blocked_or_flagged():
    # x perfectly separates the outcome.
    df = pd.DataFrame({"outcome": [0] * 10 + [1] * 10, "x": list(range(20))})
    args = {"target_col": "outcome", "feature_cols": ["x"]}
    plugin, raw = _fit(df, args)

    if raw["status"] == "blocked":
        assert raw["error_code"] in {"PERFECT_SEPARATION"}
    else:
        # If it fit, the heuristic must flag separation and a guardrail must fire.
        assert raw["details"]["metrics"]["separation_detected"] is True
        titles = [f["title"].lower() for f in _findings(plugin, raw, args)]
        assert any("separation" in t or "did not converge" in t for t in titles)


# ----------------------------------------------------------------------
# Reuse: continuous high-cardinality predictors must not be silently dropped
# ----------------------------------------------------------------------

def test_logistic_does_not_drop_legitimate_continuous_predictors():
    # Continuous predictors have ~100% uniqueness and would be treated as
    # "id-like" by the shared prep; the plugin must recover them.
    df = _binary_dataset(n=150, seed=3)
    args = {"target_col": "outcome", "feature_cols": ["x1", "x2"]}
    plugin, raw = _fit(df, args)
    assert raw["status"] == "ok", raw
    terms = {r["term"] for r in raw["details"]["tables"]["coefficients"]}
    assert {"x1", "x2"}.issubset(terms)