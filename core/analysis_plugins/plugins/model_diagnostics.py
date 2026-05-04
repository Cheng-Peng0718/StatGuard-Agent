from typing import Any, Dict, Tuple

from core.analysis_plugins.base import AnalysisPlugin, compact_dict
from core.analysis_plugins.registry import register_plugin
from core.guardrails import evaluate_diagnostics_guardrails


def extract_model_diagnostics(
    *,
    payload: Dict[str, Any],
    arguments: Dict[str, Any],
    default_title: str,
    default_summary: str,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any]]:
    title = "Model Diagnostics"
    summary = "Computed model diagnostics."

    vif = payload.get("vif", [])
    bp = payload.get("breusch_pagan", {})

    vif_values = [
        row.get("vif")
        for row in vif
        if isinstance(row, dict) and row.get("vif") is not None
    ]

    metrics = compact_dict({
        "max_vif": max(vif_values) if vif_values else None,
        "breusch_pagan_lm_statistic": bp.get("lm_statistic") if isinstance(bp, dict) else None,
        "breusch_pagan_lm_p_value": bp.get("lm_p_value") if isinstance(bp, dict) else None,
        "breusch_pagan_f_statistic": bp.get("f_statistic") if isinstance(bp, dict) else None,
        "breusch_pagan_f_p_value": bp.get("f_p_value") if isinstance(bp, dict) else None,
        "heteroscedasticity_flag_0_05": bp.get("heteroscedasticity_flag_0_05") if isinstance(bp, dict) else None,
    })

    tables: Dict[str, Any] = {}

    if vif:
        tables["vif"] = vif
    if bp:
        tables["breusch_pagan"] = bp

    return title, summary, metrics, tables


PLUGIN = register_plugin(AnalysisPlugin(
    tool_name="regression_diagnostics",
    display_name="Model Diagnostics",
    extractor=extract_model_diagnostics,
    guardrail_evaluators=[
        evaluate_diagnostics_guardrails,
    ],
))