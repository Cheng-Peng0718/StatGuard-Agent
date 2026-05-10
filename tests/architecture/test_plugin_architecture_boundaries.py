from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def read_file(relative_path: str) -> str:
    path = PROJECT_ROOT / relative_path
    assert path.exists(), f"Expected file to exist: {relative_path}"
    return path.read_text(encoding="utf-8")


def assert_forbidden_terms_absent(
    *,
    relative_path: str,
    forbidden_terms: list[str],
    allowed_false_positives: list[str] | None = None,
) -> None:
    text = read_file(relative_path)
    allowed_false_positives = allowed_false_positives or []

    violations = []

    for term in forbidden_terms:
        if term not in text:
            continue

        # Allow known harmless substrings.
        if any(term in allowed for allowed in allowed_false_positives):
            continue

        violations.append(term)

    assert not violations, (
        f"{relative_path} contains forbidden architecture-specific terms: "
        f"{violations}. Move method-specific logic into "
        f"core/analysis_tool_plugins/plugins/<method>.py instead."
    )


def test_report_builder_is_method_agnostic():
    """
    report_builder.py must remain a generic renderer.

    It should not know about statistical method names, dataset columns,
    table names from a specific method, or variable-specific formatting.
    """
    forbidden_terms = [
        "GPA",
        "SATM",
        "OLS",
        "VIF",
        "Breusch",
        "coef_table",
        "diagnostic_flags",
        "const",
        "run_multiple_regression",
        "regression_diagnostics",
        "generate_residual_histogram",
    ]

    assert_forbidden_terms_absent(
        relative_path="core/report_builder.py",
        forbidden_terms=forbidden_terms,
    )


def test_analysis_runs_is_only_dispatcher():
    """
    analysis_runs.py should only dispatch to plugins.

    It should not branch on tool_name or contain method-specific extraction logic.
    """
    forbidden_terms = [
        "if tool_name",
        "elif tool_name",
        "run_multiple_regression",
        "regression_diagnostics",
        "generate_residual_histogram",
        "GPA",
        "SATM",
        "coef_table",
        "diagnostic_flags",
    ]

    assert_forbidden_terms_absent(
        relative_path="core/analysis_runs.py",
        forbidden_terms=forbidden_terms,
    )


def test_analysis_plugin_base_is_method_agnostic():
    """
    base.py defines the plugin framework only.

    It may contain generic words like 'unregistered tools', but it should not
    contain concrete method names, dataset columns, or method-specific table keys.
    """
    forbidden_terms = [
        "GPA",
        "SATM",
        "OLS",
        "VIF",
        "Breusch",
        "coef_table",
        "diagnostic_flags",
        "run_multiple_regression",
        "regression_diagnostics",
        "generate_residual_histogram",
    ]

    assert_forbidden_terms_absent(
        relative_path="core/analysis_tool_plugins/base.py",
        forbidden_terms=forbidden_terms,
    )


def test_analysis_plugin_base_stays_minimal_runtime_wrapper():
    text = read_file("core/analysis_tool_plugins/base.py")

    forbidden_terms = [
        "def format_p_value",
        "def format_number",
        "def build_generic_report_blocks",
        "def default_extractor",
        "def metric_rows_from_dict_with_display",
        "def normalize_table_from_list_with_display",
        "def build_analysis_run",
        "def evaluate_guardrails",
        "class ArgumentSchema",
        "class DisplayConfig",
        "class MetricDisplayConfig",
        "class TableDisplayConfig",
        "class VariableRoleSpec",
        "class ApplicabilityResult",
        "class VersioningPolicy",
        "class RepairPolicy",
        "class PlanningPolicy",
    ]

    violations = [term for term in forbidden_terms if term in text]
    assert not violations, (
        "core/analysis_tool_plugins/base.py must only define the minimal "
        f"AnalysisToolPlugin wrapper. Found: {violations}"
    )


def test_analysis_plugin_support_modules_exist():
    expected = [
        "arguments.py",
        "display.py",
        "reporting.py",
        "roles.py",
        "applicability.py",
        "policy_types.py",
        "result_builder.py",
    ]

    plugin_dir = PROJECT_ROOT / "core" / "analysis_tool_plugins"

    for filename in expected:
        assert (plugin_dir / filename).exists(), f"Missing plugin support module: {filename}"


def test_plugin_files_are_allowed_to_contain_method_specific_terms():
    """
    Plugin files are the correct place for method-specific logic.

    This test documents the intended architecture boundary.
    """
    plugin_dir = PROJECT_ROOT / "core" / "analysis_tool_plugins" / "plugins"
    assert plugin_dir.exists(), "Expected plugin directory to exist."

    plugin_files = list(plugin_dir.glob("*.py"))
    assert plugin_files, "Expected at least one plugin file."

    assert any(path.name == "linear_model.py" for path in plugin_files), (
        "Expected linear_model.py plugin to exist."
    )

def test_analysis_runs_does_not_import_legacy_analysis_plugins():
    text = read_file("core/analysis_runs.py")

    assert "core.analysis_plugins" not in text
    assert "analysis_plugins" not in text
    assert "get_legacy_plugin" not in text
    assert "_build_legacy_analysis_run" not in text

def test_argument_schema_no_longer_exposes_legacy_schema_adapter():
    text = read_file("core/analysis_tool_plugins/arguments.py")

    assert "to_legacy_schema_dict" not in text
    assert "to_contract_dict" in text
