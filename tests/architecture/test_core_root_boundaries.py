from pathlib import Path


BACKEND_CORE_FILES = [
    Path("core/graph.py"),
    Path("core/analysis_runs.py"),
    Path("core/context_builder.py"),
    Path("core/data_versions.py"),
    Path("core/responses.py"),
    Path("core/schema.py"),
    Path("core/state.py"),
    Path("core/interaction_intent.py"),
]


def test_backend_core_does_not_import_report_builder():
    offenders = []

    for path in BACKEND_CORE_FILES:
        text = path.read_text(encoding="utf-8")
        if "core.report_builder" in text or "report_builder" in text:
            offenders.append(str(path))

    assert offenders == []


def test_backend_core_does_not_import_ui_app():
    offenders = []

    for path in BACKEND_CORE_FILES:
        text = path.read_text(encoding="utf-8")
        if "import app" in text or "from app" in text:
            offenders.append(str(path))

    assert offenders == []


def test_backend_core_does_not_import_cli_config_directly():
    offenders = []

    for path in BACKEND_CORE_FILES:
        text = path.read_text(encoding="utf-8")
        if "core.config" in text:
            offenders.append(str(path))

    assert offenders == []