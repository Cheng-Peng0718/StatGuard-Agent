from pathlib import Path


def test_future_ui_entrypoint_is_documented():
    text = Path("docs/17_new_ui_skeleton_design.md").read_text(
        encoding="utf-8"
    )

    assert "ui/app_v2.py" in text


def test_legacy_app_is_not_required_for_backend_tests():
    offenders = []

    for path in Path("tests").rglob("*.py"):
        normalized = path.as_posix()

        if "/ui_legacy/" in normalized or "/ui/" in normalized:
            continue

        text = path.read_text(encoding="utf-8")

        # Avoid substring checks like "import app" inside documentation strings.
        # This test only checks actual import statements.
        for line_no, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()

            if stripped == "import app" or stripped.startswith("import app "):
                offenders.append(f"{path}:{line_no}: {line}")

            if stripped == "from app import" or stripped.startswith("from app import"):
                offenders.append(f"{path}:{line_no}: {line}")

    assert offenders == []


def test_ui_directory_may_exist_but_backend_must_not_depend_on_it():
    ui_dir = Path("ui")

    # It is okay if ui/ does not exist yet.
    # S17B/S17C may create it later.
    if ui_dir.exists():
        assert ui_dir.is_dir()

    core_texts = []
    for path in Path("core").rglob("*.py"):
        core_texts.append(path.read_text(encoding="utf-8"))

    combined = "\n".join(core_texts)

    assert "from ui" not in combined
    assert "import ui" not in combined