from pathlib import Path


def test_legacy_deliverables_py_removed():
    assert not Path("core/deliverables.py").exists()


def test_check_deliverables_removed_from_core():
    offenders = []

    for path in Path("core").rglob("*.py"):
        text = path.read_text(encoding="utf-8")

        if "check_deliverables" in text:
            offenders.append(str(path))

    assert offenders == []