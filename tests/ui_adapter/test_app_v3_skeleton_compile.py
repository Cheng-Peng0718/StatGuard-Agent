import py_compile
from pathlib import Path


def test_app_v3_skeleton_compiles():
    targets = [
        "ui/app_v3.py",
        "ui/components/system_status.py",
        "ui/components/chat_panel.py",
        "ui/components/plan_timeline.py",
        "ui/components/active_workspace.py",
        "ui/components/action_bar.py",
        "ui/components/debug_panel.py",
        "ui/components/report_panel.py",
    ]

    for target in targets:
        py_compile.compile(
            str(Path(target)),
            doraise=True,
        )