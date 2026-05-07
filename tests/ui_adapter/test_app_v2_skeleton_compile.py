import py_compile
from pathlib import Path


def test_app_v2_skeleton_compiles():
    py_compile.compile(
        str(Path("ui/app_v2.py")),
        doraise=True,
    )