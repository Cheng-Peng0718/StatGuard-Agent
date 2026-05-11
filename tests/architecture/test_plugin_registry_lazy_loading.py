import subprocess
import sys


def test_importing_analysis_tool_plugins_does_not_import_concrete_plugin_modules():
    code = """
import sys
import core.analysis_tool_plugins

loaded_plugin_modules = [
    name
    for name in sys.modules
    if name.startswith("core.analysis_tool_plugins.plugins.")
]

assert loaded_plugin_modules == [], loaded_plugin_modules
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr