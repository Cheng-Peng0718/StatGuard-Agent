import subprocess
import sys


def test_importing_core_graph_does_not_create_compiled_app():
    code = """
import core.graph

# Importing graph may define workflow wiring, but should not eagerly compile
# a runnable app as a global side effect.
assert "app" not in core.graph.__dict__, "core.graph should expose create_graph_app(), not global app"
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr