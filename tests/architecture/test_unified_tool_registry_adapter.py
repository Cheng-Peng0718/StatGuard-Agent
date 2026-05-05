from core.analysis_tool_plugins import (
    PLUGIN_REGISTRY,
    get_plugin,
    get_tool_specs_for_llm,
)


def test_clean_data_registered_in_plugin_registry():
    plugin = get_plugin("clean_data")

    assert plugin is not None
    assert plugin.tool_name == "clean_data"
    assert plugin.requires_confirmation is True


def test_plugin_registry_exports_tool_specs_for_llm():
    specs = get_tool_specs_for_llm()

    names = [spec["name"] for spec in specs]

    assert "clean_data" in names

    clean_data_spec = next(spec for spec in specs if spec["name"] == "clean_data")

    assert clean_data_spec["requires_confirmation"] is True
    assert "arguments" in clean_data_spec
    assert "allowed_values" in clean_data_spec["arguments"]
    assert "conditional_allowed_values" in clean_data_spec["arguments"]
    assert "value_aliases" in clean_data_spec["arguments"]

from core.analysis_tool_plugins import (
    PLUGIN_REGISTRY,
    get_plugin,
    get_tool_specs_for_llm,
)


def test_clean_data_registered_in_plugin_registry():
    plugin = get_plugin("clean_data")

    assert plugin is not None
    assert plugin.tool_name == "clean_data"
    assert plugin.requires_confirmation is True
    assert plugin.execute is not None


def test_plugin_registry_exports_tool_specs_for_llm():
    specs = get_tool_specs_for_llm()

    names = [spec["name"] for spec in specs]

    assert "clean_data" in names

    clean_data_spec = next(spec for spec in specs if spec["name"] == "clean_data")

    assert clean_data_spec["requires_confirmation"] is True
    assert "arguments" in clean_data_spec
    assert "allowed_values" in clean_data_spec["arguments"]
    assert "conditional_allowed_values" in clean_data_spec["arguments"]
    assert "value_aliases" in clean_data_spec["arguments"]


def test_llm_tool_specs_do_not_expose_non_executable_plugins():
    specs = get_tool_specs_for_llm()

    for spec in specs:
        plugin = get_plugin(spec["name"])

        assert plugin is not None
        assert plugin.execute is not None