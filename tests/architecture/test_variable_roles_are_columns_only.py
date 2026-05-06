from core.analysis_tool_plugins import PLUGIN_REGISTRY


NON_COLUMN_ROLE_NAMES = {
    "action_type",
    "strategy",
    "method",
    "output_path",
    "group1_val",
    "group2_val",
}


def test_variable_roles_do_not_include_non_column_choices():
    offenders = []

    for tool_name, plugin in PLUGIN_REGISTRY.items():
        for role in getattr(plugin, "variable_roles", []) or []:
            if role.role_name in NON_COLUMN_ROLE_NAMES:
                offenders.append((tool_name, role.role_name))

    assert offenders == []