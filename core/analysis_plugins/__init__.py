from core.analysis_plugins.registry import (
    PLUGIN_REGISTRY,
    get_plugin,
    register_plugin,
)


def load_plugins() -> None:
    """
    Auto-discover plugin modules under core.analysis_plugins.plugins.

    Adding a new method should only require adding one plugin file.
    """
    import importlib
    import pkgutil
    import core.analysis_plugins.plugins as plugins_pkg

    for module_info in pkgutil.iter_modules(plugins_pkg.__path__):
        module_name = module_info.name

        if module_name.startswith("_"):
            continue

        importlib.import_module(f"{plugins_pkg.__name__}.{module_name}")


load_plugins()