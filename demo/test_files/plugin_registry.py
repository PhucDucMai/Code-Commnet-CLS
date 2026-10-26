"""Plugin system for the command-line tool."""


class PluginRegistry:
    """Keeps track of every plugin installed in the application.

    Plugins register themselves at import time with the register decorator.

    :param namespace: prefix added to every plugin name.
    :param strict: raise an error when two plugins use the same name.

    Usage::

        registry = PluginRegistry("cli")

        @registry.register("export")
        class ExportPlugin: ...

    .. warning:: plugins are loaded in import order, which is not guaranteed to be stable.
    """

    def __init__(self, namespace, strict=False):
        self.namespace = namespace
        self.strict = strict
        self._plugins = {}

    def register(self, name):
        """Decorator that adds a plugin class to the registry."""
        def wrapper(cls):
            self._plugins[f"{self.namespace}.{name}"] = cls
            return cls
        return wrapper


class _InternalCache:
    # A class without a docstring: nothing to classify.
    pass
