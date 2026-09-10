"""Errors at the orchestration backend boundary."""


class BackendDependencyError(ImportError):
    """An optional backend dependency is unavailable or incomplete."""


class BackendConfigurationError(ValueError):
    """A backend cannot construct or satisfy its orchestration contract."""
