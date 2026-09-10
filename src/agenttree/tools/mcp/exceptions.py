"""Errors for optional MCP dependencies, configuration, and transport calls."""


class MCPError(Exception):
    """Base error for AgentTree's real MCP transport boundary."""


class MCPConfigurationError(MCPError, ValueError):
    """Invalid configuration or misuse of explicit client lifecycle."""


class MCPDependencyError(MCPConfigurationError, ImportError):
    """The optional MCP transport SDK is unavailable."""


class MCPRuntimeError(MCPError, RuntimeError):
    """A transport operation or tool call could not complete."""


class MCPConnectionError(MCPRuntimeError):
    """Server startup, initialization, or session communication failed."""
