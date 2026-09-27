"""Per-run agent provider selection for provider-backed decision strategies."""

from contextvars import ContextVar
from typing import Mapping

from agenttree.providers import BaseProvider


ProviderBinding = tuple[BaseProvider, str | None]

_active_provider_bindings: ContextVar[Mapping[str, ProviderBinding]] = ContextVar(
    "agenttree_provider_bindings", default={})
_active_root_id: ContextVar[str | None] = ContextVar("agenttree_root_id", default=None)


def resolve_provider(agent_id: str | None, fallback: BaseProvider) -> ProviderBinding:
    if agent_id is None:
        return fallback, None
    return _active_provider_bindings.get().get(agent_id, (fallback, None))


def root_id() -> str | None:
    return _active_root_id.get()
