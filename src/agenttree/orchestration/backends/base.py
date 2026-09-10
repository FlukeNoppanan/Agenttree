"""Synchronous orchestration backend extension contract."""

from abc import ABC, abstractmethod

from agenttree.orchestration.backends.context import OrchestrationContext
from agenttree.orchestration.models import FinalResult


class BaseOrchestrationBackend(ABC):
    """Coordinate one run through backend-neutral phase operations."""

    @abstractmethod
    def run(self, context: OrchestrationContext) -> FinalResult:
        """Complete the context's workflow or propagate its original error."""
        raise NotImplementedError
