"""Default synchronous phase coordination without optional dependencies."""

from agenttree.orchestration.backends.base import BaseOrchestrationBackend
from agenttree.orchestration.backends.context import OrchestrationContext
from agenttree.orchestration.backends.errors import BackendConfigurationError
from agenttree.orchestration.models import FinalResult


class SequentialOrchestrationBackend(BaseOrchestrationBackend):
    """Preserve the Step 17 phase order, early exits, reviews, and state."""

    def run(self, context: OrchestrationContext) -> FinalResult:
        """Call each existing phase once; engine methods own revision loops."""
        for phase in (
            context.planning, context.delegation, context.execution,
            context.manager_review, context.final_review,
        ):
            phase()
            if context.result is not None:
                return context.result
        raise BackendConfigurationError("Sequential backend produced no FinalResult")
