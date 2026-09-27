"""Cooperative cancellation shared by orchestration, providers, and Tools."""

from contextvars import ContextVar
from datetime import datetime, timezone
from threading import Event


class ExecutionCancelled(RuntimeError):
    """A cancellation request or UTC deadline stopped execution progress."""


class ExecutionControl:
    def __init__(self, cancelled: Event, deadline_at: datetime | None = None) -> None:
        self.cancelled = cancelled
        self.deadline_at = deadline_at

    def check(self) -> None:
        if self.deadline_at is not None and datetime.now(timezone.utc) >= self.deadline_at:
            self.cancelled.set()
        if self.cancelled.is_set():
            raise ExecutionCancelled("Execution cancelled")

    def remaining(self) -> float | None:
        if self.deadline_at is None:
            return None
        return max(0.0, (self.deadline_at - datetime.now(timezone.utc)).total_seconds())


_active_execution_control: ContextVar[ExecutionControl | None] = ContextVar(
    "agenttree_execution_control", default=None)


def check_execution() -> None:
    control = _active_execution_control.get()
    if control is not None:
        control.check()
