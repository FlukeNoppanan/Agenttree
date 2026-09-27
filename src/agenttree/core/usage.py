"""Per-run provider usage collection without changing strategy interfaces."""

from contextvars import ContextVar
from dataclasses import dataclass, field

from agenttree.providers import ProviderResponse, ProviderUsage


@dataclass
class UsageCollector:
    calls: list[dict[str, object]] = field(default_factory=list)

    def record(self, stage: str, response: ProviderResponse) -> None:
        self.calls.append({"stage": stage, "provider": response.provider,
                           "model": response.model, "usage": response.usage})

    def summary(self) -> dict[str, object]:
        reported = [call["usage"] for call in self.calls
                    if isinstance(call["usage"], ProviderUsage)]
        def total(name: str) -> int | None:
            values = [getattr(item, name) for item in reported]
            return sum(values) if values and all(value is not None for value in values) else None
        return {"input_tokens": total("input_tokens"),
                "output_tokens": total("output_tokens"),
                "total_tokens": total("total_tokens"),
                "unreported_calls": len(self.calls) - len(reported),
                "calls": tuple(self.calls)}


_active_usage: ContextVar[UsageCollector | None] = ContextVar("agenttree_usage", default=None)


def record_usage(stage: str, response: ProviderResponse) -> None:
    collector = _active_usage.get()
    if collector is not None:
        collector.record(stage, response)
