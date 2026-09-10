"""Standard-library result contracts for deterministic evaluation output."""

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ScenarioEvaluation:
    """Measured framework properties for one predefined offline scenario."""

    scenario_id: str
    domain: str
    status: str
    final_status: str
    routing_passed: bool
    traceability_passed: bool
    provider_independence_passed: bool
    notes: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return isolated JSON-compatible fields in declaration order."""
        return deepcopy(asdict(self))


@dataclass(frozen=True)
class EvaluationReport:
    """Scenario rows, exact measurement definitions, and aggregate outcomes."""

    schema_version: str
    framework_api: str
    metric_definitions: dict[str, str]
    scenarios: tuple[ScenarioEvaluation, ...]
    overall: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic machine-readable report data without timestamps."""
        return {
            "schema_version": self.schema_version,
            "framework_api": self.framework_api,
            "metric_definitions": deepcopy(self.metric_definitions),
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
            "overall": deepcopy(self.overall),
        }
