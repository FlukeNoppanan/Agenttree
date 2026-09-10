"""IT/technical application configuration using only public AgentTree APIs."""

try:
    from examples.domains._shared import (
        DomainConfiguration, DomainRun, DomainScenario, print_summary, run_scenario,
    )
except ModuleNotFoundError:  # Direct ``python examples/domains/...`` execution.
    from _shared import (  # type: ignore[no-redef]
        DomainConfiguration, DomainRun, DomainScenario, print_summary, run_scenario,
    )


def count_signals(lines: tuple[str, ...]) -> dict[str, int]:
    """Count synthetic diagnostic lines without touching infrastructure."""
    return {"signal_count": len(lines)}


CONFIGURATION = DomainConfiguration(
    domain_id="it",
    domain_name="IT / Technical",
    manager_name="IT Incident Manager",
    manager_capability="incident_analysis",
    specialists=(
        ("Network Specialist", "network_analysis", "Network path is stable in the synthetic evidence."),
        ("Log Specialist", "log_analysis", "Synthetic logs indicate a bounded worker saturation event."),
    ),
    unrelated_specialist=("Capacity Specialist", "capacity_forecasting"),
    templates=(
        ("Analyze the synthetic network evidence", "network_analysis"),
        ("Analyze the synthetic service logs", "log_analysis"),
    ),
    scenarios=(
        DomainScenario(
            "it-incident-review",
            "Analyze a service incident and recommend next actions.",
            {"signals": ["latency elevated", "worker queue recovered"], "environment": "synthetic"},
        ),
        DomainScenario(
            "it-follow-up-review",
            "Review synthetic incident follow-up evidence.",
            {"signals": ["health checks stable", "error rate nominal"], "environment": "synthetic"},
        ),
    ),
    tool=("count_signals", count_signals, {"lines": ("latency", "queue")}),
)


def build_example() -> DomainRun:
    """Construct and execute the default IT example through AgentTree.run()."""
    return run_scenario(CONFIGURATION)


if __name__ == "__main__":
    print_summary(build_example())
