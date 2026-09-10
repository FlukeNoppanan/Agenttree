"""Business/product application configuration using synthetic market inputs."""

try:
    from examples.domains._shared import (
        DomainConfiguration, DomainRun, DomainScenario, print_summary, run_scenario,
    )
except ModuleNotFoundError:  # Direct ``python examples/domains/...`` execution.
    from _shared import (  # type: ignore[no-redef]
        DomainConfiguration, DomainRun, DomainScenario, print_summary, run_scenario,
    )


CONFIGURATION = DomainConfiguration(
    domain_id="product",
    domain_name="Business / Product",
    manager_name="Product Evaluation Manager",
    manager_capability="product_recommendation",
    specialists=(
        ("Market Specialist", "market_analysis", "Synthetic research shows a focused early-adopter segment."),
        ("Product Specialist", "product_analysis", "A small validation experiment is the recommended next step."),
    ),
    unrelated_specialist=("Finance Specialist", "financial_modeling"),
    templates=(
        ("Analyze the synthetic market assumptions", "market_analysis"),
        ("Assess the synthetic product concept", "product_analysis"),
    ),
    scenarios=(
        DomainScenario(
            "product-idea-review",
            "Analyze a product idea and produce a structured recommendation.",
            {"idea": "Synthetic collaborative planning assistant", "market_notes": ["small team fixture"]},
        ),
        DomainScenario(
            "product-experiment-review",
            "Assess a synthetic product validation experiment.",
            {"experiment": "Prototype interview fixture", "responses": 8},
        ),
    ),
)


def build_example() -> DomainRun:
    """Construct and execute the default product example through AgentTree.run()."""
    return run_scenario(CONFIGURATION)


if __name__ == "__main__":
    print_summary(build_example())
