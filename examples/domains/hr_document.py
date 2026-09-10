"""HR/document application configuration with entirely synthetic records."""

try:
    from examples.domains._shared import (
        DomainConfiguration, DomainRun, DomainScenario, print_summary, run_scenario,
    )
except ModuleNotFoundError:  # Direct ``python examples/domains/...`` execution.
    from _shared import (  # type: ignore[no-redef]
        DomainConfiguration, DomainRun, DomainScenario, print_summary, run_scenario,
    )


CONFIGURATION = DomainConfiguration(
    domain_id="hr",
    domain_name="HR / Document",
    manager_name="Document Review Manager",
    manager_capability="application_review",
    specialists=(
        ("Document Specialist", "document_analysis", "Synthetic documents are complete and internally consistent."),
        ("Qualification Specialist", "qualification_review", "Synthetic qualifications match the stated criteria."),
    ),
    unrelated_specialist=("Scheduling Specialist", "interview_scheduling"),
    templates=(
        ("Extract information from the synthetic documents", "document_analysis"),
        ("Compare synthetic qualifications with criteria", "qualification_review"),
    ),
    scenarios=(
        DomainScenario(
            "hr-application-summary",
            "Review synthetic application documents and produce a structured summary.",
            {"candidate": "Candidate A", "documents": ["resume fixture", "cover-letter fixture"]},
        ),
        DomainScenario(
            "hr-criteria-review",
            "Review a synthetic application against published criteria.",
            {"candidate": "Candidate B", "criteria": ["writing sample", "project experience"]},
        ),
    ),
)


def build_example() -> DomainRun:
    """Construct and execute the default document example through AgentTree.run()."""
    return run_scenario(CONFIGURATION)


if __name__ == "__main__":
    print_summary(build_example())
