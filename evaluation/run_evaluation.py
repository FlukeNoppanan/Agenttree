"""Run the offline multi-domain evaluation and write deterministic JSON."""

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

# Support the documented direct-script command from the repository root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.models import EvaluationReport
from evaluation.scenarios import build_report


DEFAULT_OUTPUT = Path(__file__).with_name("results.json")


def write_report(report: EvaluationReport, output: Path = DEFAULT_OUTPUT) -> None:
    """Write stable UTF-8 JSON suitable for reports and presentation tables."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def print_report(report: EvaluationReport) -> None:
    """Print compact per-scenario and aggregate results."""
    print(f"{'Domain / Scenario':45} {'Status':7} {'Routing':8} {'Trace':5} {'Provider':8}")
    for item in report.scenarios:
        print(
            f"{(item.domain + ' / ' + item.scenario_id):45} "
            f"{item.status:7} "
            f"{('PASS' if item.routing_passed else 'FAIL'):8} "
            f"{('PASS' if item.traceability_passed else 'FAIL'):5} "
            f"{('PASS' if item.provider_independence_passed else 'FAIL'):8}"
        )
    print("\nOverall")
    for name in (
        "reusability", "extensibility", "capability_based_routing",
        "provider_independence", "traceability", "tool_integration",
        "langgraph_equivalence", "all_scenarios",
    ):
        print(f"{name.replace('_', ' ').title():30} {report.overall[name]['status']}")


def main(argv: Sequence[str] | None = None) -> int:
    """Execute all scenarios, write JSON, and return nonzero on measured failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = build_report()
    write_report(report, args.output)
    print_report(report)
    required = (
        "reusability", "extensibility", "capability_based_routing",
        "provider_independence", "traceability", "tool_integration", "all_scenarios",
    )
    passed = all(report.overall[name]["status"] == "PASS" for name in required)
    if report.overall["langgraph_equivalence"]["available"]:
        passed = passed and report.overall["langgraph_equivalence"]["status"] == "PASS"
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
