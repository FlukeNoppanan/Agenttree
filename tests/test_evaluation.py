"""Offline evidence tests for the Step 20 multi-domain examples."""

from dataclasses import fields
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from agenttree import AgentTree
from agenttree.orchestration import FinalResult, FinalStatus
from evaluation.models import EvaluationReport, ScenarioEvaluation
from evaluation.run_evaluation import main, write_report
from evaluation.scenarios import (
    CONFIGURATIONS,
    REQUIRED_EVENTS,
    _extension_probe,
    _selected_names,
    build_report,
)
from examples.domains._shared import DomainRun, build_framework, run_scenario


@pytest.fixture(scope="module")
def domain_runs() -> tuple[DomainRun, ...]:
    """Run every synthetic domain scenario once for shared assertions."""
    return tuple(
        run_scenario(configuration, scenario)
        for configuration in CONFIGURATIONS
        for scenario in configuration.scenarios
    )


@pytest.fixture(scope="module")
def report() -> EvaluationReport:
    """Build the complete deterministic evaluation once."""
    return build_report()


@pytest.mark.parametrize("configuration", CONFIGURATIONS)
def test_each_domain_constructs_the_same_framework_type(configuration) -> None:
    framework, _, _ = build_framework(configuration)

    assert isinstance(framework, AgentTree)


def test_domain_examples_use_the_high_level_sdk_boundary() -> None:
    repository = Path(__file__).resolve().parents[1]
    shared_source = (repository / "examples/domains/_shared.py").read_text("utf-8")
    modules = (
        repository / "examples/domains/it_technical.py",
        repository / "examples/domains/hr_document.py",
        repository / "examples/domains/business_product.py",
    )

    assert "framework.run(task)" in shared_source
    assert "OrchestrationEngine" not in shared_source
    for path in modules:
        assert "OrchestrationEngine" not in path.read_text("utf-8")


def test_all_predefined_domain_scenarios_complete(domain_runs) -> None:
    assert len(domain_runs) == 6
    assert all(run.result.status is FinalStatus.COMPLETED for run in domain_runs)
    assert all(run.result.success for run in domain_runs)
    assert all(run.task_unchanged for run in domain_runs)


def test_expected_capability_routing_excludes_unrelated_agents(domain_runs) -> None:
    for run in domain_runs:
        managers, specialists = _selected_names(run)
        assert managers == run.expected_manager_names
        assert specialists == run.expected_specialist_names
        assert set(managers + specialists).isdisjoint(run.unrelated_agent_names)


def test_mixed_provider_bindings_execute_through_normalized_results(domain_runs) -> None:
    for run in domain_runs:
        bindings = run.framework.provider_bindings
        bound_provider_names = {
            bindings[specialist.id]
            for specialist in run.framework.specialists
            if specialist.name in run.expected_specialist_names
        }
        assert len(bound_provider_names) == len(run.expected_specialist_names)
        state = run.framework.last_state
        assert state is not None and state.execution_result is not None
        for manager_execution in state.execution_result.manager_executions:
            for execution in manager_execution.specialist_executions:
                assert execution.agent_result is not None
                assert execution.agent_result.success


def test_final_result_shape_is_consistent_across_domains(domain_runs) -> None:
    expected = tuple(field.name for field in fields(FinalResult))

    assert all(
        tuple(field.name for field in fields(run.result)) == expected
        for run in domain_runs
    )


def test_major_trace_events_exist_in_order_across_domains(domain_runs) -> None:
    for run in domain_runs:
        labels = [event.event_type for event in run.result.trace.events]
        positions = [labels.index(label) for label in REQUIRED_EVENTS]
        assert positions == sorted(positions)
        assert run.result.trace.task_id == run.task.id
        assert all(event.task_id == run.task.id for event in run.result.trace.events)


def test_offline_function_tool_produces_existing_tool_trace(domain_runs) -> None:
    tool_runs = [run for run in domain_runs if run.tool_result is not None]

    assert len(tool_runs) == 2
    assert all(run.tool_result.success for run in tool_runs)
    assert all(run.tool_result.output == {"signal_count": 2} for run in tool_runs)
    assert all(run.tool_trace.event_count == 2 for run in tool_runs)


def test_new_application_agent_and_capability_need_no_core_change() -> None:
    probe = _extension_probe()

    assert probe["status"] == "PASS"
    assert probe["final_status"] == FinalStatus.COMPLETED.value
    assert probe["new_capability"] == "new_application_synthesis"


def test_domain_customization_paths_are_outside_core(report) -> None:
    evidence = report.overall["reusability"]

    assert evidence["customization_paths_outside_core"]
    assert all(
        not path.startswith("src/agenttree/")
        for path in evidence["customization_paths"]
    )
    assert set(evidence["core_source_files_modified_per_domain"].values()) == {0}


def test_evaluation_rows_serialize_to_isolated_json_data(report) -> None:
    payload = report.to_dict()
    json.dumps(payload)
    payload["overall"]["all_scenarios"]["status"] = "CHANGED"
    payload["scenarios"][0]["metrics"]["selected_managers"].append("changed")

    fresh = report.to_dict()
    assert fresh["overall"]["all_scenarios"]["status"] == "PASS"
    assert "changed" not in fresh["scenarios"][0]["metrics"]["selected_managers"]


def test_scenario_result_contract_serializes_without_heavy_dependencies() -> None:
    row = ScenarioEvaluation(
        scenario_id="fixture",
        domain="Synthetic",
        status="PASS",
        final_status="completed",
        routing_passed=True,
        traceability_passed=True,
        provider_independence_passed=True,
        metrics={"count": 1},
    )

    assert row.to_dict()["metrics"] == {"count": 1}


def test_evaluation_report_is_deterministic(report) -> None:
    second = build_report()

    assert report.to_dict() == second.to_dict()


def test_evaluation_runner_writes_machine_readable_output(capsys) -> None:
    with TemporaryDirectory(dir=Path.cwd()) as directory:
        output = Path(directory) / "results.json"

        assert main(["--output", str(output)]) == 0
        payload = json.loads(output.read_text("utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["overall"]["all_scenarios"] == {
        "passed": 6,
        "status": "PASS",
        "total": 6,
    }
    assert "Overall" in capsys.readouterr().out


def test_write_report_produces_identical_bytes(report) -> None:
    with TemporaryDirectory(dir=Path.cwd()) as directory:
        first = Path(directory) / "first.json"
        second = Path(directory) / "second.json"

        write_report(report, first)
        write_report(report, second)

        assert first.read_bytes() == second.read_bytes()
