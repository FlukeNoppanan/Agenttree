"""Offline backend contracts and sequential/LangGraph semantic equivalence."""

from copy import deepcopy
import importlib.util
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from agenttree import AgentTree, AgentTreeConfig, ManagerAgent, RootAgent, SpecialistAgent, Task
from agenttree.core import (
    ProviderFinalReviewer, ProviderManagerReviewer, ProviderTaskDecomposer,
    ProviderTaskTriage, RuleBasedTaskTriage, StaticFinalReviewer,
    StaticManagerReviewer, StaticTaskDecomposer,
)
from agenttree.exceptions import DecisionParseError
from agenttree.models import ReviewDecision, SubtaskTemplate, WorkflowPhase
from agenttree.orchestration.backends import (
    BackendConfigurationError, BackendDependencyError, BaseOrchestrationBackend,
    LangGraphOrchestrationBackend, SequentialOrchestrationBackend,
)
from agenttree.orchestration.backends import langgraph as adapter
from agenttree.providers import MockProvider, ProviderRequest, ProviderResponse, ProviderRuntimeError


class FakeGraph:
    """Minimal recording implementation of the used graph API, with no SDK."""

    def __init__(self, schema: type) -> None:
        self.schema = schema
        self.nodes: dict[str, Any] = {}
        self.edges: dict[str, str] = {}
        self.branches: dict[str, Any] = {}
        self.visited: list[str] = []

    def add_node(self, name: str, action: Any) -> None:
        self.nodes[name] = action

    def add_edge(self, start: str, end: str) -> None:
        self.edges[start] = end

    def add_conditional_edges(self, start: str, path: Any, paths: dict[bool, str]) -> None:
        self.branches[start] = (path, paths)

    def compile(self) -> "FakeGraph":
        return self

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        state = dict(state)
        self.visited = []
        node = self.edges["START"]
        while node != "END":
            assert len(self.visited) < 6, "Unexpected graph cycle"
            self.visited.append(node)
            state.update(self.nodes[node](state))
            if node in self.branches:
                path, paths = self.branches[node]
                node = paths[path(state)]
            else:
                node = self.edges[node]
        return state


@pytest.fixture
def fake_api(monkeypatch: pytest.MonkeyPatch) -> list[FakeGraph]:
    graphs: list[FakeGraph] = []

    def create(schema: type) -> FakeGraph:
        graph = FakeGraph(schema)
        graphs.append(graph)
        return graph

    original = adapter.importlib.import_module

    def load(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "langgraph.graph":
            return SimpleNamespace(StateGraph=create, START="START", END="END")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(adapter.importlib, "import_module", load)
    return graphs


def make_framework(backend: BaseOrchestrationBackend, scenario: str = "success") -> AgentTree:
    options: dict[str, Any] = {
        "root_agent": RootAgent(name="Root", id="root"),
        "triage": RuleBasedTaskTriage({}, fallback_capabilities=("manage",)),
        "decomposer": StaticTaskDecomposer((SubtaskTemplate("Work", ("work",)),)),
        "manager_reviewer": StaticManagerReviewer(),
        "final_reviewer": StaticFinalReviewer(),
        "orchestration_backend": backend,
    }
    revise_pass = ((ReviewDecision.REVISE, "Improve"), (ReviewDecision.PASS, "Accepted"))
    if scenario == "manager_revision":
        options["manager_reviewer"] = StaticManagerReviewer(outcomes=revise_pass)
    if scenario == "final_revision":
        options["final_reviewer"] = StaticFinalReviewer(outcomes=revise_pass)
    if scenario == "final_fail":
        options["final_reviewer"] = StaticFinalReviewer(ReviewDecision.FAIL, "Incomplete")
    if scenario == "revision_limit":
        options["final_reviewer"] = StaticFinalReviewer(ReviewDecision.REVISE, "Improve")
        options["config"] = AgentTreeConfig(max_final_revisions=0)
    if scenario == "manager_limit":
        options["manager_reviewer"] = StaticManagerReviewer(ReviewDecision.REVISE)
        options["final_reviewer"] = StaticFinalReviewer(ReviewDecision.FAIL)
        options["config"] = AgentTreeConfig(max_manager_revisions=0)
    if scenario == "mixed":
        a = MockProvider(response_content=
            '{"objective":"Work","required_capabilities":["manage"],"decision":"pass","feedback":"ok"}')
        b = MockProvider(response_content=
            '{"subtasks":[{"objective":"Work","required_capabilities":["work"]}],"decision":"pass","feedback":"ok"}')
        options.update(triage=ProviderTaskTriage(a), decomposer=ProviderTaskDecomposer(b),
                       manager_reviewer=ProviderManagerReviewer(a), final_reviewer=ProviderFinalReviewer(b))
    tree = AgentTree(**options)
    if scenario != "no_manager":
        manager = ManagerAgent(name="Manager", id="manager", capabilities=("manage",))
        tree.register_manager(manager)
        if scenario != "no_specialist":
            specialist = SpecialistAgent(name="Worker", id="worker", capabilities=("work",))
            tree.register_specialist(manager, specialist)
            provider = MockProvider(response_content="Finished")
            tree.register_provider(provider)
            tree.bind_provider(specialist, provider)
    return tree


SCENARIOS = (
    "success", "no_manager", "no_specialist", "manager_revision", "final_revision",
    "final_fail", "revision_limit", "manager_limit", "mixed",
)


def normalized(value: Any) -> Any:
    """Ignore timestamps only; all identities are fixed by the test fixture."""
    if isinstance(value, dict):
        return {key: normalized(item) for key, item in value.items() if key != "timestamp"}
    if isinstance(value, (list, tuple)):
        return [normalized(item) for item in value]
    return value


def assert_equivalent(backend: BaseOrchestrationBackend, scenario: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # Both runs create the same single subtask identity; production still uses UUIDs.
    monkeypatch.setattr("agenttree.models.subtask.uuid4", lambda: "subtask")
    task = Task(objective="Work", id="task", metadata={"nested": [1]})
    before = deepcopy(task)
    sequential = make_framework(SequentialOrchestrationBackend(), scenario)
    graph = make_framework(backend, scenario)
    expected = sequential.run(task)
    actual = graph.run(task)
    assert actual.status == expected.status
    assert actual.success == expected.success
    assert actual.content == expected.content
    assert actual.manager_results == expected.manager_results
    assert normalized(graph.last_state.to_dict()) == normalized(sequential.last_state.to_dict())
    assert normalized(actual.trace.to_dict()) == normalized(expected.trace.to_dict())
    assert graph.last_state.final_result == actual
    assert task == before


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_faked_graph_sdk_equivalence(
    fake_api: list[FakeGraph], scenario: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = LangGraphOrchestrationBackend()
    assert_equivalent(backend, scenario, monkeypatch)
    graph = fake_api[0]
    assert list(graph.nodes) == ["planning", "delegation", "execution", "manager_review", "final_review"]
    assert graph.edges == {"START": "planning", "execution": "manager_review",
                           "manager_review": "final_review", "final_review": "END"}
    assert graph.branches["planning"][1] == {True: "END", False: "delegation"}
    assert graph.branches["delegation"][1] == {True: "END", False: "execution"}
    expected = list(graph.nodes)
    if scenario == "no_manager":
        expected = expected[:1]
    if scenario == "no_specialist":
        expected = expected[:2]
    assert graph.visited == expected  # Revisions never become graph cycles.


@pytest.mark.skipif(importlib.util.find_spec("langgraph") is None, reason="Optional LangGraph is not installed")
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_real_langgraph_offline_equivalence(scenario: str, monkeypatch: pytest.MonkeyPatch) -> None:
    assert_equivalent(LangGraphOrchestrationBackend(), scenario, monkeypatch)


def test_graph_backend_can_be_reused_without_run_state_leakage(fake_api: list[FakeGraph]) -> None:
    backend = LangGraphOrchestrationBackend()
    first = make_framework(backend)
    second = make_framework(backend, "no_manager")
    assert first.run(Task(objective="First")).success
    state = first.last_state
    assert not second.run(Task(objective="Second")).success
    assert first.last_state.to_dict() == state.to_dict()
    assert first.run(Task(objective="Third")).success


def test_missing_dependency_has_install_hint_and_original_cause(monkeypatch: pytest.MonkeyPatch) -> None:
    cause = ModuleNotFoundError("No module named langgraph")

    def missing(name: str) -> Any:
        raise cause

    monkeypatch.setattr(adapter.importlib, "import_module", missing)
    with pytest.raises(BackendDependencyError, match=r"agenttree\[langgraph\]") as caught:
        LangGraphOrchestrationBackend()
    assert caught.value.__cause__ is cause


def test_graph_construction_failure_is_configuration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    cause = ValueError("Graph construction failed")

    def broken(schema: type) -> Any:
        raise cause

    monkeypatch.setattr(adapter.importlib, "import_module", lambda name: SimpleNamespace(StateGraph=broken))
    with pytest.raises(BackendConfigurationError) as caught:
        LangGraphOrchestrationBackend()
    assert caught.value.__cause__ is cause


@pytest.mark.parametrize("stage", ["triage", "decomposition", "manager", "final"])
@pytest.mark.parametrize("runtime", [False, True])
def test_graph_does_not_wrap_phase_errors(fake_api: list[FakeGraph], stage: str, runtime: bool) -> None:
    from tests.test_framework import framework, register_pair

    error = ProviderRuntimeError("Offline failure")

    class Broken(MockProvider):
        def generate(self, request: ProviderRequest) -> ProviderResponse:
            raise error

    provider = Broken() if runtime else MockProvider(response_content="malformed")
    name, strategy, phase = {
        "triage": ("triage", ProviderTaskTriage, WorkflowPhase.TRIAGE),
        "decomposition": ("decomposer", ProviderTaskDecomposer, WorkflowPhase.DELEGATION),
        "manager": ("manager_reviewer", ProviderManagerReviewer, WorkflowPhase.MANAGER_REVIEW),
        "final": ("final_reviewer", ProviderFinalReviewer, WorkflowPhase.FINAL_REVIEW),
    }[stage]
    tree = framework(orchestration_backend=LangGraphOrchestrationBackend(), **{name: strategy(provider)})
    _, specialist = register_pair(tree)
    worker = MockProvider()
    tree.register_provider(worker)
    tree.bind_provider(specialist, worker)
    with pytest.raises(ProviderRuntimeError if runtime else DecisionParseError) as caught:
        tree.run(Task(objective="Work"))
    if runtime:
        assert caught.value is error
    assert tree.last_state.failed_phase is phase
    assert tree.last_state.current_phase is WorkflowPhase.FAILED


def test_graph_preserves_missing_binding_error(fake_api: list[FakeGraph]) -> None:
    from tests.test_framework import framework, register_pair

    tree = framework(orchestration_backend=LangGraphOrchestrationBackend())
    register_pair(tree)
    with pytest.raises(KeyError):
        tree.run(Task(objective="Unbound"))
    assert tree.last_state.failed_phase is WorkflowPhase.EXECUTION
    assert tree.last_state.delegation_result is not None


@pytest.mark.parametrize("graph", [False, True])
def test_last_state_tracks_progress_inside_callbacks(fake_api: list[FakeGraph], graph: bool) -> None:
    from tests.test_framework import ready

    class ObservingReviewer(StaticManagerReviewer):
        def review(self, task: Any, subtask: Any, manager: Any, specialist_executions: Any) -> Any:
            assert tree.last_state.current_phase is WorkflowPhase.MANAGER_REVIEW
            assert tree.last_state.execution_result is not None
            return super().review(task, subtask, manager, specialist_executions)

    backend = LangGraphOrchestrationBackend() if graph else SequentialOrchestrationBackend()
    tree, _ = ready(orchestration_backend=backend, manager_reviewer=ObservingReviewer())
    assert tree.run(Task(objective="Work")).success


def test_backend_contract_and_sdk_type_validation() -> None:
    from tests.test_framework import framework

    with pytest.raises(TypeError):
        BaseOrchestrationBackend()
    with pytest.raises(TypeError, match="BaseOrchestrationBackend"):
        framework(orchestration_backend="langgraph")


def test_graph_compile_failure_is_chained(fake_api: list[FakeGraph], monkeypatch: pytest.MonkeyPatch) -> None:
    cause = RuntimeError("Compile failed")

    def broken(self: FakeGraph) -> Any:
        raise cause

    monkeypatch.setattr(FakeGraph, "compile", broken)
    with pytest.raises(BackendConfigurationError) as caught:
        LangGraphOrchestrationBackend()
    assert caught.value.__cause__ is cause


def test_invalid_backend_result_is_rejected() -> None:
    class InvalidBackend(BaseOrchestrationBackend):
        def run(self, context: Any) -> Any:
            return "success"

    tree = make_framework(InvalidBackend())
    with pytest.raises(BackendConfigurationError):
        tree.run(Task(objective="Work"))
    assert tree.last_state.current_phase is WorkflowPhase.FAILED


@pytest.mark.skipif(importlib.util.find_spec("langgraph") is None, reason="Optional LangGraph is not installed")
def test_real_graph_propagates_original_provider_exception() -> None:
    from tests.test_framework import framework

    cause = ProviderRuntimeError("Offline failure")

    class Broken(MockProvider):
        def generate(self, request: ProviderRequest) -> ProviderResponse:
            raise cause

    tree = framework(orchestration_backend=LangGraphOrchestrationBackend(), triage=ProviderTaskTriage(Broken()))
    with pytest.raises(ProviderRuntimeError) as caught:
        tree.run(Task(objective="Work"))
    assert caught.value is cause
    assert tree.last_state.failed_phase is WorkflowPhase.TRIAGE


def test_base_import_and_sequential_run_with_langgraph_imports_blocked() -> None:
    script = '''
import importlib.abc
import sys
class BlockLangGraph(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'langgraph':
            raise ModuleNotFoundError('Blocked optional LangGraph')
sys.meta_path.insert(0, BlockLangGraph())
from agenttree import AgentTree
from agenttree.orchestration.backends import (
    SequentialOrchestrationBackend, LangGraphOrchestrationBackend, BackendDependencyError,
)
from examples.basic_hierarchy import main
main()
try:
    LangGraphOrchestrationBackend()
except BackendDependencyError as error:
    assert 'agenttree[langgraph]' in str(error)
else:
    raise AssertionError('Expected missing dependency error')
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Status: completed" in result.stdout
