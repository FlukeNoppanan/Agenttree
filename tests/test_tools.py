"""Tool abstractions, Python function tools, bindings, and execution tests."""

from types import MappingProxyType
from typing import Any, Mapping
from uuid import UUID

import pytest

from agenttree.agents import SpecialistAgent
from agenttree.models import ExecutionTrace
from agenttree.tools import (
    BaseTool,
    FunctionTool,
    ToolBindingRegistry,
    ToolExecutor,
    ToolRegistry,
    ToolResult,
)
from agenttree.tracing import ExecutionEventType


class EchoTool(BaseTool):
    """Minimal custom tool used to verify the public base contract."""

    def invoke(self, arguments: Mapping[str, Any]) -> ToolResult:
        return ToolResult(tool_id=self.id, success=True, output=dict(arguments))


def _add(a: int, b: int) -> int:
    return a + b


def test_base_tool_is_abstract_and_supports_custom_implementations() -> None:
    with pytest.raises(TypeError):
        BaseTool(name="base")  # type: ignore[abstract]

    class IncompleteTool(BaseTool):
        pass

    with pytest.raises(TypeError):
        IncompleteTool(name="incomplete")
    custom = EchoTool(name="echo", description="Echo arguments")
    assert custom.invoke({"value": 3}).output == {"value": 3}


def test_function_tool_creation_and_stable_unique_ids() -> None:
    first = FunctionTool(name="add", description="Add values", function=_add)
    second = FunctionTool(name="sum", function=_add)
    configured = FunctionTool(name="configured", function=_add, tool_id="tool-1")
    assert first.name == "add"
    assert first.description == "Add values"
    assert first.id != second.id
    assert UUID(first.id).version == 4
    assert configured.id == "tool-1"


@pytest.mark.parametrize("invalid", (None, 3, "callable name"))
def test_function_tool_rejects_non_callable_values(invalid: object) -> None:
    with pytest.raises(TypeError, match="callable"):
        FunctionTool(name="invalid", function=invalid)  # type: ignore[arg-type]


def test_function_tool_rejects_async_functions() -> None:
    async def async_function(value: int) -> int:
        return value

    with pytest.raises(TypeError, match="async"):
        FunctionTool(name="async", function=async_function)


def test_signature_inspection_captures_parameters_defaults_and_annotations() -> None:
    def configured(
        required: int,
        optional: str = "value",
        *,
        enabled: bool = True,
        **extras: object,
    ) -> dict[str, object]:
        return {"required": required, "optional": optional, **extras}

    tool = FunctionTool(name="configured", function=configured)
    parameters = tool.input_spec.parameters
    assert tuple(item.name for item in parameters) == (
        "required", "optional", "enabled",
    )
    assert tuple(item.required for item in parameters) == (True, False, False)
    assert tuple(item.annotation for item in parameters) == ("int", "str", "bool")
    assert tuple(item.default for item in parameters) == (None, "'value'", "True")
    assert tuple(item.kind for item in parameters) == (
        "positional_or_keyword", "positional_or_keyword", "keyword_only",
    )
    assert tool.input_spec.accepts_var_keyword is True


def test_nonserializable_annotations_and_defaults_have_safe_text() -> None:
    marker = object()

    def configured(value: "ForwardType" = marker) -> object:
        return value

    parameter = FunctionTool(
        name="configured", function=configured,
    ).input_spec.parameters[0]
    assert parameter.annotation == "ForwardType"
    assert parameter.default == "<builtins.object>"
    assert "0x" not in parameter.default


def test_function_tool_success_and_optional_default_behavior() -> None:
    def greeting(name: str, punctuation: str = "!") -> str:
        return f"Hello {name}{punctuation}"

    tool = FunctionTool(name="greeting", function=greeting)
    result = tool.invoke({"name": "Ada"})
    assert result == ToolResult(
        tool_id=tool.id, success=True, output="Hello Ada!",
    )


def test_function_tool_preserves_arbitrary_structured_output() -> None:
    def structured(value: int) -> dict[str, object]:
        return {"items": [{"value": value}], "complete": True}

    tool = FunctionTool(name="structured", function=structured)
    assert tool.invoke({"value": 4}).output == {
        "items": [{"value": 4}], "complete": True,
    }


def test_function_exception_becomes_failed_tool_result() -> None:
    def failing(value: int) -> int:
        raise RuntimeError(f"cannot process {value}")

    tool = FunctionTool(name="failing", function=failing)
    result = tool.invoke({"value": 7})
    assert result.success is False
    assert result.output is None
    assert result.error == "RuntimeError: cannot process 7"
    assert result.metadata == {"error_type": "RuntimeError"}


def test_argument_binding_rejects_missing_and_unknown_arguments() -> None:
    tool = FunctionTool(name="add", function=_add)
    with pytest.raises(TypeError, match="missing a required argument"):
        tool.invoke({"a": 1})
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        tool.invoke({"a": 1, "b": 2, "extra": 3})


def test_var_keyword_parameters_allow_additional_arguments() -> None:
    def collect(required: int, **values: int) -> dict[str, object]:
        return {"required": required, "values": values}

    tool = FunctionTool(name="collect", function=collect)
    result = tool.invoke({"required": 1, "first": 2, "second": 3})
    assert result.output == {
        "required": 1, "values": {"first": 2, "second": 3},
    }


def test_positional_only_and_var_positional_signatures_are_rejected() -> None:
    def positional_only(value: int, /) -> int:
        return value

    def var_positional(*values: int) -> tuple[int, ...]:
        return values

    with pytest.raises(ValueError, match="positional-only"):
        FunctionTool(name="positional", function=positional_only)
    with pytest.raises(ValueError, match="variadic positional"):
        FunctionTool(name="variadic", function=var_positional)


def test_tool_result_and_tool_metadata_are_isolated() -> None:
    output = {"nested": {"value": 1}}
    metadata = {"labels": ["local"]}
    result = ToolResult(
        tool_id="tool", success=True, output=output, metadata=metadata,
    )
    tool = EchoTool(name="echo", metadata=metadata)
    output["nested"]["value"] = 2
    metadata["labels"].append("changed")
    assert result.output == {"nested": {"value": 1}}
    assert result.metadata == {"labels": ["local"]}
    assert tool.metadata == {"labels": ["local"]}
    with pytest.raises(TypeError):
        tool.metadata["new"] = True  # type: ignore[index]


def test_tool_registry_registers_gets_and_unregisters_by_id_and_name() -> None:
    registry = ToolRegistry()
    tool = FunctionTool(name="  Add Values  ", function=_add, tool_id="add-1")
    registry.register(tool)
    assert registry.get("add-1") is tool
    assert registry.get_by_id(" add-1 ") is tool
    assert registry.get_by_name("add values") is tool
    assert registry.get_by_name(" ADD VALUES ") is tool
    assert registry.unregister("add-1") is tool
    assert registry.tools == ()
    with pytest.raises(KeyError):
        registry.get_by_name("add values")


def test_tool_registry_rejects_duplicate_ids_and_normalized_names() -> None:
    registry = ToolRegistry()
    original = FunctionTool(name="Add", function=_add, tool_id="first")
    registry.register(original)
    with pytest.raises(ValueError, match="ID already"):
        registry.register(FunctionTool(name="Other", function=_add, tool_id="first"))
    with pytest.raises(ValueError, match="name already"):
        registry.register(FunctionTool(name=" add ", function=_add, tool_id="second"))
    assert registry.tools == (original,)


def test_tool_registry_order_and_snapshots_are_safe() -> None:
    registry = ToolRegistry()
    first = FunctionTool(name="First", function=_add)
    second = FunctionTool(name="Second", function=_add)
    registry.register(first)
    tools_snapshot = registry.tools
    ids_snapshot = registry.ids
    names_snapshot = registry.names
    map_snapshot = registry.tool_map
    registry.register(second)
    assert tools_snapshot == (first,)
    assert ids_snapshot == (first.id,)
    assert names_snapshot == ("first",)
    assert isinstance(map_snapshot, MappingProxyType)
    with pytest.raises(TypeError):
        map_snapshot[second.id] = second  # type: ignore[index]
    assert registry.tools == (first, second)


def test_tool_registry_rejects_non_tools() -> None:
    with pytest.raises(TypeError, match="BaseTool"):
        ToolRegistry().register(object())  # type: ignore[arg-type]


def test_external_bindings_support_many_to_many_ordered_assignments() -> None:
    first_specialist = SpecialistAgent(name="First")
    second_specialist = SpecialistAgent(name="Second")
    bindings = ToolBindingRegistry()
    bindings.assign(first_specialist.id, "tool-a")
    bindings.assign(first_specialist.id, "tool-b")
    bindings.assign(second_specialist.id, "tool-a")
    assert bindings.tool_ids_for(first_specialist.id) == ("tool-a", "tool-b")
    assert bindings.tool_ids_for(second_specialist.id) == ("tool-a",)
    assert bindings.bindings == {
        first_specialist.id: ("tool-a", "tool-b"),
        second_specialist.id: ("tool-a",),
    }
    assert isinstance(bindings.bindings, MappingProxyType)


def test_bindings_reject_duplicates_and_support_unassign() -> None:
    specialist = SpecialistAgent(name="Worker")
    bindings = ToolBindingRegistry()
    bindings.assign(specialist.id, "tool")
    with pytest.raises(ValueError, match="already assigned"):
        bindings.assign(specialist.id, "tool")
    bindings.unassign(specialist.id, "tool")
    assert bindings.tool_ids_for(specialist.id) == ()
    with pytest.raises(KeyError, match="not assigned"):
        bindings.unassign(specialist.id, "tool")


def test_tool_executor_enforces_assignment_and_invokes_by_name_or_id() -> None:
    specialist = SpecialistAgent(name="Worker")
    tool = FunctionTool(name="Add", function=_add)
    registry = ToolRegistry()
    registry.register(tool)
    bindings = ToolBindingRegistry()
    executor = ToolExecutor(registry=registry, bindings=bindings)
    with pytest.raises(PermissionError, match="not assigned"):
        executor.execute(
            specialist=specialist, tool_name="add", arguments={"a": 2, "b": 3},
        )
    bindings.assign(specialist.id, tool.id)
    assert executor.execute(
        specialist=specialist, tool_name="add", arguments={"a": 2, "b": 3},
    ).output == 5
    assert executor.execute(
        specialist=specialist, tool_id=tool.id, arguments={"a": 4, "b": 5},
    ).output == 9


def test_tool_executor_requires_explicit_single_selection() -> None:
    specialist = SpecialistAgent(name="Worker")
    registry = ToolRegistry()
    executor = ToolExecutor(registry=registry)
    with pytest.raises(ValueError, match="exactly one"):
        executor.execute(specialist=specialist, arguments={})
    with pytest.raises(ValueError, match="exactly one"):
        executor.execute(
            specialist=specialist,
            tool_id="id",
            tool_name="name",
            arguments={},
        )


def test_executor_without_bindings_allows_only_explicit_registered_tools() -> None:
    specialist = SpecialistAgent(name="Worker")
    tool = FunctionTool(name="Add", function=_add)
    registry = ToolRegistry()
    registry.register(tool)
    executor = ToolExecutor(registry=registry)
    result = executor.execute(
        specialist=specialist,
        tool_id=tool.id,
        arguments={"a": 10, "b": 5},
    )
    assert result.output == 15
    with pytest.raises(KeyError):
        executor.execute(
            specialist=specialist, tool_id="unknown", arguments={},
        )


def test_tool_execution_does_not_mutate_specialist_configuration() -> None:
    specialist = SpecialistAgent(
        name="Worker", capabilities=("calculate",), metadata={"fixed": True},
    )
    before = (
        specialist.id,
        specialist.name,
        specialist.capabilities,
        specialist.metadata.copy(),
    )
    tool = FunctionTool(name="Add", function=_add)
    registry = ToolRegistry()
    registry.register(tool)
    ToolExecutor(registry=registry).execute(
        specialist=specialist,
        tool_id=tool.id,
        arguments={"a": 1, "b": 2},
    )
    assert (
        specialist.id,
        specialist.name,
        specialist.capabilities,
        specialist.metadata,
    ) == before


def test_tool_executor_records_success_and_failure_trace_events() -> None:
    def failing() -> None:
        raise RuntimeError("failed")

    specialist = SpecialistAgent(name="Worker")
    success = FunctionTool(name="Add", function=_add)
    failure = FunctionTool(name="Fail", function=failing)
    registry = ToolRegistry()
    registry.register(success)
    registry.register(failure)
    executor = ToolExecutor(registry=registry)
    trace = ExecutionTrace(task_id="task")
    executor.execute(
        specialist=specialist,
        tool_id=success.id,
        arguments={"a": 2, "b": 3},
        trace=trace,
    )
    failed_result = executor.execute(
        specialist=specialist,
        tool_id=failure.id,
        arguments={},
        trace=trace,
    )
    assert failed_result.success is False
    assert tuple(event.event_type for event in trace.events) == (
        ExecutionEventType.TOOL_EXECUTION_STARTED.value,
        ExecutionEventType.TOOL_EXECUTION_COMPLETED.value,
        ExecutionEventType.TOOL_EXECUTION_STARTED.value,
        ExecutionEventType.TOOL_EXECUTION_FAILED.value,
    )
    assert trace.last_event.actor_id == specialist.id  # type: ignore[union-attr]
    assert trace.last_event.metadata["tool_id"] == failure.id  # type: ignore[union-attr]


def test_argument_contract_error_is_traced_and_remains_distinguishable() -> None:
    specialist = SpecialistAgent(name="Worker")
    tool = FunctionTool(name="Add", function=_add)
    registry = ToolRegistry()
    registry.register(tool)
    trace = ExecutionTrace(task_id="task")
    with pytest.raises(TypeError, match="missing a required argument"):
        ToolExecutor(registry=registry).execute(
            specialist=specialist,
            tool_id=tool.id,
            arguments={"a": 1},
            trace=trace,
        )
    assert tuple(event.event_type for event in trace.events) == (
        ExecutionEventType.TOOL_EXECUTION_STARTED.value,
        ExecutionEventType.TOOL_EXECUTION_FAILED.value,
    )
