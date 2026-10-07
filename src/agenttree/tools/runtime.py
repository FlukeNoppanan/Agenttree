"""One policy-enforced model tool runtime for every agent role."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from contextvars import ContextVar
from contextvars import copy_context
from copy import deepcopy
import json
import re
from threading import Event, Thread
from time import monotonic
from typing import Any

from agenttree.agents import BaseAgent, RootAgent, ManagerAgent
from agenttree.config import AgentTreeConfig
from agenttree.models import ExecutionEvent
from agenttree.providers import BaseProvider, ProviderRequest, ProviderResponse
from agenttree.providers.exceptions import ProviderConfigurationError
from agenttree.tools.bindings import ToolBindingRegistry
from agenttree.tools.models import ToolCall, ToolResult
from agenttree.tools.registry import ToolRegistry


_active_tool_session: ContextVar[ToolSession | None] = ContextVar("agenttree_tool_session", default=None)
_active_tool_cancellation: ContextVar[Event | None] = ContextVar("agenttree_tool_cancellation", default=None)
_SECRET_KEYS = {"password", "secret", "token", "api_key", "apikey", "authorization", "credential", "private_key"}
_SECRET_PATTERN = re.compile(r"(?i)(bearer\s+\S+|(?:api[_-]?key|token|password|secret)\s*[:=]\s*\S+)")


def _safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: "[REDACTED]" if any(part in key.casefold() for part in _SECRET_KEYS) else _safe_value(item)
                for key, item in value.items() if isinstance(key, str)}
    if isinstance(value, (tuple, list)):
        return [_safe_value(item) for item in value]
    if isinstance(value, str):
        return _SECRET_PATTERN.sub("[REDACTED]", value)
    return value


def _json_bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"))


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate tool argument key")
        result[key] = value
    return result


def _reject_constant(_: str) -> None:
    raise ValueError("Nonfinite tool argument")


def _schema(tool: Any) -> dict[str, Any]:
    definition = getattr(tool, "definition", None)
    native_schema = getattr(definition, "input_schema", None)
    if isinstance(native_schema, dict) and native_schema.get("type", "object") == "object":
        return deepcopy(native_schema)
    properties = {}
    required = []
    kinds = {"str": "string", "int": "integer", "float": "number", "bool": "boolean",
             "dict": "object", "list": "array", "None": "null"}
    for parameter in tool.input_spec.parameters:
        annotation = (parameter.annotation or "").strip()
        typ = kinds.get(annotation, annotation if annotation in {"string", "integer", "number", "boolean", "object", "array", "null"} else None)
        properties[parameter.name] = {"type": typ} if typ else {}
        if parameter.description:
            properties[parameter.name]["description"] = parameter.description
        if parameter.required:
            required.append(parameter.name)
    from agenttree.core.artifacts import ArtifactOutputTool
    if isinstance(tool, ArtifactOutputTool):
        for name in ("content", "path", "media_type", "supersedes_artifact_id"):
            properties[name]["type"] = ["string", "null"]
        properties["content"]["description"] = "Only the artifact body. JSON must be a JSON-encoded string; omit unrelated context."
        properties["media_type"]["description"] = "MIME type only, e.g. text/plain or application/json. Do not include charset parameters."
        properties["operation"]["description"] = "Use none for report outputs. File intents create/modify/delete require a safe relative path."
        properties["type"]["enum"] = ["text", "code", "json", "file", "patch", "reference"]
        properties["operation"]["enum"] = ["none", "create", "modify", "delete"]
    return {"type": "object", "properties": properties, "required": required,
            "additionalProperties": tool.input_spec.accepts_var_keyword}


def _validate_schema(value: Any, schema: dict[str, Any], depth: int = 0) -> None:
    if depth > 16 or any(key in schema for key in ("$ref", "allOf", "anyOf", "oneOf", "patternProperties")):
        raise ValueError("Unsupported tool argument schema")
    expected = schema.get("type")
    kinds = {"string": lambda x: isinstance(x, str),
             "integer": lambda x: isinstance(x, int) and not isinstance(x, bool),
             "number": lambda x: isinstance(x, (int, float)) and not isinstance(x, bool),
             "boolean": lambda x: isinstance(x, bool), "object": lambda x: isinstance(x, dict),
             "array": lambda x: isinstance(x, list), "null": lambda x: x is None}
    if expected is not None:
        allowed = expected if isinstance(expected, list) else [expected]
        if (not allowed or any(not isinstance(kind, str) or kind not in kinds for kind in allowed)):
            raise ValueError("Unsupported tool argument schema")
        if not any(kinds[kind](value) for kind in allowed):
            raise ValueError("Tool argument has invalid type")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError("Tool argument is outside allowed values")
    if isinstance(value, str):
        if not value.strip() or len(value) < schema.get("minLength", 0):
            raise ValueError("Tool argument is empty or too short")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ValueError("Tool argument is too long")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError("Tool argument is below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValueError("Tool argument is above maximum")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or ("maxItems" in schema and len(value) > schema["maxItems"]):
            raise ValueError("Tool argument array length is invalid")
        item_schema = schema.get("items", {})
        if not isinstance(item_schema, dict):
            raise ValueError("Unsupported tool argument schema")
        for item in value:
            _validate_schema(item, item_schema, depth + 1)
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", ())
        if not isinstance(properties, dict) or not isinstance(required, (list, tuple)):
            raise ValueError("Unsupported tool argument schema")
        for name in required:
            if name not in value:
                raise ValueError("A required tool argument is missing")
        extras = set(value) - set(properties)
        additional = schema.get("additionalProperties", False)
        if extras and additional is False:
            raise ValueError("Unknown tool argument")
        if extras and additional is not True and not isinstance(additional, dict):
            raise ValueError("Unsupported tool argument schema")
        for name, item in value.items():
            child = properties.get(name, additional if isinstance(additional, dict) else {})
            if not isinstance(child, dict):
                raise ValueError("Unsupported tool argument schema")
            _validate_schema(item, child, depth + 1)


def _validate(tool: Any, arguments: Any, limit: int) -> dict[str, Any]:
    if isinstance(arguments, str):
        if len(arguments.encode("utf-8")) > limit:
            raise ValueError("Tool arguments exceed size limit")
        try:
            arguments = json.loads(arguments, object_pairs_hook=_unique_pairs,
                                   parse_constant=_reject_constant)
        except ValueError:
            raise ValueError("Tool arguments are malformed JSON") from None
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a JSON object")
    try:
        if _json_bytes(arguments) > limit:
            raise ValueError("Tool arguments exceed size limit")
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise ValueError("Tool arguments must be JSON compatible") from None
    _validate_schema(arguments, _schema(tool))
    return deepcopy(arguments)


class ArtifactArgumentLimitError(ValueError):
    """Safe actionable boundary feedback, without argument content."""


def _argument_limit(tool: Any, config: AgentTreeConfig) -> int:
    from agenttree.core.artifacts import ArtifactOutputTool
    if isinstance(tool, ArtifactOutputTool):
        # JSON escaping can expand one content byte to six (e.g. a control byte).
        # This is a transport ceiling, not permission for larger content/metadata.
        return 6 * config.max_artifact_bytes + config.max_tool_argument_bytes
    return config.max_tool_argument_bytes


def _validate_arguments(tool: Any, arguments: Any, config: AgentTreeConfig) -> dict[str, Any]:
    prepared = _validate(tool, arguments, _argument_limit(tool, config))
    from agenttree.core.artifacts import ArtifactOutputTool
    if isinstance(tool, ArtifactOutputTool):
        content = prepared.get("content")
        if isinstance(content, str) and len(content.encode("utf-8")) > config.max_artifact_bytes:
            raise ArtifactArgumentLimitError(
                f"Artifact content exceeds {config.max_artifact_bytes} UTF-8 bytes; shorten the body or split it into bounded artifacts.")
        metadata = {key: value for key, value in prepared.items() if key != "content"}
        if _json_bytes(metadata) > config.max_tool_argument_bytes:
            raise ArtifactArgumentLimitError(
                f"Artifact non-content arguments exceed {config.max_tool_argument_bytes} UTF-8 bytes; remove unrelated context.")
    return prepared


class ToolSession:
    """Per-run authorization, budgets, metrics, and safe trace events."""

    def __init__(self, registry: ToolRegistry, bindings: ToolBindingRegistry,
                 agents: tuple[BaseAgent, ...], config: AgentTreeConfig, execution_id: str) -> None:
        self.registry, self.bindings, self.config, self.execution_id = registry, bindings, config, execution_id
        self.agents = {agent.id: agent for agent in agents}
        self.events: list[ExecutionEvent] = []
        self.calls: Counter[str] = Counter()
        self.metrics: Counter[str] = Counter()

    def _event(self, call: ToolCall, status: str, tool_id: str | None = None,
               duration_ms: float | None = None) -> None:
        cancelled = _active_tool_cancellation.get()
        if cancelled is not None and cancelled.is_set():
            return
        self.events.append(ExecutionEvent(
            task_id=self.execution_id, actor_id=call.agent_id,
            event_type=f"{call.agent_role}.tool.{status}",
            metadata={"execution_id": self.execution_id, "agent_role": call.agent_role,
                      "tool_id": tool_id, "call_id": call.call_id,
                      "status": status, "duration_ms": duration_ms},
        ))

    def definitions(self, agent_id: str) -> tuple[dict[str, Any], ...]:
        result = []
        for tool_id in self.bindings.tool_ids_for(agent_id):
            try:
                tool = self.registry.get(tool_id)
            except KeyError:
                continue
            if tool.enabled:
                result.append({"name": tool.name, "description": tool.description,
                               "parameters": _schema(tool)})
        return tuple(result)

    def invoke(self, call: ToolCall) -> ToolResult:
        from agenttree.core.operation_journal import current_journal
        journal = current_journal()
        if journal is None:
            return self._invoke_impl(call)
        try:
            tool = self.registry.get_by_name(call.tool_name)
            policy = tool.recovery_policy.value
        except (KeyError, TypeError, ValueError):
            tool = None
            policy = "pure"
        if (tool is not None and
                (not tool.enabled or not self.bindings.is_assigned(call.agent_id, tool.id))):
            policy = "pure"
        key = journal.child_key(f"tool:{call.call_id}")
        before_events = len(self.events)
        before_calls = dict(self.calls)
        before_metrics = dict(self.metrics)
        invoked = False

        def act():
            nonlocal invoked
            invoked = True
            result = self._invoke_impl(call)
            return (result, tuple(self.events[before_events:]),
                    {name: value - before_calls.get(name, 0) for name, value in self.calls.items()},
                    {name: value - before_metrics.get(name, 0) for name, value in self.metrics.items()})

        def reconcile(operation):
            if tool is None:
                return None
            done = Event()
            received = []
            errors = []
            context = copy_context()
            def work():
                try:
                    received.append(tool.reconcile(operation))
                except Exception as error:
                    errors.append(error)
                finally:
                    done.set()
            Thread(target=lambda: context.run(work), daemon=True,
                   name="agenttree-tool-reconciliation").start()
            from agenttree.core.execution_control import _active_execution_control, check_execution
            control = _active_execution_control.get()
            remaining = control.remaining() if control is not None else None
            wait_time = min(self.config.tool_timeout, remaining) if remaining is not None else self.config.tool_timeout
            if not done.wait(wait_time):
                check_execution()
                raise TimeoutError("Tool reconciliation timed out")
            check_execution()
            if errors:
                raise errors[0]
            outcome = received[0] if received else None
            from agenttree.tools.base import ToolReconciliation, ToolReconciliationStatus
            from agenttree.core.operation_journal import RECONCILED_NOT_COMPLETED, ReconciledFailure
            if isinstance(outcome, ToolReconciliation):
                if outcome.status is ToolReconciliationStatus.NOT_COMPLETED:
                    return RECONCILED_NOT_COMPLETED
                if outcome.status is ToolReconciliationStatus.FAILED:
                    raise ReconciledFailure()
                outcome = outcome.result if outcome.status is ToolReconciliationStatus.COMPLETED else None
            if outcome is None:
                return None
            if (not isinstance(outcome, ToolResult) or outcome.tool_id != tool.id or
                    outcome.call_id not in (None, call.call_id)):
                raise TypeError("Tool reconciliation must return a matching ToolResult")
            output = _safe_value(outcome.output)
            if _json_bytes(output) > self.config.max_tool_result_bytes:
                raise ValueError("Reconciled Tool result exceeds size limit")
            outcome = replace(outcome, call_id=call.call_id, output=output)
            event = ExecutionEvent(task_id=self.execution_id, actor_id=call.agent_id,
                                   event_type=f"{call.agent_role}.tool.reconciled",
                                   metadata={"execution_id": self.execution_id,
                                             "tool_id": tool.id, "call_id": call.call_id,
                                             "status": "reconciled"})
            return outcome, (event,), {call.agent_id: 1}, {"calls": 1,
                "executed": 1, "completed" if outcome.success else "failed": 1}

        result, events, call_delta, metric_delta = journal.run(
            key, "tool.call", {"agent_id": call.agent_id, "tool_name": call.tool_name,
                               "call_id": call.call_id, "arguments": _safe_value(call.arguments)},
            act, policy=policy, agent_id=call.agent_id,
            reconcile=reconcile if policy == "reconcilable" else None)
        if not invoked:
            self.events.extend(events)
            self.calls.update(call_delta)
            self.metrics.update(metric_delta)
        return result

    def _invoke_impl(self, call: ToolCall) -> ToolResult:
        from agenttree.core.execution_control import check_execution
        check_execution()
        cancelled = _active_tool_cancellation.get()
        if cancelled is not None and cancelled.is_set():
            return ToolResult(tool_id=call.tool_name or "unknown", call_id=call.call_id,
                              success=False, error="ToolTimeout", error_type="ToolTimeout")
        self._event(call, "requested")
        if call.agent_id not in self.agents or call.execution_id != self.execution_id:
            return self._deny(call, None, "ToolNotAssigned")
        agent = self.agents[call.agent_id]
        role = "root" if isinstance(agent, RootAgent) else "manager" if isinstance(agent, ManagerAgent) else "specialist"
        if call.agent_role != role:
            return self._deny(call, None, "ToolNotAssigned")
        if self.calls[call.agent_id] >= self.config.max_tool_calls:
            self._event(call, "budget_exhausted")
            return self._error(call, None, "ToolBudgetExceeded")
        self.calls[call.agent_id] += 1
        self.metrics["calls"] += 1
        try:
            tool = self.registry.get_by_name(call.tool_name)
        except (KeyError, TypeError, ValueError):
            return self._deny(call, None, "ToolNotFound")
        if not self.bindings.is_assigned(call.agent_id, tool.id):
            return self._deny(call, tool.id, "ToolNotAssigned")
        if not tool.enabled:
            return self._deny(call, tool.id, "ToolDisabled")
        try:
            arguments = _validate_arguments(tool, call.arguments, self.config)
        except ArtifactArgumentLimitError as error:
            return replace(self._deny(call, tool.id, "ToolArgumentValidationError"), error=str(error))
        except ValueError:
            return self._deny(call, tool.id, "ToolArgumentValidationError")
        self.metrics["executed"] += 1
        self._event(call, "authorized", tool.id)
        self._event(call, "started", tool.id)
        started = monotonic()
        completed = Event()
        outcome: list[Any] = []

        def run() -> None:
            from agenttree.core.artifacts import _active_artifact_producer
            producer_token = _active_artifact_producer.set(("tool", call.agent_id))
            try:
                outcome.append(tool.invoke(arguments))
            except Exception:
                outcome.append(None)
            finally:
                _active_artifact_producer.reset(producer_token)
                completed.set()

        context = copy_context()
        Thread(target=lambda: context.run(run), daemon=True, name="agenttree-tool").start()
        from agenttree.core.execution_control import _active_execution_control
        control = _active_execution_control.get()
        remaining = control.remaining() if control is not None else None
        wait_time = min(self.config.tool_timeout, remaining) if remaining is not None else self.config.tool_timeout
        if not completed.wait(wait_time):
            check_execution()
            duration = round((monotonic() - started) * 1000, 2)
            self.metrics["timeouts"] += 1
            self._event(call, "timeout", tool.id, duration)
            return self._error(call, tool.id, "ToolTimeout", duration)
        if cancelled is not None and cancelled.is_set():
            return self._error(call, tool.id, "ToolTimeout")
        check_execution()
        duration = round((monotonic() - started) * 1000, 2)
        result = outcome[0] if outcome else None
        if not isinstance(result, ToolResult) or result.tool_id != tool.id:
            self._event(call, "failed", tool.id, duration)
            return self._error(call, tool.id, "ToolExecutionError", duration)
        if not result.success:
            self._event(call, "failed", tool.id, duration)
            from agenttree.core.artifacts import ArtifactOutputTool
            failed = self._error(call, tool.id, "ToolExecutionError", duration)
            if isinstance(tool, ArtifactOutputTool) and result.metadata.get("error_type") == "ArtifactValidationError":
                # Static remediation only; never forward arbitrary Tool exception text.
                return replace(failed, error="Artifact output rejected. Use a supported type, valid JSON text for JSON, "
                               "operation=none for reports, and a MIME type without charset parameters. "
                               "Keep the body within the configured artifact byte limit.")
            return failed
        try:
            output = _safe_value(result.output)
            if _json_bytes(output) > self.config.max_tool_result_bytes:
                self._event(call, "failed", tool.id, duration)
                return self._error(call, tool.id, "ToolResultTooLarge", duration)
        except (TypeError, ValueError, OverflowError):
            self._event(call, "failed", tool.id, duration)
            return self._error(call, tool.id, "ToolExecutionError", duration)
        self.metrics["completed"] += 1
        self._event(call, "completed", tool.id, duration)
        return ToolResult(tool_id=tool.id, call_id=call.call_id, success=True,
                          output=output, duration_ms=duration)

    def _error(self, call: ToolCall, tool_id: str | None, kind: str,
               duration: float | None = None) -> ToolResult:
        self.metrics["failed"] += 1
        return ToolResult(tool_id=tool_id or call.tool_name, call_id=call.call_id,
                          success=False, error=kind, error_type=kind, duration_ms=duration)

    def _deny(self, call: ToolCall, tool_id: str | None, kind: str) -> ToolResult:
        self.metrics["denied"] += 1
        self._event(call, "denied", tool_id)
        return self._error(call, tool_id, kind)

    def generate(self, agent_id: str, provider: BaseProvider, request: ProviderRequest,
                 strategy: str) -> ProviderResponse:
        from agenttree.core.execution_control import check_execution
        check_execution()
        definitions = self.definitions(agent_id)
        if definitions and provider.capabilities.tool_calling is not True:
            raise ProviderConfigurationError("Selected provider has no declared tool calling support")
        model_name = request.model or provider.config.model
        cached_models = getattr(provider, "_model_cache", None)
        if definitions and cached_models is not None:
            selected = next((model for model in cached_models if model.id == model_name), None)
            if (selected is not None and selected.capabilities is not None and
                    selected.capabilities.tool_calling is False):
                raise ProviderConfigurationError("Selected model does not support tool calling")
        history: tuple[dict[str, Any], ...] = ()
        for round_number in range(self.config.max_tool_rounds + 1):
            check_execution()
            cancelled = _active_tool_cancellation.get()
            if cancelled is not None and cancelled.is_set():
                raise ProviderConfigurationError("Cancelled collaboration response")
            current = replace(request, tools=definitions,
                              tool_choice="auto" if definitions else None,
                              tool_history=history,
                              tool_argument_limits={definition["name"]: _argument_limit(
                                  self.registry.get_by_name(definition["name"]), self.config)
                                  for definition in definitions})
            from agenttree.core.operation_journal import current_journal
            journal = current_journal()
            response = (journal.run(journal.next_key("provider"), "provider.generate",
                                    (agent_id, provider.name, current),
                                    lambda: _provider_response(provider, current, self.config.provider_streaming,
                                                               agent_id=agent_id, strategy=strategy,
                                                               max_bytes=self.config.max_provider_stream_bytes,
                                                               max_tool_bytes=self.config.max_tool_argument_bytes), agent_id=agent_id)
                        if journal is not None else _provider_response(provider, current, self.config.provider_streaming,
                                                                       agent_id=agent_id, strategy=strategy,
                                                                       max_bytes=self.config.max_provider_stream_bytes,
                                                                       max_tool_bytes=self.config.max_tool_argument_bytes))
            check_execution()
            if cancelled is not None and cancelled.is_set():
                raise ProviderConfigurationError("Cancelled collaboration response")
            if not isinstance(response, ProviderResponse):
                raise TypeError("provider.generate must return a ProviderResponse")
            if not response.tool_calls:
                return response
            from agenttree.core.usage import record_usage
            record_usage(strategy, response)
            if len(response.tool_calls) > self.config.max_tool_calls:
                self.metrics["failed"] += 1
                self.events.append(ExecutionEvent(
                    task_id=self.execution_id, actor_id=agent_id,
                    event_type="tool.budget_exhausted",
                    metadata={"execution_id": self.execution_id, "agent_id": agent_id},
                ))
                raise ProviderConfigurationError("Tool call budget exceeded by provider response")
            if round_number == self.config.max_tool_rounds:
                for item in response.tool_calls:
                    call = self._call(agent_id, item)
                    self._event(call, "budget_exhausted")
                raise ProviderConfigurationError("Tool round budget exhausted")
            results = []
            calls = []
            for item in response.tool_calls:
                call = self._call(agent_id, item)
                safe_arguments: Any = call.arguments
                try:
                    tool = self.registry.get_by_name(call.tool_name)
                    from agenttree.providers._adapter import tool_argument_bytes
                    if tool_argument_bytes(safe_arguments) > _argument_limit(tool, self.config):
                        safe_arguments = {}
                except (KeyError, TypeError, ValueError, OverflowError, RecursionError):
                    safe_arguments = {}
                if isinstance(safe_arguments, str):
                    try:
                        safe_arguments = json.loads(safe_arguments)
                    except ValueError:
                        safe_arguments = {}
                calls.append({"id": call.call_id, "name": call.tool_name,
                              "arguments": _safe_value(safe_arguments)})
                result = self.invoke(call)
                check_execution()
                results.append({"call_id": call.call_id, "name": call.tool_name,
                                "success": result.success, "output": result.output,
                                "error": result.error or result.error_type})
            history += ({"calls": calls, "results": results},)
        raise AssertionError("Tool loop did not terminate")

    def _call(self, agent_id: str, item: Any) -> ToolCall:
        if not isinstance(item, dict):
            raise ProviderConfigurationError("Provider tool call is malformed")
        function = item.get("function") if isinstance(item.get("function"), dict) else item
        name = function.get("name")
        call_id = item.get("id")
        if not isinstance(name, str) or not name or not isinstance(call_id, str) or not call_id:
            raise ProviderConfigurationError("Provider tool call lacks name or ID")
        agent = self.agents[agent_id]
        role = "root" if isinstance(agent, RootAgent) else "manager" if isinstance(agent, ManagerAgent) else "specialist"
        return ToolCall(call_id, name, agent_id, role, function.get("arguments"), self.execution_id)


def generate_with_tools(agent_id: str | None, provider: BaseProvider,
                        request: ProviderRequest, strategy: str) -> ProviderResponse:
    from agenttree.core.execution_control import check_execution
    check_execution()
    session = _active_tool_session.get()
    if session is None or agent_id is None or agent_id not in session.agents:
        from agenttree.core.operation_journal import current_journal
        journal = current_journal()
        response = (journal.run(journal.next_key("provider"), "provider.generate",
                                (agent_id, provider.name, request),
                                lambda: _provider_response(provider, request, False, agent_id=agent_id, strategy=strategy), agent_id=agent_id)
                    if journal is not None else _provider_response(provider, request, False, agent_id=agent_id, strategy=strategy))
    else:
        response = session.generate(agent_id, provider, request, strategy)
    check_execution()
    return response


def _raw_provider_response(provider: BaseProvider, request: ProviderRequest,
                       streaming: bool, *, agent_id: str | None = None,
                       strategy: str = "internal", max_bytes: int = 1_000_000,
                       max_tool_bytes: int = 16_384) -> ProviderResponse:
    """Commit only a complete provider result; deltas never become tool calls."""
    if not streaming or provider.capabilities.streaming is not True:
        return provider.generate(request)
    from agenttree.core.execution_control import check_execution
    from agenttree.core.live_output import emit_output_delta
    from agenttree.providers.models import ProviderStreamChunk
    final = None
    accumulated = 0
    for chunk in provider.generate_stream(request):
        check_execution()
        if not isinstance(chunk, ProviderStreamChunk):
            raise TypeError("provider stream must yield ProviderStreamChunk")
        if final is not None:
            raise ValueError("provider stream yielded after final response")
        accumulated += len(chunk.delta_text.encode("utf-8"))
        if accumulated > max_bytes:
            raise ValueError("provider stream exceeded text limit")
        if chunk.delta_text and agent_id is not None and strategy in ("specialist", "root_synthesis"):
            emit_output_delta("specialist" if strategy == "specialist" else "root",
                              agent_id, chunk.delta_text)
        if chunk.response is not None:
            if len(chunk.response.content.encode("utf-8")) > max_bytes:
                raise ValueError("provider stream exceeded response limit")
            for call in chunk.response.tool_calls:
                function = call.get("function", call)
                arguments = function.get("arguments", "")
                from agenttree.providers._adapter import tool_argument_bytes, tool_argument_limit
                if tool_argument_bytes(arguments) > tool_argument_limit(request, function.get("name"), max_tool_bytes):
                    raise ValueError("provider stream exceeded tool argument limit")
            final = chunk.response
    if final is None:
        raise ValueError("provider stream ended without a final response")
    return final


def _provider_response(provider: BaseProvider, request: ProviderRequest,
                       streaming: bool, *, agent_id: str | None = None,
                       strategy: str = "internal", max_bytes: int = 1_000_000,
                       max_tool_bytes: int = 16_384) -> ProviderResponse:
    from agenttree.providers.traffic import governed, traffic_context
    with traffic_context(agent_id=agent_id, strategy=strategy):
        return _raw_provider_response(governed(provider), request, streaming,
            agent_id=agent_id, strategy=strategy, max_bytes=max_bytes,
            max_tool_bytes=max_tool_bytes)
