"""Allowlisted JSON codec for durable execution data; never uses pickle."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
import importlib
import json
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping

_MODULES = (
    "agenttree.models.task", "agenttree.models.result", "agenttree.models.execution",
    "agenttree.models.triage", "agenttree.models.subtask", "agenttree.models.state",
    "agenttree.models.collaboration", "agenttree.orchestration.models",
    "agenttree.models.artifact",
    "agenttree.providers.models",
    "agenttree.tools.models",
)
_CLASSES: dict[str, type] = {}
for _module_name in _MODULES:
    _module = importlib.import_module(_module_name)
    for _name, _type in vars(_module).items():
        if (isinstance(_type, type) and _type.__module__ == _module_name and
                (is_dataclass(_type) or issubclass(_type, Enum))):
            _CLASSES[f"{_module_name}.{_name}"] = _type

_PRIVATE = frozenset(("password", "secret", "token", "credential", "authorization",
                      "api_key", "access_token", "refresh_token", "private_key"))


def _private_key(key: str) -> bool:
    normalized = key.casefold().replace("-", "_")
    return normalized in _PRIVATE or normalized.endswith(("_password", "_secret", "_credential", "_api_key"))


def _class_for(key: str) -> type | None:
    if key not in _CLASSES and key.startswith(("agenttree.core.execution_store.",
                                               "agenttree.core.root.")):
        module_name = key.rsplit(".", 1)[0]
        module = importlib.import_module(module_name)
        name = key.rsplit(".", 1)[1]
        candidate = getattr(module, name, None)
        if (isinstance(candidate, type) and candidate.__module__ == module.__name__ and
                (is_dataclass(candidate) or issubclass(candidate, Enum))):
            _CLASSES[key] = candidate
    return _CLASSES.get(key)


class ExecutionDataError(ValueError):
    """Malformed, oversized, unsupported, or unsafe durable data."""


def _encode(value: Any, depth: int = 0) -> Any:
    if depth > 48:
        raise ExecutionDataError("Execution data is too deeply nested")
    if isinstance(value, Enum):
        key = f"{type(value).__module__}.{type(value).__name__}"
        if _class_for(key) is None:
            raise ExecutionDataError("Unsupported execution enum")
        return {"$enum": key, "value": value.value}
    if isinstance(value, str):
        from agenttree.models.collaboration import safe_message_text
        return safe_message_text(value)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ExecutionDataError("Nonfinite execution value")
        return value
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ExecutionDataError("Execution timestamp must be timezone-aware")
        return {"$datetime": value.isoformat()}
    from agenttree.models import ExecutionTrace
    if isinstance(value, ExecutionTrace):
        return {"$trace": value.task_id, "events": [_encode(item, depth + 1) for item in value.events]}
    if is_dataclass(value) and not isinstance(value, type):
        key = f"{type(value).__module__}.{type(value).__name__}"
        if _class_for(key) is None:
            raise ExecutionDataError("Unsupported execution contract")
        return {"$class": key, "fields": {
            item.name: _encode(getattr(value, item.name), depth + 1)
            for item in fields(value) if item.init and not item.name.startswith("_")}}
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ExecutionDataError("Execution mapping keys must be text")
        return {key: "[REDACTED]" if _private_key(key)
                else _encode(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return {"$tuple": [_encode(item, depth + 1) for item in value]} if isinstance(value, tuple) else [
            _encode(item, depth + 1) for item in value]
    if isinstance(value, (set, frozenset)):
        return {"$frozenset": [_encode(item, depth + 1) for item in sorted(value, key=repr)]}
    raise ExecutionDataError("Unsupported live execution object")


def _decode(value: Any, depth: int = 0) -> Any:
    if depth > 48:
        raise ExecutionDataError("Execution data is too deeply nested")
    if isinstance(value, list):
        return [_decode(item, depth + 1) for item in value]
    if not isinstance(value, dict):
        return value
    if "$datetime" in value:
        if set(value) != {"$datetime"}:
            raise ExecutionDataError("Malformed datetime")
        parsed = datetime.fromisoformat(value["$datetime"])
        if parsed.tzinfo is None:
            raise ExecutionDataError("Naive datetime")
        return parsed
    if "$enum" in value:
        if set(value) != {"$enum", "value"} or _class_for(value["$enum"]) is None:
            raise ExecutionDataError("Unknown execution enum")
        return _CLASSES[value["$enum"]](value["value"])
    if "$tuple" in value or "$frozenset" in value:
        if len(value) != 1:
            raise ExecutionDataError("Malformed sequence")
        if "$tuple" in value:
            return tuple(_decode(item, depth + 1) for item in value["$tuple"])
        return frozenset(_decode(item, depth + 1) for item in value["$frozenset"])
    if "$trace" in value:
        if set(value) != {"$trace", "events"}:
            raise ExecutionDataError("Malformed trace")
        from agenttree.models import ExecutionTrace
        trace = ExecutionTrace(value["$trace"])
        for event in value["events"]:
            trace.append(_decode(event, depth + 1))
        return trace
    if "$class" in value:
        if set(value) != {"$class", "fields"} or _class_for(value["$class"]) is None:
            raise ExecutionDataError("Unknown execution contract")
        typ = _CLASSES[value["$class"]]
        named = value["fields"]
        expected = {item.name for item in fields(typ) if item.init and not item.name.startswith("_")}
        if not isinstance(named, dict) or set(named) - expected:
            raise ExecutionDataError("Malformed execution contract")
        return typ(**{key: _decode(item, depth + 1) for key, item in named.items()})
    return {key: _decode(item, depth + 1) for key, item in value.items()}


def dumps(value: Any, *, max_bytes: int = 2_000_000) -> str:
    try:
        raw = json.dumps(_encode(value), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError, RecursionError) as error:
        raise ExecutionDataError("Execution data could not be serialized") from error
    if len(raw.encode("utf-8")) > max_bytes:
        raise ExecutionDataError("Execution data exceeds storage limit")
    return raw


def loads(raw: str, *, max_bytes: int = 2_000_000) -> Any:
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > max_bytes:
        raise ExecutionDataError("Execution data exceeds storage limit")
    try:
        return _decode(json.loads(raw))
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as error:
        raise ExecutionDataError("Malformed execution data") from error
