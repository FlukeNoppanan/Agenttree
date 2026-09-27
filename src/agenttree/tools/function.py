"""Tool wrapper for developer-provided synchronous Python callables."""

from __future__ import annotations

from inspect import (
    Parameter,
    Signature,
    formatannotation,
    iscoroutinefunction,
    signature,
)
from typing import Any, Callable, Mapping

from agenttree.tools.base import BaseTool, ToolRecoveryPolicy
from agenttree.tools.models import ToolInputSpec, ToolParameter, ToolResult


def _annotation_text(annotation: object) -> str | None:
    if annotation is Signature.empty:
        return None
    if isinstance(annotation, str):
        return annotation
    rendered = formatannotation(annotation)
    if " at 0x" in rendered:
        annotation_type = type(annotation)
        return f"<{annotation_type.__module__}.{annotation_type.__qualname__}>"
    return rendered


def _default_text(default: object) -> str | None:
    if default is Signature.empty:
        return None
    if isinstance(default, (str, int, float, bool, bytes, type(None))):
        return repr(default)
    if isinstance(default, tuple):
        values = ", ".join(_default_text(item) or "None" for item in default)
        suffix = "," if len(default) == 1 else ""
        return f"({values}{suffix})"
    if isinstance(default, list):
        values = ", ".join(_default_text(item) or "None" for item in default)
        return f"[{values}]"
    if isinstance(default, dict):
        values = ", ".join(
            f"{_default_text(key)}: {_default_text(value)}"
            for key, value in default.items()
        )
        return f"{{{values}}}"
    if isinstance(default, (set, frozenset)):
        if not default:
            return "frozenset()" if isinstance(default, frozenset) else "set()"
        values = sorted(_default_text(item) or "None" for item in default)
        rendered = ", ".join(values)
        if isinstance(default, frozenset):
            return f"frozenset({{{rendered}}})"
        return f"{{{rendered}}}"
    value_type = type(default)
    return f"<{value_type.__module__}.{value_type.__qualname__}>"


class FunctionTool(BaseTool):
    """Wrap an explicitly supplied callable behind the common tool contract."""

    def __init__(
        self,
        *,
        name: str,
        function: Callable[..., Any],
        description: str = "",
        tool_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        enabled: bool = True,
        recovery_policy: ToolRecoveryPolicy = ToolRecoveryPolicy.UNKNOWN,
    ) -> None:
        if not callable(function):
            raise TypeError("function must be callable")
        if iscoroutinefunction(function) or iscoroutinefunction(
            getattr(function, "__call__", None),
        ):
            raise TypeError("async functions are not supported")
        try:
            inspected = signature(function)
        except (TypeError, ValueError) as error:
            raise ValueError("function must expose an inspectable signature") from error
        parameters: list[ToolParameter] = []
        accepts_var_keyword = False
        for parameter in inspected.parameters.values():
            if parameter.kind is Parameter.POSITIONAL_ONLY:
                raise ValueError(
                    "positional-only parameters cannot be invoked with arguments",
                )
            if parameter.kind is Parameter.VAR_POSITIONAL:
                raise ValueError("variadic positional parameters are not supported")
            if parameter.kind is Parameter.VAR_KEYWORD:
                accepts_var_keyword = True
                continue
            parameters.append(ToolParameter(
                name=parameter.name,
                required=parameter.default is Signature.empty,
                annotation=_annotation_text(parameter.annotation),
                default=_default_text(parameter.default),
                kind=parameter.kind.name.casefold(),
            ))
        self._function = function
        self._signature = inspected
        super().__init__(
            name=name,
            description=description,
            tool_id=tool_id,
            input_spec=ToolInputSpec(
                parameters=tuple(parameters),
                accepts_var_keyword=accepts_var_keyword,
            ),
            metadata=metadata,
            enabled=enabled,
            recovery_policy=recovery_policy,
        )

    @property
    def function(self) -> Callable[..., Any]:
        """Return the explicitly registered callable."""
        return self._function

    @property
    def signature(self) -> Signature:
        """Return the inspected Python signature used for argument binding."""
        return self._signature

    def invoke(self, arguments: Mapping[str, Any]) -> ToolResult:
        """Validate keyword arguments, call the function, and normalize output."""
        if not isinstance(arguments, Mapping):
            raise TypeError("arguments must be a mapping")
        prepared = dict(arguments)
        self._signature.bind(**prepared)
        try:
            output = self._function(**prepared)
        except Exception as error:
            return ToolResult(
                tool_id=self.id,
                success=False,
                error=f"{type(error).__name__}: {error}",
                metadata={"error_type": type(error).__name__},
            )
        return ToolResult(tool_id=self.id, success=True, output=output)
