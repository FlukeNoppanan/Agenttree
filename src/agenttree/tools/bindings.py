"""External deterministic Specialist-to-Tool assignments."""

from types import MappingProxyType
from typing import Mapping

def _binding_id(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    prepared = value.strip()
    if not prepared:
        raise ValueError(f"{label} must be a non-empty string")
    return prepared


class ToolBindingRegistry:
    """Store ordered external tool-ID assignments without mutating agents."""

    def __init__(self) -> None:
        self._bindings: dict[str, list[str]] = {}

    @property
    def bindings(self) -> Mapping[str, tuple[str, ...]]:
        """Return a read-only snapshot of all assignments."""
        return MappingProxyType({
            specialist_id: tuple(tool_ids)
            for specialist_id, tool_ids in self._bindings.items()
        })

    def assign(self, specialist_id: str, tool_id: str) -> None:
        """Append one tool assignment, rejecting duplicate pairs."""
        prepared_specialist_id = _binding_id(specialist_id, "specialist_id")
        prepared_tool_id = _binding_id(tool_id, "tool_id")
        assigned = self._bindings.setdefault(prepared_specialist_id, [])
        if prepared_tool_id in assigned:
            raise ValueError(
                f"Tool {prepared_tool_id} is already assigned to "
                f"{prepared_specialist_id}",
            )
        assigned.append(prepared_tool_id)

    def unassign(self, specialist_id: str, tool_id: str) -> None:
        """Remove one assignment, raising ``KeyError`` if it is absent."""
        prepared_specialist_id = _binding_id(specialist_id, "specialist_id")
        prepared_tool_id = _binding_id(tool_id, "tool_id")
        try:
            assigned = self._bindings[prepared_specialist_id]
            assigned.remove(prepared_tool_id)
        except (KeyError, ValueError) as error:
            raise KeyError(
                f"Tool {prepared_tool_id} is not assigned to "
                f"{prepared_specialist_id}",
            ) from error
        if not assigned:
            del self._bindings[prepared_specialist_id]

    def tool_ids_for(self, specialist_id: str) -> tuple[str, ...]:
        """Return a specialist's tool IDs in assignment order."""
        prepared_specialist_id = _binding_id(specialist_id, "specialist_id")
        return tuple(self._bindings.get(prepared_specialist_id, ()))

    def is_assigned(self, specialist_id: str, tool_id: str) -> bool:
        """Return whether the exact tool ID is assigned to the specialist."""
        prepared_specialist_id = _binding_id(specialist_id, "specialist_id")
        prepared_tool_id = _binding_id(tool_id, "tool_id")
        return prepared_tool_id in self._bindings.get(prepared_specialist_id, ())
