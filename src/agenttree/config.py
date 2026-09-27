"""Configuration for the synchronous developer-facing framework."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentTreeConfig:
    """Immutable controls for existing matching and bounded revision behavior."""

    manager_match_all: bool = False
    specialist_match_all: bool = False
    max_manager_revisions: int = 2
    max_final_revisions: int = 1
    max_tool_rounds: int = 3
    max_tool_calls: int = 8
    tool_timeout: float = 5.0
    max_tool_argument_bytes: int = 16_384
    max_tool_result_bytes: int = 65_536
    max_collaboration_messages_per_manager: int = 4
    max_collaboration_messages_total: int = 12
    max_collaboration_rounds_per_thread: int = 2
    max_collaboration_payload_bytes: int = 4_096
    max_collaboration_turns_per_decision: int = 2
    collaboration_timeout: float = 5.0
    max_artifacts: int = 32
    max_artifact_bytes: int = 1_000_000
    max_total_artifact_bytes: int = 8_000_000
    max_artifact_metadata_bytes: int = 16_384
    max_artifact_path_length: int = 512
    max_artifact_name_length: int = 128
    provider_streaming: bool = False
    max_provider_stream_bytes: int = 1_000_000

    def __post_init__(self) -> None:
        for name in ("manager_match_all", "specialist_match_all", "provider_streaming"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        for name in ("max_manager_revisions", "max_final_revisions"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("max_tool_rounds", "max_tool_calls", "max_tool_argument_bytes", "max_tool_result_bytes",
                     "max_collaboration_messages_per_manager", "max_collaboration_messages_total",
                     "max_collaboration_rounds_per_thread", "max_collaboration_payload_bytes",
                     "max_collaboration_turns_per_decision", "max_artifacts",
                     "max_artifact_bytes", "max_total_artifact_bytes",
                     "max_artifact_metadata_bytes", "max_artifact_path_length",
                     "max_artifact_name_length", "max_provider_stream_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_total_artifact_bytes < self.max_artifact_bytes:
            raise ValueError("max_total_artifact_bytes must cover one artifact")
        from math import isfinite
        if isinstance(self.tool_timeout, bool) or not isinstance(self.tool_timeout, (int, float)) or not isfinite(self.tool_timeout) or self.tool_timeout <= 0:
            raise ValueError("tool_timeout must be positive and finite")
        if isinstance(self.collaboration_timeout, bool) or not isinstance(self.collaboration_timeout, (int, float)) or not isfinite(self.collaboration_timeout) or self.collaboration_timeout <= 0:
            raise ValueError("collaboration_timeout must be positive and finite")
