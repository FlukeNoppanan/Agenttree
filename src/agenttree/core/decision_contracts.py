"""Transport schemas for existing decisions; strategy validators own semantics."""

from typing import Any


def decision_format(strategy: str, context: dict[str, Any], collaboration: bool = False) -> dict[str, Any] | None:
    text = {"type": "string", "minLength": 1}
    capability = {"type": "string", "minLength": 1}
    scope = context.get("available_manager_capabilities" if strategy == "triage" else "available_specialist_capabilities")
    if isinstance(scope, (tuple, list)) and scope:
        capability = {**capability, "enum": list(scope)}
    labels = {"type": "array", "items": capability}
    if isinstance(scope, (tuple, list)) and not scope:
        labels["maxItems"] = 0
    if strategy == "root_planning":
        properties = {"delegate": {"type": "boolean"}, "direct_output": {"type": "string"}}
        required = ["delegate"]
    elif strategy == "triage":
        properties = {"objective": text, "required_capabilities": labels}
        required = ["objective", "required_capabilities"]
    elif strategy == "decomposition":
        properties = {"subtasks": {"type": "array", "items": {
            "type": "object", "properties": {"objective": text, "required_capabilities": labels},
            "required": ["objective", "required_capabilities"]}}}
        required = ["subtasks"]
    elif strategy in {"manager_review", "final_review"}:
        properties = {"decision": {"type": "string", "enum": ["pass", "revise", "fail"]}, "feedback": {"type": "string"}}
        required = ["decision", "feedback"]
    else:
        return None
    schema = {"type": "object", "properties": properties, "required": required}
    if collaboration:
        schema = {"anyOf": [schema, {"type": "object", "properties": {
            "collaboration_request": {"type": "object", "properties": {
                "target_manager_id": text, "type": {"type": "string", "enum": ["request", "context", "review_request"]},
                "subject": text, "content": text}, "required": ["target_manager_id", "type", "subject", "content"]}},
            "required": ["collaboration_request"]}]}
    return {"type": "json_schema", "json_schema": {"name": "agenttree_" + strategy, "schema": schema}}
