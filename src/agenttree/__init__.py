"""AgentTree: a hierarchical multi-agent orchestration framework."""

from agenttree.agents import ManagerAgent, RootAgent, SpecialistAgent
from agenttree.config import AgentTreeConfig
from agenttree.framework import AgentTree
from agenttree.models import ExecutionMode, Task

__version__ = "0.2.2"

__all__ = [
    "AgentTree", "AgentTreeConfig", "RootAgent", "ManagerAgent",
    "SpecialistAgent", "Task", "ExecutionMode",
]
