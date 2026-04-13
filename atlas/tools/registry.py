"""Tool registry for the agent.

Tools register themselves with a name, description, and parameter schema.
The registry generates the OpenAI function-calling format so the LLM
can discover and invoke tools dynamically.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from atlas.observability.logger import get_logger

log = get_logger(__name__)


@dataclass
class Tool:
    """A callable tool the agent can use."""

    name: str
    description: str
    parameters: dict
    func: Callable[..., str]

    def to_openai_function(self) -> dict:
        """Convert to OpenAI function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def execute(self, **kwargs: Any) -> str:
        """Execute the tool with given arguments."""
        try:
            result = self.func(**kwargs)
            return str(result)
        except Exception as e:
            return f"Error executing {self.name}: {e}"


class ToolRegistry:
    """Registry of tools available to the agent."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict,
        func: Callable[..., str],
    ) -> None:
        """Register a new tool."""
        tool = Tool(
            name=name,
            description=description,
            parameters=parameters,
            func=func,
        )
        self._tools[name] = tool
        log.info("tool_registered", name=name)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def execute(self, tool_name: str, **kwargs: Any) -> str:
        """Execute a tool by name."""
        tool = self._tools.get(tool_name)
        if tool is None:
            return f"Error: Unknown tool '{tool_name}'"
        return tool.execute(**kwargs)

    def to_openai_tools(self) -> list[dict]:
        """Get all tools in OpenAI function-calling format."""
        return [t.to_openai_function() for t in self._tools.values()]

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    @property
    def num_tools(self) -> int:
        return len(self._tools)
