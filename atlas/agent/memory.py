"""Agent memory — scratchpad for tracking reasoning across steps.

Stores the conversation history and intermediate results
so the agent can reference previous tool outputs and reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentMemory:
    """Working memory for the agent during a research session."""

    # Full message history for the LLM
    messages: list[dict] = field(default_factory=list)

    # Results from completed DAG tasks (task_id -> result)
    task_results: dict[str, str] = field(default_factory=dict)

    # Sources cited during research
    sources: list[dict] = field(default_factory=list)

    def add_system(self, content: str) -> None:
        """Add a system message."""
        self.messages.append({"role": "system", "content": content})

    def add_user(self, content: str) -> None:
        """Add a user message."""
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        """Add an assistant message."""
        self.messages.append({"role": "assistant", "content": content})

    def add_tool_call(self, tool_call_id: str, name: str, arguments: str) -> None:
        """Record a tool call made by the assistant."""
        self.messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            ],
        })

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        """Add the result of a tool call."""
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        })

    def add_task_result(self, task_id: str, result: str) -> None:
        """Store a completed task result."""
        self.task_results[task_id] = result

    def add_source(self, source: dict) -> None:
        """Track a source used during research."""
        self.sources.append(source)

    def get_context_summary(self) -> str:
        """Get a summary of completed task results for context."""
        if not self.task_results:
            return "No previous task results."

        parts = []
        for tid, result in self.task_results.items():
            # Truncate long results
            truncated = result[:500] + "..." if len(result) > 500 else result
            parts.append(f"[Task {tid}]: {truncated}")

        return "\n\n".join(parts)

    def clear(self) -> None:
        """Reset memory."""
        self.messages.clear()
        self.task_results.clear()
        self.sources.clear()
