"""Tests for tool registry and agent orchestrator."""

import pytest

from atlas.tools.registry import ToolRegistry


class TestToolRegistry:
    """Test tool registration and execution."""

    def test_register_and_execute(self):
        """Should register and execute a simple tool."""
        registry = ToolRegistry()
        registry.register(
            name="greet",
            description="Say hello",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                },
                "required": ["name"],
            },
            func=lambda name: f"Hello, {name}!",
        )

        result = registry.execute("greet", name="Pratham")
        assert result == "Hello, Pratham!"

    def test_execute_unknown_tool(self):
        """Should return error for unknown tool."""
        registry = ToolRegistry()
        result = registry.execute("nonexistent")
        assert "Error" in result
        assert "nonexistent" in result

    def test_tool_names(self):
        """Should list registered tool names."""
        registry = ToolRegistry()
        registry.register(
            name="tool_a",
            description="A",
            parameters={},
            func=lambda: "a",
        )
        registry.register(
            name="tool_b",
            description="B",
            parameters={},
            func=lambda: "b",
        )

        assert set(registry.tool_names) == {"tool_a", "tool_b"}
        assert registry.num_tools == 2

    def test_to_openai_tools(self):
        """Should generate valid OpenAI function-calling format."""
        registry = ToolRegistry()
        registry.register(
            name="search",
            description="Search for stuff",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
            func=lambda query: f"Results for: {query}",
        )

        tools = registry.to_openai_tools()
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "search"
        assert "query" in tools[0]["function"]["parameters"]["properties"]

    def test_tool_handles_exception(self):
        """Should catch exceptions and return error string."""

        def broken_tool():
            raise ValueError("Something went wrong")

        registry = ToolRegistry()
        registry.register(
            name="broken",
            description="This will fail",
            parameters={},
            func=broken_tool,
        )

        result = registry.execute("broken")
        assert "Error" in result
        assert "Something went wrong" in result


class TestDefaultTools:
    """Test the default tool suite builds correctly."""

    def test_build_default_tools(self):
        """Should create a registry with all default tools."""
        from atlas.tools.default_tools import build_default_tools

        registry = build_default_tools()

        assert registry.num_tools == 4
        assert "retrieve" in registry.tool_names
        assert "web_search" in registry.tool_names
        assert "arxiv_search" in registry.tool_names
        assert "calculator" in registry.tool_names

    def test_calculator_works(self):
        """Calculator tool should evaluate expressions."""
        from atlas.tools.default_tools import build_default_tools

        registry = build_default_tools()
        result = registry.execute("calculator", expression="(2+3)*4")
        assert result == "20"

    def test_calculator_rejects_unsafe(self):
        """Calculator should reject non-math expressions."""
        from atlas.tools.default_tools import build_default_tools

        registry = build_default_tools()
        result = registry.execute(
            "calculator", expression="__import__('os').system('ls')"
        )
        assert "Error" in result


class TestAgentIntegration:
    """Integration tests that use the real OpenAI API.

    Run with: pytest tests/unit/test_agent.py -v -k Integration
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        from atlas.config import settings

        if not settings.openai_api_key:
            pytest.skip("OPENAI_API_KEY not set")

    def test_simple_research_query(self):
        """Should answer a simple query using tools."""
        from atlas.agent.orchestrator import AgentOrchestrator

        agent = AgentOrchestrator(max_steps=5)
        result = agent.run("What is attention in transformers?")

        assert "trace_id" in result
        assert "answer" in result
        assert result["answer"] is not None
        assert len(result["answer"]) > 50

    def test_complex_research_query(self):
        """Should decompose and answer a complex query."""
        from atlas.agent.orchestrator import AgentOrchestrator

        agent = AgentOrchestrator(max_steps=5)
        result = agent.run(
            "Compare BERT and GPT architectures"
        )

        assert result["answer"] is not None
        assert len(result["answer"]) > 100
        assert result["dag_summary"]["num_tasks"] >= 2
