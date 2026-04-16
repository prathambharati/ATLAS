"""Build and return a ToolRegistry pre-loaded with all built-in tools.

This module wires up retrieval, web search, arxiv, and calculator
as callable tools the agent can invoke via function calling.
"""

from atlas.observability.logger import get_logger
from atlas.retriever.arxiv_search import ArxivSearchTool
from atlas.retriever.hybrid import HybridRetriever
from atlas.retriever.web_search import WebSearchTool
from atlas.tools.registry import ToolRegistry

log = get_logger(__name__)


def build_default_tools(
    retriever: HybridRetriever | None = None,
) -> ToolRegistry:
    """Create a ToolRegistry with all built-in tools.

    Args:
        retriever: Optional pre-initialized retriever.

    Returns:
        A ToolRegistry with search, arxiv, retrieve, and calculator tools.
    """
    registry = ToolRegistry()
    web_search = WebSearchTool()
    arxiv_search = ArxivSearchTool()

    # --- Retrieve from vector store ---
    def retrieve_tool(query: str, top_k: int = 5) -> str:
        """Search ingested documents using hybrid retrieval."""
        if retriever is None:
            return "Error: No documents have been ingested yet."
        results = retriever.retrieve(
            query=query, top_k=top_k, method="hybrid", rerank=True
        )
        if not results:
            return "No relevant results found in ingested documents."
        output = []
        for r in results:
            output.append(
                f"[{r.source}] (score: {r.score:.3f})\n{r.text}\n"
            )
        return "\n---\n".join(output)

    registry.register(
        name="retrieve",
        description=(
            "Search through ingested PDF documents using hybrid retrieval. "
            "THIS IS THE PRIMARY TOOL — always use this FIRST before web search. "
            "If the user has uploaded documents, the answer is likely here. "
            "Uses dense + BM25 + cross-encoder re-ranking for high precision."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results (default 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
        func=retrieve_tool,
    )

    # --- Web search ---
    def web_search_tool(query: str, max_results: int = 3) -> str:
        """Search the web for current information."""
        if not web_search.is_available:
            return "Web search is not configured (no Tavily API key)."
        results = web_search.search(query, max_results=max_results)
        if not results:
            return "No web results found."
        output = []
        for r in results:
            output.append(
                f"[{r['title']}]({r['url']})\n{r['content']}\n"
            )
        return "\n---\n".join(output)

    registry.register(
        name="web_search",
        description=(
            "Search the web for current information, recent events, "
            "or topics not covered in ingested documents. "
            "Returns clean text from web pages."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results (default 3)",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
        func=web_search_tool,
    )

    # --- Arxiv search ---
    def arxiv_tool(query: str, max_results: int = 3) -> str:
        """Search arxiv for academic papers."""
        results = arxiv_search.search(query, max_results=max_results)
        if not results:
            return "No arxiv papers found."
        output = []
        for r in results:
            authors = ", ".join(r["authors"][:3])
            output.append(
                f"**{r['title']}**\n"
                f"Authors: {authors}\n"
                f"Published: {r['published'][:10]}\n"
                f"arxiv: {r['arxiv_id']}\n\n"
                f"{r['abstract'][:500]}\n"
            )
        return "\n---\n".join(output)

    registry.register(
        name="arxiv_search",
        description=(
            "Search arxiv for academic papers. Use this for finding "
            "research papers, technical details, and scientific results. "
            "Returns titles, authors, and abstracts."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for academic papers",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of papers (default 3)",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
        func=arxiv_tool,
    )

    # --- Calculator ---
    def calculator_tool(expression: str) -> str:
        """Evaluate a mathematical expression safely."""
        allowed = set("0123456789+-*/.() ,")
        if not all(c in allowed for c in expression):
            return "Error: Only basic math operations are supported."
        try:
            result = eval(expression)  # noqa: S307
            return str(result)
        except Exception as e:
            return f"Error evaluating expression: {e}"

    registry.register(
        name="calculator",
        description=(
            "Evaluate mathematical expressions. "
            "Supports basic operations: +, -, *, /, parentheses. "
            "Use for any numerical calculations."
        ),
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate, e.g. '(2+3)*4'",
                },
            },
            "required": ["expression"],
        },
        func=calculator_tool,
    )

    log.info("default_tools_built", num_tools=registry.num_tools)
    return registry
