"""Web search tool using Tavily API.

Tavily is designed specifically for RAG — it returns clean extracted text
from web pages, not raw HTML. This makes it ideal for feeding results
directly into an LLM context.

Free tier: 1000 searches/month.
Get your API key at https://tavily.com
"""

from tavily import TavilyClient

from atlas.config import settings
from atlas.observability.logger import get_logger

log = get_logger(__name__)


class WebSearchTool:
    """Search the web using Tavily and return clean text results."""

    def __init__(self, api_key: str | None = None):
        """
        Args:
            api_key: Tavily API key. Falls back to settings if not provided.
        """
        key = api_key or settings.tavily_api_key
        if not key or key.strip() == "":
            log.warning("tavily_api_key_missing", msg="Web search will not be available")
            self._client = None
        else:
            self._client = TavilyClient(api_key=key)
            log.info("web_search_ready")

    @property
    def is_available(self) -> bool:
        """Check if the web search tool is configured."""
        return self._client is not None

    def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        include_raw_content: bool = False,
    ) -> list[dict]:
        """Search the web and return structured results.

        Args:
            query: Search query string.
            max_results: Number of results to return (1-10).
            search_depth: "basic" (faster) or "advanced" (more thorough).
            include_raw_content: If True, include the full page content.

        Returns:
            List of dicts with keys: title, url, content, score
        """
        if not self.is_available:
            log.warning("web_search_unavailable", query=query)
            return []

        try:
            response = self._client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_raw_content=include_raw_content,
            )

            results = []
            for item in response.get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "score": item.get("score", 0.0),
                    "source": "web",
                })

            log.info(
                "web_search_complete",
                query=query[:80],
                num_results=len(results),
            )
            return results

        except Exception as e:
            log.error("web_search_failed", query=query, error=str(e))
            return []

    def search_as_chunks(self, query: str, max_results: int = 5) -> list[dict]:
        """Search and return results in the same format as retriever chunks.

        This allows web results to be merged with vector store results
        and re-ranked together by the cross-encoder.
        """
        raw_results = self.search(query, max_results=max_results)

        chunks = []
        for i, result in enumerate(raw_results):
            chunks.append({
                "chunk_id": f"web_{i:04d}",
                "text": f"{result['title']}\n\n{result['content']}",
                "score": result.get("score", 0.0),
                "source": result.get("url", "web"),
                "metadata": {
                    "source_type": "web",
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                },
            })

        return chunks
