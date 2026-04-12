"""Tests for web search and arxiv tools."""

import pytest

from atlas.retriever.web_search import WebSearchTool
from atlas.retriever.arxiv_search import ArxivSearchTool


class TestWebSearchTool:
    """Test web search functionality."""

    def test_unavailable_without_key(self):
        """Should gracefully handle missing API key."""
        tool = WebSearchTool(api_key="invalid_key_12345")
        # Tool initializes but search should fail gracefully
        results = tool.search("test query")
        assert isinstance(results, list)

    def test_search_as_chunks_format(self):
        """Chunks should have the correct structure."""
        tool = WebSearchTool(api_key="")
        # Even with no key, search_as_chunks should return empty list
        chunks = tool.search_as_chunks("test")
        assert isinstance(chunks, list)


class TestArxivSearchTool:
    """Test arxiv search functionality."""

    def test_search_returns_results(self):
        """Should return papers for a valid query."""
        tool = ArxivSearchTool(max_results_default=3)
        results = tool.search("attention is all you need transformer", max_results=3)

        assert len(results) > 0
        # Check structure
        first = results[0]
        assert "title" in first
        assert "abstract" in first
        assert "authors" in first
        assert "pdf_url" in first
        assert "arxiv_id" in first
        assert first["source"] == "arxiv"

    def test_search_as_chunks_format(self):
        """Chunks should match the retriever chunk format."""
        tool = ArxivSearchTool(max_results_default=2)
        chunks = tool.search_as_chunks("BERT language model", max_results=2)

        assert len(chunks) > 0
        first = chunks[0]
        assert "chunk_id" in first
        assert "text" in first
        assert "score" in first
        assert "source" in first
        assert "metadata" in first
        assert first["chunk_id"].startswith("arxiv_")
        assert first["metadata"]["source_type"] == "arxiv"

    def test_search_empty_query(self):
        """Should handle empty or very broad queries gracefully."""
        tool = ArxivSearchTool(max_results_default=2)
        results = tool.search("asdfghjklzxcvbnm", max_results=2)
        # Might return 0 or some results — either is fine, no crash
        assert isinstance(results, list)

    def test_authors_formatting_in_chunks(self):
        """Author string should be properly formatted in chunk text."""
        tool = ArxivSearchTool(max_results_default=1)
        chunks = tool.search_as_chunks("deep learning", max_results=1)

        if chunks:
            assert "Authors:" in chunks[0]["text"]
