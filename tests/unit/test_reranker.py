"""Tests for the cross-encoder re-ranker."""

import pytest

from atlas.api.schemas import ChunkResult
from atlas.retriever.reranker import Reranker


def _make_chunk_results(texts: list[str]) -> list[ChunkResult]:
    """Helper to create ChunkResult objects from text strings."""
    return [
        ChunkResult(
            chunk_id=f"chunk_{i:04d}",
            text=text,
            score=0.5,  # Dummy initial score
            source="test.pdf",
            metadata={},
        )
        for i, text in enumerate(texts)
    ]


class TestReranker:
    """Test cross-encoder re-ranking."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Load the re-ranker model once for all tests."""
        self.reranker = Reranker()

    def test_rerank_returns_correct_count(self):
        """Should return top_k results."""
        chunks = _make_chunk_results(
            [
                "Machine learning is a subset of artificial intelligence.",
                "The weather today is sunny and warm.",
                "Deep neural networks use backpropagation.",
                "Cooking pasta requires boiling water.",
                "Transformers use self-attention mechanisms.",
            ]
        )
        results = self.reranker.rerank("neural network architectures", chunks, top_k=3)
        assert len(results) == 3

    def test_rerank_improves_ordering(self):
        """Relevant chunks should be ranked higher than irrelevant ones."""
        chunks = _make_chunk_results(
            [
                "The best recipe for chocolate cake involves cocoa powder.",
                "Neural networks consist of layers of interconnected nodes.",
                "My cat likes to sleep on the couch all day long.",
                "Backpropagation computes gradients through the network.",
                "The stock market closed higher today on tech gains.",
            ]
        )
        results = self.reranker.rerank("how do neural networks learn", chunks, top_k=5)

        # The ML-related chunks should be in the top 2
        top_2_texts = [r.text for r in results[:2]]
        assert any("neural" in t.lower() or "backpropagation" in t.lower() for t in top_2_texts)

    def test_rerank_empty_input(self):
        """Empty input should return empty output."""
        results = self.reranker.rerank("anything", [], top_k=5)
        assert results == []

    def test_rerank_updates_scores(self):
        """Scores should be updated to cross-encoder scores, not original."""
        chunks = _make_chunk_results(["Transformers use attention."])
        results = self.reranker.rerank("what is a transformer", chunks)
        # Score should have changed from the dummy 0.5
        assert results[0].score != 0.5

    def test_rerank_top_k_none_returns_all(self):
        """When top_k is None, all results should be returned."""
        chunks = _make_chunk_results(
            [
                "Text one.",
                "Text two.",
                "Text three.",
            ]
        )
        results = self.reranker.rerank("query", chunks, top_k=None)
        assert len(results) == 3

    def test_rerank_preserves_metadata(self):
        """Chunk metadata should be preserved after re-ranking."""
        chunks = [
            ChunkResult(
                chunk_id="test_001",
                text="Attention is all you need.",
                score=0.5,
                source="paper.pdf",
                metadata={"page_number": 3, "document_id": "abc123"},
            )
        ]
        results = self.reranker.rerank("transformer architecture", chunks)
        assert results[0].chunk_id == "test_001"
        assert results[0].source == "paper.pdf"
        assert results[0].metadata["page_number"] == 3
        assert results[0].metadata["document_id"] == "abc123"
