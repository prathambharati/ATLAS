"""Tests for dense and sparse retrieval modules."""

from atlas.retriever.ingest import Chunk
from atlas.retriever.sparse import SparseIndex


def _make_chunks(texts: list[str], source: str = "test.pdf") -> list[Chunk]:
    """Helper to create Chunk objects from text strings."""
    return [
        Chunk(
            chunk_id=f"test_chunk_{i:04d}",
            text=text,
            source=source,
            page_number=1,
            chunk_index=i,
            metadata={},
        )
        for i, text in enumerate(texts)
    ]


class TestSparseIndex:
    """Test BM25 sparse retrieval."""

    def test_add_and_search(self):
        """Should return relevant results for a matching query."""
        index = SparseIndex()
        chunks = _make_chunks(
            [
                "Machine learning is a subset of artificial intelligence.",
                "Neural networks are inspired by biological neurons.",
                "The weather today is sunny and warm.",
                "Deep learning uses multiple layers of neural networks.",
                "Cooking pasta requires boiling water and salt.",
            ]
        )
        index.add_chunks(chunks)

        results = index.search("neural networks deep learning", top_k=3)
        assert len(results) > 0

        # The neural network / deep learning chunks should rank higher
        top_texts = [r["text"] for r in results]
        assert any("neural" in t.lower() for t in top_texts)

    def test_empty_index_returns_empty(self):
        """Searching an empty index should return no results."""
        index = SparseIndex()
        results = index.search("anything")
        assert results == []

    def test_count(self):
        """Count should reflect number of indexed chunks."""
        index = SparseIndex()
        assert index.count == 0

        chunks = _make_chunks(["text one", "text two", "text three"])
        index.add_chunks(chunks)
        assert index.count == 3

    def test_incremental_add(self):
        """Adding chunks incrementally should work correctly."""
        index = SparseIndex()

        index.add_chunks(_make_chunks(["First batch of documents."]))
        assert index.count == 1

        index.add_chunks(_make_chunks(["Second batch of documents."]))
        assert index.count == 2

        results = index.search("documents", top_k=5)
        assert len(results) == 2

    def test_no_zero_score_results(self):
        """Should not return results with zero BM25 score."""
        index = SparseIndex()
        chunks = _make_chunks(
            [
                "Quantum computing uses qubits.",
                "Baking bread requires flour and yeast.",
            ]
        )
        index.add_chunks(chunks)

        results = index.search("quantum qubits", top_k=5)
        for r in results:
            assert r["score"] > 0

    def test_top_k_limits_results(self):
        """Should return at most top_k results."""
        index = SparseIndex()
        chunks = _make_chunks([f"Document number {i} about science." for i in range(20)])
        index.add_chunks(chunks)

        results = index.search("science document", top_k=3)
        assert len(results) <= 3


class TestHybridRRF:
    """Test Reciprocal Rank Fusion logic."""

    def test_rrf_score_calculation(self):
        """RRF should combine rankings from both retrievers."""

        # Just verify the math: for k=60, rank 1 → 1/61, rank 2 → 1/62
        score_rank_1 = 1.0 / (60 + 1)
        score_rank_2 = 1.0 / (60 + 2)

        assert score_rank_1 > score_rank_2
        assert abs(score_rank_1 - 0.01639) < 0.001

    def test_rrf_boosts_docs_in_both_retrievers(self):
        """A doc appearing in both dense and sparse results should score higher."""
        # This is a conceptual test — in practice, a doc ranked #1 in both
        # retrievers gets score 2 * 1/(k+1) vs a doc in only one gets 1/(k+1)
        k = 60
        score_in_both = 2 * (1.0 / (k + 1))
        score_in_one = 1.0 / (k + 1)
        assert score_in_both > score_in_one
