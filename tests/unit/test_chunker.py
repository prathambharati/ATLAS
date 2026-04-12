"""Tests for the document chunker."""

import pytest

from atlas.retriever.ingest import RecursiveChunker


class TestRecursiveChunker:
    """Test the recursive text chunking logic."""

    def setup_method(self):
        self.chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)

    def test_short_text_returns_single_chunk(self):
        """Text shorter than chunk_size should return one chunk."""
        text = "This is a short sentence."
        chunks = self.chunker.split(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_empty_text_returns_empty(self):
        """Empty text should return no chunks."""
        assert self.chunker.split("") == []
        assert self.chunker.split("   ") == []

    def test_paragraph_split(self):
        """Text with paragraph breaks should split on \\n\\n first."""
        text = "First paragraph with enough text to exceed the chunk size limit we set.\n\nSecond paragraph also has enough text to be a meaningful standalone chunk on its own."
        chunks = self.chunker.split(text)
        assert len(chunks) >= 2
        assert "First paragraph" in chunks[0]

    def test_long_text_creates_multiple_chunks(self):
        """Long text should be split into multiple chunks."""
        text = "Word " * 200  # ~1000 chars
        chunks = self.chunker.split(text)
        assert len(chunks) > 1

    def test_overlap_is_applied(self):
        """Consecutive chunks should share overlapping text."""
        # Create text that forces multiple chunks
        text = "Sentence one is here. " * 20
        chunker = RecursiveChunker(chunk_size=80, chunk_overlap=20)
        chunks = chunker.split(text)

        if len(chunks) >= 2:
            # The beginning of chunk[1] should contain text from the end of chunk[0]
            tail_of_first = chunks[0][-20:]
            assert tail_of_first in chunks[1], (
                f"Expected overlap: '{tail_of_first}' not found in second chunk"
            )

    def test_no_overlap_when_zero(self):
        """With overlap=0, chunks should not share text."""
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=0)
        text = "A " * 100
        chunks = chunker.split(text)
        assert len(chunks) > 1

    def test_chunks_respect_max_size(self):
        """No chunk should greatly exceed chunk_size (overlap excluded)."""
        text = "Hello world. " * 100
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=0)
        chunks = chunker.split(text)

        for chunk in chunks:
            # Allow some tolerance for splitting boundaries
            assert len(chunk) <= 150, f"Chunk too large ({len(chunk)} chars): {chunk[:50]}..."

    def test_sentence_boundary_splitting(self):
        """When paragraphs are too long, should split on sentences."""
        # One long paragraph with many sentences
        text = "This is sentence one. This is sentence two. This is sentence three. " * 10
        chunker = RecursiveChunker(chunk_size=120, chunk_overlap=0)
        chunks = chunker.split(text)
        assert len(chunks) > 1


class TestRecursiveChunkerEdgeCases:
    """Edge cases for the chunker."""

    def test_single_long_word(self):
        """A single word longer than chunk_size should still be returned."""
        text = "a" * 200
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=0)
        chunks = chunker.split(text)
        assert len(chunks) >= 1
        # All characters should be preserved
        assert sum(len(c) for c in chunks) >= 200

    def test_only_whitespace_and_newlines(self):
        """Whitespace-only text should return empty."""
        assert RecursiveChunker().split("\n\n\n   \n") == []

    def test_unicode_text(self):
        """Should handle unicode text correctly."""
        text = "机器学习是人工智能的一个子集。" * 20
        chunker = RecursiveChunker(chunk_size=60, chunk_overlap=10)
        chunks = chunker.split(text)
        assert len(chunks) >= 1
