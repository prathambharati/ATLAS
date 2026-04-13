"""Sparse retrieval using BM25 (Okapi BM25).

Provides keyword-based retrieval as a complement to dense (semantic) search.
Together they form the hybrid retrieval pipeline.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from rank_bm25 import BM25Okapi

from atlas.observability.logger import get_logger

if TYPE_CHECKING:
    from atlas.retriever.ingest import Chunk

log = get_logger(__name__)


class SparseIndex:
    """BM25-based sparse retrieval index."""

    def __init__(self):
        self._chunks: list[Chunk] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer with lowercasing."""
        text = text.lower()
        # Remove punctuation, keep alphanumeric and spaces
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = text.split()
        # Remove very short tokens
        return [t for t in tokens if len(t) > 1]

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Add chunks to the BM25 index."""
        if not chunks:
            return

        self._chunks.extend(chunks)
        new_tokenized = [self._tokenize(c.text) for c in chunks]
        self._tokenized_corpus.extend(new_tokenized)

        # Rebuild BM25 index (BM25Okapi doesn't support incremental adds)
        self._bm25 = BM25Okapi(self._tokenized_corpus)

        log.info("sparse_index_updated", num_added=len(chunks), total=len(self._chunks))

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Search the BM25 index for relevant chunks.

        Args:
            query: Search query string.
            top_k: Number of top results to return.

        Returns:
            List of dicts with keys: chunk_id, text, score, source, metadata
        """
        if self._bm25 is None or not self._chunks:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-k indices sorted by score (descending)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] == 0:
                continue  # Skip zero-score results

            chunk = self._chunks[idx]
            results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "score": round(float(scores[idx]), 4),
                    "source": chunk.source,
                    "metadata": {
                        "page_number": chunk.page_number,
                        "chunk_index": chunk.chunk_index,
                        **chunk.metadata,
                    },
                }
            )

        return results

    @property
    def count(self) -> int:
        return len(self._chunks)
