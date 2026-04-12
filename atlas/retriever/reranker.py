"""Cross-encoder re-ranker for retrieval results.

After hybrid retrieval (dense + BM25 + RRF) returns candidate chunks,
the re-ranker scores each (query, chunk) pair using a cross-encoder model.
Cross-encoders are much more accurate than bi-encoders because they see
the query and document together, enabling full attention between them.

This typically improves Precision@5 by 30-60% over embedding-only retrieval.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sentence_transformers import CrossEncoder

from atlas.config import settings
from atlas.observability.logger import get_logger

if TYPE_CHECKING:
    from atlas.api.schemas import ChunkResult

log = get_logger(__name__)

# Default cross-encoder model — small but effective
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """Re-rank retrieval results using a cross-encoder model.

    Cross-encoders process (query, document) pairs jointly through
    a transformer, producing a relevance score. This is more accurate
    than bi-encoder similarity but slower — hence we only re-rank
    the top-k candidates from the initial retrieval stage.
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        """
        Args:
            model_name: HuggingFace model ID for the cross-encoder.
                        Default is ms-marco-MiniLM which is fast and accurate.
        """
        self.model_name = model_name
        log.info("loading_reranker", model=model_name)
        self._model = CrossEncoder(model_name)
        log.info("reranker_ready", model=model_name)

    def rerank(
        self,
        query: str,
        results: list[ChunkResult],
        top_k: int | None = None,
    ) -> list[ChunkResult]:
        """Re-rank retrieval results using the cross-encoder.

        Args:
            query: The original search query.
            results: List of ChunkResult from hybrid retrieval.
            top_k: Number of top results to return after re-ranking.
                   If None, returns all results re-sorted.

        Returns:
            Re-ranked list of ChunkResult with updated scores.
        """
        if not results:
            return []

        if top_k is None:
            top_k = len(results)

        # Build (query, chunk_text) pairs for the cross-encoder
        pairs = [(query, r.text) for r in results]

        # Score all pairs
        scores = self._model.predict(pairs)

        # Attach scores and sort descending
        scored_results = list(zip(results, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Update scores and return top_k
        reranked = []
        for result, score in scored_results[:top_k]:
            # Create a new ChunkResult with the cross-encoder score
            reranked.append(
                result.model_copy(update={"score": round(float(score), 6)})
            )

        log.info(
            "reranking_complete",
            query=query[:80],
            candidates=len(results),
            returned=len(reranked),
            top_score=reranked[0].score if reranked else 0,
        )

        return reranked
