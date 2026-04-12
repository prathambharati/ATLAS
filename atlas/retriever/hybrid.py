"""Hybrid retrieval: Dense + Sparse with Reciprocal Rank Fusion (RRF) + Re-ranking.

Combines semantic (dense) and keyword (sparse/BM25) retrieval using
Reciprocal Rank Fusion, then optionally re-ranks with a cross-encoder.

RRF formula: score(d) = Σ 1 / (k + rank_i(d))
where k is a constant (default 60) and rank_i is the rank from retriever i.
"""

from atlas.api.schemas import ChunkResult
from atlas.retriever.dense import DenseIndex
from atlas.retriever.sparse import SparseIndex
from atlas.retriever.reranker import Reranker
from atlas.observability.logger import get_logger

log = get_logger(__name__)


class HybridRetriever:
    """Hybrid retriever combining dense and sparse search with RRF + re-ranking."""

    def __init__(self, rrf_k: int = 60, use_reranker: bool = True):
        """
        Args:
            rrf_k: RRF constant. Higher values reduce the impact of rank position.
                   Standard value is 60 (from the original RRF paper).
            use_reranker: Whether to load and use the cross-encoder re-ranker.
        """
        self.dense_index = DenseIndex()
        self.sparse_index = SparseIndex()
        self.rrf_k = rrf_k

        # Lazy-load reranker (it takes a few seconds to load the model)
        self._reranker: Reranker | None = None
        self._use_reranker = use_reranker

    @property
    def reranker(self) -> Reranker:
        """Lazy-load the re-ranker on first use."""
        if self._reranker is None:
            self._reranker = Reranker()
        return self._reranker

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        method: str = "hybrid",
        rerank: bool = True,
    ) -> list[ChunkResult]:
        """Retrieve relevant chunks using the specified method.

        Args:
            query: The search query.
            top_k: Number of final results to return.
            method: One of "dense", "sparse", or "hybrid".
            rerank: Whether to apply cross-encoder re-ranking.

        Returns:
            List of ChunkResult sorted by relevance.
        """
        # Fetch more candidates when re-ranking (re-ranker picks the best from a larger pool)
        candidate_k = top_k * 4 if rerank else top_k

        if method == "dense":
            raw_results = self.dense_index.search(query, top_k=candidate_k)
        elif method == "sparse":
            raw_results = self.sparse_index.search(query, top_k=candidate_k)
        elif method == "hybrid":
            raw_results = self._hybrid_search(query, top_k=candidate_k)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'dense', 'sparse', or 'hybrid'.")

        results = [
            ChunkResult(
                chunk_id=r["chunk_id"],
                text=r["text"],
                score=r["score"],
                source=r["source"],
                metadata=r.get("metadata", {}),
            )
            for r in raw_results
        ]

        # Apply re-ranking if enabled
        if rerank and self._use_reranker and results:
            results = self.reranker.rerank(query=query, results=results, top_k=top_k)
            log.info(
                "retrieval_with_rerank",
                query=query[:80],
                method=method,
                candidates=len(raw_results),
                final=len(results),
            )
        else:
            results = results[:top_k]
            log.info(
                "retrieval_no_rerank",
                query=query[:80],
                method=method,
                num_results=len(results),
            )

        return results

    def _hybrid_search(self, query: str, top_k: int = 5) -> list[dict]:
        """Run both dense and sparse search, fuse with RRF.

        Fetches more candidates from each retriever than needed (3x top_k),
        then fuses and returns the top_k overall results.
        """
        candidate_k = top_k * 3

        dense_results = self.dense_index.search(query, top_k=candidate_k)
        sparse_results = self.sparse_index.search(query, top_k=candidate_k)

        log.info(
            "hybrid_candidates",
            dense_count=len(dense_results),
            sparse_count=len(sparse_results),
        )

        # Build RRF scores
        rrf_scores: dict[str, float] = {}
        chunk_data: dict[str, dict] = {}

        # Score dense results by rank
        for rank, result in enumerate(dense_results):
            cid = result["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            chunk_data[cid] = result

        # Score sparse results by rank
        for rank, result in enumerate(sparse_results):
            cid = result["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            if cid not in chunk_data:
                chunk_data[cid] = result

        # Sort by RRF score and take top_k
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]

        results = []
        for cid in sorted_ids:
            data = chunk_data[cid]
            data["score"] = round(rrf_scores[cid], 6)
            results.append(data)

        return results