"""Hybrid retrieval: Dense + Sparse with Reciprocal Rank Fusion (RRF).

Combines semantic (dense) and keyword (sparse/BM25) retrieval using
Reciprocal Rank Fusion to get the best of both worlds.

RRF formula: score(d) = Σ 1 / (k + rank_i(d))
where k is a constant (default 60) and rank_i is the rank from retriever i.
"""

from atlas.api.schemas import ChunkResult
from atlas.retriever.dense import DenseIndex
from atlas.retriever.sparse import SparseIndex
from atlas.observability.logger import get_logger

log = get_logger(__name__)


class HybridRetriever:
    """Hybrid retriever combining dense and sparse search with RRF."""

    def __init__(self, rrf_k: int = 60):
        """
        Args:
            rrf_k: RRF constant. Higher values reduce the impact of rank position.
                   Standard value is 60 (from the original RRF paper).
        """
        self.dense_index = DenseIndex()
        self.sparse_index = SparseIndex()
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        method: str = "hybrid",
    ) -> list[ChunkResult]:
        """Retrieve relevant chunks using the specified method.

        Args:
            query: The search query.
            top_k: Number of results to return.
            method: One of "dense", "sparse", or "hybrid".

        Returns:
            List of ChunkResult sorted by relevance.
        """
        if method == "dense":
            raw_results = self.dense_index.search(query, top_k=top_k)
        elif method == "sparse":
            raw_results = self.sparse_index.search(query, top_k=top_k)
        elif method == "hybrid":
            raw_results = self._hybrid_search(query, top_k=top_k)
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

        log.info(
            "retrieval_complete",
            query=query[:80],
            method=method,
            num_results=len(results),
        )
        return results

    def _hybrid_search(self, query: str, top_k: int = 5) -> list[dict]:
        """Run both dense and sparse search, fuse with RRF.

        Fetches more candidates from each retriever than needed (2x top_k),
        then fuses and returns the top_k overall results.
        """
        candidate_k = top_k * 3  # Fetch extra candidates for better fusion

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
