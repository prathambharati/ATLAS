"""Dense retrieval using sentence-transformers embeddings + ChromaDB.

Embeds chunks and queries with a sentence-transformer model,
stores/retrieves from ChromaDB for semantic similarity search.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import chromadb
from sentence_transformers import SentenceTransformer

from atlas.config import settings
from atlas.observability.logger import get_logger

if TYPE_CHECKING:
    from atlas.retriever.ingest import Chunk

log = get_logger(__name__)


class DenseIndex:
    """Dense vector index backed by ChromaDB + sentence-transformers."""

    def __init__(
        self,
        model_name: str = settings.embedding_model,
        collection_name: str = settings.chroma_collection,
        persist: bool = False,
    ):
        self.model_name = model_name
        self.collection_name = collection_name

        # Load embedding model
        log.info("loading_embedding_model", model=model_name)
        self._model = SentenceTransformer(model_name)

        # Initialize ChromaDB (in-memory for dev, persistent for prod)
        if persist:
            self._client = chromadb.HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port,
            )
        else:
            self._client = chromadb.Client()

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        log.info(
            "dense_index_ready",
            model=model_name,
            collection=collection_name,
            existing_count=self._collection.count(),
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings for a list of texts."""
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Compute embedding for a single query."""
        return self._model.encode(query).tolist()

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Embed and store chunks in ChromaDB."""
        if not chunks:
            return

        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {
                "source": c.source,
                "page_number": c.page_number,
                "chunk_index": c.chunk_index,
                **{k: str(v) for k, v in c.metadata.items()},
            }
            for c in chunks
        ]

        # Embed in batches of 64
        batch_size = 64
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self.embed(batch)
            all_embeddings.extend(embeddings)

        # Upsert into ChromaDB
        self._collection.upsert(
            ids=ids,
            embeddings=all_embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        log.info("dense_index_updated", num_added=len(chunks), total=self._collection.count())

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Search for the most similar chunks to a query.

        Returns:
            List of dicts with keys: chunk_id, text, score, source, metadata
        """
        query_embedding = self.embed_query(query)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        output = []
        for i, chunk_id in enumerate(results["ids"][0]):
            # ChromaDB returns cosine distance; convert to similarity
            distance = results["distances"][0][i]
            similarity = 1 - distance

            output.append({
                "chunk_id": chunk_id,
                "text": results["documents"][0][i],
                "score": round(similarity, 4),
                "source": results["metadatas"][0][i].get("source", "unknown"),
                "metadata": results["metadatas"][0][i],
            })

        return output

    @property
    def count(self) -> int:
        return self._collection.count()
