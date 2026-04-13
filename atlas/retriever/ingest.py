"""Document ingestion pipeline.

Reads PDFs, splits into chunks, computes embeddings, and stores
in both ChromaDB (dense) and a BM25 index (sparse).
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

from atlas.config import settings
from atlas.observability.logger import get_logger
from atlas.retriever.dense import DenseIndex
from atlas.retriever.sparse import SparseIndex

log = get_logger(__name__)


@dataclass
class Chunk:
    """A text chunk from a document."""

    chunk_id: str
    text: str
    source: str
    page_number: int
    chunk_index: int
    metadata: dict


class RecursiveChunker:
    """Split text into overlapping chunks using recursive character splitting.

    Tries to split on paragraph breaks first, then sentences, then words,
    preserving semantic coherence within each chunk.
    """

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split(self, text: str) -> list[str]:
        """Split text into chunks recursively."""
        chunks = self._recursive_split(text, self.separators)
        # Merge small chunks and add overlap
        return self._merge_with_overlap(chunks)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """Try each separator in order until chunks are small enough."""
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        # Try the first separator
        sep = separators[0]
        remaining_seps = separators[1:] if len(separators) > 1 else separators

        if sep == "":
            # Hard split by character count (last resort)
            return [
                text[i : i + self.chunk_size].strip()
                for i in range(0, len(text), self.chunk_size)
                if text[i : i + self.chunk_size].strip()
            ]

        splits = text.split(sep)
        result = []
        current = ""

        for piece in splits:
            test = current + sep + piece if current else piece
            if len(test) <= self.chunk_size:
                current = test
            else:
                if current:
                    result.append(current.strip())
                # If the single piece is too big, recurse with next separator
                if len(piece) > self.chunk_size:
                    result.extend(self._recursive_split(piece, remaining_seps))
                else:
                    current = piece

        if current.strip():
            result.append(current.strip())

        return result

    def _merge_with_overlap(self, chunks: list[str]) -> list[str]:
        """Add overlap between consecutive chunks for context continuity."""
        if not chunks or self.chunk_overlap == 0:
            return chunks

        merged = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            # Take the last `chunk_overlap` characters from previous chunk
            overlap = prev[-self.chunk_overlap :] if len(prev) > self.chunk_overlap else prev
            merged.append(overlap + " " + chunks[i])

        return merged


class DocumentIngestor:
    """Ingest PDF documents into dense + sparse indices."""

    def __init__(self):
        self.chunker = RecursiveChunker()
        self.dense_index = DenseIndex()
        self.sparse_index = SparseIndex()
        self._ingested: dict[str, list[Chunk]] = {}

    def ingest(self, file_path: str, metadata: dict | None = None) -> dict:
        """Ingest a PDF file: extract text, chunk, embed, index.

        Args:
            file_path: Path to the PDF file.
            metadata: Optional metadata to attach to all chunks.

        Returns:
            Dict with document_id and num_chunks.
        """
        metadata = metadata or {}
        path = Path(file_path)
        doc_id = self._generate_doc_id(path)

        log.info("ingesting_document", file=path.name, doc_id=doc_id)

        # 1. Extract text from PDF
        pages = self._extract_text(path)
        log.info("extracted_pages", file=path.name, num_pages=len(pages))

        # 2. Chunk each page
        all_chunks: list[Chunk] = []
        chunk_idx = 0
        for page_num, page_text in enumerate(pages, start=1):
            if not page_text.strip():
                continue
            text_chunks = self.chunker.split(page_text)
            for text in text_chunks:
                chunk = Chunk(
                    chunk_id=f"{doc_id}_chunk_{chunk_idx:04d}",
                    text=text,
                    source=path.name,
                    page_number=page_num,
                    chunk_index=chunk_idx,
                    metadata={**metadata, "document_id": doc_id},
                )
                all_chunks.append(chunk)
                chunk_idx += 1

        log.info("chunked_document", file=path.name, num_chunks=len(all_chunks))

        if not all_chunks:
            log.warning("no_chunks_created", file=path.name)
            return {"document_id": doc_id, "num_chunks": 0}

        # 3. Add to dense index (embeddings + ChromaDB)
        self.dense_index.add_chunks(all_chunks)

        # 4. Add to sparse index (BM25)
        self.sparse_index.add_chunks(all_chunks)

        # 5. Store reference
        self._ingested[doc_id] = all_chunks

        log.info(
            "ingestion_complete",
            file=path.name,
            doc_id=doc_id,
            num_chunks=len(all_chunks),
        )

        return {"document_id": doc_id, "num_chunks": len(all_chunks)}

    def _extract_text(self, path: Path) -> list[str]:
        """Extract text from each page of a PDF."""
        pages = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                pages.append(text)
        return pages

    def _generate_doc_id(self, path: Path) -> str:
        """Generate a deterministic document ID from file content hash."""
        hasher = hashlib.sha256()
        hasher.update(path.read_bytes())
        return hasher.hexdigest()[:16]

    @property
    def num_documents(self) -> int:
        return len(self._ingested)

    @property
    def num_total_chunks(self) -> int:
        return sum(len(chunks) for chunks in self._ingested.values())
