"""API routes for ATLAS."""

from fastapi import APIRouter, HTTPException
from pathlib import Path

from atlas.api.schemas import (
    IngestRequest,
    IngestResponse,
    RetrievalRequest,
    RetrievalResponse,
    ResearchRequest,
    ResearchResponse,
)
from atlas.retriever.ingest import DocumentIngestor
from atlas.retriever.hybrid import HybridRetriever
from atlas.observability.logger import get_logger
from atlas.observability.tracer import trace_store

log = get_logger(__name__)

router = APIRouter()

# Initialize components (lazy singletons)
_ingestor: DocumentIngestor | None = None
_retriever: HybridRetriever | None = None


def get_ingestor() -> DocumentIngestor:
    global _ingestor
    if _ingestor is None:
        _ingestor = DocumentIngestor()
    return _ingestor


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: IngestRequest):
    """Ingest a PDF document into the vector store."""
    file_path = Path(request.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    if not file_path.suffix.lower() == ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        ingestor = get_ingestor()
        result = ingestor.ingest(str(file_path), metadata=request.metadata)
        log.info("document_ingested", file=str(file_path), chunks=result["num_chunks"])
        return IngestResponse(
            document_id=result["document_id"],
            num_chunks=result["num_chunks"],
            message=f"Successfully ingested {file_path.name} into {result['num_chunks']} chunks",
        )
    except Exception as e:
        log.error("ingest_failed", file=str(file_path), error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrieve", response_model=RetrievalResponse)
async def retrieve_chunks(request: RetrievalRequest):
    """Retrieve relevant chunks for a query."""
    try:
        retriever = get_retriever()
        results = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            method=request.method,
        )
        log.info("retrieval_complete", query=request.query, num_results=len(results))
        return RetrievalResponse(
            query=request.query,
            method=request.method,
            results=results,
            num_results=len(results),
        )
    except Exception as e:
        log.error("retrieval_failed", query=request.query, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/research", response_model=ResearchResponse)
async def run_research(request: ResearchRequest):
    """Run the full research agent pipeline. (Stub — built in Week 4)"""
    trace = trace_store.create(query=request.query)
    return ResearchResponse(
        trace_id=trace.trace_id,
        query=request.query,
        status="not_implemented",
        report=None,
        confidence=None,
        sources=[],
    )


@router.get("/traces")
async def list_traces(n: int = 10):
    """List recent agent traces."""
    return trace_store.list_recent(n=n)


@router.get("/traces/{trace_id}")
async def get_trace(trace_id: str):
    """Get a specific agent trace."""
    trace = trace_store.get(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace.summary()
