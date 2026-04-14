"""API routes for ATLAS."""

from pathlib import Path

from fastapi import APIRouter, HTTPException

from atlas.agent.orchestrator import AgentOrchestrator
from atlas.api.schemas import (
    IngestRequest,
    IngestResponse,
    ResearchRequest,
    ResearchResponse,
    RetrievalRequest,
    RetrievalResponse,
)
from atlas.observability.logger import get_logger
from atlas.observability.tracer import trace_store
from atlas.report.generator import ReportGenerator
from atlas.retriever.hybrid import HybridRetriever
from atlas.retriever.ingest import DocumentIngestor

log = get_logger(__name__)

router = APIRouter()

# Initialize components (lazy singletons)
_ingestor: DocumentIngestor | None = None
_retriever: HybridRetriever | None = None
_agent: AgentOrchestrator | None = None
_report_gen: ReportGenerator | None = None


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


def get_agent() -> AgentOrchestrator:
    global _agent
    if _agent is None:
        _agent = AgentOrchestrator()
    return _agent


def get_report_generator() -> ReportGenerator:
    global _report_gen
    if _report_gen is None:
        _report_gen = ReportGenerator()
    return _report_gen


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: IngestRequest):
    """Ingest a PDF document into the vector store."""
    file_path = Path(request.file_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=404, detail=f"File not found: {file_path}"
        )

    if not file_path.suffix.lower() == ".pdf":
        raise HTTPException(
            status_code=400, detail="Only PDF files are supported"
        )

    try:
        ingestor = get_ingestor()
        result = ingestor.ingest(str(file_path), metadata=request.metadata)
        log.info(
            "document_ingested",
            file=str(file_path),
            chunks=result["num_chunks"],
        )
        return IngestResponse(
            document_id=result["document_id"],
            num_chunks=result["num_chunks"],
            message=(
                f"Successfully ingested {file_path.name} "
                f"into {result['num_chunks']} chunks"
            ),
        )
    except Exception as e:
        log.error("ingest_failed", file=str(file_path), error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrieve", response_model=RetrievalResponse)
async def retrieve_chunks(request: RetrievalRequest):
    """Retrieve relevant chunks for a query with optional re-ranking."""
    try:
        retriever = get_retriever()
        results = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            method=request.method,
            rerank=request.rerank,
        )
        log.info(
            "retrieval_complete",
            query=request.query,
            num_results=len(results),
            reranked=request.rerank,
        )
        return RetrievalResponse(
            query=request.query,
            method=request.method,
            reranked=request.rerank,
            results=results,
            num_results=len(results),
        )
    except Exception as e:
        log.error("retrieval_failed", query=request.query, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/research", response_model=ResearchResponse)
async def run_research(request: ResearchRequest):
    """Run the full autonomous research agent pipeline.

    1. Decomposes query into sub-tasks (DAG)
    2. Executes each task with ReAct agent + tools
    3. Generates structured report with citations
    """
    try:
        agent = get_agent()
        report_gen = get_report_generator()

        # Run the agent
        result = agent.run(query=request.query)

        # Generate structured report
        report = report_gen.generate(
            query=result["query"],
            answer=result["answer"],
            sources=result["sources"],
            dag_summary=result.get("dag_summary"),
        )

        return ResearchResponse(
            trace_id=result["trace_id"],
            query=result["query"],
            status="completed",
            report=report,
            confidence=None,
            sources=result["sources"],
        )
    except Exception as e:
        log.error("research_failed", query=request.query, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


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
