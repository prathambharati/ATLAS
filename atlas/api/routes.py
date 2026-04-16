"""API routes for ATLAS."""

import json
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from atlas.agent.orchestrator import AgentOrchestrator
from atlas.api.schemas import (
    IngestRequest,
    IngestResponse,
    ResearchRequest,
    ResearchResponse,
    RetrievalRequest,
    RetrievalResponse,
)
from atlas.evaluator.evaluator import HallucinationEvaluator
from atlas.evaluator.self_corrector import SelfCorrector
from atlas.observability.logger import get_logger
from atlas.observability.metrics import cost_tracker
from atlas.observability.tracer import trace_store
from atlas.report.generator import ReportGenerator
from atlas.retriever.hybrid import HybridRetriever
from atlas.retriever.ingest import DocumentIngestor
from atlas.tools.default_tools import build_default_tools

log = get_logger(__name__)

router = APIRouter()

# Initialize components (lazy singletons)
_ingestor: DocumentIngestor | None = None
_retriever: HybridRetriever | None = None
_agent: AgentOrchestrator | None = None
_report_gen: ReportGenerator | None = None
_evaluator: HallucinationEvaluator | None = None
_corrector: SelfCorrector | None = None

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


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
        retriever = get_retriever()
        tools = build_default_tools(retriever=retriever)
        _agent = AgentOrchestrator(tool_registry=tools)
    return _agent


def _rebuild_agent() -> None:
    """Rebuild the agent with the current retriever (call after ingestion)."""
    global _agent
    retriever = get_retriever()
    tools = build_default_tools(retriever=retriever)
    _agent = AgentOrchestrator(tool_registry=tools)


def get_report_generator() -> ReportGenerator:
    global _report_gen
    if _report_gen is None:
        _report_gen = ReportGenerator()
    return _report_gen


def get_evaluator() -> HallucinationEvaluator:
    global _evaluator
    if _evaluator is None:
        _evaluator = HallucinationEvaluator()
    return _evaluator


def get_corrector() -> SelfCorrector:
    global _corrector
    if _corrector is None:
        _corrector = SelfCorrector(max_cycles=2)
    return _corrector


# --- Document Ingestion ---


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
        # Share with agent's retriever
        retriever = get_retriever()
        retriever.dense_index = ingestor.dense_index
        retriever.sparse_index = ingestor.sparse_index

        # Rebuild agent so its retrieve tool uses the updated indices
        _rebuild_agent()

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


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and ingest a PDF file from the browser."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Only PDF files are supported"
        )
    try:
        save_path = UPLOAD_DIR / file.filename
        content = await file.read()
        save_path.write_bytes(content)

        ingestor = get_ingestor()
        result = ingestor.ingest(str(save_path))

        # Share indices with retriever so agent can search uploaded docs
        retriever = get_retriever()
        retriever.dense_index = ingestor.dense_index
        retriever.sparse_index = ingestor.sparse_index

        # Rebuild agent so its retrieve tool uses the updated indices
        _rebuild_agent()

        log.info(
            "pdf_uploaded_and_ingested",
            file=file.filename,
            chunks=result["num_chunks"],
        )
        return {
            "filename": file.filename,
            "document_id": result["document_id"],
            "num_chunks": result["num_chunks"],
            "message": (
                f"Ingested {file.filename} "
                f"into {result['num_chunks']} chunks"
            ),
        }
    except Exception as e:
        log.error("upload_failed", file=file.filename, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# --- Retrieval ---


@router.post("/retrieve", response_model=RetrievalResponse)
async def retrieve_chunks(request: RetrievalRequest):
    """Retrieve relevant chunks for a query."""
    try:
        retriever = get_retriever()
        results = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            method=request.method,
            rerank=request.rerank,
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


# --- Research ---


@router.post("/research", response_model=ResearchResponse)
async def run_research(request: ResearchRequest):
    """Run research (non-streaming)."""
    try:
        agent = get_agent()
        report_gen = get_report_generator()
        result = agent.run(query=request.query)

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


@router.post("/research/stream")
async def run_research_stream(request: ResearchRequest):
    """Research with streaming updates + self-correction."""

    async def event_stream():
        agent = get_agent()
        report_gen = get_report_generator()

        def send(event_type: str, data: dict) -> str:
            return f"data: {json.dumps({'type': event_type, **data})}\n\n"

        try:
            # Plan
            yield send("status", {"message": "Analyzing query..."})
            dag = agent._decomposer.decompose(request.query)
            yield send("plan", {
                "message": f"Created {dag.num_tasks} sub-tasks",
                "tasks": [
                    {"id": t.task_id, "query": t.query}
                    for t in dag.tasks
                ],
            })

            # Execute
            from atlas.agent.memory import AgentMemory

            memory = AgentMemory()
            trace = trace_store.create(query=request.query)
            batches = dag.execution_order()
            total_tokens_in = 0
            total_tokens_out = 0

            for batch_idx, batch in enumerate(batches):
                for task in batch:
                    yield send("task_start", {
                        "message": f"Researching: {task.query}",
                        "task_id": task.task_id,
                    })

                    task_result = agent._execute_task(
                        task_query=task.query,
                        task_description=task.description,
                        memory=memory,
                        trace=trace,
                    )
                    dag.mark_completed(task.task_id, task_result)
                    memory.add_task_result(task.task_id, task_result)

                    yield send("task_done", {
                        "message": f"Completed: {task.query}",
                        "task_id": task.task_id,
                    })

            # Synthesize
            yield send("status", {"message": "Synthesizing findings..."})
            if dag.num_tasks > 1:
                answer = agent._synthesize(
                    request.query, memory, trace
                )
            else:
                results = dag.get_results()
                answer = (
                    list(results.values())[0]
                    if results
                    else "No answer generated."
                )

            # Build evidence from sources
            evidence_chunks = []
            for source in memory.sources:
                text = source.get(
                    "full_text", source.get("result_preview", "")
                )
                if text and "Error" not in text:
                    evidence_chunks.append({
                        "text": text[:1000],
                        "source": source.get("tool", "unknown"),
                    })

            # Evaluate
            evaluation = None
            if evidence_chunks:
                yield send("status", {
                    "message": "Evaluating groundedness..."
                })
                try:
                    evaluator = get_evaluator()
                    evaluation = evaluator.evaluate(
                        answer, evidence_chunks
                    )
                    yield send("evaluation", {
                        "message": "Initial evaluation complete",
                        "confidence": evaluation.overall_confidence,
                        "total_claims": evaluation.total_claims,
                        "supported": evaluation.supported_claims,
                        "unsupported": evaluation.unsupported_claims,
                        "contradicted": evaluation.contradicted_claims,
                        "claims": evaluation.claim_results,
                    })

                    # Self-correction if confidence is low
                    # Self-correction (disabled — needs tuning)
                    if False and (
                        evaluation.overall_confidence < 0.8
                        and evaluation.unsupported_claims > 0
                    ):
                        yield send("status", {
                            "message": (
                                "Confidence below 80% — "
                                "starting self-correction..."
                            ),
                        })

                        corrector = get_corrector()

                        def on_cycle(num, cycle):
                            pass  # Logged internally

                        correction = corrector.correct(
                            answer=answer,
                            evidence_chunks=evidence_chunks,
                            trace=trace,
                            on_cycle=on_cycle,
                        )

                        answer = correction.corrected_answer

                        yield send("correction", {
                            "message": (
                                f"Self-correction complete: "
                                f"{correction.original_confidence:.0%} → "
                                f"{correction.final_confidence:.0%}"
                            ),
                            "original_confidence": (
                                correction.original_confidence
                            ),
                            "final_confidence": (
                                correction.final_confidence
                            ),
                            "num_cycles": correction.num_cycles,
                            "cycles": [
                                c.to_dict()
                                for c in correction.cycles
                            ],
                            "converged": correction.converged,
                        })

                        # Re-evaluate corrected answer
                        evaluation = correction.final_evaluation
                        if evaluation:
                            yield send("evaluation", {
                                "message": "Post-correction evaluation",
                                "confidence": (
                                    evaluation.overall_confidence
                                ),
                                "total_claims": evaluation.total_claims,
                                "supported": (
                                    evaluation.supported_claims
                                ),
                                "unsupported": (
                                    evaluation.unsupported_claims
                                ),
                                "contradicted": (
                                    evaluation.contradicted_claims
                                ),
                                "claims": evaluation.claim_results,
                            })

                except Exception as e:
                    log.error("eval_stream_error", error=str(e))

            # Track cost
            for step in trace.steps:
                total_tokens_in += step.tokens_in
                total_tokens_out += step.tokens_out

            cost_entry = cost_tracker.record(
                query=request.query,
                tokens_in=total_tokens_in,
                tokens_out=total_tokens_out,
                trace_id=trace.trace_id,
            )

            # Generate report
            yield send("status", {"message": "Generating report..."})
            report = report_gen.generate(
                query=request.query,
                answer=answer,
                sources=memory.sources,
                evaluation=evaluation,
                dag_summary=dag.summary(),
            )

            yield send("complete", {
                "message": "Research complete",
                "trace_id": trace.trace_id,
                "report": report,
                "sources": memory.sources,
                "confidence": (
                    evaluation.overall_confidence
                    if evaluation
                    else None
                ),
                "cost": cost_entry.to_dict(),
            })

        except Exception as e:
            yield send("error", {"message": str(e)})

    return StreamingResponse(
        event_stream(), media_type="text/event-stream"
    )


# --- Cost & Traces ---


@router.get("/costs")
async def get_costs():
    """Get cost tracking summary."""
    return cost_tracker.get_summary()


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
