"""Pydantic models for API request/response schemas."""

from pydantic import BaseModel, Field


# --- Ingestion ---

class IngestRequest(BaseModel):
    """Request to ingest a document into the vector store."""
    file_path: str = Field(..., description="Path to PDF file to ingest")
    metadata: dict = Field(default_factory=dict, description="Optional metadata")


class IngestResponse(BaseModel):
    """Response after ingesting a document."""
    document_id: str
    num_chunks: int
    message: str


# --- Retrieval ---

class RetrievalRequest(BaseModel):
    """Request to retrieve relevant chunks for a query."""
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=5, ge=1, le=20)
    method: str = Field(default="hybrid", pattern="^(dense|sparse|hybrid)$")


class ChunkResult(BaseModel):
    """A single retrieved chunk with score."""
    chunk_id: str
    text: str
    score: float
    source: str
    metadata: dict = Field(default_factory=dict)


class RetrievalResponse(BaseModel):
    """Response containing retrieved chunks."""
    query: str
    method: str
    results: list[ChunkResult]
    num_results: int


# --- Research (full pipeline — will be built in later weeks) ---

class ResearchRequest(BaseModel):
    """Request to run the full research agent."""
    query: str = Field(..., description="Research question")
    max_steps: int = Field(default=10, ge=1, le=50)
    include_web_search: bool = Field(default=True)
    include_arxiv: bool = Field(default=True)


class ResearchResponse(BaseModel):
    """Response from the research agent."""
    trace_id: str
    query: str
    status: str
    report: str | None = None
    confidence: float | None = None
    sources: list[dict] = Field(default_factory=list)
