<p align="center">
  <h1 align="center">ATLAS</h1>
  <p align="center"><strong>Autonomous Tool-using LLM Agent for Synthesis</strong></p>
  <p align="center">
    An end-to-end autonomous research agent with multi-source hybrid retrieval, agentic planning, tool use, hallucination detection via NLI + SHAP attribution, and production-grade observability.
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.111+-009688?style=flat-square&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/LLM-GPT--4o%20%7C%20Llama%203-purple?style=flat-square" />
  <img src="https://img.shields.io/badge/Vector%20DB-ChromaDB-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" />
</p>

---

## The Problem

LLMs hallucinate. RAG pipelines help, but most implementations are shallow — single-source retrieval, no verification, no transparency into what the model actually grounded its answer on.

**ATLAS** is a research agent that doesn't just retrieve-and-generate. It **plans** multi-step research strategies, **retrieves** from multiple sources simultaneously, **executes code** for data analysis, and then **audits its own output** — scoring every claim against retrieved evidence using Natural Language Inference and producing interpretable attribution trails via SHAP.

## Architecture

```
                            ┌─────────────────────┐
                            │     User Query       │
                            └──────────┬──────────┘
                                       │
                            ┌──────────▼──────────┐
                            │   Planner Agent      │
                            │  Query → Sub-task    │
                            │  DAG (topological)   │
                            └──────────┬──────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                   │
          ┌─────────▼────────┐ ┌──────▼───────┐ ┌────────▼────────┐
          │  Hybrid Retriever│ │  Web Search  │ │  Arxiv Search   │
          │  Dense + BM25    │ │  (Tavily)    │ │  (API)          │
          │  + RRF Fusion    │ └──────┬───────┘ └────────┬────────┘
          └─────────┬────────┘        │                  │
                    │                 │                  │
          ┌─────────▼────────┐        │                  │
          │  Cross-Encoder   │◄───────┴──────────────────┘
          │  Re-ranker       │
          └─────────┬────────┘
                    │
          ┌─────────▼─────────────────────────────┐
          │         Orchestrator (ReAct Loop)       │
          │  Reason → Act → Observe → Repeat       │
          │  Tools: code exec, search, calculator   │
          └─────────┬─────────────────────────────┘
                    │
          ┌─────────▼─────────────────────────────┐
          │    Hallucination & Grounding Evaluator  │
          │                                         │
          │  1. Extract atomic claims from output   │
          │  2. NLI entailment scoring per claim    │
          │  3. SHAP attribution → which chunk      │
          │     influenced which claim              │
          │  4. Confidence score per section         │
          └─────────┬─────────────────────────────┘
                    │
          ┌─────────▼─────────────────────────────┐
          │         Report Generator                │
          │  Structured markdown/PDF with:          │
          │  • Inline citations [1], [2]            │
          │  • Per-section confidence bars           │
          │  • Auto-generated visualizations         │
          └─────────────────────────────────────────┘
```

## Key Technical Contributions

### Hybrid Retrieval with Reciprocal Rank Fusion
Combines dense retrieval (sentence-transformers embeddings + ChromaDB) with sparse retrieval (BM25) using [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf). The fusion formula `score(d) = Σ 1/(k + rank_i(d))` across retrievers consistently outperforms either method alone, particularly on queries mixing technical terms with natural language.

### Claim-Level Hallucination Detection
Unlike binary "hallucinated or not" approaches, ATLAS extracts atomic claims from generated text and scores each independently against retrieved evidence using a Natural Language Inference model (BART-MNLI). Claims below a configurable threshold are flagged with the specific evidence gap identified.

### SHAP-Based Evidence Attribution
Uses SHAP (SHapley Additive exPlanations) on the cross-encoder to produce interpretable attribution maps — showing exactly which retrieved chunk influenced each generated claim. This provides an auditable evidence trail from source → retrieval → generation.

### Agentic DAG-Based Planning
The planner decomposes complex research questions into a Directed Acyclic Graph of sub-tasks with explicit dependencies. Tasks without dependencies execute in parallel, while dependent tasks wait for upstream results. Topological sorting ensures correct execution order.

### Production Observability
Every LLM call, retrieval step, and tool invocation is traced with latency, token counts, and cost. Traces stream to the frontend via WebSocket for a real-time "agent thinking" UI.

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **LLM Inference** | GPT-4o-mini / Llama 3.1 8B via vLLM | Cloud + local fallback for cost control |
| **Embeddings** | `all-MiniLM-L6-v2` (384-dim) | Best speed/quality tradeoff for retrieval |
| **Vector Store** | ChromaDB (HNSW, cosine) | In-process for dev, HTTP for production |
| **Sparse Search** | BM25 (Okapi) via `rank_bm25` | Catches exact keyword matches dense misses |
| **Re-ranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 6x precision improvement over embedding-only |
| **NLI Model** | `facebook/bart-large-mnli` | Entailment scoring for hallucination detection |
| **Explainability** | SHAP (Transformer Explainer) | Chunk-level attribution for evidence trails |
| **Backend** | FastAPI + Uvicorn | Async-native, WebSocket support, auto-docs |
| **Frontend** | React + Tailwind CSS | Live agent trace streaming |
| **Code Sandbox** | subprocess with resource limits | Safe execution of agent-generated code |
| **PDF Processing** | pdfplumber + unstructured | Layout-aware text extraction with table support |
| **Web Search** | Tavily API | Optimized for RAG (returns clean text, not HTML) |
| **Containerization** | Docker + docker-compose | One-command deployment (app + ChromaDB) |
| **CI/CD** | GitHub Actions | Lint + test on every PR |
| **Logging** | structlog (JSON) | Structured, parseable, production-ready |

## Modules

```
atlas/
├── retriever/          # Hybrid retrieval: dense + sparse + reranker + web + arxiv
├── planner/            # Query decomposition into executable DAG
├── agent/              # ReAct orchestrator with tool-use loop
├── tools/              # Sandboxed code execution, search, calculator
├── evaluator/          # NLI grounding, SHAP attribution, confidence scoring
├── report/             # Structured report generation with citations
├── observability/      # Call tracing, token/cost metrics, structured logging
└── api/                # FastAPI routes, Pydantic schemas, WebSocket streaming
```

## Quick Start

```bash
git clone https://github.com/prathambharati/ATLAS.git
cd ATLAS

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
cp .env.example .env           # Add your API keys

# Start the server
uvicorn atlas.main:app --reload --port 8080

# Ingest a document
curl -X POST http://localhost:8080/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_path": "path/to/paper.pdf"}'

# Search with hybrid retrieval
curl -X POST http://localhost:8080/api/v1/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "transformer attention mechanism", "top_k": 5, "method": "hybrid"}'
```

**Interactive API docs:** http://localhost:8080/docs

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/ingest` | Ingest PDF → chunk → embed → index |
| `POST` | `/api/v1/retrieve` | Hybrid retrieval with RRF fusion |
| `POST` | `/api/v1/research` | Full autonomous research pipeline |
| `GET` | `/api/v1/traces` | List agent execution traces |
| `GET` | `/api/v1/traces/{id}` | Detailed trace with per-step metrics |

## Evaluation

| Metric | Method | Target |
|--------|--------|--------|
| Answer Relevance | LLM-as-judge (1-5 scale) | ≥ 4.0 |
| Groundedness | % claims with NLI entailment > 0.7 | ≥ 85% |
| Retrieval Precision@5 | Human-labeled relevance | ≥ 70% |
| Faithfulness | % claims traceable to source | ≥ 90% |
| End-to-end Latency | Query → report | < 60s |

## Development

```bash
python -m pytest tests/ -v          # Run tests
ruff check atlas/ tests/            # Lint
ruff format atlas/ tests/           # Format
docker-compose up -d                # Start with Docker
```

## Project Roadmap

- [x] Hybrid retrieval engine (dense + sparse + RRF)
- [x] PDF ingestion with recursive chunking
- [x] FastAPI server with Swagger docs
- [x] Observability layer (tracing, structured logging)
- [x] Unit tests (19 passing)
- [ ] Cross-encoder re-ranking
- [ ] Web search + Arxiv integration
- [ ] Query decomposition planner (DAG)
- [ ] ReAct agent orchestrator with tool use
- [ ] Hallucination detection (NLI + SHAP)
- [ ] Report generator with citations
- [ ] React frontend with live agent trace
- [ ] Evaluation suite
- [ ] Docker deployment

## License

MIT

---

<p align="center">
  Built by <a href="https://github.com/prathambharati">Pratham Bharati</a> · MS Applied Machine Learning @ University of Maryland
</p>
