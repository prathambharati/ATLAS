# ATLAS — Autonomous Tool-using LLM Agent for Synthesis

An end-to-end autonomous research agent that takes a natural language question, decomposes it into sub-tasks, retrieves evidence from multiple sources (PDFs, web, arxiv), executes code for analysis, self-evaluates outputs for hallucinations using NLI + SHAP-based attribution, and produces grounded research reports with inline citations and confidence scores.

## Architecture

```
User Query
    │
    ▼
┌──────────┐    ┌──────────────┐    ┌─────────────┐
│ Planner  │ ──▶│  Retriever   │ ──▶│  Tool-Use   │
│ (DAG)    │    │  (Hybrid)    │    │  Executor   │
└──────────┘    └──────────────┘    └─────────────┘
    │                │                     │
    └────────────────┼─────────────────────┘
                     ▼
            ┌────────────────┐
            │  Orchestrator  │
            │  (ReAct Loop)  │
            └───────┬────────┘
                    ▼
         ┌─────────────────────┐
         │  Grounding Evaluator│
         │  (NLI + SHAP)       │
         └─────────┬───────────┘
                   ▼
          ┌────────────────┐
          │ Report + PDF   │
          └────────────────┘
```

## Key Features

- **Hybrid Retrieval** — Dense (sentence-transformers + ChromaDB) + Sparse (BM25) fused with Reciprocal Rank Fusion
- **Agentic Planning** — Query decomposition into a dependency DAG with topological execution
- **Tool Use** — Sandboxed code execution, web search (Tavily), arxiv lookup
- **Hallucination Detection** — NLI entailment scoring + SHAP chunk attribution per claim
- **Observability** — Full LLM call tracing, token/latency/cost tracking, structured JSON logging
- **Production-Ready** — FastAPI, Docker, CI/CD, WebSocket live trace streaming

## Quick Start

```bash
# Clone
git clone https://github.com/prathambharati/atlas.git
cd atlas

# Setup
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env with your API keys

# Run
make run
# API available at http://localhost:8080
# Health check: http://localhost:8080/health
# Docs: http://localhost:8080/docs
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/ingest` | Ingest a PDF document |
| `POST` | `/api/v1/retrieve` | Retrieve relevant chunks |
| `POST` | `/api/v1/research` | Run full research agent |
| `GET` | `/api/v1/traces` | List recent agent traces |
| `GET` | `/api/v1/traces/{id}` | Get specific trace |

## Development

```bash
make test        # Run tests
make lint        # Lint with ruff
make format      # Auto-format
make docker-up   # Start with Docker
```

## Tech Stack

Python 3.11+ · FastAPI · ChromaDB · sentence-transformers · BM25 · OpenAI · Tavily · SHAP · Docker

## License

MIT
