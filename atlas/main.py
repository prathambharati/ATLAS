"""ATLAS FastAPI application entrypoint."""

from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from atlas.api.routes import router
from atlas.config import settings
from atlas.observability.logger import setup_logging

FRONTEND_PATH = Path(__file__).parent.parent / "frontend" / "index.html"


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    setup_logging(level=settings.log_level)

    app = FastAPI(
        title="ATLAS",
        description="Autonomous Tool-using LLM Agent for Synthesis",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "version": "0.1.0"}

    @app.get("/", response_class=HTMLResponse)
    async def serve_frontend():
        """Serve the ATLAS frontend."""
        if FRONTEND_PATH.exists():
            return FRONTEND_PATH.read_text(encoding="utf-8")
        return HTMLResponse(
            content="<h1>ATLAS</h1><p>Frontend not found. "
            "Visit <a href='/docs'>/docs</a> for the API.</p>",
            status_code=200,
        )

    return app


app = create_app()


def run():
    """Run the server (called via CLI: `atlas`)."""
    uvicorn.run(
        "atlas.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )


if __name__ == "__main__":
    run()
