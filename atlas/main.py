"""ATLAS FastAPI application entrypoint."""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from atlas.config import settings
from atlas.api.routes import router
from atlas.observability.logger import setup_logging


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
