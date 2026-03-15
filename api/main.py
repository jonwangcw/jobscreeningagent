"""FastAPI app factory.

Architecture notes:
- No business logic here. Route handlers call agent/ functions only.
- Frontend (built React app) is served as static files from /
- Scheduler is started at app startup via lifespan context
"""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from agent.main import load_config
from agent.scheduler import start_scheduler, stop_scheduler, trigger_run
from api.routes.generate import get_generate_router
from api.routes.jobs import router as jobs_router

logger = logging.getLogger(__name__)

_config: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global _config
    config_path = os.environ.get("CONFIG_PATH", "config.yml")
    _config = load_config(config_path)
    start_scheduler(_config)
    logger.info("App started. Scheduler running.")
    yield
    # Shutdown
    stop_scheduler()
    logger.info("App stopped.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Job Application Agent",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API routes
    app.include_router(jobs_router)

    config_path = os.environ.get("CONFIG_PATH", "config.yml")
    cfg = load_config(config_path)
    app.include_router(get_generate_router(cfg))

    # POST /run — manual trigger
    @app.post("/run", tags=["scheduler"])
    def manual_run():
        stats = trigger_run(_config)
        if stats is None:
            return {"message": "A pipeline run is already in progress"}
        return {"message": "Pipeline run complete", "stats": stats}

    # Health check
    @app.get("/health", tags=["health"])
    def health():
        return {"status": "ok"}

    # Serve built React frontend as static files
    frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")

    return app


app = create_app()
