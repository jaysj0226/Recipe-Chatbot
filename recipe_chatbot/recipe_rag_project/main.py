# -*- coding: utf-8 -*-
"""Main FastAPI application factory and router composition."""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from config.settings import STATIC_DIR

from routes.health import router as health_router
from routes.ask import router as ask_router
from routes.session import router as session_router
from routes.debug import router as test_router
from routes.root import router as root_router


# FastAPI App
app = FastAPI(
    title="Recipe RAG System",
    description="Modular RAG system with Router → Rewrite → Retrieve → Generate pipeline",
    version="2.0.0",
)

# Routers
app.include_router(health_router)
app.include_router(ask_router)
app.include_router(session_router)
app.include_router(test_router)
app.include_router(root_router)

# Static files (if exists)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# Main Entry Point
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )

# For Gunicorn:
# gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
