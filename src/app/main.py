"""FastAPI application entry point"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.router import router
from src.core.config import settings


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""

    app = FastAPI(
        title="ScopeIQ AI Backend",
        description="AI-powered document processing and chat service for construction projects",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include consolidated v1 router
    app.include_router(router, prefix="/api/v1", tags=["api-v1"])

    return app


# Create app instance
app = create_app()
