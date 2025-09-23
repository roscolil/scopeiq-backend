"""Health monitoring service"""

import os
import time
from typing import Dict, Any
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings


class HealthService:
    """Service for monitoring application health"""

    def __init__(self):
        self.start_time = time.time()

    def check_pinecone_health(self) -> bool:
        """Check if Pinecone is accessible"""
        try:
            pinecone_api_key = os.environ.get("PINECONE_API_KEY")
            pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME")

            if not pinecone_api_key or not pinecone_index_name:
                return False

            pc = Pinecone(api_key=pinecone_api_key)
            pc.Index(pinecone_index_name).describe_index_stats()
            return True
        except Exception:
            return False

    def check_openai_health(self) -> bool:
        """Check if OpenAI API key is configured"""
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            return False

        # Basic validation: OpenAI API keys start with 'sk-' and are typically 51+ characters
        return openai_api_key.startswith("sk-") and len(openai_api_key) >= 51

    def get_uptime(self) -> int:
        """Get server uptime in seconds"""
        return int(time.time() - self.start_time)

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        vector_store_up = self.check_pinecone_health()
        ai_models_up = self.check_openai_health()

        # Check if we're in development mode (no API keys set)
        is_development = not (
            os.environ.get("PINECONE_API_KEY") and os.environ.get("OPENAI_API_KEY")
        )

        if is_development:
            # In development mode, return healthy even without API keys
            status = "healthy"
        elif vector_store_up and ai_models_up:
            status = "healthy"
        elif vector_store_up or ai_models_up:
            status = "degraded"
        else:
            status = "unhealthy"

        return {
            "status": status,
            "version": "1.0.0",
            "services": {
                "vector_store": "up" if vector_store_up else "down",
                "ai_models": "up" if ai_models_up else "down",
            },
            "uptime_seconds": self.get_uptime(),
            "development_mode": is_development,
        }


# Global service instance
health_service = HealthService()
