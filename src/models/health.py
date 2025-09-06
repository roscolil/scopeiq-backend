"""Health check models"""

from pydantic import BaseModel
from typing import Literal


class ServicesStatus(BaseModel):
    vector_store: Literal["up", "down"]
    ai_models: Literal["up", "down"]


class HealthData(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str = "1.0.0"
    services: ServicesStatus
    uptime_seconds: int


class HealthResponse(BaseModel):
    success: bool
    data: HealthData
