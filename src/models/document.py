"""Document-related models"""

from pydantic import BaseModel
from typing import Dict, Any, Optional, Literal


class ProcessingResults(BaseModel):
    chunks_created: Optional[int] = None
    embeddings_generated: Optional[int] = None
    enhanced_analysis_completed: Optional[bool] = None
    search_ready: Optional[bool] = None


class DocumentProgressData(BaseModel):
    document_id: str
    status: Literal["processing", "completed", "failed"]
    progress_percentage: int
    current_stage: str
    error_message: Optional[str] = None
    processing_results: Optional[ProcessingResults] = None


class DocumentUploadResponse(BaseModel):
    success: bool
    data: Dict[str, Any]


class DocumentProgressResponse(BaseModel):
    success: bool
    data: DocumentProgressData
