"""Chat-related models"""

from pydantic import BaseModel
from typing import List, Optional, Literal
from datetime import datetime


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime


class ChatRequest(BaseModel):
    query: str
    project_id: str
    document_id: Optional[str] = None
    conversation_history: List[ChatMessage] = []
    context_type: Literal["document", "project"] = "document"
    include_search_results: bool = True


class ChatMetadata(BaseModel):
    sources_used: List[str]


class ChatData(BaseModel):
    response: str
    metadata: ChatMetadata


class ChatResponse(BaseModel):
    success: bool
    data: ChatData
