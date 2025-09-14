"""API v1 router"""

import uuid
import asyncio
from fastapi import APIRouter, BackgroundTasks, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from src.services.health import health_service
from src.services.document import document_processing_service
from src.services.chat import chat_service
from src.models.health import HealthResponse
from src.models.document import (
    DocumentUploadResponse,
    DocumentUploadData,
    DocumentProgressResponse,
    ProcessingResults,
)
from src.models.chat import ChatRequest, ChatResponse
from src.utils.validators import validate_file_upload
from src.api.abbreviation_router import router as abbreviation_router
from src.api.category_router import router as category_router

router = APIRouter()

# Include abbreviation and category routers
router.include_router(abbreviation_router)
router.include_router(category_router)


async def process_document_background(
    file_content: bytes,
    document_id: str,
    project_id: str,
    company_id: str,
    document_name: str = None,
):
    """Wrapper function to run async process_document in background task"""
    await document_processing_service.process_document(
        file_content, document_id, project_id, company_id, document_name
    )


# Health Check Endpoint
@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Monitor backend health and service status"""
    health_data = health_service.get_health_status()

    # Set appropriate HTTP status code
    if health_data["status"] == "healthy":
        status_code = 200
    elif health_data["status"] == "degraded":
        status_code = 200  # Still operational
    else:
        status_code = 503  # Service unavailable

    return JSONResponse(
        status_code=status_code, content={"success": True, "data": health_data}
    )


# Document Upload Endpoint
@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    project_id: str = Form(...),
    company_id: str = Form(...),
    document_name: str = Form(None),
):
    """Upload PDF, DOC/DOCX, and TXT documents for AI processing and indexing"""

    # Validate file upload
    validate_file_upload(file)

    # Read file content
    file_content = await file.read()

    # Generate document ID
    document_id = f"doc_{uuid.uuid4().hex[:12]}"

    # Get original filename
    original_filename = document_name or file.filename

    # Sanitize filename for storage
    sanitized_filename = document_processing_service.sanitize_filename(
        original_filename
    )

    # Start background processing
    background_tasks.add_task(
        process_document_background,
        file_content,
        document_id,
        project_id,
        company_id,
        original_filename,
    )

    # Generate S3 key using sanitized filename
    s3_service = document_processing_service.s3_service
    s3_key = s3_service.generate_s3_key(company_id, project_id, sanitized_filename)
    s3_url = f"https://{s3_service.bucket_name}.s3.{s3_service.region_name}.amazonaws.com/{s3_key}"

    return DocumentUploadResponse(
        success=True,
        data=DocumentUploadData(
            document_id=document_id,
            processing_status="processing",
            original_filename=original_filename,
            sanitized_filename=sanitized_filename,
            s3_key=s3_key,
            s3_url=s3_url,
            message="Document uploaded and processing started",
        ),
    )


# Document Progress Endpoint
@router.get(
    "/documents/{document_id}/progress", response_model=DocumentProgressResponse
)
async def get_document_progress(document_id: str):
    """Check processing status and progress of uploaded documents"""

    progress_data = document_processing_service.get_document_progress(document_id)

    if "error" in progress_data:
        if progress_data["error"] == "Document not found":
            raise HTTPException(status_code=404, detail="Document not found")
        else:
            raise HTTPException(status_code=500, detail=progress_data["error"])

    # Add document_id to the progress data and handle processing_results
    progress_data["document_id"] = document_id

    # Convert processing_results to ProcessingResults model if it exists and has data
    if progress_data.get("processing_results") and progress_data["processing_results"]:
        progress_data["processing_results"] = ProcessingResults(
            **progress_data["processing_results"]
        )
    else:
        progress_data["processing_results"] = None

    return DocumentProgressResponse(success=True, data=progress_data)


# Chat Conversation Endpoint
@router.post("/chat/conversation", response_model=ChatResponse)
async def chat_conversation(request: ChatRequest):
    """Handle AI conversations with document context and search capabilities"""

    try:
        # Process the chat query
        chat_result = await chat_service.process_chat_query(
            query=request.query,
            project_id=request.project_id,
            document_id=request.document_id,
            conversation_history=request.conversation_history,
            context_type=request.context_type,
            include_search_results=request.include_search_results,
        )

        return ChatResponse(success=True, data=chat_result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")
