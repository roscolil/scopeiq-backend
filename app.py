from dotenv import load_dotenv

load_dotenv()

import asyncio
from fastapi import FastAPI, BackgroundTasks
from graph import graph
from ingest import start_ingestion, get_ingestion_progress
from pydantic import BaseModel

app = FastAPI()


class ChatInput(BaseModel):
    messages: list[str]
    thread_id: str


class IngestInput(BaseModel):
    file_path: str


class IngestResponse(BaseModel):
    task_id: str
    message: str


class ProgressResponse(BaseModel):
    task_id: str
    status: str
    progress: int
    file_path: str
    chunks_created: int
    chunks_added: int
    error: str = None


@app.post("/chat")
async def chat(input: ChatInput):
    config = {"configurable": {"thread_id": input.thread_id}}
    response = await graph.ainvoke({"messages": input.messages}, config=config)
    return response["messages"][-1].content


async def run_ingestion(file_path: str):
    """Background task to run ingestion"""
    await start_ingestion(file_path)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(input: IngestInput, background_tasks: BackgroundTasks):
    """Start ingestion process and return task ID for progress tracking"""
    task_id = await start_ingestion(input.file_path)
    return IngestResponse(
        task_id=task_id, message="Ingestion started. Use the task_id to check progress."
    )


@app.get("/ingest/progress/{task_id}", response_model=ProgressResponse)
async def get_progress(task_id: str):
    """Get progress information for an ingestion task"""
    progress_data = get_ingestion_progress(task_id)

    if "error" in progress_data and progress_data["error"] == "Task not found":
        return ProgressResponse(
            task_id=task_id,
            status="not_found",
            progress=0,
            file_path="",
            chunks_created=0,
            chunks_added=0,
            error="Task not found",
        )

    return ProgressResponse(
        task_id=task_id,
        status=progress_data["status"],
        progress=progress_data["progress"],
        file_path=progress_data["file_path"],
        chunks_created=progress_data["chunks_created"],
        chunks_added=progress_data["chunks_added"],
        error=progress_data.get("error"),
    )


# Health check
@app.get("/api/v1/health")
async def health_check():
    return {
        "success": True,
        "data": {
            "status": "healthy",
            "version": "1.0.0",
            "services": {"database": "up", "vector_store": "up", "ai_models": "up"},
            "uptime_seconds": 3600,
        },
    }
