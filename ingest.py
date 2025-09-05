import os
import uuid
from typing import Dict, Any
from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings


def init_pinecone():
    # Check if required environment variables are set
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME")

    if not pinecone_api_key or not pinecone_index_name:
        print(
            "Warning: PINECONE_API_KEY and PINECONE_INDEX_NAME environment variables are not set."
        )
        print(
            "Please set these environment variables to use the retriever functionality."
        )
        return None

    pc = Pinecone(api_key=pinecone_api_key)
    pc_index = pc.Index(pinecone_index_name)
    return pc_index


def init_vector_store(pc_index):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = PineconeVectorStore(index=pc_index, embedding=embeddings)
    return vector_store


# In-memory storage for progress tracking
ingestion_progress: Dict[str, Dict[str, Any]] = {}

pc_index = init_pinecone()
vector_store = init_vector_store(pc_index)


def parse_and_chunk_pdf(file_path, task_id=None):
    if task_id:
        ingestion_progress[task_id]["status"] = "parsing"
        ingestion_progress[task_id]["progress"] = 10

    loader = PyPDFLoader(file_path)
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
        if task_id:
            # Update progress based on pages loaded
            ingestion_progress[task_id]["progress"] = min(30, 10 + (len(pages) * 2))

    if task_id:
        ingestion_progress[task_id]["status"] = "chunking"
        ingestion_progress[task_id]["progress"] = 40

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )
    all_splits = text_splitter.split_documents(pages)

    if task_id:
        ingestion_progress[task_id]["progress"] = 60
        ingestion_progress[task_id]["chunks_created"] = len(all_splits)

    return all_splits


def add_to_vector_store(all_splits, task_id=None):
    if task_id:
        ingestion_progress[task_id]["status"] = "vectorizing"
        ingestion_progress[task_id]["progress"] = 70

    vector_store.add_documents(all_splits)

    if task_id:
        ingestion_progress[task_id]["status"] = "completed"
        ingestion_progress[task_id]["progress"] = 100
        ingestion_progress[task_id]["chunks_added"] = len(all_splits)


async def start_ingestion(file_path: str) -> str:
    """Start ingestion process and return task ID for progress tracking"""
    task_id = str(uuid.uuid4())

    # Initialize progress tracking
    ingestion_progress[task_id] = {
        "status": "starting",
        "progress": 0,
        "file_path": file_path,
        "chunks_created": 0,
        "chunks_added": 0,
        "error": None,
    }

    try:
        # Run ingestion process
        all_splits = await parse_and_chunk_pdf(file_path, task_id)
        await add_to_vector_store(all_splits, task_id)
    except Exception as e:
        ingestion_progress[task_id]["status"] = "error"
        ingestion_progress[task_id]["error"] = str(e)

    return task_id


def get_ingestion_progress(task_id: str) -> Dict[str, Any]:
    """Get progress information for a specific ingestion task"""
    return ingestion_progress.get(task_id, {"error": "Task not found"})
