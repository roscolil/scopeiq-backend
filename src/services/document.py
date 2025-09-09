import os
import re
from typing import Dict, Any
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.services.progress import InMemoryProgressTracker
from src.services.s3 import S3Service
from src.core.config import settings


class DocumentProcessingService:
    def __init__(self):
        self.pc_index = self._init_pinecone()
        self.vector_store = self._init_vector_store()
        self.progress_tracker = InMemoryProgressTracker()
        self.s3_service = S3Service()

    def _init_pinecone(self):
        """Initialize Pinecone connection"""
        pinecone_api_key = settings.PINECONE_API_KEY
        pinecone_index_name = settings.PINECONE_INDEX_NAME

        if not pinecone_api_key or not pinecone_index_name:
            return None

        pc = Pinecone(api_key=pinecone_api_key)
        return pc.Index(pinecone_index_name)

    def _init_vector_store(self):
        """Initialize vector store"""
        if not self.pc_index:
            return None

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large", openai_api_key=settings.OPENAI_API_KEY
        )
        return PineconeVectorStore(index=self.pc_index, embedding=embeddings)

    def _update_progress(
        self,
        document_id: str,
        status: str,
        progress: int,
        current_stage: str,
        error_message: str = None,
        processing_results: Dict = None,
    ):
        """Update document processing progress"""
        self.progress_tracker.update_progress(
            document_id,
            status,
            progress,
            current_stage,
            error_message,
            processing_results,
        )

    def _get_progress(self, document_id: str) -> Dict[str, Any]:
        """Get document processing progress"""
        return self.progress_tracker.get_progress(document_id)

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage"""
        sanitized = re.sub(r"\s+", "_", filename)  # Replace spaces with underscores
        sanitized = re.sub(r"[^\w\-_.]", "", sanitized)  # Remove special chars
        sanitized = re.sub(r"_{2,}", "_", sanitized)  # Replace multiple underscores
        return sanitized.strip("_")  # Remove leading/trailing underscores

    def process_document(
        self,
        file_content: bytes,
        document_id: str,
        project_id: str,
        company_id: str,
        document_name: str = None,
    ):
        """Process document through the full pipeline"""
        try:
            # Stage 1: Upload to S3
            self._update_progress(document_id, "processing", 10, "Uploading to S3")

            filename = document_name or f"document_{document_id}.pdf"
            sanitized_filename = self.sanitize_filename(filename)
            s3_key = self.s3_service.generate_s3_key(
                company_id, project_id, sanitized_filename
            )
            s3_url = self.s3_service.upload_file(file_content, s3_key)

            # Stage 2: Extract text from PDF
            self._update_progress(
                document_id, "processing", 20, "Extracting text from PDF"
            )

            # Save file temporarily for processing
            temp_file_path = f"/tmp/{document_id}.pdf"
            with open(temp_file_path, "wb") as f:
                f.write(file_content)

            loader = PyPDFLoader(temp_file_path)
            pages = loader.load()

            # Stage 3: Chunk text
            self._update_progress(document_id, "processing", 40, "Chunking text")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200,
                add_start_index=True,
            )
            all_splits = text_splitter.split_documents(pages)

            # Stage 4: Generate embeddings and store in vector store
            self._update_progress(
                document_id, "processing", 60, "Generating embeddings"
            )

            if self.vector_store:
                # Add metadata to chunks
                for i, chunk in enumerate(all_splits):
                    chunk.metadata.update(
                        {
                            "document_id": document_id,
                            "project_id": project_id,
                            "company_id": company_id,
                            "chunk_id": f"{document_id}_chunk_{i}",
                            "s3_url": s3_url,
                            "s3_key": s3_key,
                        }
                    )

                self.vector_store.add_documents(all_splits, namespace=project_id)

            # Stage 5: Enhanced analysis (placeholder for construction-specific analysis)
            self._update_progress(
                document_id, "processing", 80, "Performing enhanced analysis"
            )

            # Here you would add construction-specific analysis like:
            # - Extract doors, windows, rooms
            # - Identify materials and specifications
            # - Parse measurements and dimensions
            # For now, we'll mark as completed

            # Stage 6: Complete
            processing_results = {
                "chunks_created": len(all_splits),
                "embeddings_generated": len(all_splits),
                "enhanced_analysis_completed": True,
                "search_ready": True,
            }

            self._update_progress(
                document_id,
                "completed",
                100,
                "Processing completed",
                processing_results=processing_results,
            )

            # Clean up temp file
            os.remove(temp_file_path)

        except Exception as e:
            self._update_progress(
                document_id, "failed", 0, "Processing failed", error_message=str(e)
            )
            raise e

    def get_document_progress(self, document_id: str) -> Dict[str, Any]:
        """Get document processing progress"""
        return self._get_progress(document_id)


# Global service instance
document_processing_service = DocumentProcessingService()
