import os
import re
import asyncio
from pathlib import Path
from functools import partial
from typing import Dict, Any, List, Tuple
from PIL import Image
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pdf2image import convert_from_path
import pymupdf
import pymupdf4llm
from src.services.progress import InMemoryProgressTracker
from src.services.s3 import S3Service
from src.services.vision_pipeline import vision_pipeline_service
from src.services.page_classifier import page_classifier
from src.core.config import settings


def clamp_linebreaks(text: str, max_consecutive: int = 3) -> str:
    pattern = r"\n{" + str(max_consecutive + 1) + r",}"
    replacement = "\n" * max_consecutive
    return re.sub(pattern, replacement, text)


def remove_contiguous_spaces(text: str) -> str:
    """Replace runs of spaces with a single space."""
    return re.sub(r" {2,}", " ", text)


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

    def _detect_file_type(self, filename: str) -> str:
        """Detect file type based on filename extension"""
        if not filename:
            return "pdf"  # Default to PDF if no filename

        filename_lower = filename.lower()
        if filename_lower.endswith((".doc", ".docx")):
            return "docx"
        elif filename_lower.endswith(".pdf"):
            return "pdf"
        elif filename_lower.endswith(".txt"):
            return "txt"
        else:
            return "pdf"  # Default to PDF for unknown types

    async def _load_docx_document(self, temp_file_path: str):
        """Load DOC/DOCX document using UnstructuredWordDocumentLoader"""
        loader = UnstructuredWordDocumentLoader(
            temp_file_path,
            include_page_breaks=True,
        )

        file_name = Path(temp_file_path).name
        docx_pages = []
        async for page in loader.alazy_load():
            page.metadata["source"] = file_name
            page.metadata["file_type"] = "docx"
            docx_pages.append(page)

        return docx_pages

    async def _load_pdf_document(
        self,
        temp_file_path: str,
        document_id: str,
        project_id: str,
        company_id: str,
    ) -> Tuple[List[Document], List[Document]]:
        """
        Load PDF document and classify pages into text and drawing pages.

        Returns:
            Tuple of (text_pages, drawing_pages)
        """
        file_name = Path(temp_file_path).name
        text_pages_list: List[Document] = []
        drawing_pages_list: List[Document] = []

        # Convert PDF pages to images for classification and vision processing
        loop = asyncio.get_event_loop()
        page_images = await loop.run_in_executor(
            None, partial(convert_from_path, temp_file_path, dpi=200)
        )

        with pymupdf.open(temp_file_path) as source_doc:
            total_pages = min(source_doc.page_count, len(page_images))

            for idx in range(total_pages):
                single_page_doc = pymupdf.open()
                single_page_doc.insert_pdf(source_doc, from_page=idx, to_page=idx)
                page_image = page_images[idx]

                try:
                    # Classify page type
                    page_type = page_classifier.classify_page(page_image)

                    # Get page number (0-indexed from PDF)
                    page_number = idx + 1

                    # Base metadata for the page
                    base_metadata = {
                        "source": file_name,
                        "file_type": "pdf",
                        "document_id": document_id,
                        "project_id": project_id,
                        "company_id": company_id,
                        "page_number": page_number,
                    }

                    print(f"Processing Page {page_number}: {page_type}")

                    if page_type == "drawing":
                        # Process through vision pipeline
                        processed_drawing_doc = (
                            await vision_pipeline_service.process_drawing_page(
                                page_image=page_image,
                                page_number=page_number,
                                document_id=document_id,
                                metadata=base_metadata,
                            )
                        )
                        drawing_pages_list.append(processed_drawing_doc)
                    else:
                        text_page_doc = Document(
                            # page_content=clamp_linebreaks(
                            #     pymupdf4llm.to_markdown(single_page_doc)
                            # ),
                            # TODO: Fix this to something more useful (pure text extraction)
                            page_content=clamp_linebreaks(
                                remove_contiguous_spaces(
                                    single_page_doc[0].get_text(sort=True)
                                )
                            ),
                            metadata={**base_metadata, "page_type": "text"},
                        )
                        text_pages_list.append(text_page_doc)
                finally:
                    single_page_doc.close()

        return text_pages_list, drawing_pages_list

    async def _load_txt_document(self, temp_file_path: str):
        """Load TXT document using TextLoader"""
        loader = TextLoader(temp_file_path)
        file_name = Path(temp_file_path).name
        pages = []
        async for page in loader.alazy_load():
            page.metadata["source"] = file_name
            page.metadata["file_type"] = "txt"
            pages.append(page)
        return pages

    async def process_document(
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

            # Detect file type and set appropriate extension
            file_type = self._detect_file_type(document_name)
            if file_type == "docx":
                default_extension = "docx"
            elif file_type == "txt":
                default_extension = "txt"
            else:
                default_extension = "pdf"
            filename = document_name or f"document_{document_id}.{default_extension}"
            sanitized_filename = self.sanitize_filename(filename)
            s3_key = self.s3_service.generate_s3_key(
                company_id, project_id, sanitized_filename
            )
            s3_url = self.s3_service.upload_file(file_content, s3_key)

            # Stage 2: Extract text from document
            if file_type == "docx":
                extraction_stage = "Extracting text from DOC/DOCX"
            elif file_type == "txt":
                extraction_stage = "Extracting text from TXT"
            else:
                extraction_stage = "Extracting text from PDF"
            self._update_progress(document_id, "processing", 20, extraction_stage)

            # Save file temporarily for processing
            temp_file_path = f"/tmp/{document_id}.{default_extension}"
            with open(temp_file_path, "wb") as f:
                f.write(file_content)

            # Load document based on file type
            if file_type == "docx":
                pages = await self._load_docx_document(temp_file_path)
                drawing_pages = []
            elif file_type == "txt":
                pages = await self._load_txt_document(temp_file_path)
                drawing_pages = []
            else:
                # PDF: separate text and drawing pages
                pages, drawing_pages = await self._load_pdf_document(
                    temp_file_path, document_id, project_id, company_id
                )

            # Stage 3: Process drawing pages through vision pipeline
            if drawing_pages:
                self._update_progress(
                    document_id,
                    "processing",
                    35,
                    f"Processing {len(drawing_pages)} drawing pages through vision pipeline",
                )
                # Drawing pages are already processed by vision pipeline

            # Stage 4: Chunk text pages
            self._update_progress(document_id, "processing", 40, "Chunking text pages")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200,
                add_start_index=True,
            )

            if default_extension in ["docx", "txt"]:
                # chunk docx and txt files as pdf is already paged
                pages = text_splitter.split_documents(pages)

            # Combine all pages (text and drawing)
            all_pages = pages + drawing_pages

            # Stage 5: Generate embeddings and store in vector store
            self._update_progress(
                document_id, "processing", 60, "Generating embeddings"
            )

            if self.vector_store:
                # Add metadata to all chunks/pages
                for i, chunk in enumerate(all_pages):
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

                self.vector_store.add_documents(all_pages, namespace=project_id)

            # Stage 6: Complete
            processing_results = {
                "chunks_created": len(pages),
                "embeddings_generated": len(pages),
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
