"""Input validation utilities"""

from fastapi import HTTPException, UploadFile
from src.core.config import settings


def validate_file_upload(file: UploadFile) -> None:
    """Validate uploaded file"""

    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Note: File size validation should be done after reading the file
    # as we need the actual content size, not just the filename
