"""Abbreviation-related models"""

from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class AbbreviationBase(BaseModel):
    """Base abbreviation model"""

    abbreviation: str
    full_form: str


class AbbreviationCreate(AbbreviationBase):
    """Model for creating abbreviations"""

    pass


class AbbreviationUpdate(BaseModel):
    """Model for updating abbreviations"""

    abbreviation: Optional[str] = None
    full_form: Optional[str] = None


class Abbreviation(AbbreviationBase):
    """Complete abbreviation model with metadata"""

    id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class AbbreviationResponse(BaseModel):
    """Response model for abbreviation operations"""

    success: bool
    data: Abbreviation


class AbbreviationListResponse(BaseModel):
    """Response model for listing abbreviations"""

    success: bool
    data: list[Abbreviation]
    total: int
    skip: int
    limit: int
