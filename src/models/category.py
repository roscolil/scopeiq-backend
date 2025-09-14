"""Category-related models"""

from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class CategoryBase(BaseModel):
    """Base category model"""

    name: str
    description: Optional[str] = None
    parent_id: Optional[str] = None


class CategoryCreate(CategoryBase):
    """Model for creating categories"""

    pass


class CategoryUpdate(BaseModel):
    """Model for updating categories"""

    name: Optional[str] = None
    description: Optional[str] = None
    parent_id: Optional[str] = None


class Category(CategoryBase):
    """Complete category model with metadata"""

    id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class CategoryResponse(BaseModel):
    """Response model for category operations"""

    success: bool
    data: Category


class CategoryListResponse(BaseModel):
    """Response model for listing categories"""

    success: bool
    data: list[Category]
    total: int
    skip: int
    limit: int
