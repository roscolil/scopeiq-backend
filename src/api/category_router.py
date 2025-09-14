"""Category API endpoints"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from src.services.service_factory import get_category_service
from src.models.category import (
    CategoryCreate,
    CategoryUpdate,
    CategoryResponse,
    CategoryListResponse,
)


router = APIRouter(prefix="/categories", tags=["categories"])


@router.post("/", response_model=CategoryResponse, status_code=201)
async def create_category(category_data: CategoryCreate):
    """Create a new category"""
    try:
        service = get_category_service()
        category = await service.create_category(category_data)
        return CategoryResponse(success=True, data=category)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to create category: {str(e)}"
        )


@router.get("/{category_id}", response_model=CategoryResponse)
async def get_category(category_id: str):
    """Get a category by ID"""
    service = get_category_service()
    category = await service.get_category(category_id)
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")

    return CategoryResponse(success=True, data=category)


@router.get("/", response_model=CategoryListResponse)
async def list_categories(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(
        100, ge=1, le=1000, description="Maximum number of records to return"
    ),
    parent_id: Optional[str] = Query(
        None,
        description="Filter by parent category ID (empty string for root categories)",
    ),
):
    """List categories with parent filtering and pagination"""
    try:
        service = get_category_service()
        categories = await service.get_categories(
            skip=skip, limit=limit, parent_id=parent_id
        )
        total = await service.get_total_count(parent_id=parent_id)

        return CategoryListResponse(
            success=True, data=categories, total=total, skip=skip, limit=limit
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list categories: {str(e)}"
        )


@router.get("/{category_id}/children", response_model=CategoryListResponse)
async def get_child_categories(category_id: str):
    """Get all child categories of a parent category"""
    try:
        service = get_category_service()
        categories = await service.get_child_categories(category_id)
        return CategoryListResponse(
            success=True,
            data=categories,
            total=len(categories),
            skip=0,
            limit=len(categories),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get child categories: {str(e)}"
        )


@router.put("/{category_id}", response_model=CategoryResponse)
async def update_category(category_id: str, update_data: CategoryUpdate):
    """Update a category"""
    try:
        service = get_category_service()
        category = await service.update_category(category_id, update_data)
        if not category:
            raise HTTPException(status_code=404, detail="Category not found")

        return CategoryResponse(success=True, data=category)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to update category: {str(e)}"
        )


@router.delete("/{category_id}", status_code=204)
async def delete_category(category_id: str):
    """Delete a category"""
    try:
        service = get_category_service()
        deleted = await service.delete_category(category_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Category not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete category: {str(e)}"
        )
