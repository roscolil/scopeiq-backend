"""Abbreviation API endpoints"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from src.services.service_factory import get_abbreviation_service
from src.models.abbreviation import (
    AbbreviationCreate,
    AbbreviationUpdate,
    AbbreviationResponse,
    AbbreviationListResponse,
)


router = APIRouter(prefix="/abbreviations", tags=["abbreviations"])


@router.post("/", response_model=AbbreviationResponse, status_code=201)
async def create_abbreviation(abbreviation_data: AbbreviationCreate):
    """Create a new abbreviation"""
    try:
        service = get_abbreviation_service()
        abbreviation = await service.create_abbreviation(abbreviation_data)
        return AbbreviationResponse(success=True, data=abbreviation)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to create abbreviation: {str(e)}"
        )


@router.get("/{abbreviation_id}", response_model=AbbreviationResponse)
async def get_abbreviation(abbreviation_id: str):
    """Get an abbreviation by ID"""
    service = get_abbreviation_service()
    abbreviation = await service.get_abbreviation(abbreviation_id)
    if not abbreviation:
        raise HTTPException(status_code=404, detail="Abbreviation not found")

    return AbbreviationResponse(success=True, data=abbreviation)


@router.get("/", response_model=AbbreviationListResponse)
async def list_abbreviations(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(
        100, ge=1, le=1000, description="Maximum number of records to return"
    ),
):
    """List abbreviations with pagination"""
    try:
        service = get_abbreviation_service()
        abbreviations = await service.get_abbreviations(skip=skip, limit=limit)
        total = await service.get_total_count()

        return AbbreviationListResponse(
            success=True, data=abbreviations, total=total, skip=skip, limit=limit
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list abbreviations: {str(e)}"
        )


@router.put("/{abbreviation_id}", response_model=AbbreviationResponse)
async def update_abbreviation(abbreviation_id: str, update_data: AbbreviationUpdate):
    """Update an abbreviation"""
    try:
        service = get_abbreviation_service()
        abbreviation = await service.update_abbreviation(abbreviation_id, update_data)
        if not abbreviation:
            raise HTTPException(status_code=404, detail="Abbreviation not found")

        return AbbreviationResponse(success=True, data=abbreviation)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to update abbreviation: {str(e)}"
        )


@router.delete("/{abbreviation_id}", status_code=204)
async def delete_abbreviation(abbreviation_id: str):
    """Delete an abbreviation"""
    try:
        service = get_abbreviation_service()
        deleted = await service.delete_abbreviation(abbreviation_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Abbreviation not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete abbreviation: {str(e)}"
        )
