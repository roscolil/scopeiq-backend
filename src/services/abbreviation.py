"""Abbreviation service for CRUD operations using DynamoDB"""

import uuid
from datetime import datetime
from typing import List, Optional
from pynamodb.exceptions import DoesNotExist, PutError, UpdateError, DeleteError

from src.models.abbreviation import (
    Abbreviation,
    AbbreviationCreate,
    AbbreviationUpdate,
)
from src.models.dynamodb_models import AbbreviationModel


class AbbreviationService:
    """Service class for abbreviation CRUD operations using DynamoDB"""

    def __init__(self):
        # DynamoDB-only backend
        pass

    async def create_abbreviation(
        self, abbreviation_data: AbbreviationCreate
    ) -> Abbreviation:
        """Create a new abbreviation"""
        abbreviation_id = str(uuid.uuid4())
        now = datetime.utcnow()

        try:
            db_item = AbbreviationModel(
                id=abbreviation_id,
                abbreviation=abbreviation_data.abbreviation,
                full_form=abbreviation_data.full_form,
                created_at=now,
                updated_at=now,
            )
            db_item.save()

            return Abbreviation(
                id=abbreviation_id,
                abbreviation=abbreviation_data.abbreviation,
                full_form=abbreviation_data.full_form,
                created_at=now,
                updated_at=now,
            )
        except PutError as e:
            raise Exception(f"Failed to create abbreviation: {str(e)}")

    async def get_abbreviation(self, abbreviation_id: str) -> Optional[Abbreviation]:
        """Get an abbreviation by ID"""
        try:
            db_item = AbbreviationModel.get(abbreviation_id)
            return Abbreviation(
                id=db_item.id,
                abbreviation=db_item.abbreviation,
                full_form=db_item.full_form,
                created_at=db_item.created_at,
                updated_at=db_item.updated_at,
            )
        except DoesNotExist:
            return None

    async def get_abbreviations(
        self, skip: int = 0, limit: int = 100
    ) -> List[Abbreviation]:
        """Get list of abbreviations with pagination"""
        try:
            abbreviations = []

            # Scan all items
            for item in AbbreviationModel.scan():
                abbreviations.append(
                    Abbreviation(
                        id=item.id,
                        abbreviation=item.abbreviation,
                        full_form=item.full_form,
                        created_at=item.created_at,
                        updated_at=item.updated_at,
                    )
                )

            # Apply pagination
            return abbreviations[skip : skip + limit]

        except Exception as e:
            raise Exception(f"Failed to get abbreviations: {str(e)}")

    async def update_abbreviation(
        self, abbreviation_id: str, update_data: AbbreviationUpdate
    ) -> Optional[Abbreviation]:
        """Update an abbreviation"""
        try:
            # Get existing item
            db_item = AbbreviationModel.get(abbreviation_id)

            # Update fields if provided
            update_dict = update_data.dict(exclude_unset=True)
            for field, value in update_dict.items():
                setattr(db_item, field, value)

            db_item.updated_at = datetime.utcnow()
            db_item.save()

            return Abbreviation(
                id=db_item.id,
                abbreviation=db_item.abbreviation,
                full_form=db_item.full_form,
                created_at=db_item.created_at,
                updated_at=db_item.updated_at,
            )

        except DoesNotExist:
            return None
        except UpdateError as e:
            raise Exception(f"Failed to update abbreviation: {str(e)}")

    async def delete_abbreviation(self, abbreviation_id: str) -> bool:
        """Delete an abbreviation"""
        try:
            db_item = AbbreviationModel.get(abbreviation_id)
            db_item.delete()
            return True
        except DoesNotExist:
            return False
        except DeleteError as e:
            raise Exception(f"Failed to delete abbreviation: {str(e)}")

    async def get_total_count(self) -> int:
        """Get total count of abbreviations"""
        try:
            count = 0

            # Count all items
            for _ in AbbreviationModel.scan():
                count += 1

            return count

        except Exception as e:
            raise Exception(f"Failed to get abbreviation count: {str(e)}")


# Global service instance
abbreviation_service = AbbreviationService()
