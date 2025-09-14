"""Category service for CRUD operations using DynamoDB"""

import uuid
from datetime import datetime
from typing import List, Optional
from pynamodb.exceptions import DoesNotExist, PutError, UpdateError, DeleteError

from src.models.category import (
    Category,
    CategoryCreate,
    CategoryUpdate,
)
from src.models.dynamodb_models import CategoryModel


class CategoryService:
    """Service class for category CRUD operations using DynamoDB"""

    def __init__(self):
        # DynamoDB-only backend
        pass

    async def create_category(self, category_data: CategoryCreate) -> Category:
        """Create a new category"""
        category_id = str(uuid.uuid4())
        now = datetime.utcnow()

        try:
            db_item = CategoryModel(
                id=category_id,
                name=category_data.name,
                description=category_data.description,
                parent_id=category_data.parent_id,
                created_at=now,
                updated_at=now,
            )
            db_item.save()

            return Category(
                id=category_id,
                name=category_data.name,
                description=category_data.description,
                parent_id=category_data.parent_id,
                created_at=now,
                updated_at=now,
            )
        except PutError as e:
            raise Exception(f"Failed to create category: {str(e)}")

    async def get_category(self, category_id: str) -> Optional[Category]:
        """Get a category by ID"""
        try:
            db_item = CategoryModel.get(category_id)
            return Category(
                id=db_item.id,
                name=db_item.name,
                description=db_item.description,
                parent_id=db_item.parent_id,
                created_at=db_item.created_at,
                updated_at=db_item.updated_at,
            )
        except DoesNotExist:
            return None

    async def get_categories(
        self,
        skip: int = 0,
        limit: int = 100,
        parent_id: Optional[str] = None,
    ) -> List[Category]:
        """Get list of categories with parent filtering and pagination"""
        try:
            categories = []

            if parent_id is not None:
                # Use GSI for parent filtering
                if parent_id == "":
                    # Get root categories (no parent)
                    for item in CategoryModel.scan():
                        if item.parent_id is None:
                            categories.append(
                                Category(
                                    id=item.id,
                                    name=item.name,
                                    description=item.description,
                                    parent_id=item.parent_id,
                                    created_at=item.created_at,
                                    updated_at=item.updated_at,
                                )
                            )
                else:
                    # Get categories with specific parent
                    for item in CategoryModel.parent_id_index.query(parent_id):
                        categories.append(
                            Category(
                                id=item.id,
                                name=item.name,
                                description=item.description,
                                parent_id=item.parent_id,
                                created_at=item.created_at,
                                updated_at=item.updated_at,
                            )
                        )
            else:
                # Scan all items if no parent filter
                for item in CategoryModel.scan():
                    categories.append(
                        Category(
                            id=item.id,
                            name=item.name,
                            description=item.description,
                            parent_id=item.parent_id,
                            created_at=item.created_at,
                            updated_at=item.updated_at,
                        )
                    )

            # Apply pagination
            return categories[skip : skip + limit]

        except Exception as e:
            raise Exception(f"Failed to get categories: {str(e)}")

    async def update_category(
        self, category_id: str, update_data: CategoryUpdate
    ) -> Optional[Category]:
        """Update a category"""
        try:
            # Get existing item
            db_item = CategoryModel.get(category_id)

            # Update fields if provided
            update_dict = update_data.dict(exclude_unset=True)
            for field, value in update_dict.items():
                setattr(db_item, field, value)

            db_item.updated_at = datetime.utcnow()
            db_item.save()

            return Category(
                id=db_item.id,
                name=db_item.name,
                description=db_item.description,
                parent_id=db_item.parent_id,
                created_at=db_item.created_at,
                updated_at=db_item.updated_at,
            )

        except DoesNotExist:
            return None
        except UpdateError as e:
            raise Exception(f"Failed to update category: {str(e)}")

    async def delete_category(self, category_id: str) -> bool:
        """Delete a category"""
        try:
            db_item = CategoryModel.get(category_id)
            db_item.delete()
            return True
        except DoesNotExist:
            return False
        except DeleteError as e:
            raise Exception(f"Failed to delete category: {str(e)}")

    async def get_total_count(self, parent_id: Optional[str] = None) -> int:
        """Get total count of categories"""
        try:
            categories = []

            if parent_id is not None:
                if parent_id == "":
                    for item in CategoryModel.scan():
                        if item.parent_id is None:
                            categories.append(
                                Category(
                                    id=item.id,
                                    name=item.name,
                                    description=item.description,
                                    parent_id=item.parent_id,
                                    created_at=item.created_at,
                                    updated_at=item.updated_at,
                                )
                            )
                else:
                    for item in CategoryModel.parent_id_index.query(parent_id):
                        categories.append(
                            Category(
                                id=item.id,
                                name=item.name,
                                description=item.description,
                                parent_id=item.parent_id,
                                created_at=item.created_at,
                                updated_at=item.updated_at,
                            )
                        )
            else:
                for item in CategoryModel.scan():
                    categories.append(
                        Category(
                            id=item.id,
                            name=item.name,
                            description=item.description,
                            parent_id=item.parent_id,
                            created_at=item.created_at,
                            updated_at=item.updated_at,
                        )
                    )

            return len(categories)

        except Exception as e:
            raise Exception(f"Failed to get category count: {str(e)}")

    async def get_child_categories(self, parent_id: str) -> List[Category]:
        """Get all child categories of a parent category"""
        try:
            categories = []
            for item in CategoryModel.parent_id_index.query(parent_id):
                categories.append(
                    Category(
                        id=item.id,
                        name=item.name,
                        description=item.description,
                        parent_id=item.parent_id,
                        created_at=item.created_at,
                        updated_at=item.updated_at,
                    )
                )
            return categories
        except Exception as e:
            raise Exception(f"Failed to get child categories: {str(e)}")


# Global service instance
category_service = CategoryService()
