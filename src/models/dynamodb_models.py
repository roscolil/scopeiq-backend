"""DynamoDB models using PynamoDB ORM"""

from datetime import datetime
from typing import Optional
from pynamodb.models import Model
from pynamodb.attributes import UnicodeAttribute, UTCDateTimeAttribute
from pynamodb.indexes import GlobalSecondaryIndex, AllProjection
from src.core.config import settings


class AbbreviationIndex(GlobalSecondaryIndex):
    """Global Secondary Index for searching abbreviations by abbreviation text"""

    class Meta:
        index_name = "abbreviation-index"
        read_capacity_units = 5
        write_capacity_units = 5
        projection = AllProjection()

    abbreviation = UnicodeAttribute(hash_key=True)


class ParentIdIndex(GlobalSecondaryIndex):
    """Global Secondary Index for searching categories by parent_id"""

    class Meta:
        index_name = "parent-id-index"
        read_capacity_units = 5
        write_capacity_units = 5
        projection = AllProjection()

    parent_id = UnicodeAttribute(hash_key=True)


class AbbreviationModel(Model):
    """DynamoDB model for abbreviations using PynamoDB ORM"""

    class Meta:
        table_name = "abbreviations"
        region = settings.AWS_REGION  # Use AWS_REGION from config.py
        billing_mode = "PAY_PER_REQUEST"  # On-demand billing

    # Primary key
    id = UnicodeAttribute(hash_key=True)

    # Attributes
    abbreviation = UnicodeAttribute()
    full_form = UnicodeAttribute()
    created_at = UTCDateTimeAttribute(default=datetime.utcnow)
    updated_at = UTCDateTimeAttribute(default=datetime.utcnow)

    # Global Secondary Index
    abbreviation_index = AbbreviationIndex()


class CategoryModel(Model):
    """DynamoDB model for categories using PynamoDB ORM"""

    class Meta:
        table_name = "categories"
        region = settings.AWS_REGION  # Use AWS_REGION from config.py
        billing_mode = "PAY_PER_REQUEST"  # On-demand billing

    # Primary key
    id = UnicodeAttribute(hash_key=True)

    # Attributes
    name = UnicodeAttribute()
    description = UnicodeAttribute(null=True)
    parent_id = UnicodeAttribute(null=True)
    created_at = UTCDateTimeAttribute(default=datetime.utcnow)
    updated_at = UTCDateTimeAttribute(default=datetime.utcnow)

    # Global Secondary Index
    parent_id_index = ParentIdIndex()
