"""DynamoDB configuration and initialization"""

import os
from typing import Optional
from src.models.dynamodb_models import AbbreviationModel, CategoryModel
from src.core.config import settings


class DynamoDBConfig:
    """Configuration for DynamoDB connection"""

    def __init__(
        self,
        region_name: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        # Use settings from config.py as defaults, allow override via parameters
        self.region_name = region_name or settings.AWS_REGION
        self.endpoint_url = endpoint_url or settings.DYNAMODB_ENDPOINT_URL
        self.aws_access_key_id = aws_access_key_id or settings.AWS_ACCESS_KEY_ID
        self.aws_secret_access_key = (
            aws_secret_access_key or settings.AWS_SECRET_ACCESS_KEY
        )

        # Override environment variables if explicitly provided
        if self.aws_access_key_id:
            os.environ["AWS_ACCESS_KEY_ID"] = self.aws_access_key_id
        if self.aws_secret_access_key:
            os.environ["AWS_SECRET_ACCESS_KEY"] = self.aws_secret_access_key
        if self.region_name:
            os.environ["AWS_DEFAULT_REGION"] = self.region_name

    def initialize_models(self):
        """Initialize DynamoDB models with configuration"""
        # Clear any existing connections to force reinitialization
        AbbreviationModel._connection = None
        CategoryModel._connection = None

        # Update model Meta classes with configuration
        AbbreviationModel.Meta.region = self.region_name
        CategoryModel.Meta.region = self.region_name

        # Only set custom endpoint for local development
        if self.endpoint_url:
            # Set the host endpoint for local development
            AbbreviationModel.Meta.host = self.endpoint_url
            CategoryModel.Meta.host = self.endpoint_url

            # Set region to None for local development
            AbbreviationModel.Meta.region = None
            CategoryModel.Meta.region = None

            print(f"🔧 Using local DynamoDB endpoint: {self.endpoint_url}")
        else:
            print(f"☁️ Using AWS DynamoDB in region: {self.region_name}")

    async def create_tables(self):
        """Create DynamoDB tables if they don't exist"""
        try:
            # Initialize models first
            self.initialize_models()

            # Create tables
            if not AbbreviationModel.exists():
                AbbreviationModel.create_table(wait=True)
                print("✅ Abbreviations table created successfully")
            else:
                print("✅ Abbreviations table already exists")

            if not CategoryModel.exists():
                CategoryModel.create_table(wait=True)
                print("✅ Categories table created successfully")
            else:
                print("✅ Categories table already exists")

        except Exception as e:
            print(f"❌ Error creating tables: {str(e)}")
            raise

    async def delete_tables(self):
        """Delete DynamoDB tables (useful for testing)"""
        try:
            if AbbreviationModel.exists():
                AbbreviationModel.delete_table()
                print("✅ Abbreviations table deleted")

            if CategoryModel.exists():
                CategoryModel.delete_table()
                print("✅ Categories table deleted")

        except Exception as e:
            print(f"❌ Error deleting tables: {str(e)}")
            raise


# Global configuration instance
dynamodb_config = DynamoDBConfig()
