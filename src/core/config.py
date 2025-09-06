"""Core configuration settings"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List
import os
from dotenv import load_dotenv

# Load .env file into os.environ for libraries that expect it
load_dotenv()


class Settings(BaseSettings):
    """Core application settings"""

    # Environment
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=True, env="DEBUG")

    # API Configuration
    API_V1_STR: str = Field(default="/api/v1", env="API_V1_STR")
    PROJECT_NAME: str = Field(default="ScopeIQ AI Backend", env="PROJECT_NAME")

    # CORS
    ALLOWED_ORIGINS: List[str] = Field(default=["*"], env="ALLOWED_ORIGINS")

    # AWS Configuration
    AWS_ACCESS_KEY_ID: str = Field(default="", env="AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: str = Field(default="", env="AWS_SECRET_ACCESS_KEY")
    AWS_REGION: str = Field(default="us-east-1", env="AWS_REGION")
    S3_BUCKET_NAME: str = Field(default="", env="S3_BUCKET_NAME")

    # Pinecone Configuration
    PINECONE_API_KEY: str = Field(default="", env="PINECONE_API_KEY")
    PINECONE_INDEX_NAME: str = Field(default="", env="PINECONE_INDEX_NAME")

    # OpenAI Configuration
    OPENAI_API_KEY: str = Field(default="", env="OPENAI_API_KEY")

    # langsmith
    LANGSMITH_API_KEY: str = Field(default="", env="LANGSMITH_API_KEY")
    LANGCHAIN_PROJECT: str = Field(default="", env="LANGCHAIN_PROJECT")
    LANGSMITH_TRACING: str = Field(default="", env="LANGSMITH_TRACING")

    # File Upload Configuration
    MAX_FILE_SIZE: int = Field(default=10 * 1024 * 1024, env="MAX_FILE_SIZE")  # 10MB
    ALLOWED_FILE_TYPES: List[str] = Field(default=[".pdf"], env="ALLOWED_FILE_TYPES")

    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "extra": "ignore",  # Ignore extra environment variables
    }


settings = Settings()


def sync_settings_to_env():
    """Sync Pydantic settings back to os.environ for libraries that expect them"""
    # This ensures that libraries expecting environment variables can find them
    env_vars = {
        "OPENAI_API_KEY": settings.OPENAI_API_KEY,
        "PINECONE_API_KEY": settings.PINECONE_API_KEY,
        "PINECONE_INDEX_NAME": settings.PINECONE_INDEX_NAME,
        "AWS_ACCESS_KEY_ID": settings.AWS_ACCESS_KEY_ID,
        "AWS_SECRET_ACCESS_KEY": settings.AWS_SECRET_ACCESS_KEY,
        "AWS_REGION": settings.AWS_REGION,
        "S3_BUCKET_NAME": settings.S3_BUCKET_NAME,
        "LANGSMITH_API_KEY": settings.LANGSMITH_API_KEY,
        "LANGCHAIN_PROJECT": settings.LANGCHAIN_PROJECT,
        "LANGSMITH_TRACING": settings.LANGSMITH_TRACING,
    }

    for key, value in env_vars.items():
        if value and key not in os.environ:
            os.environ[key] = value


# Sync settings to environment variables
sync_settings_to_env()
