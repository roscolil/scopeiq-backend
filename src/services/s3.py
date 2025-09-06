import boto3
from datetime import datetime
from src.core.config import settings


class S3Service:
    def __init__(self):
        self.region_name = settings.AWS_REGION
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=self.region_name,
        )
        self.bucket_name = settings.S3_BUCKET_NAME

    def upload_file(self, file_content: bytes, s3_key: str) -> str:
        """Upload file to S3 and return URL"""
        if not self.bucket_name:
            raise ValueError("S3_BUCKET_NAME environment variable not set")

        self.s3_client.put_object(
            Bucket=self.bucket_name, Key=s3_key, Body=file_content
        )

        return f"https://{self.bucket_name}.s3.{settings.AWS_REGION}.amazonaws.com/{s3_key}"

    def generate_s3_key(self, company_id: str, project_id: str, filename: str) -> str:
        """Generate S3 key with proper naming convention"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{company_id}/{project_id}/files/{timestamp}_{filename}"
