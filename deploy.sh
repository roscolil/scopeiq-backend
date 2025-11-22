#!/bin/bash

# Quick deployment script for ScopeIQ AI Backend
# Simple version for fast deployments

set -e

# Configuration
AWS_REGION="ap-southeast-2"
AWS_ACCOUNT_ID="405995996508"
ECR_REPOSITORY="scopeiq-ai-backend"
IMAGE_TAG="latest"

echo "🚀 Building and pushing ScopeIQ AI Backend to ECR..."

# Login to ECR
echo "📝 Logging in to ECR..."
aws ecr get-login-password --region $AWS_REGION | \
    docker login --username AWS --password-stdin \
    $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Build image
echo "🔨 Building Docker image for linux/amd64 platform..."
docker build --platform linux/amd64 -t scopeiq-ai-backend:$IMAGE_TAG .

# Tag for ECR
echo "🏷️  Tagging image..."
docker tag scopeiq-ai-backend:$IMAGE_TAG \
    $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:$IMAGE_TAG

# Push to ECR
echo "📤 Pushing to ECR..."
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:$IMAGE_TAG

# Cleanup
echo "🧹 Cleaning up..."
docker rmi scopeiq-ai-backend:$IMAGE_TAG
docker rmi $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:$IMAGE_TAG

echo "✅ Deployment complete!"
echo "📋 ECR Image URI: $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:$IMAGE_TAG"
