# ScopeIQ AI Backend

AI-powered document processing and chat service for construction projects.

## Quick Start

### Prerequisites

- Python 3.11+
- Virtual environment (recommended)

### Installation

1. **Clone and setup**
   ```bash
   git clone <repository-url>
   cd scopeiq-ai-backend
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Environment Configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run the application**
   ```bash
   # Development mode (no API keys required)
   uvicorn src.app.main:app --reload
   
   # Or use the startup script
   python run.py
   ```

4. **Access the API**
   - Health Check: http://localhost:8000/api/v1/health

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Service health status |
| `/api/v1/documents/upload` | POST | Upload PDF documents |
| `/api/v1/documents/{id}/progress` | GET | Check processing progress |
| `/api/v1/chat/conversation` | POST | AI chat with document context |

## Environment Variables

### Required for Production
```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_REGION=us-east-1
S3_BUCKET_NAME=your_bucket_name

# AI Services
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=your_index_name
OPENAI_API_KEY=your_openai_key

# Optional
LANGSMITH_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=your_project_name
```

## Deployment

### Docker (Recommended)
```bash
docker build -t scopeiq-ai-backend .
docker run -p 8000:8000 --env-file .env scopeiq-ai-backend
```

### Manual Deployment
```bash
# Production server
uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Initialize DynamoDB Tables
```bash
# Create tables
python setup_dynamodb.py
```

## **Data Model**

### Abbreviations Table
- **Primary Key**: `id` (String)
- **Attributes**: 
  - `abbreviation` (String)
  - `full_form` (String)
  - `created_at` (DateTime)
  - `updated_at` (DateTime)
- **Global Secondary Index**: `abbreviation-index` on `abbreviation`

### Categories Table
- **Primary Key**: `id` (String)
- **Attributes**:
  - `name` (String)
  - `description` (String, nullable)
  - `parent_id` (String, nullable)
  - `created_at` (DateTime)
  - `updated_at` (DateTime)
- **Global Secondary Index**: `parent-id-index` on `parent_id`