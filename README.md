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
| `/api/v1/documents/upload` | POST | Upload PDF, DOC/DOCX, or TXT documents |
| `/api/v1/documents/{id}/progress` | GET | Check processing progress |
| `/api/v1/chat/conversation` | POST | AI chat with document context (RAG-powered) |
| `/api/v1/abbreviations/*` | GET/POST | Manage abbreviations |
| `/api/v1/categories/*` | GET/POST | Manage categories |

## Environment Variables

### Required for Production
```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_REGION=ap-southeast-2
S3_BUCKET_NAME=your_bucket_name

# AI Services
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=your_index_name
OPENAI_API_KEY=your_openai_key

# Vision Pipeline (DashScope/Qwen)
DASHSCOPE_API_KEY=your_dashscope_key
VLM_MODEL_NAME=qwen3-vl-235b-a22b-instruct
VLM_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1

# Optional
LANGSMITH_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=your_project_name
YOLO_MODEL_PATH=yolo11n.pt  # Path to YOLO model (if using YOLO detection)
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

## Document Processing Pipeline

The system processes documents through a multi-stage pipeline that intelligently handles different document types and page content.

### Supported File Types
- **PDF**: Full support with page classification and vision processing
- **DOC/DOCX**: Text extraction and processing
- **TXT**: Direct text processing

### Processing Stages

#### 1. Document Upload & Storage
- Files are uploaded to S3 with organized key structure: `{company_id}/{project_id}/{filename}`
- Filenames are sanitized for safe storage

#### 2. Page Classification (PDF only)
- Each PDF page is classified as either:
  - **Text page**: Contains mostly text content (schedules, tables, specifications)
  - **Drawing page**: Contains architectural/engineering drawings, plans, or schematics
- Classification uses Qwen vision model (`qwen3-vl-8b-instruct`)

#### 3. Text Page Processing
- Text extraction using PyMuPDF
- Text cleaning (removes excessive line breaks and spaces)
- Chunking with RecursiveCharacterTextSplitter (2000 chars, 200 overlap)

#### 4. Drawing Page Processing (Vision Pipeline)
Drawing pages go through a multi-stage vision processing pipeline:

**Stage 1: Legend Detection**
- Uses Vision Language Model (VLM) to detect legend bounding boxes
- Outputs detected legend boxes

**Stage 1.1: [Temp] Legend Summarization**
- Extracts and summarizes legend items from detected regions
- Creates structured descriptions of symbols and their meanings
- Currently used to assist Stage 

**Stage 2: Legend Item Detection**
- Uses a single-class YOLO object detection model to detect legend items - symbols & descriptions within a legend box
- Performs Row Clustering to group and relate detected symbols and descriptions 
- Then, performs Column Detection to handle multi-column legends and differentiate between Symbols and Descriptions
- Outputs a list bounding boxes with detected row and column 
- Currently not used but will be part of longer-term solution

**Stage 3: Symbol Query**
- Temporary solution to use a reasoning VLM (`qwen3-vl-235b-a22b-thinking`) to:
  - Analyze symbol occurrences in the drawing
  - Identify locations (rooms, areas) where symbols appear
  - Count total occurrences of each symbol
  - Provide detailed spatial analysis
- Longer-term solution to employ "query by example" methods to search symbols 

#### 5. Embedding & Vector Storage
- All processed content (text chunks and drawing descriptions) are embedded using OpenAI `text-embedding-3-large`
- Stored in Pinecone vector database with namespace per project
- Metadata includes document ID, project ID, company ID, page numbers, and processing type

## Chat Service (RAG)

The chat service uses LangGraph for Retrieval Augmented Generation (RAG):

### Architecture
1. **Retrieval**: Queries Pinecone vector store using MMR (Maximal Marginal Relevance) search
2. **Generation**: Uses GPT-4o to generate answers based on retrieved context

### Features
- Project-scoped document retrieval (namespace isolation)
- Context-aware responses based on uploaded documents
- Supports conversation history
- Specialized prompts for construction document analysis

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