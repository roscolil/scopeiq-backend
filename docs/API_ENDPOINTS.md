# ScopeIQ AI Backend - CRUD API Endpoints

This document provides comprehensive documentation for all CRUD endpoints for abbreviations and categories, designed for frontend integration.

## Base Configuration

- **Base URL**: `http://localhost:8000/api/v1`
- **Content-Type**: `application/json`
- **Response Format**: All endpoints return JSON responses with consistent structure

## Response Format

### Success Response
```json
{
  "success": true,
  "data": { ... }
}
```

### List Response (with pagination)
```json
{
  "success": true,
  "data": [...],
  "total": 25,
  "skip": 0,
  "limit": 10
}
```

### Error Response
```json
{
  "detail": "Error message here"
}
```

---

## 📋 Abbreviations API

### Data Model
```typescript
interface Abbreviation {
  id: string;
  abbreviation: string;
  full_form: string;
  created_at: string; // ISO 8601 datetime
  updated_at: string; // ISO 8601 datetime
}
```

### 1. Create Abbreviation

**Endpoint**: `POST /api/v1/abbreviations`

**Request Body**:
```json
{
  "abbreviation": "API",
  "full_form": "Application Programming Interface"
}
```

**Response** (201 Created):
```json
{
  "success": true,
  "data": {
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "abbreviation": "API",
    "full_form": "Application Programming Interface",
    "created_at": "2024-01-01T12:00:00Z",
    "updated_at": "2024-01-01T12:00:00Z"
  }
}
```

**cURL Example**:
```bash
curl -X POST http://localhost:8000/api/v1/abbreviations \
  -H "Content-Type: application/json" \
  -d '{"abbreviation": "API", "full_form": "Application Programming Interface"}'
```

### 2. Get Abbreviation by ID

**Endpoint**: `GET /api/v1/abbreviations/{abbreviation_id}`

**Response** (200 OK):
```json
{
  "success": true,
  "data": {
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "abbreviation": "API",
    "full_form": "Application Programming Interface",
    "created_at": "2024-01-01T12:00:00Z",
    "updated_at": "2024-01-01T12:00:00Z"
  }
}
```

**Response** (404 Not Found):
```json
{
  "detail": "Abbreviation not found"
}
```

**cURL Example**:
```bash
curl http://localhost:8000/api/v1/abbreviations/123e4567-e89b-12d3-a456-426614174000
```

### 3. List Abbreviations (with Pagination)

**Endpoint**: `GET /api/v1/abbreviations`

**Query Parameters**:
- `skip` (optional): Number of records to skip (default: 0)
- `limit` (optional): Maximum number of records to return (default: 100, max: 1000)

**Response** (200 OK):
```json
{
  "success": true,
  "data": [
    {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "abbreviation": "API",
      "full_form": "Application Programming Interface",
      "created_at": "2024-01-01T12:00:00Z",
      "updated_at": "2024-01-01T12:00:00Z"
    }
  ],
  "total": 25,
  "skip": 0,
  "limit": 10
}
```

**cURL Example**:
```bash
curl "http://localhost:8000/api/v1/abbreviations?skip=0&limit=10"
```

### 4. Update Abbreviation

**Endpoint**: `PUT /api/v1/abbreviations/{abbreviation_id}`

**Request Body** (all fields optional):
```json
{
  "abbreviation": "API",
  "full_form": "Application Programming Interface (Updated)"
}
```

**Response** (200 OK):
```json
{
  "success": true,
  "data": {
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "abbreviation": "API",
    "full_form": "Application Programming Interface (Updated)",
    "created_at": "2024-01-01T12:00:00Z",
    "updated_at": "2024-01-01T12:05:00Z"
  }
}
```

**Response** (404 Not Found):
```json
{
  "detail": "Abbreviation not found"
}
```

**cURL Example**:
```bash
curl -X PUT http://localhost:8000/api/v1/abbreviations/123e4567-e89b-12d3-a456-426614174000 \
  -H "Content-Type: application/json" \
  -d '{"full_form": "Application Programming Interface (Updated)"}'
```

### 5. Delete Abbreviation

**Endpoint**: `DELETE /api/v1/abbreviations/{abbreviation_id}`

**Response** (204 No Content): Empty response body

**Response** (404 Not Found):
```json
{
  "detail": "Abbreviation not found"
}
```

**cURL Example**:
```bash
curl -X DELETE http://localhost:8000/api/v1/abbreviations/123e4567-e89b-12d3-a456-426614174000
```

---

## 📁 Categories API

### Data Model
```typescript
interface Category {
  id: string;
  name: string;
  description: string;
  parent_id: string | null;
  created_at: string; // ISO 8601 datetime
  updated_at: string; // ISO 8601 datetime
}
```

### 1. Create Category

**Endpoint**: `POST /api/v1/categories`

**Request Body**:
```json
{
  "name": "Building Permits",
  "description": "Documents related to building permits",
  "parent_id": null
}
```

**Response** (201 Created):
```json
{
  "success": true,
  "data": {
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "name": "Building Permits",
    "description": "Documents related to building permits",
    "parent_id": null,
    "created_at": "2024-01-01T12:00:00Z",
    "updated_at": "2024-01-01T12:00:00Z"
  }
}
```

**cURL Example**:
```bash
curl -X POST http://localhost:8000/api/v1/categories \
  -H "Content-Type: application/json" \
  -d '{"name": "Building Permits", "description": "Documents related to building permits", "parent_id": null}'
```

### 2. Get Category by ID

**Endpoint**: `GET /api/v1/categories/{category_id}`

**Response** (200 OK):
```json
{
  "success": true,
  "data": {
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "name": "Building Permits",
    "description": "Documents related to building permits",
    "parent_id": null,
    "created_at": "2024-01-01T12:00:00Z",
    "updated_at": "2024-01-01T12:00:00Z"
  }
}
```

**Response** (404 Not Found):
```json
{
  "detail": "Category not found"
}
```

**cURL Example**:
```bash
curl http://localhost:8000/api/v1/categories/123e4567-e89b-12d3-a456-426614174000
```

### 3. List Categories (with Pagination and Parent Filtering)

**Endpoint**: `GET /api/v1/categories`

**Query Parameters**:
- `skip` (optional): Number of records to skip (default: 0)
- `limit` (optional): Maximum number of records to return (default: 100, max: 1000)
- `parent_id` (optional): Filter by parent category ID
  - `parent_id=""` (empty string): Get only root categories (no parent)
  - `parent_id="uuid"`: Get categories with specific parent
  - `parent_id=null` (not provided): Get all categories

**Response** (200 OK):
```json
{
  "success": true,
  "data": [
    {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "name": "Building Permits",
      "description": "Documents related to building permits",
      "parent_id": null,
      "created_at": "2024-01-01T12:00:00Z",
      "updated_at": "2024-01-01T12:00:00Z"
    },
    {
      "id": "456e7890-e89b-12d3-a456-426614174001",
      "name": "Architectural",
      "description": "Documents related to architectural under drawings",
      "parent_id": "789e0123-e89b-12d3-a456-426614174002",
      "created_at": "2024-01-01T12:00:00Z",
      "updated_at": "2024-01-01T12:00:00Z"
    }
  ],
  "total": 25,
  "skip": 0,
  "limit": 10
}
```

**cURL Examples**:
```bash
# Get all categories
curl "http://localhost:8000/api/v1/categories?skip=0&limit=10"

# Get only root categories
curl "http://localhost:8000/api/v1/categories?parent_id=&skip=0&limit=10"

# Get categories with specific parent
curl "http://localhost:8000/api/v1/categories?parent_id=789e0123-e89b-12d3-a456-426614174002&skip=0&limit=10"
```

### 4. Get Child Categories

**Endpoint**: `GET /api/v1/categories/{category_id}/children`

**Response** (200 OK):
```json
{
  "success": true,
  "data": [
    {
      "id": "456e7890-e89b-12d3-a456-426614174001",
      "name": "Architectural",
      "description": "Documents related to architectural under drawings",
      "parent_id": "789e0123-e89b-12d3-a456-426614174002",
      "created_at": "2024-01-01T12:00:00Z",
      "updated_at": "2024-01-01T12:00:00Z"
    }
  ],
  "total": 1,
  "skip": 0,
  "limit": 1
}
```

**cURL Example**:
```bash
curl http://localhost:8000/api/v1/categories/789e0123-e89b-12d3-a456-426614174002/children
```

### 5. Update Category

**Endpoint**: `PUT /api/v1/categories/{category_id}`

**Request Body** (all fields optional):
```json
{
  "name": "Building Permits",
  "description": "Updated description for building permits",
  "parent_id": null
}
```

**Response** (200 OK):
```json
{
  "success": true,
  "data": {
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "name": "Building Permits",
    "description": "Updated description for building permits",
    "parent_id": null,
    "created_at": "2024-01-01T12:00:00Z",
    "updated_at": "2024-01-01T12:05:00Z"
  }
}
```

**Response** (404 Not Found):
```json
{
  "detail": "Category not found"
}
```

**cURL Example**:
```bash
curl -X PUT http://localhost:8000/api/v1/categories/123e4567-e89b-12d3-a456-426614174000 \
  -H "Content-Type: application/json" \
  -d '{"description": "Updated description for building permits"}'
```

### 6. Delete Category

**Endpoint**: `DELETE /api/v1/categories/{category_id}`

**Response** (204 No Content): Empty response body

**Response** (404 Not Found):
```json
{
  "detail": "Category not found"
}
```

**cURL Example**:
```bash
curl -X DELETE http://localhost:8000/api/v1/categories/123e4567-e89b-12d3-a456-426614174000
```

---

### Error Response Format
```json
{
  "detail": "Specific error message here"
}
```