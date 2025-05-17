#!/usr/bin/env python3

from fastapi import FastAPI, HTTPException, Query, Path, BackgroundTasks, Request, status, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
import weaviate
import requests
import uvicorn
import os
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
import json
import time
from datetime import datetime
import uuid
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# Import utility functions from the other files
from weaviate_classes_objects_check import (
    get_weaviate_meta_info,
    get_weaviate_health,
    count_objects_via_graphql,
    get_sample_objects,
    run_benchmark_query,
    analyze_property_types,
    estimate_collection_size,
    format_size
)

# Load environment variables
load_dotenv()
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8090")

# Initialize FastAPI app
app = FastAPI(
    title="Weaviate Dashboard",
    description="REST API and Web Dashboard for managing and inspecting Weaviate databases",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Output directory for reports
OUTPUT_DIR = "weaviate_inspection_reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Setup template and static file directories
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Error handling
@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    if request.url.path.startswith("/api"):
        # API requests should return JSON error responses
        return await http_exception_handler(request, exc)
    
    # Web requests should return HTML error page
    return templates.TemplateResponse(
        "error.html",
        {
            "request": request,
            "status_code": exc.status_code,
            "title": "Error",
            "message": exc.detail
        },
        status_code=exc.status_code
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    if request.url.path.startswith("/api"):
        # API requests should return JSON error responses
        return await request_validation_exception_handler(request, exc)
    
    # Web requests should return HTML error page
    return templates.TemplateResponse(
        "error.html",
        {
            "request": request,
            "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY,
            "title": "Validation Error",
            "message": "Invalid request parameters"
        },
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    
    if request.url.path.startswith("/api"):
        # API requests should return JSON error responses
        return JSONResponse(
            status_code=status_code,
            content={"detail": str(exc)}
        )
    
    # Web requests should return HTML error page
    return templates.TemplateResponse(
        "error.html",
        {
            "request": request,
            "status_code": status_code,
            "title": "Server Error",
            "message": "An unexpected error occurred: " + str(exc)
        },
        status_code=status_code
    )

# API models
class ClassDeleteRequest(BaseModel):
    class_name: str = Field(..., description="Name of the class to delete")

class ClassInfo(BaseModel):
    class_name: str = Field(..., description="Name of the class")
    object_count: int = Field(..., description="Number of objects in the class")
    property_count: int = Field(..., description="Number of properties in the class")
    vector_type: str = Field(..., description="Type of vector index")
    vector_dimension: Any = Field(..., description="Dimension of vectors")

class ClassProperty(BaseModel):
    name: str = Field(..., description="Name of the property")
    dataType: List[str] = Field(..., description="Data type of the property")
    description: Optional[str] = Field(None, description="Description of the property")
    tokenization: Optional[str] = Field(None, description="Tokenization strategy")
    indexInverted: Optional[bool] = Field(True, description="Whether to index the property")

class CreateClassRequest(BaseModel):
    class_name: str = Field(..., description="Name of the class to create")
    description: Optional[str] = Field(None, description="Description of the class")
    vectorizer: str = Field("none", description="Type of vectorizer")
    vector_dimension: Optional[int] = Field(None, description="Dimension of vectors")
    properties: List[ClassProperty] = Field([], description="Properties of the class")

class UpdateClassConfigRequest(BaseModel):
    vectorizer: Optional[str] = Field(None, description="Type of vectorizer to use (or None)")
    vector_index_config: Optional[Dict[str, Any]] = Field(None, description="Vector index configuration")
    module_config: Optional[Dict[str, Any]] = Field(None, description="Module configuration")
    distance: Optional[str] = Field(None, description="Distance metric for vector search (cosine, dot, l2-squared)")
    max_connections: Optional[int] = Field(None, description="Maximum connections per node (HNSW index)")
    ef_construction: Optional[int] = Field(None, description="efConstruction parameter (HNSW index)")
    vector_dimension: Optional[int] = Field(None, description="Vector dimension")
    vectorize_class_name: Optional[bool] = Field(None, description="Whether to vectorize the class name")
    # New fields for property updates
    add_properties: Optional[List[ClassProperty]] = Field(None, description="Properties to add to the collection")
    update_index_config: Optional[Dict[str, Any]] = Field(None, description="Index configuration updates")
    inverted_index_config: Optional[Dict[str, Any]] = Field(None, description="Inverted index configuration updates")

class InspectionRequest(BaseModel):
    include_samples: bool = Field(False, description="Whether to include sample objects in the report")
    run_benchmarks: bool = Field(False, description="Whether to run benchmark queries")

class AddObjectRequest(BaseModel):
    properties: Dict[str, Any] = Field(..., description="Properties of the object")

class QueryRequest(BaseModel):
    class_name: str = Field(..., description="Name of the class to query")
    query_text: str = Field(..., description="Query text")
    limit: int = Field(5, description="Maximum number of results")
    search_type: str = Field("vector", description="Type of search: 'vector', 'keyword', or 'combined'")

# Frontend routes
@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    """Serve the dashboard frontend"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/results", response_class=HTMLResponse)
async def get_results_page(request: Request):
    """Serve the search results page"""
    return templates.TemplateResponse("results.html", {"request": request})

@app.get("/inspect/{report_id}", response_class=HTMLResponse)
async def get_inspection_page(request: Request, report_id: str):
    """Serve the inspection details page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/objects/{class_name}", response_class=HTMLResponse)
async def get_objects_page(request: Request, class_name: str):
    """Serve the objects browser page"""
    return templates.TemplateResponse("objects.html", {"request": request, "class_name": class_name})

# Helper functions
def create_weaviate_headers():
    """Create headers for Weaviate API requests"""
    headers = {"Content-Type": "application/json"}
    if WEAVIATE_API_KEY:
        headers["Authorization"] = f"Bearer {WEAVIATE_API_KEY}"
    return headers

def get_schema(base_url=WEAVIATE_URL):
    """Get the schema from Weaviate"""
    try:
        response = requests.get(f"{base_url}/v1/schema", headers=create_weaviate_headers())
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=f"Error getting schema: {response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving schema: {str(e)}")

def create_class(class_data: dict, base_url=WEAVIATE_URL):
    """Create a new class in Weaviate"""
    try:
        response = requests.post(
            f"{base_url}/v1/schema",
            headers=create_weaviate_headers(),
            json=class_data
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            error_msg = f"Error creating class: {response.status_code}"
            if response.text:
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg = f"Error creating class: {error_data['error']}"
                except:
                    error_msg = f"Error creating class: {response.text}"
            
            raise HTTPException(status_code=response.status_code, detail=error_msg)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating class: {str(e)}")

def add_object_to_class(class_name: str, obj_data: dict, base_url=WEAVIATE_URL):
    """Add an object to a class in Weaviate"""
    try:
        response = requests.post(
            f"{base_url}/v1/objects",
            headers=create_weaviate_headers(),
            json={
                "class": class_name,
                "properties": obj_data
            }
        )
        
        if response.status_code in [200, 201]:
            return response.json()
        else:
            error_msg = f"Error adding object: {response.status_code}"
            if response.text:
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg = f"Error adding object: {error_data['error']}"
                except:
                    error_msg = f"Error adding object: {response.text}"
            
            raise HTTPException(status_code=response.status_code, detail=error_msg)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding object: {str(e)}")

def delete_class(class_name, base_url=WEAVIATE_URL):
    """Delete a class from Weaviate"""
    try:
        response = requests.delete(
            f"{base_url}/v1/schema/{class_name}",
            headers=create_weaviate_headers()
        )
        
        if response.status_code in [200, 204]:
            return True
        else:
            raise HTTPException(
                status_code=response.status_code, 
                detail=f"Error deleting class: {response.text}"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during class deletion: {str(e)}")

def list_classes_with_counts(base_url=WEAVIATE_URL):
    """List all classes with their object counts"""
    schema = get_schema(base_url)
    
    if "classes" not in schema or not schema["classes"]:
        return []
    
    classes_with_counts = []
    for class_info in schema["classes"]:
        class_name = class_info.get("class", "Unknown")
        
        # Count objects
        object_count, _ = count_objects_via_graphql(class_name, base_url, create_weaviate_headers())
        
        # Get vector config
        vector_config_type = class_info.get("vectorizer", "None")
        vector_dimension = "N/A"
        if "vectorIndexConfig" in class_info:
            vector_config = class_info.get("vectorIndexConfig", {})
            vector_dimension = vector_config.get("dimension", "N/A")
        
        # Get property count
        properties = class_info.get("properties", [])
        property_count = len(properties)
        
        classes_with_counts.append({
            "class_name": class_name,
            "object_count": object_count,
            "property_count": property_count,
            "vector_type": vector_config_type,
            "vector_dimension": vector_dimension
        })
    
    return classes_with_counts

def get_embedding_model():
    """Get the embedding model for vector search"""
    model_name = "BAAI/bge-m3"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    
    try:
        embedding_model = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            cache_folder=None
        )
        return embedding_model
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        return None

def perform_vector_search(class_name, query_text, limit=5, base_url=WEAVIATE_URL):
    """Perform a vector search in the specified class"""
    try:
        # Get embedding model
        embedding_model = get_embedding_model()
        if not embedding_model:
            return {"status": "error", "message": "Failed to initialize embedding model"}
        
        # Generate embedding for the query
        query_vector = embedding_model.embed_query(query_text)
        
        # First get schema to properly construct query
        schema_response = requests.get(f"{base_url}/v1/schema/{class_name}", headers=create_weaviate_headers())
        if schema_response.status_code != 200:
            return {"status": "error", "message": f"Error getting schema: {schema_response.status_code}"}
            
        schema = schema_response.json()
        properties = [prop.get("name") for prop in schema.get("properties", [])]
        
        # Create properties query string
        properties_query = " ".join(properties)
        
        # Prepare GraphQL query with all properties explicitly listed
        graphql_query = {
            "query": f"""
            {{
              Get {{
                {class_name}(
                  nearVector: {{
                    vector: {json.dumps(query_vector)}
                    certainty: 0.7
                  }}
                  limit: {limit}
                ) {{
                  {properties_query}
                  _additional {{
                    id
                    certainty
                    creationTimeUnix
                    vector
                  }}
                }}
              }}
            }}
            """
        }
        
        # Execute query
        response = requests.post(
            f"{base_url}/v1/graphql",
            headers=create_weaviate_headers(),
            json=graphql_query
        )
        
        if response.status_code == 200:
            result = response.json()
            if "errors" in result:
                return {"status": "error", "message": f"GraphQL errors: {result['errors']}"}
                
            results = result.get("data", {}).get("Get", {}).get(class_name, [])
            return {"status": "success", "results": results}
        else:
            return {"status": "error", "message": f"Error in search: {response.status_code} - {response.text}"}
    except Exception as e:
        return {"status": "error", "message": f"Error in vector search: {str(e)}"}

def perform_keyword_search(class_name, query_text, limit=5, base_url=WEAVIATE_URL):
    """Perform a keyword search in the specified class"""
    try:
        # First get all property names to find properties containing text
        schema_response = requests.get(f"{base_url}/v1/schema/{class_name}", headers=create_weaviate_headers())
        if schema_response.status_code != 200:
            return {"status": "error", "message": f"Error getting schema: {schema_response.status_code}"}
            
        schema = schema_response.json()
        properties = [prop.get("name") for prop in schema.get("properties", [])]
        text_properties = []
        
        # Find text properties to search in
        for prop in schema.get("properties", []):
            if "text" in prop.get("dataType", []) or "string" in prop.get("dataType", []):
                text_properties.append(prop.get("name"))
        
        if not text_properties:
            return {"status": "error", "message": "No text properties found for keyword search"}
        
        # We'll search in the first text property for simplicity
        # In a more comprehensive implementation, you might want to search across multiple properties
        search_property = text_properties[0]
        
        # Create properties query string
        properties_query = " ".join(properties)
        
        # Prepare GraphQL query
        graphql_query = {
            "query": f"""
            {{
              Get {{
                {class_name}(
                  where: {{
                    operator: Like
                    path: ["{search_property}"]
                    valueText: "*{query_text.replace('"', '\\"')}*"
                  }}
                  limit: {limit}
                ) {{
                  {properties_query}
                  _additional {{
                    id
                    creationTimeUnix
                    vector
                  }}
                }}
              }}
            }}
            """
        }
        
        # Execute query
        response = requests.post(
            f"{base_url}/v1/graphql",
            headers=create_weaviate_headers(),
            json=graphql_query
        )
        
        if response.status_code == 200:
            result = response.json()
            if "errors" in result:
                return {"status": "error", "message": f"GraphQL errors: {result['errors']}"}
                
            results = result.get("data", {}).get("Get", {}).get(class_name, [])
            return {"status": "success", "results": results}
        else:
            return {"status": "error", "message": f"Error in search: {response.status_code} - {response.text}"}
    except Exception as e:
        return {"status": "error", "message": f"Error in keyword search: {str(e)}"}

def perform_combined_search(class_name, query_text, limit=5, base_url=WEAVIATE_URL):
    """Perform both vector and keyword search and combine results"""
    try:
        # Perform vector search
        vector_result = perform_vector_search(class_name, query_text, limit, base_url)
        
        # Perform keyword search
        keyword_result = perform_keyword_search(class_name, query_text, limit, base_url)
        
        # Check for errors
        if vector_result.get("status") == "error" and keyword_result.get("status") == "error":
            return {
                "status": "error", 
                "message": f"Vector search error: {vector_result.get('message')}. Keyword search error: {keyword_result.get('message')}"
            }
        
        # Combine results
        combined_results = []
        
        # Add vector results
        if vector_result.get("status") == "success" and vector_result.get("results"):
            for result in vector_result.get("results", []):
                # Mark the result as coming from vector search
                if "_additional" not in result:
                    result["_additional"] = {}
                result["_additional"]["search_type"] = "vector"
                combined_results.append(result)
        
        # Add keyword results that aren't already in the combined results
        if keyword_result.get("status") == "success" and keyword_result.get("results"):
            for result in keyword_result.get("results", []):
                # Check if this result is already in combined_results
                result_id = result.get("_additional", {}).get("id")
                
                if result_id and not any(r.get("_additional", {}).get("id") == result_id for r in combined_results):
                    # Mark the result as coming from keyword search
                    if "_additional" not in result:
                        result["_additional"] = {}
                    result["_additional"]["search_type"] = "keyword"
                    combined_results.append(result)
        
        # Limit total results
        combined_results = combined_results[:limit]
        
        return {"status": "success", "results": combined_results}
    except Exception as e:
        return {"status": "error", "message": f"Error in combined search: {str(e)}"}

# API Routes
@app.get("/api")
async def api_root():
    return {"message": "Welcome to Weaviate Dashboard API"}

@app.get("/api/health")
async def check_health():
    """Check if the Weaviate instance is healthy"""
    try:
        response = requests.get(f"{WEAVIATE_URL}/v1/.well-known/ready", headers=create_weaviate_headers(), timeout=5)
        if response.status_code == 200:
            return {"status": "ready", "message": "Weaviate instance is ready"}
        else:
            return {"status": "not_ready", "message": f"Weaviate instance not ready: {response.status_code}"}
    except requests.exceptions.Timeout:
        return {"status": "timeout", "message": "Connection to Weaviate timed out after 5 seconds"}
    except Exception as e:
        return {"status": "error", "message": f"Error connecting to Weaviate: {str(e)}"}

@app.get("/api/meta")
async def get_meta():
    """Get metadata about the Weaviate instance"""
    try:
        meta_info = get_weaviate_meta_info(WEAVIATE_URL, create_weaviate_headers())
        health_info = get_weaviate_health(WEAVIATE_URL, create_weaviate_headers())
        
        return {
            "meta": meta_info,
            "health": health_info
        }
    except Exception as e:
        return {
            "meta": {"status": "error", "message": f"Error getting metadata: {str(e)}"},
            "health": {"status": "error", "message": f"Error getting health information: {str(e)}"}
        }

@app.post("/api/classes")
async def create_class_endpoint(class_data: CreateClassRequest):
    """Create a new class in Weaviate"""
    try:
        # Format the request data for Weaviate API
        request_data = {
            "class": class_data.class_name
        }
        
        # Add optional fields
        if class_data.description:
            request_data["description"] = class_data.description
        
        # Add vectorizer
        request_data["vectorizer"] = class_data.vectorizer
        
        # Add vectorIndexConfig if dimension is provided
        if class_data.vector_dimension:
            request_data["vectorIndexConfig"] = {
                "dimension": class_data.vector_dimension
            }
        
        # Add properties
        if class_data.properties:
            request_data["properties"] = [prop.dict(exclude_none=True) for prop in class_data.properties]
        
        # Create class
        result = create_class(request_data)
        
        return {
            "status": "success",
            "message": f"Class '{class_data.class_name}' successfully created",
            "data": result
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating class: {str(e)}")

@app.post("/api/objects/{class_name}")
async def add_object(class_name: str, obj_data: AddObjectRequest):
    """Add an object to a class"""
    try:
        result = add_object_to_class(class_name, obj_data.properties)
        
        return {
            "status": "success",
            "message": f"Object successfully added to class '{class_name}'",
            "data": result
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding object: {str(e)}")

@app.get("/api/classes", response_model=List[ClassInfo])
async def get_classes():
    """Get all classes with their object counts"""
    try:
        classes = list_classes_with_counts()
        return classes
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting classes: {str(e)}")

@app.get("/api/classes/{class_name}")
async def get_class_details(class_name: str = Path(..., description="Name of the class to get details for")):
    """Get detailed information about a specific class"""
    try:
        schema = get_schema()
        
        # Find the specific class
        class_info = None
        for cls in schema.get("classes", []):
            if cls.get("class") == class_name:
                class_info = cls
                break
        
        if not class_info:
            raise HTTPException(status_code=404, detail=f"Class '{class_name}' not found")
        
        # Get object count
        object_count, count_time = count_objects_via_graphql(class_name, WEAVIATE_URL, create_weaviate_headers())
        
        # Get properties
        properties = class_info.get("properties", [])
        
        # Get vector config
        vector_config = {
            "type": class_info.get("vectorizer", "None"),
            "config": class_info.get("vectorIndexConfig", {})
        }
        
        # Get sample objects
        sample_objects, sample_time = get_sample_objects(class_name, WEAVIATE_URL, create_weaviate_headers(), limit=3)
        
        # Analyze property types
        property_types = analyze_property_types(properties)
        
        # Estimate storage size if objects exist
        storage_estimate = None
        if object_count > 0 and sample_objects and "dimension" in vector_config["config"]:
            vector_dimension = vector_config["config"]["dimension"]
            storage_estimate = estimate_collection_size(class_name, object_count, sample_objects, vector_dimension)
        
        return {
            "class_name": class_name,
            "object_count": object_count,
            "count_query_time": count_time,
            "properties": properties,
            "property_types": property_types,
            "vector_config": vector_config,
            "sample_objects": sample_objects[:3] if sample_objects else [],
            "sample_query_time": sample_time,
            "storage_estimate": storage_estimate
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting class details: {str(e)}")

@app.delete("/api/classes/{class_name}")
async def delete_class_endpoint(class_name: str = Path(..., description="Name of the class to delete")):
    """Delete a class from Weaviate"""
    try:
        result = delete_class(class_name)
        return {"status": "success", "message": f"Class '{class_name}' successfully deleted"}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting class: {str(e)}")

@app.get("/api/objects/{class_name}")
async def get_objects(
    class_name: str = Path(..., description="Name of the class to get objects from"),
    limit: int = Query(50, description="Maximum number of objects to return"),
    offset: int = Query(0, description="Offset for pagination")
):
    """Get objects from a class"""
    try:
        # First get all property names
        schema_response = requests.get(f"{WEAVIATE_URL}/v1/schema/{class_name}", headers=create_weaviate_headers())
        if schema_response.status_code != 200:
            raise HTTPException(status_code=schema_response.status_code, detail=f"Error getting schema for {class_name}")
            
        schema = schema_response.json()
        properties = [prop.get("name") for prop in schema.get("properties", [])]
        
        # Create dynamic GraphQL query with all properties
        properties_query = " ".join(properties)
        
        graphql_query = {
            "query": f"""
            {{
              Get {{
                {class_name}(limit: {limit}, offset: {offset}) {{
                  {properties_query}
                  _additional {{
                    id
                    creationTimeUnix
                    vector
                    certainty
                  }}
                }}
              }}
            }}
            """
        }
        
        response = requests.post(
            f"{WEAVIATE_URL}/v1/graphql",
            headers=create_weaviate_headers(),
            json=graphql_query
        )
        
        if response.status_code == 200:
            result = response.json()
            if "errors" in result:
                raise HTTPException(status_code=400, detail=f"GraphQL errors: {result['errors']}")
                
            # Extract objects from response
            objects = result.get("data", {}).get("Get", {}).get(class_name, [])
            
            # Get total count
            count_query = {
                "query": f"""
                {{
                  Aggregate {{
                    {class_name} {{
                      meta {{
                        count
                      }}
                    }}
                  }}
                }}
                """
            }
            
            count_response = requests.post(
                f"{WEAVIATE_URL}/v1/graphql",
                headers=create_weaviate_headers(),
                json=count_query
            )
            
            total_count = 0
            if count_response.status_code == 200:
                count_result = count_response.json()
                if "errors" not in count_result:
                    count_data = count_result.get("data", {}).get("Aggregate", {}).get(class_name, [{}])
                    if count_data and len(count_data) > 0:
                        total_count = count_data[0].get("meta", {}).get("count", 0)
            
            return {
                "objects": objects,
                "meta": {
                    "total": total_count,
                    "count": len(objects),
                    "offset": offset,
                    "limit": limit,
                    "class_name": class_name,
                    "properties": properties
                }
            }
        else:
            raise HTTPException(status_code=response.status_code, detail=f"Error fetching objects: {response.text}")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting objects: {str(e)}")

@app.post("/api/inspect")
async def run_inspection(background_tasks: BackgroundTasks, request: InspectionRequest = None):
    """Run a full database inspection"""
    # Generate a unique ID for this inspection run
    if request is None:
        request = InspectionRequest()
        
    inspection_id = str(uuid.uuid4())[:8]
    inspection_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_id = f"{inspection_timestamp}_{inspection_id}"
    
    # Create a skeleton of the report
    inspection_report = {
        "id": report_id,
        "timestamp": datetime.now().isoformat(),
        "status": "started",
        "server_info": {},
        "health_check": {},
        "collections": [],
        "performance": {},
        "storage": {}
    }
    
    # Save the initial report
    report_path = os.path.join(OUTPUT_DIR, f"weaviate_inspection_{report_id}.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(inspection_report, f, indent=2, default=str)
    
    # Run the inspection in the background
    background_tasks.add_task(
        run_full_inspection,
        report_id=report_id,
        report_path=report_path,
        include_samples=request.include_samples,
        run_benchmarks=request.run_benchmarks
    )
    
    return {
        "report_id": report_id,
        "status": "started",
        "message": "Inspection started in the background",
        "result_path": report_path
    }

@app.get("/api/inspect/{report_id}")
async def get_inspection_report(report_id: str = Path(..., description="ID of the inspection report")):
    """Get an inspection report by ID"""
    report_path = os.path.join(OUTPUT_DIR, f"weaviate_inspection_{report_id}.json")
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail=f"Report with ID {report_id} not found")
    
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading report: {str(e)}")

@app.get("/api/inspect")
async def list_inspection_reports():
    """List all inspection reports"""
    try:
        reports = []
        for filename in os.listdir(OUTPUT_DIR):
            if filename.startswith("weaviate_inspection_") and filename.endswith(".json"):
                report_path = os.path.join(OUTPUT_DIR, filename)
                try:
                    with open(report_path, 'r', encoding='utf-8') as f:
                        report = json.load(f)
                    
                    reports.append({
                        "id": report.get("id"),
                        "timestamp": report.get("timestamp"),
                        "status": report.get("status", "unknown"),
                        "collections_count": len(report.get("collections", [])),
                        "file_path": report_path
                    })
                except Exception as e:
                    # Skip corrupt reports
                    continue
        
        return {"reports": sorted(reports, key=lambda x: x.get("timestamp", ""), reverse=True)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing reports: {str(e)}")

@app.get("/api/datatypes")
async def get_data_types():
    """Get available data types for class properties"""
    return {
        "primitive": ["text", "string", "int", "number", "boolean", "date", "geoCoordinates", "phoneNumber", "blob", "uuid"],
        "reference": ["cross-reference"],
        "vectorizers": ["none", "text2vec-contextionary", "text2vec-transformers", "text2vec-openai", "text2vec-cohere", "text2vec-huggingface"]
    }

@app.post("/api/query")
async def query_class(request: QueryRequest):
    """Query a class with vector or keyword search"""
    try:
        if request.search_type == "vector":
            result = perform_vector_search(request.class_name, request.query_text, request.limit)
            return result
        elif request.search_type == "keyword":
            result = perform_keyword_search(request.class_name, request.query_text, request.limit)
            return result
        elif request.search_type == "combined":
            result = perform_combined_search(request.class_name, request.query_text, request.limit)
            return result
        else:
            raise HTTPException(status_code=400, detail=f"Invalid search type: {request.search_type}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying class: {str(e)}")

# Background task function
async def run_full_inspection(report_id, report_path, include_samples=False, run_benchmarks=False):
    """Run a full inspection of the Weaviate database"""
    try:
        # Update the report status
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        report["status"] = "running"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Get Weaviate metadata
        meta_info = get_weaviate_meta_info(WEAVIATE_URL, create_weaviate_headers())
        health_info = get_weaviate_health(WEAVIATE_URL, create_weaviate_headers())
        
        report["server_info"] = meta_info
        report["health_check"] = health_info
        
        # Get schema
        schema = get_schema()
        
        # Process collections info from schema
        total_objects = 0
        total_storage_bytes = 0
        benchmark_results = {}
        
        if "classes" in schema and schema["classes"]:
            # Process each class
            for class_info in schema["classes"]:
                class_name = class_info.get("class", "Unknown")
                
                collection_data = {
                    "name": class_name,
                    "properties": [],
                    "objects": {},
                    "vector_config": {},
                    "storage": {},
                    "performance": {}
                }
                
                # Run benchmark queries if requested
                if run_benchmarks:
                    benchmark = run_benchmark_query(class_name, WEAVIATE_URL, create_weaviate_headers())
                    benchmark_results[class_name] = benchmark
                    collection_data["performance"] = benchmark
                
                # Count objects
                object_count, count_time = count_objects_via_graphql(class_name, WEAVIATE_URL, create_weaviate_headers())
                total_objects += object_count
                
                collection_data["objects"]["count"] = object_count
                collection_data["objects"]["count_query_time"] = count_time
                
                # Get properties
                properties = class_info.get("properties", [])
                collection_data["properties"] = properties
                
                # Analyze property types
                property_types = analyze_property_types(properties)
                collection_data["property_types"] = property_types
                
                # Get vector config
                vector_config_type = class_info.get("vectorizer", "None")
                vector_dimension = "N/A"
                if "vectorIndexConfig" in class_info:
                    vector_config = class_info.get("vectorIndexConfig", {})
                    vector_dimension = vector_config.get("dimension", "N/A")
                    
                collection_data["vector_config"] = {
                    "type": vector_config_type,
                    "dimension": vector_dimension
                }
                
                # Get sample objects if requested and objects exist
                if include_samples and object_count > 0:
                    sample_objects, sample_time = get_sample_objects(class_name, WEAVIATE_URL, create_weaviate_headers(), limit=3)
                    
                    collection_data["objects"]["sample_query_time"] = sample_time
                    
                    if sample_objects:
                        collection_data["objects"]["samples"] = []
                        
                        for obj in sample_objects:
                            obj_summary = {"properties": {}}
                            
                            # Get object ID
                            obj_id = obj.get("_additional", {}).get("id", "Unknown")
                            obj_summary["id"] = obj_id
                            
                            # Get creation time
                            creation_time = obj.get("_additional", {}).get("creationTimeUnix", 0)
                            if creation_time:
                                try:
                                    creation_date = datetime.fromtimestamp(creation_time/1000).strftime('%Y-%m-%d %H:%M:%S')
                                    obj_summary["created"] = creation_date
                                except:
                                    obj_summary["created"] = creation_time
                            
                            # Add key properties
                            for prop, value in obj.items():
                                if prop != "_additional":
                                    obj_summary["properties"][prop] = value
                            
                            collection_data["objects"]["samples"].append(obj_summary)
                        
                        # Estimate storage size
                        if isinstance(vector_dimension, (int, float)) and vector_dimension > 0:
                            size_estimate = estimate_collection_size(class_name, object_count, sample_objects, vector_dimension)
                            total_storage_bytes += size_estimate["estimated_size_bytes"]
                            collection_data["storage"] = size_estimate
                
                # Add collection data to the report
                report["collections"].append(collection_data)
                
                # Periodically update the report file
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, default=str)
            
            # Update final storage statistics
            report["storage"]["total_bytes"] = total_storage_bytes
            report["storage"]["total_formatted"] = format_size(total_storage_bytes)
            report["performance"]["benchmarks"] = benchmark_results
        
        # Update the report status to completed
        report["status"] = "completed"
        report["end_time"] = datetime.now().isoformat()
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
            
    except Exception as e:
        # Update the report with error status
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                report = json.load(f)
            
            report["status"] = "error"
            report["error"] = str(e)
            report["end_time"] = datetime.now().isoformat()
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
        except:
            pass

# Add a new function to update the class configuration
def update_class_config(class_name: str, config: dict, base_url=WEAVIATE_URL):
    """Update the configuration of a Weaviate class/collection"""
    try:
        # Convert from our API format to Weaviate's expected format
        weaviate_config = {}
        
        # Handle vectorizer
        if "vectorizer" in config:
            weaviate_config["vectorizer"] = config["vectorizer"]
        
        # Handle vector index config
        vector_index_config = {}
        
        if "max_connections" in config and config["max_connections"] is not None:
            vector_index_config["maxConnections"] = config["max_connections"]
            
        if "ef_construction" in config and config["ef_construction"] is not None:
            vector_index_config["efConstruction"] = config["ef_construction"]
            
        if "distance" in config and config["distance"] is not None:
            vector_index_config["distance"] = config["distance"]
            
        if "vector_dimension" in config and config["vector_dimension"] is not None:
            vector_index_config["dimension"] = config["vector_dimension"]
            
        if vector_index_config:
            weaviate_config["vectorIndexConfig"] = vector_index_config

        # Handle inverted index config
        if "inverted_index_config" in config and config["inverted_index_config"]:
            weaviate_config["invertedIndexConfig"] = config["inverted_index_config"]
            
        # Handle module config
        if "module_config" in config and config["module_config"]:
            weaviate_config["moduleConfig"] = config["module_config"]
        elif "vectorize_class_name" in config and config["vectorize_class_name"] is not None:
            # Special case for vectorizeClassName which is common
            if "vectorizer" in config and config["vectorizer"] and config["vectorizer"] != "none":
                weaviate_config["moduleConfig"] = {
                    config["vectorizer"]: {
                        "vectorizeClassName": config["vectorize_class_name"]
                    }
                }
        
        # Weaviate v1 API endpoint for updating class config
        url = f"{base_url}/v1/schema/{class_name}"
        
        # Handle property additions if specified (using separate endpoint)
        property_results = []
        if "add_properties" in config and config["add_properties"]:
            print(f"Adding properties to class {class_name}: {config['add_properties']}")
            for prop in config["add_properties"]:
                try:
                    # For debugging, print out the property data
                    print(f"Processing property for addition: {prop}")
                    
                    # Verify property has required fields
                    if "name" not in prop or not prop["name"]:
                        raise ValueError("Property must have a name")
                    
                    if "dataType" not in prop or not prop["dataType"]:
                        raise ValueError("Property must have a dataType")
                    
                    # Convert from our API model to Weaviate's expected format
                    prop_data = {
                        "dataType": prop["dataType"] if isinstance(prop["dataType"], list) else [prop["dataType"]],
                        "name": prop["name"],
                    }
                    
                    if "description" in prop and prop["description"]:
                        prop_data["description"] = prop["description"]
                        
                    if "tokenization" in prop and prop["tokenization"]:
                        prop_data["tokenization"] = prop["tokenization"]
                        
                    if "indexInverted" in prop:
                        prop_data["indexInverted"] = prop["indexInverted"]
                    
                    # Add property using Weaviate v4 approach
                    prop_url = f"{base_url}/v1/schema/{class_name}/properties"
                    
                    print(f"Adding property to {class_name}: {json.dumps(prop_data)}")
                    
                    prop_response = requests.post(
                        prop_url, 
                        json=prop_data,
                        headers=create_weaviate_headers()
                    )
                    
                    print(f"Property add response: {prop_response.status_code} - {prop_response.text}")
                    
                    if prop_response.status_code == 200:
                        property_results.append({
                            "name": prop["name"],
                            "status": "success"
                        })
                    else:
                        # Try to extract meaningful error message
                        error_message = "Unknown error"
                        try:
                            error_data = prop_response.json()
                            if "error" in error_data:
                                error_message = error_data["error"]
                            elif "message" in error_data:
                                error_message = error_data["message"]
                            else:
                                error_message = prop_response.text
                        except:
                            error_message = prop_response.text
                            
                        property_results.append({
                            "name": prop["name"],
                            "status": "error",
                            "message": error_message
                        })
                        print(f"Error adding property {prop['name']}: {prop_response.status_code} - {error_message}")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    property_results.append({
                        "name": prop.get("name", "unknown"),
                        "status": "error",
                        "message": str(e)
                    })
                    print(f"Exception adding property: {str(e)}")
        
        # Update class config if we have settings to update
        config_response = None
        config_result = None
        if weaviate_config:
            print(f"Updating class {class_name} config with: {json.dumps(weaviate_config)}")
            config_response = requests.put(
                url, 
                json=weaviate_config, 
                headers=create_weaviate_headers()
            )
            
            print(f"Config update response: {config_response.status_code} - {config_response.text}")
            
            try:
                config_result = config_response.json()
            except:
                config_result = {"text": config_response.text}
        
        # Prepare the response
        result = {"status": "success"}
        
        if config_response:
            if config_response.status_code == 200:
                result["config_update"] = "success"
                result["config_data"] = config_result
            else:
                result["config_update"] = "error"
                try:
                    # Try to extract a meaningful error message
                    config_error = config_response.json()
                    if "error" in config_error:
                        result["config_message"] = config_error["error"]
                    elif "message" in config_error:
                        result["config_message"] = config_error["message"]
                    else:
                        result["config_message"] = config_response.text
                except:
                    result["config_message"] = config_response.text
        
        if property_results:
            result["property_updates"] = property_results
            
        # Check whether anything succeeded
        if ((config_response and config_response.status_code != 200) and 
            (property_results and all(p["status"] == "error" for p in property_results))):
            result["status"] = "error"
            result["message"] = "All updates failed"
            
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error in update_class_config: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error updating class configuration: {str(e)}"
        )

# Add the API endpoint for updating class configurations
@app.put("/api/classes/{class_name}/config")
async def update_class_config_endpoint(
    class_name: str = Path(..., description="Name of the class to update"),
    config: UpdateClassConfigRequest = Body(..., description="Configuration to update")
):
    """Update the configuration of a class in Weaviate"""
    try:
        # Convert the model to a dictionary
        config_dict = config.model_dump(exclude_none=True)
        
        # Make sure we extract property objects correctly
        if "add_properties" in config_dict and config_dict["add_properties"] is not None:
            # Ensure each property is properly formatted for the API request
            for prop in config_dict["add_properties"]:
                print(f"Processing property: {prop}")
                
                # No further processing needed, the update_class_config function handles this
        
        # Call the function to update the class configuration
        result = update_class_config(class_name, config_dict)
        return result
    except Exception as e:
        print(f"Error in update_class_config_endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error updating class: {str(e)}"
        )

if __name__ == "__main__":
    import socket
    import argparse
    from contextlib import closing
    
    def find_free_port(default_port, max_attempts=10):
        """Find a free port starting from the default port"""
        for port in range(default_port, default_port + max_attempts):
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
                try:
                    sock.bind(('0.0.0.0', port))
                    return port
                except socket.error:
                    print(f"Port {port} is already in use, trying next port...")
        raise RuntimeError(f"Could not find a free port after {max_attempts} attempts")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Weaviate Dashboard')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the dashboard on (default: 8000)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the dashboard on (default: 0.0.0.0)')
    parser.add_argument('--auto-port', action='store_true', help='Automatically find a free port if the default is in use')
    args = parser.parse_args()
    
    port = args.port
    if args.auto_port:
        try:
            port = find_free_port(port)
            print(f"Using port {port}")
        except RuntimeError as e:
            print(f"Error: {e}")
            import sys
            sys.exit(1)
    
    # Run the server
    uvicorn.run("new_dashboard:app", host=args.host, port=port, reload=True)
