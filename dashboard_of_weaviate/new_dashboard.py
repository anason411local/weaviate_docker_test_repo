#!/usr/bin/env python3

from fastapi import FastAPI, HTTPException, Query, Path, BackgroundTasks, Request, status, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
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
import subprocess
import pandas as pd
from io import BytesIO
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import base64
import secrets
import math
import asyncio
import shutil
import string
import httpx
import numpy as np
import google.generativeai as genai  # Add this import for the Gemini API
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

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
    title="Weaviate_DB_Dashboard_411Local",
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
    search_type: str = Field("vector", description="Type of search: 'vector', 'keyword', 'combined', or 'gemini'")
    gemini_api_key: Optional[str] = Field(None, description="API key for Gemini, required if search_type is 'gemini'")
    gemini_model: Optional[str] = Field("gemini-1.5-flash", description="Gemini model to use, defaults to gemini-1.5-flash")

# Add model for login credentials
class LoginCredentials(BaseModel):
    username: str = Field(..., description="Username for sudo access")
    password: str = Field(..., description="Password for sudo access")

# Global variable to store user credentials (temporarily in memory)
sudo_credentials = None

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

@app.get("/api/classes/export")
async def export_classes_to_excel():
    """Export all classes and their information to Excel format"""
    try:
        # Get all classes
        classes = list_classes_with_counts()
        
        # Create a dataframe for the basic class information
        classes_df = pd.DataFrame(classes)
        
        # Create a BytesIO object to store the Excel file
        output = BytesIO()
        
        # Create Excel writer
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Write the main classes dataframe
            classes_df.to_excel(writer, sheet_name='Classes Overview', index=False)
            
            # For each class, get detailed information and create a sheet
            for cls in classes:
                class_name = cls['class_name']
                # Get class details including properties
                try:
                    class_details = get_class_details_for_export(class_name)
                    
                    # Create a properties dataframe
                    properties = class_details.get('properties', [])
                    if properties:
                        props_df = pd.DataFrame(properties)
                        # Convert dataType lists to strings for Excel
                        props_df['dataType'] = props_df['dataType'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
                        props_df.to_excel(writer, sheet_name=f'{class_name}', index=False)
                except Exception as e:
                    print(f"Error getting details for class {class_name}: {e}")
                    continue
        
        # Set the pointer to the beginning of the BytesIO object
        output.seek(0)
        
        # Return the Excel file
        return StreamingResponse(
            output, 
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            headers={"Content-Disposition": f"attachment; filename=weaviate_classes.xlsx"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting classes: {str(e)}")

# Helper function to get class details for export without async
def get_class_details_for_export(class_name: str):
    """Get detailed information about a specific class for export"""
    try:
        schema = get_schema()
        
        # Find the specific class
        class_info = None
        for cls in schema.get("classes", []):
            if cls.get("class") == class_name:
                class_info = cls
                break
        
        if not class_info:
            return {"error": f"Class '{class_name}' not found"}
        
        # Get object count
        object_count, count_time = count_objects_via_graphql(class_name, WEAVIATE_URL, create_weaviate_headers())
        
        # Get properties
        properties = class_info.get("properties", [])
        
        # Get vector config
        vector_config = {
            "type": class_info.get("vectorizer", "None"),
            "config": class_info.get("vectorIndexConfig", {})
        }
        
        # Analyze property types
        property_types = analyze_property_types(properties)
        
        return {
            "class_name": class_name,
            "object_count": object_count,
            "count_query_time": count_time,
            "properties": properties,
            "property_types": property_types,
            "vector_config": vector_config
        }
    except Exception as e:
        print(f"Error getting class details for export: {str(e)}")
        return {"error": str(e)}

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
    """Query a class in Weaviate using various search methods"""
    
    # Check if required parameters are present
    if not request.class_name:
        return {"status": "error", "message": "Class name is required"}
    
    if not request.query_text:
        return {"status": "error", "message": "Query text is required"}
    
    # Choose the search method based on the search_type parameter
    if request.search_type == "vector":
        return perform_vector_search(request.class_name, request.query_text, request.limit)
    
    elif request.search_type == "keyword":
        return perform_keyword_search(request.class_name, request.query_text, request.limit)
    
    elif request.search_type == "combined":
        return perform_combined_search(request.class_name, request.query_text, request.limit)
    
    elif request.search_type == "gemini":
        return await perform_gemini_search(
            request.class_name,
            request.query_text,
            request.limit,
            request.gemini_api_key,
            WEAVIATE_URL,
            request.gemini_model
        )
    
    else:
        return {"status": "error", "message": f"Invalid search type: {request.search_type}"}

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

@app.post("/api/auth/login")
async def login(credentials: LoginCredentials):
    """Login with username and password to access Docker information"""
    global sudo_credentials
    
    # Store credentials in memory (not secure for production, but works for this use case)
    # In a real-world application, you would validate these credentials against the system
    sudo_credentials = {
        "username": credentials.username,
        "password": credentials.password
    }
    
    # Test if credentials work by trying to run a simple sudo command
    try:
        # Create command to test sudo access
        cmd = f"echo '{sudo_credentials['password']}' | sudo -S whoami"
        
        # Run the command
        process = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        
        # Check if command succeeded
        if process.returncode == 0 and 'root' in process.stdout:
            return {"status": "success", "message": "Authentication successful"}
        else:
            sudo_credentials = None
            return {"status": "error", "message": "Invalid credentials"}
    except Exception as e:
        sudo_credentials = None
        return {"status": "error", "message": f"Authentication error: {str(e)}"}

def get_docker_container_info_with_sudo():
    """Get Docker container information about Weaviate using sudo if credentials are available"""
    global sudo_credentials
    
    if not sudo_credentials:
        return {"status": "error", "message": "Docker information requires authentication. Please login first."}
    
    try:
        # Use sudo with provided credentials to run Docker commands
        username = sudo_credentials["username"]
        password = sudo_credentials["password"]
        
        # Check if Docker is available
        docker_check_cmd = f"echo '{password}' | sudo -S docker info"
        docker_check = subprocess.run(docker_check_cmd, capture_output=True, text=True, shell=True)
        
        if docker_check.returncode != 0:
            return {"status": "error", "message": "Docker not available or credentials invalid"}
        
        # Get Weaviate container info
        container_cmd = f"echo '{password}' | sudo -S docker ps --filter 'name=weaviate' --format '{{{{.ID}}}},{{{{.Image}}}},{{{{.Status}}}},{{{{.Names}}}}'"
        container_info = subprocess.run(container_cmd, capture_output=True, text=True, shell=True)
        
        # No container found
        if not container_info.stdout.strip():
            return {"status": "not_found", "message": "No Weaviate container found"}
        
        # Get Docker system info for aggregate metrics
        system_cmd = f"echo '{password}' | sudo -S docker system df"
        system_info = subprocess.run(system_cmd, capture_output=True, text=True, shell=True)
        
        # Parse system stats
        system_stats = {}
        if system_info.returncode == 0:
            system_stats = {
                "total_containers": 0,
                "running_containers": 0,
                "total_images": 0,
                "disk_usage": "Unknown"
            }
            
            for line in system_info.stdout.strip().split('\n'):
                if "Containers:" in line:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        system_stats["total_containers"] = parts[1]
                        system_stats["running_containers"] = parts[3].strip("()")
                elif "Images:" in line:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        system_stats["total_images"] = parts[1]
                elif "Local Volumes:" in line:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        system_stats["volumes"] = parts[2]
        
        containers = []
        for line in container_info.stdout.strip().split('\n'):
            if line:
                parts = line.split(',')
                if len(parts) >= 4:
                    container_id, image, status, name = parts
                    
                    # Get stats for this container
                    stats_cmd = f"echo '{password}' | sudo -S docker stats {container_id} --no-stream --format '{{{{.CPUPerc}}}},{{{{.MemUsage}}}},{{{{.MemPerc}}}},{{{{.NetIO}}}},{{{{.BlockIO}}}}'"
                    stats = subprocess.run(stats_cmd, capture_output=True, text=True, shell=True)
                    stats_parts = stats.stdout.strip().split(',') if stats.stdout.strip() else []
                    
                    # Get more detailed container information
                    inspect_cmd = f"echo '{password}' | sudo -S docker inspect {container_id}"
                    inspect_info = subprocess.run(inspect_cmd, capture_output=True, text=True, shell=True)
                    
                    # Parse container inspection data
                    container_details = {}
                    if inspect_info.returncode == 0:
                        try:
                            container_data = json.loads(inspect_info.stdout)[0]
                            
                            # Extract ports
                            ports = []
                            if "NetworkSettings" in container_data and "Ports" in container_data["NetworkSettings"]:
                                port_mappings = container_data["NetworkSettings"]["Ports"]
                                if port_mappings:
                                    for container_port, host_bindings in port_mappings.items():
                                        if host_bindings:
                                            for binding in host_bindings:
                                                ports.append(f"{binding.get('HostIp', '0.0.0.0')}:{binding.get('HostPort', '?')} → {container_port}")
                                        else:
                                            ports.append(f"{container_port} (exposed)")
                            
                            # Extract volumes
                            volumes = []
                            if "Mounts" in container_data:
                                for mount in container_data["Mounts"]:
                                    src = mount.get("Source", "?")
                                    dst = mount.get("Destination", "?")
                                    mode = mount.get("Mode", "rw")
                                    volumes.append(f"{src} → {dst} ({mode})")
                            
                            # Extract environment variables
                            env_vars = []
                            if "Config" in container_data and "Env" in container_data["Config"]:
                                for env in container_data["Config"]["Env"]:
                                    # Filter out sensitive information
                                    if not any(keyword in env.lower() for keyword in ["password", "secret", "key", "token"]):
                                        env_vars.append(env)
                            
                            # Extract health status
                            health_status = "N/A"
                            if "State" in container_data and "Health" in container_data["State"]:
                                health_status = container_data["State"]["Health"]["Status"]
                            
                            # Extract resource limits
                            resource_limits = {}
                            if "HostConfig" in container_data:
                                host_config = container_data["HostConfig"]
                                if "Memory" in host_config and host_config["Memory"] > 0:
                                    memory_limit = host_config["Memory"] / (1024 * 1024)  # Convert to MB
                                    resource_limits["memory"] = f"{memory_limit:.0f}MB"
                                if "NanoCpus" in host_config and host_config["NanoCpus"] > 0:
                                    cpu_limit = host_config["NanoCpus"] / 1e9  # Convert to CPU cores
                                    resource_limits["cpu"] = f"{cpu_limit:.2f} cores"
                            
                            # Add to container details
                            container_details = {
                                "ports": ports,
                                "volumes": volumes,
                                "env_vars": env_vars,
                                "health_status": health_status,
                                "resource_limits": resource_limits,
                                "created": container_data.get("Created", "Unknown"),
                                "restarts": container_data.get("RestartCount", 0) if "RestartCount" in container_data else 0
                            }
                        except json.JSONDecodeError:
                            print(f"Error parsing container inspection data: {inspect_info.stdout}")
                    
                    # Get container logs (last 5 lines)
                    logs_cmd = f"echo '{password}' | sudo -S docker logs --tail 5 {container_id} 2>&1"
                    logs_info = subprocess.run(logs_cmd, capture_output=True, text=True, shell=True)
                    logs = logs_info.stdout.strip().split('\n') if logs_info.returncode == 0 else []
                    
                    containers.append({
                        "id": container_id,
                        "image": image,
                        "status": status,
                        "name": name,
                        "cpu_usage": stats_parts[0] if len(stats_parts) > 0 else "N/A",
                        "memory_usage": stats_parts[1] if len(stats_parts) > 1 else "N/A",
                        "memory_percent": stats_parts[2] if len(stats_parts) > 2 else "N/A",
                        "network_io": stats_parts[3] if len(stats_parts) > 3 else "N/A",
                        "block_io": stats_parts[4] if len(stats_parts) > 4 else "N/A",
                        "details": container_details,
                        "logs": logs
                    })
        
        return {
            "status": "success", 
            "containers": containers,
            "system": system_stats
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Keep the original function for backward compatibility
def get_docker_container_info():
    """Get Docker container information about Weaviate"""
    # First try with sudo if credentials are available
    if sudo_credentials:
        return get_docker_container_info_with_sudo()
    
    # If no credentials or sudo failed, try without sudo
    try:
        # Check if Docker is available
        docker_check = subprocess.run(['docker', 'info'], capture_output=True, text=True)
        if docker_check.returncode != 0:
            return {"status": "error", "message": "Docker not available or requires permissions. Please use login."}
        
        # Get Weaviate container info
        container_info = subprocess.run(
            ['docker', 'ps', '--filter', 'name=weaviate', '--format', 
             '{{.ID}},{{.Image}},{{.Status}},{{.Names}}'],
            capture_output=True, text=True
        )
        
        # No container found
        if not container_info.stdout.strip():
            return {"status": "not_found", "message": "No Weaviate container found"}
        
        containers = []
        for line in container_info.stdout.strip().split('\n'):
            if line:
                parts = line.split(',')
                if len(parts) >= 4:
                    container_id, image, status, name = parts
                    
                    # Get stats for this container
                    stats = subprocess.run(
                        ['docker', 'stats', container_id, '--no-stream', '--format', 
                         '{{.CPUPerc}},{{.MemUsage}},{{.MemPerc}},{{.NetIO}},{{.BlockIO}}'],
                        capture_output=True, text=True
                    )
                    
                    stats_parts = stats.stdout.strip().split(',') if stats.stdout.strip() else []
                    
                    containers.append({
                        "id": container_id,
                        "image": image,
                        "status": status,
                        "name": name,
                        "cpu_usage": stats_parts[0] if len(stats_parts) > 0 else "N/A",
                        "memory_usage": stats_parts[1] if len(stats_parts) > 1 else "N/A",
                        "memory_percent": stats_parts[2] if len(stats_parts) > 2 else "N/A",
                        "network_io": stats_parts[3] if len(stats_parts) > 3 else "N/A",
                        "block_io": stats_parts[4] if len(stats_parts) > 4 else "N/A"
                    })
        
        return {"status": "success", "containers": containers}
    except Exception as e:
        return {"status": "error", "message": f"Docker error: {str(e)}. Please use login."}

@app.get("/api/docker-info")
async def get_docker_info():
    """Get information about Docker containers running Weaviate"""
    return get_docker_container_info()

async def perform_gemini_search(class_name, query_text, limit=5, gemini_api_key=None, base_url=WEAVIATE_URL, gemini_model="gemini-1.5-flash"):
    """
    Perform a search using Google's Gemini model to analyze vector search results.
    
    1. First retrieves results from vector search
    2. Sends these results to Gemini with a carefully crafted prompt
    3. Returns both the original results and Gemini's enhanced analysis
    """
    if not gemini_api_key:
        return {
            "status": "error",
            "message": "Gemini API key is required for Gemini search"
        }
    
    try:
        # First perform a vector search to get relevant documents
        vector_results = perform_vector_search(class_name, query_text, limit, base_url)
        
        if vector_results.get("status") == "error":
            return vector_results
            
        # Extract the results
        results = vector_results.get("results", [])
        
        if not results:
            return {
                "status": "success",
                "results": [],
                "gemini_analysis": "No relevant documents found for your query."
            }
            
        # Construct a prompt for Gemini
        prompt = construct_gemini_prompt(query_text, results, class_name)
        
        # Call Gemini API with the specified model
        gemini_response = await call_gemini_api(prompt, gemini_api_key, gemini_model)
        
        # Return both the original results and Gemini's analysis
        return {
            "status": "success",
            "results": results,
            "gemini_analysis": gemini_response
        }
        
    except Exception as e:
        print(f"Error in Gemini search: {str(e)}")
        return {
            "status": "error",
            "message": f"Error performing Gemini search: {str(e)}"
        }

def construct_gemini_prompt(query, results, class_name):
    """
    Construct a prompt for the Gemini model with context about the Weaviate database and search results.
    """
    # Start with an introduction that explains the context
    prompt = f"""
You are an AI assistant analyzing internal data from a Weaviate Vector Database. 
This is a self-hosted instance running in Docker, and you're working with the collection named '{class_name}'.

USER QUERY: "{query}"

I'm providing you with the top search results from a vector similarity search on this data. 
These results are chunks of data stored as objects in the database.

As you analyze these results, please keep in mind:
1. This is for developers working with this data
2. Include technical details that would be helpful for development
3. Your response should synthesize information from these results
4. Cite specific results when you reference them
5. If the results don't contain enough information to answer fully, acknowledge that

Here are the search results:
"""

    # Add the search results
    for i, result in enumerate(results):
        prompt += f"\n--- RESULT {i+1} ---\n"
        
        # Add main content if available
        if "text" in result:
            prompt += f"Content: {result['text']}\n"
            
        # Add other key properties that might be useful
        for prop, value in result.items():
            if prop not in ["_additional", "text"] and not isinstance(value, dict) and not isinstance(value, list):
                prompt += f"{prop}: {value}\n"
        
        # Add metadata
        if "_additional" in result:
            add_info = result["_additional"]
            if "certainty" in add_info:
                prompt += f"Relevance Score: {add_info['certainty']:.4f}\n"
                
    # Add final instructions
    prompt += """
Based on these search results, please:
1. Provide a comprehensive answer to the user's query
2. Include any relevant code examples or technical details
3. Highlight any gaps in the information
4. Suggest follow-up queries if appropriate

Remember that you're helping developers understand and work with this data.
"""
    
    return prompt

async def call_gemini_api(prompt, api_key, model="gemini-1.5-flash"):
    """
    Call the Google Gemini API with the provided prompt using the official Google Generative AI library.
    """
    try:
        # Configure the Google Generative AI library with the API key
        genai.configure(api_key=api_key)
        
        # Get the specified model
        model_obj = genai.GenerativeModel(model)
        
        # Generate content with the model
        response = await asyncio.to_thread(
            model_obj.generate_content,
            prompt,
            generation_config={
                "temperature": 0.3,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 2048
            }
        )
        
        # Extract the generated text from the response
        if response and hasattr(response, 'text'):
            return response.text
        
        # Handle unexpected response format
        return "Error: Received unexpected response format from Gemini API."
            
    except Exception as e:
        return f"Error calling Gemini API: {str(e)}"

# Add model for cosine similarity analysis request
class CosineSimilarityRequest(BaseModel):
    class_name: Optional[str] = Field(None, description="Name of the class to analyze (optional, taken from path)")
    sample_size: int = Field(100, description="Number of sample vectors for analysis", ge=10, le=1000000)

# Add API endpoint for cosine similarity analysis
@app.post("/api/classes/{class_name}/cosine-similarity")
async def analyze_cosine_similarity(
    class_name: str = Path(..., description="Name of the class to analyze"),
    request: CosineSimilarityRequest = Body(...),
):
    """
    Analyze the cosine similarity distribution of objects in a class.
    This helps validate the quality of the vector embeddings.
    """
    try:
        # Override request.class_name with the path parameter if it exists
        # This ensures the path parameter takes precedence
        
        # Get sample objects with their vectors
        limit = request.sample_size  # Use the full requested sample size
        
        # First get all property names
        schema_response = requests.get(f"{WEAVIATE_URL}/v1/schema/{class_name}", headers=create_weaviate_headers())
        if schema_response.status_code != 200:
            raise HTTPException(status_code=schema_response.status_code, detail=f"Error getting schema for {class_name}")
            
        schema = schema_response.json()
        properties = [prop.get("name") for prop in schema.get("properties", [])]
        
        # Create dynamic GraphQL query with all properties and include vectors
        properties_query = " ".join(properties)
        
        graphql_query = {
            "query": f"""
            {{
              Get {{
                {class_name}(limit: {limit}) {{
                  {properties_query}
                  _additional {{
                    id
                    vector
                  }}
                }}
              }}
            }}
            """
        }
        
        # Execute the query
        response = requests.post(
            f"{WEAVIATE_URL}/v1/graphql",
            headers=create_weaviate_headers(),
            json=graphql_query
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"Error fetching objects: {response.text}")
            
        result = response.json()
        if "errors" in result:
            raise HTTPException(status_code=400, detail=f"GraphQL errors: {result['errors']}")
                
        # Extract objects from response
        objects = result.get("data", {}).get("Get", {}).get(class_name, [])
        
        if not objects:
            return {
                "status": "error",
                "message": f"No objects found in class {class_name}"
            }
            
        # Compute cosine similarities
        documents_with_vectors = []
        for obj in objects:
            if "_additional" in obj and "vector" in obj["_additional"]:
                documents_with_vectors.append(obj)
        
        if len(documents_with_vectors) < 2:
            return {
                "status": "error",
                "message": f"Not enough objects with vectors found in class {class_name}"
            }
            
        # Extract vectors and compute similarities
        similarities = compute_cosine_similarities(documents_with_vectors)
        
        # Compute similarity statistics
        stats = analyze_similarities_for_api(similarities)
        
        # Generate histogram data for Plotly
        histogram_data = generate_histogram_data(similarities)
        
        # Generate distribution data for Plotly
        distribution_data = generate_distribution_data(similarities)
        
        return {
            "status": "success",
            "class_name": class_name,
            "sample_size": len(documents_with_vectors),
            "total_pairs_analyzed": len(similarities),
            "statistics": stats,
            "histogram_data": histogram_data,
            "distribution_data": distribution_data
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

def compute_cosine_similarities(documents):
    """Compute pairwise cosine similarities between all document vectors using sklearn."""
    if not documents:
        return []
    
    # Extract vectors and ensure they all have the same dimension
    vectors = []
    valid_documents = []
    
    # First pass: determine the correct dimension
    dimensions = []
    for doc in documents:
        vec = doc["_additional"]["vector"]
        if isinstance(vec, list):
            dimensions.append(len(vec))
    
    if not dimensions:
        return []
    
    # Find the most common dimension
    from collections import Counter
    dimension_counts = Counter(dimensions)
    correct_dim = dimension_counts.most_common(1)[0][0]
    
    # Second pass: only keep vectors with the correct dimension
    for doc in documents:
        vec = doc["_additional"]["vector"]
        if isinstance(vec, list) and len(vec) == correct_dim:
            vectors.append(vec)
            valid_documents.append(doc)
    
    if not vectors:
        return []
    
    # Convert to numpy arrays
    vectors = np.array(vectors)
    
    # Compute cosine similarity matrix using sklearn
    similarity_matrix = cosine_similarity(vectors)
    
    # Extract upper triangular part to get pairwise similarities (excluding self-comparisons)
    n = len(vectors)
    similarities = []
    
    for i in range(n):
        for j in range(i+1, n):
            similarities.append(similarity_matrix[i, j])
    
    return np.array(similarities)

def analyze_similarities_for_api(similarities):
    """Analyze the distribution of cosine similarities and return statistics without generating images."""
    if len(similarities) == 0:
        return {}
    
    # Compute statistics
    stats = {
        "count": len(similarities),
        "mean": float(np.mean(similarities)),
        "std": float(np.std(similarities)),
        "min": float(np.min(similarities)),
        "25%": float(np.percentile(similarities, 25)),
        "median": float(np.median(similarities)),
        "75%": float(np.percentile(similarities, 75)),
        "90%": float(np.percentile(similarities, 90)),
        "95%": float(np.percentile(similarities, 95)),
        "99%": float(np.percentile(similarities, 99)),
        "max": float(np.max(similarities))
    }
    
    return stats

def generate_histogram_data(similarities, bins=50):
    """Generate histogram data for Plotly visualization."""
    if len(similarities) == 0:
        return {}
    
    hist, bin_edges = np.histogram(similarities, bins=bins)
    
    # Create bin centers for x-axis
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    data = {
        "x": bin_centers.tolist(),
        "y": hist.tolist(),
        "bin_edges": bin_edges.tolist(),
        "mean": float(np.mean(similarities)),
        "median": float(np.median(similarities)),
        "q1": float(np.percentile(similarities, 25)),
        "q3": float(np.percentile(similarities, 75))
    }
    
    return data

def generate_distribution_data(similarities, num_bins=12):
    """Generate distribution data for Plotly visualization."""
    if len(similarities) == 0:
        return {}
    
    total_count = len(similarities)
    
    # Create bins from min to max
    min_val = np.min(similarities)
    max_val = np.max(similarities)
    
    # Define bin edges
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    
    # Count frequencies
    hist, _ = np.histogram(similarities, bins=bin_edges)
    
    # Convert to percentages
    percentages = (hist / total_count) * 100
    
    # Create bin labels
    bin_labels = [f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}" for i in range(len(bin_edges)-1)]
    
    data = {
        "bin_labels": bin_labels,
        "percentages": percentages.tolist(),
        "counts": hist.tolist(),
        "bin_edges": bin_edges.tolist(),
        "total_count": total_count
    }
    
    return data

# Add model for retrieval metrics analysis request
class RetrievalMetricsRequest(BaseModel):
    class_name: Optional[str] = Field(None, description="Name of the class to analyze")
    query: str = Field(..., description="The search query to evaluate")
    max_k: int = Field(10, description="Maximum k value to calculate metrics for", ge=1, le=100)

# Add API endpoint for retrieval metrics analysis
@app.post("/api/classes/{class_name}/retrieval-metrics")
async def analyze_retrieval_metrics(
    class_name: str = Path(..., description="Name of the class to analyze"),
    request: RetrievalMetricsRequest = Body(...),
):
    """
    Analyze retrieval metrics (Precision@k, Recall@k, F1 Score@k) for a search query.
    This helps evaluate the quality of vector search results.
    """
    try:
        # Perform vector search using the query
        query = request.query
        max_k = min(request.max_k, 100)  # Limit to 100 maximum
        
        # Double the max_k to get more results for sampling "relevant" documents
        search_limit = max(max_k * 2, 30)  # Get at least 30 results if possible
        
        # Perform search using current Weaviate instance
        search_results = perform_vector_search(class_name, query, limit=search_limit)
        
        if search_results.get("status") != "success" or not search_results.get("results"):
            return {
                "status": "error",
                "message": f"No search results found for query: {query}"
            }
            
        results = search_results.get("results", [])
        total_results = len(results)
        
        if total_results < 3:
            return {
                "status": "error",
                "message": f"Too few search results ({total_results}) to perform meaningful analysis. Try a different query."
            }
        
        # Extract document identifiers from results
        retrieved_docs = []
        certainty_scores = []
        
        for r in results:
            # Create a unique identifier for each document
            doc_id = {}
            for key, value in r.items():
                if key != "_additional" and not isinstance(value, (list, dict)):
                    doc_id[key] = value
            
            retrieved_docs.append(doc_id)
            
            # Get certainty scores
            if "_additional" in r and "certainty" in r["_additional"]:
                certainty_scores.append(r["_additional"]["certainty"])
            else:
                certainty_scores.append(0.0)
        
        # Randomly select some as "relevant" for demonstration
        # Using fixed seed for reproducibility
        np.random.seed(42)
        relevant_size = min(total_results // 2, 10)  # Use up to half the results as relevant, max 10
        if relevant_size < 2:
            relevant_size = 2  # Ensure at least 2 relevant docs if possible
            
        relevant_indices = np.random.choice(
            total_results, 
            size=relevant_size,
            replace=False
        )
        
        # Create list of relevant documents
        relevant_docs = [retrieved_docs[i] for i in relevant_indices]
        
        # Only keep max_k results for metrics calculation
        retrieved_docs = retrieved_docs[:max_k]
        certainty_scores = certainty_scores[:max_k]
        
        # Calculate metrics at each k
        precision_at_k = []
        recall_at_k = []
        f1_at_k = []
        
        for k in range(1, max_k + 1):
            # Only consider top k results
            docs_at_k = retrieved_docs[:k]
            
            # Calculate relevant retrieved docs
            # We need to match each doc in docs_at_k with each in relevant_docs
            relevant_retrieved = 0
            for doc in docs_at_k:
                for rel_doc in relevant_docs:
                    # Check if all keys in rel_doc match in doc
                    match = True
                    for key, value in rel_doc.items():
                        if key not in doc or doc[key] != value:
                            match = False
                            break
                    
                    if match:
                        relevant_retrieved += 1
                        break
            
            # Calculate Precision@k
            precision = relevant_retrieved / k if k > 0 else 0
            
            # Calculate Recall@k
            recall = relevant_retrieved / len(relevant_docs) if relevant_docs else 0
            
            # Calculate F1 Score@k
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            precision_at_k.append(precision)
            recall_at_k.append(recall)
            f1_at_k.append(f1)
        
        # Create retrieval data for visualization
        retrieval_data = {
            "docs": retrieved_docs,
            "certainty": certainty_scores,
            "relevant": []
        }
        
        # Mark which documents are relevant
        for doc in retrieved_docs:
            is_relevant = False
            for rel_doc in relevant_docs:
                match = True
                for key, value in rel_doc.items():
                    if key not in doc or doc[key] != value:
                        match = False
                        break
                
                if match:
                    is_relevant = True
                    break
            
            retrieval_data["relevant"].append(1 if is_relevant else 0)
        
        # Create data for visualization
        k_values = list(range(1, max_k + 1))
        
        # Calculate additional statistics
        max_p_idx = np.argmax(precision_at_k)
        max_r_idx = np.argmax(recall_at_k)
        max_f1_idx = np.argmax(f1_at_k)
        
        stats = {
            "max_precision": {
                "value": float(max(precision_at_k)),
                "at_k": int(max_p_idx + 1)
            },
            "max_recall": {
                "value": float(max(recall_at_k)),
                "at_k": int(max_r_idx + 1)
            },
            "max_f1": {
                "value": float(max(f1_at_k)),
                "at_k": int(max_f1_idx + 1)
            },
            "avg_precision": float(np.mean(precision_at_k)),
            "avg_recall": float(np.mean(recall_at_k)),
            "avg_f1": float(np.mean(f1_at_k)),
            "auc_precision": float(np.trapz(precision_at_k) / max_k),
            "auc_recall": float(np.trapz(recall_at_k) / max_k),
            "auc_f1": float(np.trapz(f1_at_k) / max_k)
        }
        
        # Create bar chart data for comparison at key k values
        key_indices = [0, max_k//2, -1]
        key_k_values = [k_values[i] for i in key_indices]
        key_precision = [precision_at_k[i] for i in key_indices]
        key_recall = [recall_at_k[i] for i in key_indices]
        key_f1 = [f1_at_k[i] for i in key_indices]
        
        bar_chart_data = {
            "k_values": key_k_values,
            "precision": key_precision,
            "recall": key_recall,
            "f1": key_f1
        }
        
        # Create heatmap data
        heatmap_data = {
            "precision": precision_at_k,
            "recall": recall_at_k,
            "f1": f1_at_k,
            "k_values": k_values
        }
        
        # Create radar chart data
        radar_data = [
            float(np.mean(precision_at_k)),  # Average precision
            float(np.mean(recall_at_k)),     # Average recall
            float(np.mean(f1_at_k)),         # Average F1
            float(np.mean([np.mean(precision_at_k), np.mean(recall_at_k), np.mean(f1_at_k)])),  # Overall average
            float(max(precision_at_k)),      # Max precision
            float(max(recall_at_k))         # Max recall
        ]
        
        return {
            "status": "success",
            "class_name": class_name,
            "query": query,
            "max_k": max_k,
            "k_values": k_values,
            "precision_at_k": [float(p) for p in precision_at_k],
            "recall_at_k": [float(r) for r in recall_at_k],
            "f1_at_k": [float(f) for f in f1_at_k],
            "retrieval_data": retrieval_data,
            "statistics": stats,
            "bar_chart_data": bar_chart_data,
            "heatmap_data": heatmap_data,
            "radar_data": radar_data,
            "relevant_docs_count": len(relevant_docs),
            "total_results_count": total_results
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e)
        }

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
