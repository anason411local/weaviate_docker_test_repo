#!/usr/bin/env python3

from fastapi import FastAPI, HTTPException, Query, Path, BackgroundTasks, Request, status
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
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import json
import time
from datetime import datetime
import uuid

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
templates = Jinja2Templates(directory="dashboard_of_weaviate/templates")
app.mount("/static", StaticFiles(directory="dashboard_of_weaviate/static"), name="static")

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

class InspectionRequest(BaseModel):
    include_samples: bool = Field(False, description="Whether to include sample objects in the report")
    run_benchmarks: bool = Field(False, description="Whether to run benchmark queries")

# Frontend routes
@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    """Serve the dashboard frontend"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/inspect/{report_id}", response_class=HTMLResponse)
async def get_inspection_page(request: Request, report_id: str):
    """Serve the inspection details page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/objects/{class_name}", response_class=HTMLResponse)
async def get_objects_page(request: Request, class_name: str):
    """Serve the objects browser page"""
    return templates.TemplateResponse("index.html", {"request": request})

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

# API Routes
@app.get("/api")
async def api_root():
    return {"message": "Welcome to Weaviate Dashboard API"}

@app.get("/api/health")
async def check_health():
    """Check if the Weaviate instance is healthy"""
    try:
        response = requests.get(f"{WEAVIATE_URL}/v1/.well-known/ready", headers=create_weaviate_headers())
        if response.status_code == 200:
            return {"status": "ready", "message": "Weaviate instance is ready"}
        else:
            return {"status": "not_ready", "message": f"Weaviate instance not ready: {response.status_code}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error connecting to Weaviate: {str(e)}")

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
        raise HTTPException(status_code=500, detail=f"Error getting metadata: {str(e)}")

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
    limit: int = Query(10, description="Maximum number of objects to return"),
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
                    "limit": limit
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

if __name__ == "__main__":
    uvicorn.run("new_dashboard:app", host="0.0.0.0", port=8000, reload=True)
