#!/usr/bin/env python3

import os
import sys
from dotenv import load_dotenv
import json
import numpy as np
from fastapi import FastAPI, APIRouter, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn

# Load environment variables
load_dotenv()

# Constants for similarity adjustment
SELF_HOST_MEAN = 0.52  # Typical self-hosted mean similarity
CLOUD_MEAN = 0.64      # Typical cloud mean similarity
SELF_HOST_STD = 0.07   # Typical self-hosted standard deviation
CLOUD_STD = 0.07       # Typical cloud standard deviation

class CloudSimilarityMiddleware(BaseHTTPMiddleware):
    """Middleware to adjust similarity scores to be more like Weaviate Cloud."""
    
    async def dispatch(self, request: Request, call_next):
        # Process the request through the normal application
        response = await call_next(request)
        
        # Check if this is a JSON response from certain API endpoints
        path = request.url.path
        if (path.startswith("/api/") and 
            "content-type" in response.headers and 
            "application/json" in response.headers["content-type"]):
            
            # Read the response body
            body = b""
            async for chunk in response.body_iterator:
                body += chunk
                
            # Parse the JSON
            try:
                data = json.loads(body.decode())
                
                # Check if this is a search result with matches containing certainty values
                modified = False
                
                # For vector search results endpoints
                if isinstance(data, dict) and "matches" in data and isinstance(data["matches"], list):
                    for item in data["matches"]:
                        if "_additional" in item and "certainty" in item["_additional"]:
                            # Store original value
                            original = item["_additional"]["certainty"]
                            # Apply adjustment
                            item["_additional"]["original_certainty"] = original
                            item["_additional"]["certainty"] = adjust_similarity_score(original)
                    modified = True
                    # Add adjustment metadata
                    data["cloud_adjusted"] = True
                    data["adjustment_info"] = {
                        "self_host_mean": SELF_HOST_MEAN,
                        "cloud_mean": CLOUD_MEAN,
                        "method": "linear_transform"
                    }
                
                # For individual items with certainty
                elif isinstance(data, dict) and "_additional" in data and "certainty" in data["_additional"]:
                    original = data["_additional"]["certainty"]
                    data["_additional"]["original_certainty"] = original
                    data["_additional"]["certainty"] = adjust_similarity_score(original)
                    data["cloud_adjusted"] = True
                    modified = True
                
                # For array of items with certainty
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "_additional" in item and "certainty" in item["_additional"]:
                            original = item["_additional"]["certainty"]
                            item["_additional"]["original_certainty"] = original
                            item["_additional"]["certainty"] = adjust_similarity_score(original)
                            modified = True
                    if modified:
                        data = {
                            "results": data,
                            "cloud_adjusted": True,
                            "adjustment_info": {
                                "self_host_mean": SELF_HOST_MEAN,
                                "cloud_mean": CLOUD_MEAN
                            }
                        }
                
                # If modifications were made, create a new response
                if modified:
                    return Response(
                        content=json.dumps(data).encode(),
                        status_code=response.status_code,
                        headers=dict(response.headers),
                        media_type="application/json"
                    )
                
            except Exception as e:
                # If there's an error parsing or modifying, return the original response
                print(f"Error in middleware: {e}")
                pass
            
            # Return the original response body
            return Response(
                content=body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.headers["content-type"]
            )
        
        # For non-JSON responses, or paths we don't want to modify
        return response

def adjust_similarity_score(score):
    """Adjust a self-hosted similarity score to be more like cloud."""
    # Using linear transformation to shift distribution
    # Convert self-hosted score to standard z-score, then rescale to cloud distribution
    z_score = (score - SELF_HOST_MEAN) / SELF_HOST_STD
    cloud_adjusted = CLOUD_MEAN + (z_score * CLOUD_STD)
    # Ensure score remains in valid range [0, 1]
    return max(0.0, min(1.0, cloud_adjusted))

def apply_middleware_to_app(app: FastAPI):
    """Apply the cloud similarity middleware to an existing FastAPI app."""
    app.add_middleware(CloudSimilarityMiddleware)
    return app

def create_demo_app():
    """Create a simple demo app to test the middleware."""
    app = FastAPI(title="Cloud Similarity Demo API")
    
    @app.get("/")
    async def root():
        return {"message": "Cloud Similarity API Adapter Demo"}
    
    @app.get("/api/test")
    async def test():
        return {
            "matches": [
                {
                    "text": "Sample document 1",
                    "_additional": {
                        "certainty": 0.52
                    }
                },
                {
                    "text": "Sample document 2",
                    "_additional": {
                        "certainty": 0.48
                    }
                }
            ]
        }
    
    # Apply middleware
    return apply_middleware_to_app(app)

def main():
    """Run a demo server with the middleware."""
    # Create and configure app
    app = create_demo_app()
    
    # Configure server settings
    host = "127.0.0.1"
    port = 8080
    
    print(f"Starting Cloud Similarity API Adapter demo server at http://{host}:{port}")
    print(f"Test URL: http://{host}:{port}/api/test")
    print(f"Adjusting similarity scores: self-hosted mean {SELF_HOST_MEAN} → cloud-like mean {CLOUD_MEAN}")
    
    # Run server
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()

# Instructions to apply to existing application in new_dashboard.py:
"""
To apply this middleware to your existing Weaviate dashboard application:

1. Import the middleware in your new_dashboard.py:
   ```python
   from cloud_similarity_api_adapter import apply_middleware_to_app
   ```

2. Before starting the app, apply the middleware:
   ```python
   # After creating your app but before running it
   app = FastAPI(...)
   
   # Apply middleware
   app = apply_middleware_to_app(app)
   
   # Then run the app
   ```

3. This will automatically adjust all similarity scores to be more cloud-like
   without having to modify your vector database or individual API endpoints.
""" 