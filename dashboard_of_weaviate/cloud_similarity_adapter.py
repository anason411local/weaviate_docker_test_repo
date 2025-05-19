#!/usr/bin/env python3

import os
import numpy as np
from dotenv import load_dotenv
import requests
import json
import argparse
from transformers import AutoModel, AutoTokenizer
import torch

# Load environment variables
load_dotenv()

# Get Weaviate connection info from environment
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8090")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
COLLECTION_NAME = "Area_expansion_Dep_Anas"

# Constants for similarity adjustment
SELF_HOST_MEAN = 0.52  # Typical self-hosted mean
CLOUD_MEAN = 0.64      # Typical cloud mean
SELF_HOST_STD = 0.07   # Typical self-hosted std
CLOUD_STD = 0.07       # Typical cloud std

def create_weaviate_headers():
    """Create headers for Weaviate API requests."""
    headers = {"Content-Type": "application/json"}
    if WEAVIATE_API_KEY:
        headers["Authorization"] = f"Bearer {WEAVIATE_API_KEY}"
    return headers

def adjust_similarity_score(score):
    """Adjust a self-hosted similarity score to be more like cloud."""
    # Using linear transformation to shift distribution
    # Convert self-hosted score to standard z-score, then rescale to cloud distribution
    z_score = (score - SELF_HOST_MEAN) / SELF_HOST_STD
    cloud_adjusted = CLOUD_MEAN + (z_score * CLOUD_STD)
    # Ensure score remains in valid range [0, 1]
    return max(0.0, min(1.0, cloud_adjusted))

def vector_search(query_text, limit=5):
    """Perform vector search with adjusted cloud-like similarity scores."""
    # First, get embedding for the query
    embedding = generate_embedding(query_text)
    if embedding is None:
        return {"error": "Failed to generate embedding for query"}
    
    # Now use the embedding for search
    headers = create_weaviate_headers()
    
    # GraphQL query for vector search
    graphql_query = f"""
    {{
      Get {{
        {COLLECTION_NAME}(
          nearVector: {{
            vector: {embedding}
          }}
          limit: {limit}
        ) {{
          text
          filename
          page
          source
          _additional {{
            id
            certainty
            distance
          }}
        }}
      }}
    }}
    """
    
    try:
        response = requests.post(
            f"{WEAVIATE_URL}/v1/graphql",
            headers=headers,
            json={"query": graphql_query}
        )
        
        if response.status_code == 200:
            result = response.json()
            if "errors" in result:
                print(f"GraphQL errors: {result['errors']}")
                return {"error": str(result["errors"])}
                
            documents = result.get("data", {}).get("Get", {}).get(COLLECTION_NAME, [])
            
            # Adjust similarity scores to be more cloud-like
            for doc in documents:
                if "_additional" in doc:
                    # Certainty is already between 0-1 where 1 is perfect match
                    original_certainty = doc["_additional"].get("certainty", 0)
                    # Apply adjustment
                    doc["_additional"]["original_certainty"] = original_certainty
                    doc["_additional"]["certainty"] = adjust_similarity_score(original_certainty)
            
            return {
                "matches": documents,
                "adjusted": True,
                "adjustment_info": {
                    "self_host_mean": SELF_HOST_MEAN,
                    "cloud_mean": CLOUD_MEAN,
                    "method": "linear_transform"
                }
            }
        else:
            print(f"Error in vector search: {response.status_code}")
            print(response.text)
            return {"error": f"Search error: {response.status_code}"}
    except Exception as e:
        print(f"Exception in vector search: {e}")
        return {"error": str(e)}

def generate_embedding(text):
    """Generate embedding for query text."""
    try:
        # Try to use existing embedding endpoint first
        headers = create_weaviate_headers()
        response = requests.post(
            f"{WEAVIATE_URL}/v1/modules/text2vec-transformers/vectorize",
            headers=headers,
            json={"text": text}
        )
        
        if response.status_code == 200:
            return response.json().get("vector")
        
        # If that fails, try to use local model
        print("Using local model to generate embedding as Weaviate endpoint failed")
        return generate_local_embedding(text)
    except Exception as e:
        print(f"Error generating embedding: {e}")
        # Try local fallback
        return generate_local_embedding(text)

def generate_local_embedding(text):
    """Generate embedding using local model."""
    try:
        # Use the BAAI/bge-m3 model by default
        model_name = "BAAI/bge-m3"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Prepare the embeddings with the correct format
        encoded_input = tokenizer(['Represent this document for retrieval: ' + text], 
                                 padding=True, truncation=True, 
                                 return_tensors='pt', max_length=512)
        
        # Get embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)
            # Use CLS token embedding
            embeddings = model_output.last_hidden_state[:, 0, :].numpy()
        
        # Normalize the embedding to unit length
        embedding = embeddings[0]
        embedding_norm = np.linalg.norm(embedding)
        if embedding_norm > 0:
            embedding = embedding / embedding_norm
            
        return embedding.tolist()
    except Exception as e:
        print(f"Error generating local embedding: {e}")
        return None

def main():
    # Global variables that will be modified
    global SELF_HOST_MEAN, CLOUD_MEAN
    
    parser = argparse.ArgumentParser(description="Perform vector search with cloud-like similarity scores")
    parser.add_argument("--query", required=True, help="Search query text")
    parser.add_argument("--limit", type=int, default=5, help="Number of results to return")
    parser.add_argument("--self-mean", type=float, default=SELF_HOST_MEAN, 
                        help=f"Self-hosted mean similarity (default: {SELF_HOST_MEAN})")
    parser.add_argument("--cloud-mean", type=float, default=CLOUD_MEAN, 
                        help=f"Target cloud mean similarity (default: {CLOUD_MEAN})")
    
    args = parser.parse_args()
    
    # Update global constants if custom values provided
    SELF_HOST_MEAN = args.self_mean
    CLOUD_MEAN = args.cloud_mean
    
    print(f"Performing search with query: '{args.query}'")
    print(f"Adjusting similarity scores from mean {SELF_HOST_MEAN} to cloud-like mean {CLOUD_MEAN}")
    
    results = vector_search(args.query, args.limit)
    
    if "error" in results:
        print(f"Error: {results['error']}")
    else:
        print("\nSearch Results (adjusted to cloud-like similarity):")
        print("----------------------------------------------------")
        
        for i, doc in enumerate(results["matches"]):
            original = doc["_additional"].get("original_certainty", "N/A")
            adjusted = doc["_additional"].get("certainty", "N/A")
            
            print(f"\n[{i+1}] Score: {adjusted:.4f} (original: {original:.4f})")
            print(f"    Text: {doc.get('text', '')[:150]}...")
            print(f"    File: {doc.get('filename', 'Unknown')}, Page: {doc.get('page', 'Unknown')}")

if __name__ == "__main__":
    main() 