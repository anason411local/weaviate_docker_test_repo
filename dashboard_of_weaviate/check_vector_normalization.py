#!/usr/bin/env python3

import os
import requests
import json
import numpy as np
from dotenv import load_dotenv
import argparse

# Load environment variables
load_dotenv()

# Get Weaviate connection info from environment
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8090")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

def create_weaviate_headers():
    """Create headers for Weaviate API requests."""
    headers = {"Content-Type": "application/json"}
    if WEAVIATE_API_KEY:
        headers["Authorization"] = f"Bearer {WEAVIATE_API_KEY}"
    return headers

def check_vector_normalization(vector):
    """Check if a vector is normalized (has unit length)."""
    vec_np = np.array(vector)
    magnitude = np.linalg.norm(vec_np)
    return abs(magnitude - 1.0) < 0.01  # Allow for small floating point differences

def fetch_sample_vectors(class_name, limit=100):
    """Fetch a sample of vectors from the specified class."""
    headers = create_weaviate_headers()
    
    graphql_query = f"""
    {{
      Get {{
        {class_name}(
          limit: {limit}
        ) {{
          _additional {{
            id
            vector
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
                return []
                
            documents = result.get("data", {}).get("Get", {}).get(class_name, [])
            print(f"Retrieved {len(documents)} documents with vectors")
            return documents
        else:
            print(f"Error fetching vectors: {response.status_code}")
            print(response.text)
            return []
    except Exception as e:
        print(f"Error fetching vectors: {e}")
        return []

def analyze_normalization(documents):
    """Analyze what percentage of vectors are normalized."""
    if not documents:
        print("No documents to analyze")
        return
    
    normalized_count = 0
    non_normalized_count = 0
    vector_lengths = []
    
    for doc in documents:
        if "_additional" in doc and "vector" in doc["_additional"]:
            vec = doc["_additional"]["vector"]
            if isinstance(vec, list):
                vec_np = np.array(vec)
                magnitude = np.linalg.norm(vec_np)
                vector_lengths.append(magnitude)
                
                if check_vector_normalization(vec):
                    normalized_count += 1
                else:
                    non_normalized_count += 1
    
    total = normalized_count + non_normalized_count
    if total == 0:
        print("No vectors found in documents")
        return
    
    print(f"Total vectors analyzed: {total}")
    print(f"Normalized vectors: {normalized_count} ({normalized_count/total*100:.2f}%)")
    print(f"Non-normalized vectors: {non_normalized_count} ({non_normalized_count/total*100:.2f}%)")
    
    if vector_lengths:
        print(f"Average vector length: {np.mean(vector_lengths):.6f}")
        print(f"Min vector length: {np.min(vector_lengths):.6f}")
        print(f"Max vector length: {np.max(vector_lengths):.6f}")
        print(f"Std dev of vector lengths: {np.std(vector_lengths):.6f}")

def main():
    parser = argparse.ArgumentParser(description="Check vector normalization in Weaviate database")
    parser.add_argument("--class-name", default="Area_expansion_Dep_Anas", 
                        help="Name of the class to check (default: Area_expansion_Dep_Anas)")
    parser.add_argument("--limit", type=int, default=100, 
                        help="Number of vectors to sample (default: 100)")
    parser.add_argument("--url", 
                        help="Weaviate URL (default: from .env or http://localhost:8090)")
    parser.add_argument("--api-key", 
                        help="Weaviate API key (default: from .env)")
    
    args = parser.parse_args()
    
    # Override defaults if provided
    global WEAVIATE_URL, WEAVIATE_API_KEY
    if args.url:
        WEAVIATE_URL = args.url
    if args.api_key:
        WEAVIATE_API_KEY = args.api_key
    
    print(f"Checking vector normalization in {WEAVIATE_URL}, class {args.class_name}")
    docs = fetch_sample_vectors(args.class_name, args.limit)
    analyze_normalization(docs)

if __name__ == "__main__":
    main() 