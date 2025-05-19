#!/usr/bin/env python3

import os
import requests
import json
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
import argparse
import time

# Load environment variables
load_dotenv()

def create_weaviate_headers(api_key=None):
    """Create headers for Weaviate API requests."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers

def fetch_objects_with_vectors(url, api_key, class_name="Area_expansion_Dep_Anas", limit=100, batch_size=25):
    """Fetch objects with their vectors from Weaviate."""
    headers = create_weaviate_headers(api_key)
    
    # First count total objects to determine pagination
    count_query = f"""
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
    
    try:
        response = requests.post(
            f"{url}/v1/graphql",
            headers=headers,
            json={"query": count_query}
        )
        
        if response.status_code == 200:
            result = response.json()
            if "errors" in result:
                print(f"GraphQL errors: {result['errors']}")
                return []
                
            total_count = result.get("data", {}).get("Aggregate", {}).get(class_name, [{}])[0].get("meta", {}).get("count", 0)
            print(f"Total objects in {class_name}: {total_count}")
            
            # Limit to the specified number if needed
            total_count = min(total_count, limit)
            
            # Fetch in batches
            all_objects = []
            
            for offset in tqdm(range(0, total_count, batch_size), desc="Fetching objects"):
                # Fetch a batch of objects with their vectors
                batch_size_adj = min(batch_size, total_count - offset)
                
                graphql_query = f"""
                {{
                  Get {{
                    {class_name}(
                      limit: {batch_size_adj}
                      offset: {offset}
                    ) {{
                      _additional {{
                        id
                        vector
                      }}
                    }}
                  }}
                }}
                """
                
                batch_response = requests.post(
                    f"{url}/v1/graphql",
                    headers=headers,
                    json={"query": graphql_query}
                )
                
                if batch_response.status_code == 200:
                    batch_result = batch_response.json()
                    if "errors" in batch_result:
                        print(f"GraphQL errors: {batch_result['errors']}")
                        continue
                        
                    batch_objects = batch_result.get("data", {}).get("Get", {}).get(class_name, [])
                    all_objects.extend(batch_objects)
                else:
                    print(f"Error fetching batch at offset {offset}: {batch_response.status_code}")
                    print(batch_response.text)
            
            return all_objects
        else:
            print(f"Error counting objects: {response.status_code}")
            print(response.text)
            return []
    except Exception as e:
        print(f"Error fetching objects: {e}")
        return []

def normalize_vector(vector):
    """Normalize a vector to unit length."""
    vec_np = np.array(vector)
    norm = np.linalg.norm(vec_np)
    if norm > 0:
        return (vec_np / norm).tolist()
    return vector

def update_object_vector(url, api_key, class_name, object_id, vector):
    """Update an object's vector in Weaviate."""
    headers = create_weaviate_headers(api_key)
    
    try:
        # Use the PATCH endpoint to update the object
        update_url = f"{url}/v1/objects/{class_name}/{object_id}"
        
        # Weaviate expects the vector in a specific format
        update_data = {
            "vector": vector
        }
        
        response = requests.patch(
            update_url,
            headers=headers,
            json=update_data
        )
        
        if response.status_code in [200, 204]:
            return True
        else:
            print(f"Error updating object {object_id}: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"Error updating object {object_id}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Normalize vectors in a Weaviate instance')
    parser.add_argument('--url', type=str, required=True, 
                        help='URL for Weaviate instance')
    parser.add_argument('--api-key', type=str, 
                        help='API key for Weaviate instance')
    parser.add_argument('--class-name', type=str, default="Area_expansion_Dep_Anas", 
                        help='Class name to process (default: Area_expansion_Dep_Anas)')
    parser.add_argument('--limit', type=int, default=1000, 
                        help='Maximum number of objects to process (default: 1000)')
    parser.add_argument('--batch-size', type=int, default=25, 
                        help='Number of objects to fetch in each batch (default: 25)')
    parser.add_argument('--dry-run', action='store_true', 
                        help='Check normalization without updating objects')
    
    args = parser.parse_args()
    
    # Fetch objects with vectors
    print(f"Fetching objects from Weaviate at {args.url}...")
    objects = fetch_objects_with_vectors(args.url, args.api_key, args.class_name, args.limit, args.batch_size)
    
    if not objects:
        print("No objects found. Exiting.")
        return
    
    # Check normalization status
    normalized_count = 0
    non_normalized_count = 0
    total_with_vectors = 0
    
    for obj in objects:
        if "_additional" in obj and "vector" in obj["_additional"]:
            vec = obj["_additional"]["vector"]
            if isinstance(vec, list):
                total_with_vectors += 1
                # Calculate vector magnitude
                vec_np = np.array(vec)
                magnitude = np.linalg.norm(vec_np)
                # Check if the magnitude is close to 1.0 (normalized)
                if abs(magnitude - 1.0) < 0.01:  # Allow for small floating point differences
                    normalized_count += 1
                else:
                    non_normalized_count += 1
    
    print(f"\nNormalization status:")
    print(f"Total objects with vectors: {total_with_vectors}")
    normalized_pct = normalized_count/total_with_vectors*100 if total_with_vectors > 0 else 0
    non_normalized_pct = non_normalized_count/total_with_vectors*100 if total_with_vectors > 0 else 0
    print(f"Already normalized: {normalized_count} ({normalized_pct:.2f}%)")
    print(f"Need normalization: {non_normalized_count} ({non_normalized_pct:.2f}%)")
    
    if args.dry_run:
        print("\nDry run completed. No objects were updated.")
        return
    
    if non_normalized_count == 0:
        print("\nAll vectors are already normalized. No updates needed.")
        return
    
    # Normalize vectors and update objects
    print(f"\nNormalizing {non_normalized_count} vectors...")
    
    success_count = 0
    error_count = 0
    
    for obj in tqdm(objects, desc="Updating vectors"):
        if "_additional" in obj and "vector" in obj["_additional"] and "id" in obj["_additional"]:
            vec = obj["_additional"]["vector"]
            obj_id = obj["_additional"]["id"]
            
            if isinstance(vec, list):
                # Calculate vector magnitude
                vec_np = np.array(vec)
                magnitude = np.linalg.norm(vec_np)
                
                # Only normalize if needed
                if abs(magnitude - 1.0) >= 0.01:
                    # Normalize the vector
                    normalized_vec = normalize_vector(vec)
                    
                    # Update the object
                    success = update_object_vector(args.url, args.api_key, args.class_name, obj_id, normalized_vec)
                    
                    if success:
                        success_count += 1
                    else:
                        error_count += 1
                    
                    # Rate limiting - be gentle with the API
                    time.sleep(0.1)
    
    print(f"\nNormalization complete:")
    print(f"Successfully normalized: {success_count}")
    print(f"Errors: {error_count}")

if __name__ == "__main__":
    main() 