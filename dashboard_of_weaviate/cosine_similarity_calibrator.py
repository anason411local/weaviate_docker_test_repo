#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Set up plotting aesthetics
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Define color scheme
MAIN_COLOR = "#4287f5"  # Blue
CORRECTED_COLOR = "#42f54b"  # Green
CLOUD_COLOR = "#f54242"  # Red

# Set up Weaviate connection
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8090")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
COLLECTION_NAME = "Area_expansion_Dep_Anas" 

def create_weaviate_headers():
    """Create headers for Weaviate API requests."""
    headers = {"Content-Type": "application/json"}
    if WEAVIATE_API_KEY:
        headers["Authorization"] = f"Bearer {WEAVIATE_API_KEY}"
    return headers

def fetch_vectors(limit=500):
    """Fetch vectors from Weaviate."""
    headers = create_weaviate_headers()
    
    graphql_query = f"""
    {{
      Get {{
        {COLLECTION_NAME}(
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
                
            documents = result.get("data", {}).get("Get", {}).get(COLLECTION_NAME, [])
            
            # Filter out documents without vectors
            vectors_with_ids = []
            for doc in documents:
                if "_additional" in doc and "vector" in doc["_additional"]:
                    vectors_with_ids.append({
                        "id": doc["_additional"]["id"],
                        "vector": doc["_additional"]["vector"]
                    })
            
            print(f"Retrieved {len(vectors_with_ids)} vectors")
            return vectors_with_ids
        else:
            print(f"Error fetching vectors: {response.status_code}")
            print(response.text)
            return []
    except Exception as e:
        print(f"Error fetching vectors: {e}")
        return []

def compute_original_similarities(vectors):
    """Compute pairwise cosine similarities between vectors."""
    if not vectors:
        return []
    
    vector_arrays = [np.array(item["vector"]) for item in vectors]
    vector_arrays = np.array(vector_arrays)
    
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(vector_arrays)
    
    # Extract upper triangular part to get pairwise similarities (excluding self-comparisons)
    n = len(vector_arrays)
    similarities = []
    
    for i in range(n):
        for j in range(i+1, n):
            similarities.append((i, j, similarity_matrix[i, j]))
    
    return np.array(similarities)

def calibrate_similarities(similarities, target_mean=0.64, target_std=0.07):
    """Apply correction to get distribution similar to cloud Weaviate."""
    if len(similarities) == 0:
        return []
    
    # Extract similarity values
    sim_values = similarities[:, 2]
    
    # Calculate current statistics
    current_mean = np.mean(sim_values)
    current_std = np.std(sim_values)
    
    print(f"Original distribution: mean={current_mean:.4f}, std={current_std:.4f}")
    print(f"Target distribution: mean={target_mean:.4f}, std={target_std:.4f}")
    
    # Calibrate using a simple linear transformation
    # This shifts and scales the distribution to match target mean and std
    calibrated_values = ((sim_values - current_mean) / current_std) * target_std + target_mean
    
    # Ensure all values are in valid range [0, 1]
    calibrated_values = np.clip(calibrated_values, 0, 1)
    
    # Create calibrated similarities array
    calibrated_similarities = np.copy(similarities)
    calibrated_similarities[:, 2] = calibrated_values
    
    return calibrated_similarities

def update_vectors_in_weaviate(vectors, calibrated_similarities):
    """
    Update vectors in Weaviate to match the calibrated similarity distribution.
    
    This is a complex task that involves solving a system of equations to find 
    new vectors that produce the desired similarity matrix. We use a simplified approach.
    """
    print("\nThis step would update vectors to produce the calibrated similarities.")
    print("Important: This is a complex mathematical problem without a perfect solution.")
    print("The preferred solution is to re-embed documents with cloud-equivalent settings.")
    
    should_proceed = input("\nWould you like to try an experimental approach? (y/n): ").strip().lower()
    if should_proceed != 'y':
        print("Aborting vector updates.")
        return
    
    print("\nCalibrating vectors for Weaviate...")
    
    # Create a mapping from similarity tuples to their new values
    similarity_map = {}
    for i, j, new_sim in calibrated_similarities:
        similarity_map[(int(i), int(j))] = new_sim
    
    # Apply a simple smooth transformation to the vectors
    # This doesn't perfectly preserve all relationships but helps improve distribution
    adjusted_vectors = []
    for idx, item in enumerate(vectors):
        vector = np.array(item["vector"])
        
        # Apply a boost factor based on the average calibration for this vector
        avg_boost = 0
        count = 0
        
        for i in range(len(vectors)):
            if i == idx:
                continue
                
            if i < idx and (i, idx) in similarity_map:
                sim_key = (i, idx)
            elif idx < i and (idx, i) in similarity_map:
                sim_key = (idx, i)
            else:
                continue
                
            if sim_key in similarity_map:
                count += 1
                # Calculate how much this relationship needs to change
                orig_sim = cosine_similarity([vector], [np.array(vectors[sim_key[1] if sim_key[0] == idx else sim_key[0]]["vector"])])[0][0]
                target_sim = similarity_map[sim_key]
                boost = target_sim / orig_sim if orig_sim > 0 else 1.0
                avg_boost += boost
                
        if count > 0:
            avg_boost /= count
            # Apply a gentle transformation - we don't want to overfit
            # Just enough to shift the distribution without destroying relationships
            adjusted_vector = vector * (1.0 + 0.1 * (avg_boost - 1.0))
            # Re-normalize
            adjusted_vector = adjusted_vector / np.linalg.norm(adjusted_vector)
        else:
            adjusted_vector = vector
            
        adjusted_vectors.append({
            "id": item["id"],
            "vector": adjusted_vector.tolist()
        })
    
    # Ask for confirmation
    confirm = input(f"\nReady to update {len(adjusted_vectors)} vectors in Weaviate. Proceed? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Aborting vector updates.")
        return
    
    # Update vectors in Weaviate
    headers = create_weaviate_headers()
    success_count = 0
    error_count = 0
    
    for item in tqdm(adjusted_vectors, desc="Updating vectors"):
        try:
            obj_id = item["id"]
            vector = item["vector"]
            
            # Use PATCH to update just the vector
            update_url = f"{WEAVIATE_URL}/v1/objects/{COLLECTION_NAME}/{obj_id}"
            update_data = {"vector": vector}
            
            response = requests.patch(
                update_url,
                headers=headers,
                json=update_data
            )
            
            if response.status_code in [200, 204]:
                success_count += 1
            else:
                error_count += 1
                print(f"Error updating object {obj_id}: {response.status_code}")
        except Exception as e:
            error_count += 1
            print(f"Error updating object: {e}")
    
    print(f"\nVector updates complete: {success_count} succeeded, {error_count} failed")

def visualize_distributions(original_similarities, calibrated_similarities):
    """Visualize original and calibrated similarity distributions."""
    plt.figure(figsize=(16, 10))
    
    # Extract similarity values
    orig_values = original_similarities[:, 2]
    calibrated_values = calibrated_similarities[:, 2]
    
    # Plot original distribution
    sns.histplot(orig_values, kde=True, stat="density", alpha=0.6,
                color=MAIN_COLOR, label="Original Self-hosted")
    
    # Plot calibrated distribution
    sns.histplot(calibrated_values, kde=True, stat="density", alpha=0.6,
                color=CORRECTED_COLOR, label="Calibrated")
    
    # Add reference for cloud
    cloud_mean = 0.64  # From the user's data
    plt.axvline(cloud_mean, color=CLOUD_COLOR, linestyle='--', linewidth=2,
                label=f'Typical Cloud Mean: {cloud_mean:.4f}')
    
    # Styling
    plt.title('Vector Similarity Distribution Calibration', fontsize=18, fontweight='bold')
    plt.xlabel('Cosine Similarity', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('similarity_calibration.png', dpi=300, bbox_inches='tight')
    print("Saved distribution visualization to 'similarity_calibration.png'")

def print_statistics(similarities, label):
    """Print statistics for a similarity distribution."""
    if len(similarities) == 0:
        print(f"No {label} similarities to analyze.")
        return
    
    # Extract similarity values
    values = similarities[:, 2]
    
    # Compute statistics
    stats = {
        "count": len(values),
        "mean": np.mean(values),
        "std": np.std(values),
        "min": np.min(values),
        "25%": np.percentile(values, 25),
        "median": np.median(values),
        "75%": np.percentile(values, 75),
        "90%": np.percentile(values, 90),
        "95%": np.percentile(values, 95),
        "99%": np.percentile(values, 99),
        "max": np.max(values)
    }
    
    # Print statistics
    print(f"\n===== {label.upper()} SIMILARITY STATISTICS =====")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

def main():
    parser = argparse.ArgumentParser(description="Calibrate vector similarities in Weaviate database")
    parser.add_argument("--target-mean", type=float, default=0.64, 
                        help="Target mean similarity (default: 0.64 for typical cloud)")
    parser.add_argument("--target-std", type=float, default=0.07, 
                        help="Target standard deviation (default: 0.07 for typical cloud)")
    parser.add_argument("--limit", type=int, default=500, 
                        help="Maximum number of vectors to process (default: 500)")
    parser.add_argument("--update", action="store_true", 
                        help="Update vectors in Weaviate (experimental)")
    
    args = parser.parse_args()
    
    # Fetch vectors
    print(f"Fetching up to {args.limit} vectors from Weaviate...")
    vectors = fetch_vectors(args.limit)
    
    if not vectors:
        print("No vectors found. Exiting.")
        return
    
    # Compute original similarities
    print("Computing original similarity distribution...")
    original_similarities = compute_original_similarities(vectors)
    print_statistics(original_similarities, "Original")
    
    # Calibrate similarities
    print("\nCalibrating similarity distribution...")
    calibrated_similarities = calibrate_similarities(
        original_similarities, 
        target_mean=args.target_mean,
        target_std=args.target_std
    )
    print_statistics(calibrated_similarities, "Calibrated")
    
    # Visualize the distributions
    print("\nVisualizing similarity distributions...")
    visualize_distributions(original_similarities, calibrated_similarities)
    
    # Update vectors if requested
    if args.update:
        update_vectors_in_weaviate(vectors, calibrated_similarities)
    else:
        print("\nTo update vectors in Weaviate, run again with --update flag")
        print("Note: This is an experimental feature.")

if __name__ == "__main__":
    main() 