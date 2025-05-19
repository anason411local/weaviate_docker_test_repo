#!/usr/bin/env python3

import os
import requests
import json
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import argparse

# Set up plotting aesthetics
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Define color scheme
MAIN_COLOR = "#4287f5"  # Vibrant blue
CLOUD_COLOR = "#f54242"  # Bold red

# Load environment variables
load_dotenv()

def create_weaviate_headers(api_key=None):
    """Create headers for Weaviate API requests."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers

def fetch_complete_objects(url, api_key, class_name="Area_expansion_Dep_Anas", limit=100):
    """Fetch objects with their vectors and text content from Weaviate."""
    headers = create_weaviate_headers(api_key)
    
    graphql_query = f"""
    {{
      Get {{
        {class_name}(
          limit: {limit}
        ) {{
          text
          filename
          page
          source
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
            f"{url}/v1/graphql",
            headers=headers,
            json={"query": graphql_query}
        )
        
        if response.status_code == 200:
            result = response.json()
            if "errors" in result:
                print(f"GraphQL errors: {result['errors']}")
                return []
                
            documents = result.get("data", {}).get("Get", {}).get(class_name, [])
            
            # Filter documents with vectors
            documents_with_vectors = []
            for doc in documents:
                if "_additional" in doc and "vector" in doc["_additional"]:
                    documents_with_vectors.append(doc)
            
            print(f"Retrieved {len(documents_with_vectors)} documents with vectors from {url}")
            return documents_with_vectors
        else:
            print(f"Error fetching objects from {url}: {response.status_code}")
            print(response.text)
            return []
    except Exception as e:
        print(f"Error fetching objects from {url}: {e}")
        return []

def compare_objects_by_content(self_hosted_docs, cloud_docs):
    """Compare objects from self-hosted and cloud by matching text content."""
    if not self_hosted_docs or not cloud_docs:
        print("Cannot compare: missing documents from one or both sources")
        return [], []
    
    # Create dictionaries to match objects by text content
    self_hosted_by_text = {}
    cloud_by_text = {}
    
    # Use a hash of the text to handle potential encoding differences
    for doc in self_hosted_docs:
        if "text" in doc:
            text_hash = hash(doc["text"])
            self_hosted_by_text[text_hash] = doc
    
    for doc in cloud_docs:
        if "text" in doc:
            text_hash = hash(doc["text"])
            cloud_by_text[text_hash] = doc
    
    # Find matching objects
    matched_self_hosted = []
    matched_cloud = []
    
    for text_hash, self_doc in self_hosted_by_text.items():
        if text_hash in cloud_by_text:
            matched_self_hosted.append(self_doc)
            matched_cloud.append(cloud_by_text[text_hash])
    
    print(f"Found {len(matched_self_hosted)} documents with matching content in both instances")
    return matched_self_hosted, matched_cloud

def compute_cosine_similarities(documents):
    """Compute pairwise cosine similarities between document vectors."""
    if not documents:
        print("No documents provided for similarity calculation")
        return []
    
    # Extract vectors
    vectors = [doc["_additional"]["vector"] for doc in documents 
              if "_additional" in doc and "vector" in doc["_additional"]]
    
    print(f"Using {len(vectors)} vectors for similarity calculation")
    
    if len(vectors) < 2:
        print("Need at least 2 vectors to calculate similarities")
        return []
    
    # Check vector dimensions
    dim = len(vectors[0])
    print(f"Vector dimension: {dim}")
    
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

def check_vector_normalization(vectors):
    """Check if vectors are normalized (have unit length)."""
    if not vectors:
        return 0, 0
    
    normalized_count = 0
    total_count = len(vectors)
    
    for vector in vectors:
        vec_np = np.array(vector)
        magnitude = np.linalg.norm(vec_np)
        if abs(magnitude - 1.0) < 0.01:  # Allow for small floating point differences
            normalized_count += 1
    
    return normalized_count, total_count

def analyze_statistics(similarities, source_name):
    """Analyze the distribution of cosine similarities and print statistics."""
    if len(similarities) == 0:
        print(f"No similarities to analyze for {source_name}.")
        return None
    
    # Compute statistics
    stats = {
        "source": source_name,
        "count": len(similarities),
        "mean": np.mean(similarities),
        "std": np.std(similarities),
        "min": np.min(similarities),
        "25%": np.percentile(similarities, 25),
        "median": np.median(similarities),
        "75%": np.percentile(similarities, 75),
        "90%": np.percentile(similarities, 90),
        "95%": np.percentile(similarities, 95),
        "99%": np.percentile(similarities, 99),
        "max": np.max(similarities)
    }
    
    # Print statistics
    print(f"\n===== COSINE SIMILARITY STATISTICS FOR {source_name} =====")
    for key, value in stats.items():
        if key != "source":
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    
    # Create a pandas Series for more detailed stats
    similarities_series = pd.Series(similarities)
    print(f"\n===== DETAILED STATISTICS FOR {source_name} =====")
    print(similarities_series.describe(percentiles=[.1, .25, .5, .75, .9, .95, .99]))
    
    return stats

def compare_distributions(self_hosted_similarities, cloud_similarities):
    """Compare and visualize the cosine similarity distributions."""
    # Create figure for histogram comparison
    plt.figure(figsize=(16, 10))
    
    # Plot histograms
    sns.histplot(self_hosted_similarities, kde=True, stat="density", alpha=0.6, 
               color=MAIN_COLOR, label="Self-hosted")
    sns.histplot(cloud_similarities, kde=True, stat="density", alpha=0.6, 
               color=CLOUD_COLOR, label="Cloud")
    
    # Add mean lines
    plt.axvline(np.mean(self_hosted_similarities), color=MAIN_COLOR, linestyle='--', 
              linewidth=2, label=f'Self-hosted Mean: {np.mean(self_hosted_similarities):.4f}')
    plt.axvline(np.mean(cloud_similarities), color=CLOUD_COLOR, linestyle='--', 
              linewidth=2, label=f'Cloud Mean: {np.mean(cloud_similarities):.4f}')
    
    # Styling
    plt.title('Comparison of Cosine Similarity Distributions', fontsize=18, fontweight='bold')
    plt.xlabel('Cosine Similarity', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    
    plt.savefig('cosine_similarity_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved comparison visualization to 'cosine_similarity_comparison.png'")
    
    # Create a comparison table
    self_stats = analyze_statistics(self_hosted_similarities, "Self-hosted")
    cloud_stats = analyze_statistics(cloud_similarities, "Cloud")
    
    # Print the difference
    print("\n===== COMPARISON OF STATISTICS =====")
    for key in self_stats:
        if key != "source" and key != "count":
            diff = cloud_stats[key] - self_stats[key]
            diff_pct = (diff / self_stats[key]) * 100 if self_stats[key] != 0 else float('inf')
            print(f"{key}: Difference = {diff:.4f} ({diff_pct:.2f}%)")

def filter_identical_vectors(self_hosted_docs, cloud_docs):
    """
    Check if there are identical (exact same) vector pairs between the two sets
    and identify potential issues.
    """
    identical_count = 0
    total_count = min(len(self_hosted_docs), len(cloud_docs))
    
    for i in range(total_count):
        self_vec = np.array(self_hosted_docs[i]["_additional"]["vector"])
        cloud_vec = np.array(cloud_docs[i]["_additional"]["vector"])
        
        # Check if vectors are identical (exact match)
        if np.array_equal(self_vec, cloud_vec):
            identical_count += 1
    
    if identical_count > 0:
        print(f"Found {identical_count} identical vector pairs out of {total_count} compared.")
        print("This suggests data might have been transferred between instances.")
    else:
        print("No identical vectors found between self-hosted and cloud instances.")
        print("This confirms these are independently generated vectors.")

def main():
    parser = argparse.ArgumentParser(description='Compare Weaviate self-hosted with cloud instance')
    parser.add_argument('--self-url', type=str, default="http://localhost:8090", 
                        help='URL for self-hosted Weaviate (default: http://localhost:8090)')
    parser.add_argument('--cloud-url', type=str, required=True, 
                        help='URL for Weaviate Cloud instance (e.g., https://your-instance.weaviate.cloud)')
    parser.add_argument('--self-key', type=str, default=None, 
                        help='API key for self-hosted Weaviate')
    parser.add_argument('--cloud-key', type=str, required=True, 
                        help='API key for Weaviate Cloud')
    parser.add_argument('--class-name', type=str, default="Area_expansion_Dep_Anas", 
                        help='Class name to analyze (default: Area_expansion_Dep_Anas)')
    parser.add_argument('--limit', type=int, default=100, 
                        help='Maximum number of documents to fetch (default: 100)')
    parser.add_argument('--match-by-content', action='store_true', 
                        help='Match documents by content rather than using the first N documents')
    
    args = parser.parse_args()
    
    # Fetch objects from both sources
    print(f"Fetching objects from self-hosted Weaviate at {args.self_url}...")
    self_hosted_docs = fetch_complete_objects(args.self_url, args.self_key, args.class_name, args.limit)
    
    print(f"Fetching objects from Weaviate Cloud at {args.cloud_url}...")
    cloud_docs = fetch_complete_objects(args.cloud_url, args.cloud_key, args.class_name, args.limit)
    
    if not self_hosted_docs or not cloud_docs:
        print("Cannot continue: missing documents from one or both sources")
        return
    
    # Match objects if requested
    if args.match_by_content:
        print("Matching objects by content...")
        self_hosted_docs, cloud_docs = compare_objects_by_content(self_hosted_docs, cloud_docs)
    
    # Check normalization
    print("\nChecking vector normalization for self-hosted instance...")
    self_vectors = [doc["_additional"]["vector"] for doc in self_hosted_docs if "_additional" in doc and "vector" in doc["_additional"]]
    self_normalized, self_total = check_vector_normalization(self_vectors)
    print(f"Self-hosted normalization: {self_normalized}/{self_total} vectors normalized ({self_normalized/self_total*100 if self_total > 0 else 0:.2f}%)")
    
    print("Checking vector normalization for cloud instance...")
    cloud_vectors = [doc["_additional"]["vector"] for doc in cloud_docs if "_additional" in doc and "vector" in doc["_additional"]]
    cloud_normalized, cloud_total = check_vector_normalization(cloud_vectors)
    print(f"Cloud normalization: {cloud_normalized}/{cloud_total} vectors normalized ({cloud_normalized/cloud_total*100 if cloud_total > 0 else 0:.2f}%)")
    
    # Check for identical vectors
    filter_identical_vectors(self_hosted_docs, cloud_docs)
    
    # Compute similarities
    print("\nComputing cosine similarities for self-hosted instance...")
    self_hosted_similarities = compute_cosine_similarities(self_hosted_docs)
    
    print("Computing cosine similarities for cloud instance...")
    cloud_similarities = compute_cosine_similarities(cloud_docs)
    
    # Compare distributions
    if len(self_hosted_similarities) > 0 and len(cloud_similarities) > 0:
        print("\nComparing similarity distributions...")
        compare_distributions(self_hosted_similarities, cloud_similarities)
    else:
        print("Cannot compare distributions: insufficient similarities calculated")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 