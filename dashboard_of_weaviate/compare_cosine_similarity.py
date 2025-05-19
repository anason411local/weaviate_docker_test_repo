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

def fetch_vectors(url, api_key, class_name="PDFDocuments", limit=1000):
    """Fetch vectors from Weaviate instance."""
    headers = create_weaviate_headers(api_key)
    
    graphql_query = """
    {
      Get {
        %s(
          limit: %d
        ) {
          _additional {
            id
            vector
          }
        }
      }
    }
    """ % (class_name, limit)
    
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
            print(f"Error fetching vectors from {url}: {response.status_code}")
            print(response.text)
            return []
    except Exception as e:
        print(f"Error fetching vectors from {url}: {e}")
        return []

def compute_cosine_similarities(documents):
    """Compute pairwise cosine similarities between document vectors."""
    if not documents:
        return []
    
    # Extract vectors and ensure they all have the same dimension
    vectors = []
    
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
    print(f"Using vector dimension: {correct_dim}")
    
    # Second pass: only keep vectors with the correct dimension
    for doc in documents:
        vec = doc["_additional"]["vector"]
        if isinstance(vec, list) and len(vec) == correct_dim:
            vectors.append(vec)
    
    print(f"Using {len(vectors)} out of {len(documents)} documents with valid vectors")
    
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

def analyze_similarities(similarities, source_name):
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
    self_stats = analyze_similarities(self_hosted_similarities, "Self-hosted")
    cloud_stats = analyze_similarities(cloud_similarities, "Cloud")
    
    # Create comparison DataFrame
    comparison = pd.DataFrame([self_stats, cloud_stats])
    
    # Print the difference
    print("\n===== COMPARISON OF STATISTICS =====")
    for key in self_stats:
        if key != "source" and key != "count":
            diff = cloud_stats[key] - self_stats[key]
            diff_pct = (diff / self_stats[key]) * 100 if self_stats[key] != 0 else float('inf')
            print(f"{key}: Difference = {diff:.4f} ({diff_pct:.2f}%)")
    
    return comparison

def check_vector_normalization(documents):
    """Check if vectors are normalized (unit length)."""
    normalized_count = 0
    total_count = 0
    
    for doc in documents:
        if "_additional" in doc and "vector" in doc["_additional"]:
            vec = doc["_additional"]["vector"]
            if isinstance(vec, list):
                total_count += 1
                # Calculate vector magnitude
                vec_np = np.array(vec)
                magnitude = np.linalg.norm(vec_np)
                # Check if the magnitude is close to 1.0 (normalized)
                if abs(magnitude - 1.0) < 0.01:  # Allow for small floating point differences
                    normalized_count += 1
    
    if total_count > 0:
        normalized_pct = (normalized_count / total_count) * 100
        print(f"Vector normalization check: {normalized_count}/{total_count} vectors are normalized ({normalized_pct:.2f}%)")
    else:
        print("No vectors to check for normalization")
        
    return normalized_count, total_count

def main():
    parser = argparse.ArgumentParser(description='Compare cosine similarities between Weaviate instances')
    parser.add_argument('--self-url', type=str, default="http://localhost:8090", 
                        help='URL for self-hosted Weaviate (default: http://localhost:8090)')
    parser.add_argument('--cloud-url', type=str, required=True, 
                        help='URL for Weaviate Cloud instance')
    parser.add_argument('--self-key', type=str, default=None, 
                        help='API key for self-hosted Weaviate')
    parser.add_argument('--cloud-key', type=str, required=True, 
                        help='API key for Weaviate Cloud')
    parser.add_argument('--class-name', type=str, default="PDFDocuments", 
                        help='Class name to analyze (default: PDFDocuments)')
    parser.add_argument('--limit', type=int, default=200, 
                        help='Maximum number of documents to fetch (default: 200)')
    
    args = parser.parse_args()
    
    # Fetch vectors from both sources
    print(f"Fetching vectors from self-hosted Weaviate at {args.self_url}...")
    self_hosted_docs = fetch_vectors(args.self_url, args.self_key, args.class_name, args.limit)
    
    print(f"Fetching vectors from Weaviate Cloud at {args.cloud_url}...")
    cloud_docs = fetch_vectors(args.cloud_url, args.cloud_key, args.class_name, args.limit)
    
    # Check normalization
    print("\nChecking vector normalization for self-hosted instance...")
    self_normalized, self_total = check_vector_normalization(self_hosted_docs)
    
    print("Checking vector normalization for cloud instance...")
    cloud_normalized, cloud_total = check_vector_normalization(cloud_docs)
    
    # Compute similarities
    print("\nComputing cosine similarities for self-hosted instance...")
    self_hosted_similarities = compute_cosine_similarities(self_hosted_docs)
    
    print("Computing cosine similarities for cloud instance...")
    cloud_similarities = compute_cosine_similarities(cloud_docs)
    
    # Compare distributions
    print("\nComparing similarity distributions...")
    compare_distributions(self_hosted_similarities, cloud_similarities)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 