import os
import requests
import json
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from scipy.spatial.distance import cosine
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

# Set plot style with modern aesthetics
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Define enhanced color palette for more beautiful visuals
MAIN_COLOR = "#4287f5"  # Vibrant blue
ACCENT_COLOR = "#f54242"  # Bold red
GREEN_COLOR = "#42f54b"  # Vibrant green
PURPLE_COLOR = "#a142f5"  # Rich purple
CUSTOM_CMAP = LinearSegmentedColormap.from_list("custom_cmap", 
                                               ["#f0f9e8", "#7bccc4", "#43a2ca", "#0868ac"])
GRADIENT_CMAP = LinearSegmentedColormap.from_list("gradient_cmap",
                                                ["#ff9a9e", "#fad0c4", "#a1c4fd", "#c2e9fb"])

# Load environment variables
load_dotenv()

# --- Configuration for Weaviate ---
# Using localhost:8090 for your local self-hosted Weaviate instance
WEAVIATE_URL = "http://localhost:8090"  # Changed from environment variable to direct URL
# For Docker containers, you might need to use 'host.docker.internal' instead of 'localhost'
# WEAVIATE_URL = "http://host.docker.internal:8090"  # Uncomment if needed
# WEAVIATE_API_KEY will be loaded from .env.
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

print(f"Attempting to connect to Weaviate at: {WEAVIATE_URL}")

# Check if API key is set (only if needed)
if not WEAVIATE_API_KEY:
    print("Warning: WEAVIATE_API_KEY is not set in the .env file. Proceeding without API key.")
    print("If your self-hosted Weaviate instance requires an API key, requests will likely fail.")
else:
    print("WEAVIATE_API_KEY found in .env file.")

# Ensure URL has proper scheme
if not WEAVIATE_URL.startswith(("http://", "https://")):
    print(f"Warning: WEAVIATE_URL '{WEAVIATE_URL}' does not have a scheme. Defaulting to http://.")
    WEAVIATE_URL = f"http://{WEAVIATE_URL}"
# --- End of Weaviate Configuration ---

def create_weaviate_headers():
    """Create headers for Weaviate API requests."""
    headers = {"Content-Type": "application/json"}
    if WEAVIATE_API_KEY:
        headers["Authorization"] = f"Bearer {WEAVIATE_API_KEY}"
    return headers

def fetch_all_vectors(limit=10000):
    """Fetch all vectors from the PDFDocuments collection."""
    headers = create_weaviate_headers()
    
    # GraphQL query to fetch vectors
    graphql_query = """
    {
      Get {
        PDFDocuments(
          limit: %d
        ) {
          text
          filename
          page
          _additional {
            vector
          }
        }
      }
    }
    """ % limit
    
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
                
            documents = result.get("data", {}).get("Get", {}).get("PDFDocuments", [])
            
            # Filter out documents without vectors
            documents_with_vectors = []
            for doc in documents:
                if "_additional" in doc and "vector" in doc["_additional"]:
                    documents_with_vectors.append(doc)
            
            print(f"Retrieved {len(documents_with_vectors)} documents with vectors")
            return documents_with_vectors
        else:
            print(f"Error fetching vectors: {response.status_code}")
            print(response.text)
            return []
    except Exception as e:
        print(f"Error fetching vectors: {e}")
        return []

def compute_cosine_similarities(documents):
    """Compute pairwise cosine similarities between all document vectors using sklearn."""
    if not documents:
        print("No documents with vectors found.")
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
        print("No valid vector dimensions found.")
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
            valid_documents.append(doc)
        else:
            print(f"Skipping vector with incorrect dimension: {len(vec) if isinstance(vec, list) else 'not a list'}")
    
    print(f"Using {len(vectors)} out of {len(documents)} documents with valid vectors")
    
    if not vectors:
        print("No valid vectors found.")
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

def compute_similarity_distribution(similarities, num_bins=12):
    """Compute percentage-wise distribution of cosine similarities with enhanced visuals."""
    total_count = len(similarities)
    if total_count == 0:
        print("No similarities to analyze.")
        return None
    
    # Create bins from min to max
    min_val = np.min(similarities)
    max_val = np.max(similarities)
    
    # Define bin edges
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    
    # Count frequencies
    hist, bin_edges = np.histogram(similarities, bins=bin_edges)
    
    # Convert to percentages
    percentages = (hist / total_count) * 100
    
    # Create bin labels
    bin_labels = [f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}" for i in range(len(bin_edges)-1)]
    
    # Create a dictionary of bin labels to percentages
    distribution = {bin_labels[i]: percentages[i] for i in range(len(bin_labels))}
    
    # Print the distribution
    print("\n===== PERCENTAGE-WISE COSINE SIMILARITY DISTRIBUTION =====")
    print(f"{'Range':<20} | {'Count':<7} | {'Percentage':>10}")
    print("-" * 43)
    for i in range(len(bin_labels)):
        count = hist[i]
        percentage = percentages[i]
        print(f"{bin_labels[i]:<20} | {count:<7} | {percentage:>9.2f}%")
    
    # Create gorgeous figure for bar chart with modern aesthetic
    plt.figure(figsize=(16, 9))
    
    # Create beautiful gradient color map for bars
    colors = GRADIENT_CMAP(np.linspace(0, 1, len(bin_labels)))
    
    # Plot the distribution with enhanced styling
    bars = plt.bar(bin_labels, percentages, color=colors, width=0.8, 
                  edgecolor='white', linewidth=1.5, alpha=0.9)
    
    # Add a smooth trend curve
    x = np.arange(len(bin_labels))
    z = np.polyfit(x, percentages, 3)  # Use cubic polynomial for smoother curve
    p = np.poly1d(z)
    
    # Generate more points for a smoother curve
    x_smooth = np.linspace(0, len(bin_labels)-1, 100)
    plt.plot(x_smooth, p(x_smooth), '-', color=PURPLE_COLOR, linewidth=3, alpha=0.8)
    
    # Add percentage labels on top of bars
    for bar in bars:
        height = bar.get_height()
        if height > 0.5:  # Only add labels for bars with significant height
            plt.text(
                bar.get_x() + bar.get_width()/2., 
                height + 0.5,
                f'{height:.1f}%', 
                ha='center', 
                va='bottom', 
                rotation=0,
                fontweight='bold',
                fontsize=11,
                color='black'
            )
    
    # Add styling
    plt.title('Cosine Similarity Distribution', fontsize=22, fontweight='bold', pad=20)
    plt.xlabel('Similarity Range', fontsize=16, labelpad=15)
    plt.ylabel('Percentage of Document Pairs', fontsize=16, labelpad=15)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add annotation with statistics
    stats_text = (
        f"Total pairs: {total_count}\n"
        f"Mean: {np.mean(similarities):.4f}\n"
        f"Median: {np.median(similarities):.4f}\n"
        f"Min: {np.min(similarities):.4f}\n"
        f"Max: {np.max(similarities):.4f}\n"
        f"Std Dev: {np.std(similarities):.4f}"
    )
    plt.annotate(
        stats_text, 
        xy=(0.02, 0.96), 
        xycoords='axes fraction',
        bbox=dict(boxstyle="round,pad=0.8", facecolor='white', alpha=0.9, edgecolor='lightgray'),
        fontsize=12,
        fontweight='bold'
    )
    
    # Add a beautiful background gradient
    ax = plt.gca()
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.savefig('cosine_similarity_distribution.png', dpi=300, bbox_inches='tight')
    
    return distribution

def analyze_similarities(similarities):
    """Analyze the distribution of cosine similarities with enhanced visuals."""
    if len(similarities) == 0:
        print("No similarities to analyze.")
        return
    
    # Compute statistics
    stats = {
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
    print("\n===== COSINE SIMILARITY STATISTICS =====")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    
    # Create a pandas Series for more detailed stats
    similarities_series = pd.Series(similarities)
    print("\n===== DETAILED STATISTICS =====")
    print(similarities_series.describe(percentiles=[.1, .25, .5, .75, .9, .95, .99]))
    
    # Create a single enhanced histogram with gradient colors - MORE BEAUTIFUL!
    plt.figure(figsize=(16, 9))
    
    # Create an enhanced histogram with gradient fill
    n, bins, patches = plt.hist(similarities, bins=50, alpha=0.8, edgecolor='white', linewidth=0.8)
    
    # Add beautiful color gradient to histogram
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)
    
    # Use custom gradient colormap for a more appealing visual
    cm = plt.cm.get_cmap(GRADIENT_CMAP)
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))
    
    # Add mean and median lines with enhanced styling
    plt.axvline(stats["mean"], color=ACCENT_COLOR, linestyle='-', linewidth=3, 
                label=f'Mean: {stats["mean"]:.4f}', alpha=0.8)
    plt.axvline(stats["median"], color=GREEN_COLOR, linestyle='-', linewidth=3, 
                label=f'Median: {stats["median"]:.4f}', alpha=0.8)
    
    # Add key percentiles with more attractive styling
    plt.axvline(stats["25%"], color=PURPLE_COLOR, linestyle='--', linewidth=2, 
                label=f'25th %: {stats["25%"]:.4f}', alpha=0.7)
    plt.axvline(stats["75%"], color=PURPLE_COLOR, linestyle='--', linewidth=2, 
                label=f'75th %: {stats["75%"]:.4f}', alpha=0.7)
    
    # Add beautiful title and labels
    plt.title('Cosine Similarity Distribution', fontsize=22, fontweight='bold', pad=20)
    plt.xlabel('Cosine Similarity', fontsize=16, labelpad=15)
    plt.ylabel('Frequency', fontsize=16, labelpad=15)
    
    # Enhanced legend with shadow and rounded corners
    legend = plt.legend(fontsize=12, framealpha=0.9, facecolor='white', 
                      edgecolor='lightgray', loc='upper right')
    legend.get_frame().set_boxstyle('round,pad=0.6')
    
    # Add subtle grid for better readability
    plt.grid(alpha=0.3, linestyle='--')
    
    # Add annotation with statistics in an attractive box
    stats_text = (
        f"Total pairs: {stats['count']}\n"
        f"Mean: {stats['mean']:.4f}\n"
        f"Median: {stats['median']:.4f}\n"
        f"Min: {stats['min']:.4f}\n"
        f"Max: {stats['max']:.4f}\n"
        f"Std Dev: {stats['std']:.4f}"
    )
    plt.annotate(
        stats_text,
        xy=(0.02, 0.96),
        xycoords='axes fraction',
        bbox=dict(boxstyle="round,pad=0.8", facecolor='white', alpha=0.9, edgecolor='lightgray'),
        fontsize=12,
        fontweight='bold'
    )
    
    # Set beautiful background
    ax = plt.gca()
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.savefig('histogram_distribution.png', dpi=300, bbox_inches='tight')
    
    # Also compute the percentage distribution
    compute_similarity_distribution(similarities)
    
    return stats

def main():
    print("Fetching document vectors from Weaviate...")
    documents = fetch_all_vectors()
    
    if documents:
        print(f"Computing cosine similarities between {len(documents)} documents...")
        similarities = compute_cosine_similarities(documents)
        
        print("Analyzing similarity distribution...")
        analyze_similarities(similarities)
        
        print("\nAnalysis complete. Results saved to:")
        print("1. 'histogram_distribution.png'")
        print("2. 'cosine_similarity_distribution.png'")
    else:
        print("No documents found. Please run simplified_vectorizer.py first to populate the database.")

if __name__ == "__main__":
    main()