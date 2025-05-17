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

# Set plot style
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Define custom color palette for consistent visuals
MAIN_COLOR = "#1f77b4"  # Main blue color
ACCENT_COLOR = "#ff7f0e"  # Orange accent
GREEN_COLOR = "#2ca02c"  # Green
RED_COLOR = "#d62728"  # Red
CUSTOM_CMAP = LinearSegmentedColormap.from_list("custom_cmap", 
                                               ["#f0f9e8", "#7bccc4", "#43a2ca", "#0868ac"])

# Load environment variables
load_dotenv()

# Get Weaviate credentials
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

# Check if credentials are set
if not WEAVIATE_URL or not WEAVIATE_API_KEY:
    raise ValueError("WEAVIATE_URL and WEAVIATE_API_KEY must be set in .env file")

# Ensure URL has proper scheme
if WEAVIATE_URL and not WEAVIATE_URL.startswith(("http://", "https://")):
    WEAVIATE_URL = f"https://{WEAVIATE_URL}"
    print(f"Added https:// prefix to Weaviate URL: {WEAVIATE_URL}")

def fetch_all_vectors(limit=10000):
    """Fetch all vectors from the PDFDocuments collection."""
    headers = {
        "Authorization": f"Bearer {WEAVIATE_API_KEY}",
        "Content-Type": "application/json"
    }
    
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
    
    vectors = [doc["_additional"]["vector"] for doc in documents]
    
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

def create_heatmap(documents, max_docs=30):
    """Create a heatmap of cosine similarities between document vectors."""
    if not documents:
        print("No documents with vectors found.")
        return
    
    # Limit to max_docs for readability
    if len(documents) > max_docs:
        documents = documents[:max_docs]
        print(f"Limiting heatmap to first {max_docs} documents for readability")
    
    vectors = [doc["_additional"]["vector"] for doc in documents]
    filenames = [f"{doc['filename']}(p{doc['page']})" for doc in documents]
    
    # Create short labels
    short_labels = []
    for fn in filenames:
        # Extract just the file prefix and page number
        prefix = fn.split('_')[0] if '_' in fn else fn.split('.')[0]
        if '(' in prefix:
            prefix = prefix.split('(')[0]
        page = fn.split('p')[-1].split(')')[0] if 'p' in fn else ''
        short_label = f"{prefix[:10]}..p{page}" if len(prefix) > 10 else f"{prefix}..p{page}"
        short_labels.append(short_label)
    
    # Convert to numpy arrays
    vectors = np.array(vectors)
    
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(vectors)
    
    # Create figure for heatmap
    plt.figure(figsize=(14, 12))
    
    # Create heatmap with custom colormap
    heatmap = sns.heatmap(
        similarity_matrix, 
        annot=False,
        cmap=CUSTOM_CMAP,
        xticklabels=short_labels,
        yticklabels=short_labels,
        vmin=0, vmax=1,
        cbar_kws={'label': 'Cosine Similarity'}
    )
    
    # Set title and labels
    plt.title('Cosine Similarity Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('cosine_similarity_heatmap.png', dpi=300, bbox_inches='tight')
    
    return similarity_matrix

def compute_similarity_distribution(similarities, num_bins=10):
    """Compute percentage-wise distribution of cosine similarities."""
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
    
    # Create figure for bar chart
    plt.figure(figsize=(14, 8))
    
    # Create color gradient for bars based on similarity values
    colors = cm.viridis(np.linspace(0.1, 0.9, len(bin_labels)))
    
    # Plot the distribution with better styling
    bars = plt.bar(bin_labels, percentages, color=colors, width=0.7, edgecolor='black', linewidth=0.5)
    
    # Add a trend line
    x = np.arange(len(bin_labels))
    z = np.polyfit(x, percentages, 2)
    p = np.poly1d(z)
    plt.plot(x, p(x), '--', color=RED_COLOR, linewidth=2, alpha=0.7)
    
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
                fontsize=9,
                color='black'
            )
    
    # Add styling
    plt.title('Percentage-wise Cosine Similarity Distribution', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Cosine Similarity Range', fontsize=12, labelpad=10)
    plt.ylabel('Percentage of Document Pairs', fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add annotation with statistics
    stats_text = (
        f"Total pairs: {total_count}\n"
        f"Mean: {np.mean(similarities):.4f}\n"
        f"Median: {np.median(similarities):.4f}\n"
        f"Std Dev: {np.std(similarities):.4f}"
    )
    plt.annotate(
        stats_text, 
        xy=(0.02, 0.96), 
        xycoords='axes fraction',
        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8),
        fontsize=10
    )
    
    plt.tight_layout()
    plt.savefig('cosine_similarity_percentage_distribution.png', dpi=300, bbox_inches='tight')
    
    return distribution

def analyze_similarities(similarities):
    """Analyze the distribution of cosine similarities."""
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
    
    # Create figure with enhanced styling
    plt.figure(figsize=(16, 10))
    
    # Subplot 1: Enhanced Histogram
    plt.subplot(2, 2, 1)
    n, bins, patches = plt.hist(similarities, bins=50, alpha=0.7, color=MAIN_COLOR, edgecolor='black', linewidth=0.5)
    
    # Add color gradient to histogram
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    cm = plt.cm.get_cmap('viridis')
    col = bin_centers - min(bin_centers)
    col /= max(col)
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))
    
    # Add mean and median lines
    plt.axvline(stats["mean"], color=RED_COLOR, linestyle='--', linewidth=2, 
                label=f'Mean: {stats["mean"]:.4f}')
    plt.axvline(stats["median"], color=GREEN_COLOR, linestyle='--', linewidth=2, 
                label=f'Median: {stats["median"]:.4f}')
    
    # Add key percentiles
    plt.axvline(stats["25%"], color='purple', linestyle=':', linewidth=1.5, 
                label=f'25th %: {stats["25%"]:.4f}')
    plt.axvline(stats["75%"], color='purple', linestyle=':', linewidth=1.5, 
                label=f'75th %: {stats["75%"]:.4f}')
    
    plt.title('Cosine Similarity Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Cosine Similarity', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    # Subplot 2: Enhanced Box plot
    plt.subplot(2, 2, 2)
    box = sns.boxplot(x=similarities, width=0.5, color=MAIN_COLOR)
    
    # Add a swarm plot on top for distribution visualization
    sns.swarmplot(x=similarities, size=2, color='black', alpha=0.5)
    
    # Add annotations for key statistics
    plt.axvline(stats["mean"], color=RED_COLOR, linestyle='--', linewidth=1.5)
    plt.text(stats["mean"], 0.02, f'Mean: {stats["mean"]:.4f}', 
             ha='center', va='bottom', fontsize=10, color=RED_COLOR)
    
    # Annotate quartiles
    for perc, label, color in [
        ("25%", "Q1", "purple"), 
        ("median", "Median", GREEN_COLOR), 
        ("75%", "Q3", "purple")
    ]:
        plt.text(stats[perc], 0.05, f'{label}: {stats[perc]:.4f}', 
                ha='center', va='bottom', fontsize=10, color=color, 
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
    
    plt.title('Cosine Similarity Boxplot', fontsize=14, fontweight='bold')
    plt.xlabel('Cosine Similarity', fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    
    # Subplot 3: KDE Plot with rugplot
    plt.subplot(2, 2, 3)
    sns.kdeplot(similarities, shade=True, color=MAIN_COLOR, alpha=0.7)
    sns.rugplot(similarities, color=RED_COLOR, alpha=0.5)
    
    # Add percentile markers
    for perc, label, color in [
        ("25%", "25th", "purple"), 
        ("median", "50th", GREEN_COLOR), 
        ("75%", "75th", "purple"),
        ("90%", "90th", ACCENT_COLOR),
        ("95%", "95th", "brown")
    ]:
        plt.axvline(stats[perc], color=color, linestyle='--', linewidth=1, alpha=0.7)
        plt.text(stats[perc], 0.2, f'{label}', ha='center', va='bottom', 
                 fontsize=9, color=color, rotation=90)
    
    plt.title('Kernel Density Estimate with Percentiles', fontsize=14, fontweight='bold')
    plt.xlabel('Cosine Similarity', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.grid(alpha=0.3)
    
    # Subplot 4: ECDF (Empirical Cumulative Distribution Function)
    plt.subplot(2, 2, 4)
    
    # Sort data for ECDF
    x = np.sort(similarities)
    y = np.arange(1, len(x) + 1) / len(x)
    
    # Plot ECDF
    plt.plot(x, y, marker='.', linestyle='none', alpha=0.3, color=MAIN_COLOR)
    plt.plot(x, y, linestyle='-', linewidth=2, color=ACCENT_COLOR, alpha=0.7)
    
    # Add markers for percentiles
    key_percentiles = [25, 50, 75, 90, 95]
    for p in key_percentiles:
        perc_val = np.percentile(similarities, p)
        plt.plot([perc_val, perc_val], [0, p/100], 'k--', linewidth=1, alpha=0.5)
        plt.plot([min(similarities), perc_val], [p/100, p/100], 'k--', linewidth=1, alpha=0.5)
        plt.scatter([perc_val], [p/100], color='red', s=50, zorder=5)
        plt.annotate(f'{p}%: {perc_val:.4f}', 
                    xy=(perc_val, p/100), 
                    xytext=(5, 0), 
                    textcoords='offset points', 
                    fontsize=9)
    
    plt.title('Cumulative Distribution Function', fontsize=14, fontweight='bold')
    plt.xlabel('Cosine Similarity', fontsize=12)
    plt.ylabel('Cumulative Probability', fontsize=12)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cosine_similarity_distribution.png', dpi=300, bbox_inches='tight')
    
    # Also compute the percentage distribution
    compute_similarity_distribution(similarities)
    
    return stats

def main():
    print("Fetching document vectors from Weaviate...")
    documents = fetch_all_vectors()
    
    if documents:
        # Create heatmap visualization
        print("Creating cosine similarity heatmap...")
        create_heatmap(documents)
        
        print(f"Computing cosine similarities between {len(documents)} documents...")
        similarities = compute_cosine_similarities(documents)
        
        print("Analyzing similarity distribution...")
        analyze_similarities(similarities)
        
        print("\nAnalysis complete. Results saved to:")
        print("1. 'cosine_similarity_distribution.png'")
        print("2. 'cosine_similarity_percentage_distribution.png'")
        print("3. 'cosine_similarity_heatmap.png'")
    else:
        print("No documents found. Please run simplified_vectorizer.py first to populate the database.")

if __name__ == "__main__":
    main()