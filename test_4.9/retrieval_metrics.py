import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dotenv import load_dotenv
from simplified_query import vector_search, check_collection_exists

# Set up a beautiful visual style
plt.style.use('ggplot')
sns.set_palette("viridis")
sns.set_style("whitegrid")

def calculate_metrics_at_k(query, relevant_docs, max_k=10):
    """
    Calculate Precision@k, Recall@k, and F1 Score@k for different values of k.
    
    Args:
        query (str): The search query
        relevant_docs (list): List of doc IDs that are relevant to the query
        max_k (int): Maximum k value to calculate metrics for
    
    Returns:
        tuple: Lists of precision, recall, and f1 scores at each k
    """
    # Perform search and get results
    results = vector_search(query, limit=max_k)
    
    if not results:
        return [0] * max_k, [0] * max_k, [0] * max_k
    
    # Extract document identifiers (using filename and page as identifier)
    retrieved_docs = [(r['filename'], r['page']) for r in results]
    
    # Calculate metrics at each k
    precision_at_k = []
    recall_at_k = []
    f1_at_k = []
    
    for k in range(1, max_k + 1):
        # Only consider top k results
        docs_at_k = retrieved_docs[:k]
        
        # Calculate relevant retrieved docs
        relevant_retrieved = sum(1 for doc in docs_at_k if doc in relevant_docs)
        
        # Calculate Precision@k
        precision = relevant_retrieved / k if k > 0 else 0
        
        # Calculate Recall@k
        recall = relevant_retrieved / len(relevant_docs) if relevant_docs else 0
        
        # Calculate F1 Score@k
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_at_k.append(precision)
        recall_at_k.append(recall)
        f1_at_k.append(f1)
    
    return precision_at_k, recall_at_k, f1_at_k

def plot_metrics(precision_at_k, recall_at_k, f1_at_k, max_k=10, query=""):
    """
    Create a beautiful visualization of retrieval metrics.
    
    Args:
        precision_at_k (list): Precision values at different k
        recall_at_k (list): Recall values at different k
        f1_at_k (list): F1 scores at different k
        max_k (int): Maximum k value in the plot
        query (str): The search query used
    """
    k_values = list(range(1, max_k + 1))
    
    # Create figure with custom size and high DPI
    plt.figure(figsize=(12, 8), dpi=100)
    
    # Plot metrics with attractive styling
    plt.plot(k_values, precision_at_k, 'o-', linewidth=2.5, markersize=8, label='Precision@K')
    plt.plot(k_values, recall_at_k, 's-', linewidth=2.5, markersize=8, label='Recall@K')
    plt.plot(k_values, f1_at_k, '^-', linewidth=2.5, markersize=8, label='F1 Score@K')
    
    # Set labels and title with nice fonts
    plt.xlabel('K (Number of Retrieved Documents)', fontsize=14, fontweight='bold')
    plt.ylabel('Score', fontsize=14, fontweight='bold')
    
    if query:
        plt.title(f'Retrieval Metrics for Query: "{query}"', fontsize=16, fontweight='bold')
    else:
        plt.title('Retrieval Metrics at K', fontsize=16, fontweight='bold')
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set axis limits for cleaner look
    plt.xlim(0.5, max_k + 0.5)
    plt.ylim(-0.05, 1.05)
    
    # Set x-ticks to integers
    plt.xticks(k_values)
    
    # Add legend with nice styling
    plt.legend(loc='best', fontsize=12, frameon=True, framealpha=0.9)
    
    # Add annotations for specific points
    for i, (p, r, f) in enumerate(zip(precision_at_k, recall_at_k, f1_at_k)):
        k = i + 1
        # Only annotate some points to avoid clutter
        if k % 2 == 0 or k == 1 or k == max_k:
            plt.annotate(f'P: {p:.2f}', (k, p), textcoords="offset points", 
                         xytext=(0,10), ha='center', fontsize=9)
            plt.annotate(f'R: {r:.2f}', (k, r), textcoords="offset points", 
                         xytext=(0,10), ha='center', fontsize=9)
            plt.annotate(f'F1: {f:.2f}', (k, f), textcoords="offset points", 
                         xytext=(0,10), ha='center', fontsize=9)
    
    # Add a background color to the plot area
    ax = plt.gca()
    ax.set_facecolor('#f8f9fa')
    
    # Add a descriptive text box
    textbox = (
        "Precision@K: Fraction of retrieved documents that are relevant\n"
        "Recall@K: Fraction of relevant documents that are retrieved\n"
        "F1 Score@K: Harmonic mean of Precision and Recall"
    )
    plt.figtext(0.5, 0.01, textbox, ha="center", fontsize=10, 
                bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
    
    # Tight layout for better spacing
    plt.tight_layout(pad=3)
    
    # Save the figure
    plt.savefig('retrieval_metrics.png', bbox_inches='tight')
    
    # Show the plot
    plt.show()

def main():
    """Main function to demonstrate the metrics visualization."""
    # Check if collection exists
    if not check_collection_exists():
        return
    
    # Get user query
    query = input("Enter your search query: ")
    
    # For demonstration purposes, we'll simulate relevant documents
    # In a real application, you would need ground truth data
    print("Since we don't have ground truth data, let's simulate some relevant documents.")
    print("For a real evaluation, you would need actual relevance judgments.")
    
    # Get sample results to use as "relevant" documents for demonstration
    sample_results = vector_search(query, limit=20)
    
    # Randomly select some as "relevant" for demonstration
    np.random.seed(42)  # For reproducibility
    relevant_indices = np.random.choice(
        min(len(sample_results), 20), 
        size=min(len(sample_results) // 2, 10),
        replace=False
    )
    
    # Create relevant docs list
    relevant_docs = [(sample_results[i]['filename'], sample_results[i]['page']) 
                     for i in relevant_indices]
    
    print(f"\nUsing {len(relevant_docs)} simulated relevant documents for evaluation.")
    
    # Calculate metrics
    max_k = 10
    precision_at_k, recall_at_k, f1_at_k = calculate_metrics_at_k(query, relevant_docs, max_k)
    
    # Display metrics in the console
    print("\n" + "="*50)
    print(f"{'K':<5}{'Precision@K':<15}{'Recall@K':<15}{'F1 Score@K':<15}")
    print("-"*50)
    for k in range(max_k):
        print(f"{k+1:<5}{precision_at_k[k]:<15.4f}{recall_at_k[k]:<15.4f}{f1_at_k[k]:<15.4f}")
    
    # Plot the metrics
    plot_metrics(precision_at_k, recall_at_k, f1_at_k, max_k, query)
    print("\nVisualization saved as 'retrieval_metrics.png'")

if __name__ == "__main__":
    main() 