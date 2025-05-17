import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from dotenv import load_dotenv
from simplified_query import vector_search, check_collection_exists
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec

# Load environment variables
load_dotenv()

# Set up enhanced visual styles with modern aesthetics
plt.style.use('ggplot')
sns.set_style("whitegrid")

# Define enhanced color palette for more beautiful visuals
MAIN_COLOR = "#4287f5"     # Vibrant blue
ACCENT_COLOR = "#f54242"   # Bold red
GREEN_COLOR = "#42f54b"    # Vibrant green
PURPLE_COLOR = "#a142f5"   # Rich purple
YELLOW_COLOR = "#f5d142"   # Golden yellow
TEAL_COLOR = "#42e8f5"     # Bright teal

# Create custom colormaps for gradient effects
GRADIENT_CMAP = LinearSegmentedColormap.from_list("gradient_cmap",
                                               ["#ff9a9e", "#fad0c4", "#a1c4fd", "#c2e9fb"])
RAINBOW_CMAP = LinearSegmentedColormap.from_list("rainbow_cmap",
                                              ["#a1c4fd", "#ffbdbd", "#c2e9fb", "#ffecd2"])

def calculate_metrics_at_k(query, relevant_docs, max_k=10):
    """
    Calculate Precision@k, Recall@k, and F1 Score@k for different values of k.
    
    Args:
        query (str): The search query
        relevant_docs (list): List of doc IDs that are relevant to the query
        max_k (int): Maximum k value to calculate metrics for
    
    Returns:
        tuple: Lists of precision, recall, f1 scores, and retrieved docs at each k
    """
    # Perform search using the local Weaviate instance and get results
    results = vector_search(query, limit=max_k)
    
    if not results:
        return [0] * max_k, [0] * max_k, [0] * max_k, []
    
    # Extract document identifiers (using filename and page as identifier)
    retrieved_docs = [(r['filename'], r['page']) for r in results]
    
    # Get certainty scores for the visualization
    certainty_scores = []
    for r in results:
        if "_additional" in r and "certainty" in r["_additional"]:
            certainty_scores.append(r["_additional"]["certainty"])
        else:
            certainty_scores.append(0.0)
    
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
    
    # Include the retrieved docs and certainty scores for visualization
    retrieval_data = {
        'docs': retrieved_docs,
        'certainty': certainty_scores,
        'relevant': [1 if doc in relevant_docs else 0 for doc in retrieved_docs]
    }
    
    return precision_at_k, recall_at_k, f1_at_k, retrieval_data

def plot_metrics(precision_at_k, recall_at_k, f1_at_k, retrieval_data, max_k=10, query=""):
    """
    Create enhanced beautiful visualizations of retrieval metrics.
    
    Args:
        precision_at_k (list): Precision values at different k
        recall_at_k (list): Recall values at different k
        f1_at_k (list): F1 scores at different k
        retrieval_data (dict): Data about retrieved documents
        max_k (int): Maximum k value in the plot
        query (str): The search query used
    """
    k_values = list(range(1, max_k + 1))
    
    # Create a figure with multiple subplots using GridSpec for more control
    fig = plt.figure(figsize=(20, 16), dpi=140)
    gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1], width_ratios=[2, 1])
    
    # ==== Main Metrics Plot (Top Left) ====
    ax_main = plt.subplot(gs[0, 0])
    
    # Create gradient-filled area under the curves
    ax_main.fill_between(k_values, 0, precision_at_k, alpha=0.3, color=MAIN_COLOR, label='_nolegend_')
    ax_main.fill_between(k_values, 0, recall_at_k, alpha=0.3, color=ACCENT_COLOR, label='_nolegend_')
    ax_main.fill_between(k_values, 0, f1_at_k, alpha=0.3, color=GREEN_COLOR, label='_nolegend_')
    
    # Plot metrics with attractive styling and markers
    line_p = ax_main.plot(k_values, precision_at_k, 'o-', linewidth=3, markersize=10, 
               color=MAIN_COLOR, label='Precision@K')
    line_r = ax_main.plot(k_values, recall_at_k, 's-', linewidth=3, markersize=10, 
               color=ACCENT_COLOR, label='Recall@K')
    line_f = ax_main.plot(k_values, f1_at_k, '^-', linewidth=3, markersize=10, 
               color=GREEN_COLOR, label='F1 Score@K')
    
    # Set labels and title with enhanced styling
    ax_main.set_xlabel('K (Number of Retrieved Documents)', fontsize=16, fontweight='bold', labelpad=15)
    ax_main.set_ylabel('Score', fontsize=16, fontweight='bold', labelpad=15)
    
    if query:
        title = f'Retrieval Metrics for Query: "{query}"'
    else:
        title = 'Retrieval Metrics at K'
    
    ax_main.set_title(title, fontsize=20, fontweight='bold', pad=20)
    
    # Enhance grid for better readability
    ax_main.grid(True, linestyle='--', alpha=0.7)
    
    # Set axis limits with padding
    ax_main.set_xlim(0.5, max_k + 0.5)
    ax_main.set_ylim(-0.05, 1.05)
    
    # Set x-ticks to integers
    ax_main.set_xticks(k_values)
    ax_main.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Add legend with enhanced styling
    legend = ax_main.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  ncol=3, fontsize=14, frameon=True, framealpha=0.9,
                  shadow=True, fancybox=True)
    legend.get_frame().set_linewidth(1.5)
    legend.get_frame().set_edgecolor('gray')
    
    # Add annotations for max values
    max_p_idx = np.argmax(precision_at_k)
    max_r_idx = np.argmax(recall_at_k)
    max_f1_idx = np.argmax(f1_at_k)
    
    for idx, name, color in zip([max_p_idx, max_r_idx, max_f1_idx], 
                            ['Max Precision', 'Max Recall', 'Max F1'],
                            [MAIN_COLOR, ACCENT_COLOR, GREEN_COLOR]):
        ax_main.annotate(
            f'{name}: {max(precision_at_k if name=="Max Precision" else recall_at_k if name=="Max Recall" else f1_at_k):.3f} at K={idx+1}',
            xy=(idx+1, max(precision_at_k if name=="Max Precision" else recall_at_k if name=="Max Recall" else f1_at_k)),
            xytext=(20, -30 if name=="Max Recall" else 30),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color=color),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.8),
            fontsize=12
        )
    
    # Add a background color to the plot area
    ax_main.set_facecolor('#f8f9fa')
    
    # ==== Radar Chart (Top Right) ====
    ax_radar = plt.subplot(gs[0, 1], polar=True)
    
    # Prepare data for radar chart - we need to close the loop
    metrics = ['Precision', 'Recall', 'F1 Score', 'Avg Score', 'Max P', 'Max R']
    
    # Calculate values for radar chart (using averages and maximums)
    stats = [
        np.mean(precision_at_k),           # Average precision
        np.mean(recall_at_k),              # Average recall
        np.mean(f1_at_k),                  # Average F1
        np.mean([np.mean(precision_at_k), np.mean(recall_at_k), np.mean(f1_at_k)]),  # Overall average
        max(precision_at_k),               # Max precision
        max(recall_at_k),                  # Max recall
    ]
    
    # Need to close the loop for the radar chart
    stats = np.concatenate((stats, [stats[0]]))
    
    # Prepare angles for radar chart
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Plot radar chart with beautiful styling
    ax_radar.plot(angles, stats, 'o-', linewidth=3, markersize=10, color=PURPLE_COLOR)
    ax_radar.fill(angles, stats, alpha=0.25, color=PURPLE_COLOR)
    
    # Set radar chart labels and styling
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(metrics, fontsize=12, fontweight='bold')
    
    # Enhance radar chart appearance
    ax_radar.set_facecolor('#f8f9fa')
    ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_radar.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax_radar.set_rlim(0, 1)
    ax_radar.set_title('Metrics Overview', fontsize=16, fontweight='bold', pad=20)
    
    # ==== Bar Chart of Metrics at 3 Key K Values (Middle Left) ====
    ax_bar = plt.subplot(gs[1, 0])
    
    # Choose 3 key k values (beginning, middle, end)
    key_indices = [0, len(k_values)//2, -1]
    key_k_values = [k_values[i] for i in key_indices]
    key_precision = [precision_at_k[i] for i in key_indices]
    key_recall = [recall_at_k[i] for i in key_indices]
    key_f1 = [f1_at_k[i] for i in key_indices]
    
    # Set up bar positions
    bar_width = 0.25
    r1 = np.arange(len(key_k_values))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    
    # Create bar chart with gradient colors
    bars1 = ax_bar.bar(r1, key_precision, width=bar_width, label='Precision', color=MAIN_COLOR, edgecolor='white', linewidth=1.5)
    bars2 = ax_bar.bar(r2, key_recall, width=bar_width, label='Recall', color=ACCENT_COLOR, edgecolor='white', linewidth=1.5)
    bars3 = ax_bar.bar(r3, key_f1, width=bar_width, label='F1 Score', color=GREEN_COLOR, edgecolor='white', linewidth=1.5)
    
    # Add bar labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax_bar.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=10, fontweight='bold')
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    # Set labels and style
    ax_bar.set_xlabel('K Value', fontsize=14, fontweight='bold', labelpad=10)
    ax_bar.set_ylabel('Score', fontsize=14, fontweight='bold', labelpad=10)
    ax_bar.set_title('Comparison at Key K Values', fontsize=16, fontweight='bold', pad=15)
    ax_bar.set_xticks([r + bar_width for r in range(len(key_k_values))])
    ax_bar.set_xticklabels([f'K={k}' for k in key_k_values])
    ax_bar.set_ylim(0, 1.15)
    ax_bar.legend(loc='upper right', fontsize=12)
    ax_bar.set_facecolor('#f8f9fa')
    
    # ==== Heatmap of metrics at different K (Middle Right) ====
    ax_heat = plt.subplot(gs[1, 1])
    
    # Create data for heatmap
    heat_data = np.array([precision_at_k, recall_at_k, f1_at_k])
    
    # Plot heatmap with beautiful gradient colors
    sns.heatmap(heat_data, annot=True, fmt=".2f", cmap=RAINBOW_CMAP,
               xticklabels=k_values, yticklabels=['Precision', 'Recall', 'F1'],
               ax=ax_heat, linewidths=1, cbar=False)
    
    # Style heatmap
    ax_heat.set_title('Metrics Heatmap', fontsize=16, fontweight='bold', pad=15)
    ax_heat.set_xlabel('K Value', fontsize=14, fontweight='bold', labelpad=10)
    
    # ==== Document Relevance Bar Chart (Bottom Row) ====
    ax_docs = plt.subplot(gs[2, :])
    
    if retrieval_data and 'docs' in retrieval_data and 'relevant' in retrieval_data:
        # Get document names and relevance
        doc_names = [f"{doc[0]}(p{doc[1]})" for doc in retrieval_data['docs']]
        # Shorten document names for better display
        short_names = []
        for name in doc_names:
            if len(name) > 20:
                parts = name.split('_')
                if len(parts) > 1:
                    short_name = f"{parts[0][:15]}...(p{name.split('p')[-1].split(')')[0]})"
                else:
                    short_name = f"{name[:15]}..."
                short_names.append(short_name)
            else:
                short_names.append(name)
        
        # Prepare bar heights and colors based on relevance and certainty
        relevance = retrieval_data['relevant']
        certainty = retrieval_data.get('certainty', [0.7] * len(relevance))
        
        # Create bars with colors based on relevance
        bar_colors = [TEAL_COLOR if rel else YELLOW_COLOR for rel in relevance]
        
        # Create the bars
        bars = ax_docs.bar(range(len(short_names)), certainty, color=bar_colors)
        
        # Add relevance markers
        for i, rel in enumerate(relevance):
            if rel:
                ax_docs.annotate('✓',
                             xy=(i, certainty[i]),
                             xytext=(0, 5),
                             textcoords="offset points",
                             ha='center', va='bottom',
                             fontsize=16, fontweight='bold', color='green')
        
        # Add certainty labels
        for i, cert in enumerate(certainty):
            ax_docs.annotate(f'{cert:.3f}',
                         xy=(i, cert/2),
                         ha='center', va='center',
                         fontsize=10, fontweight='bold', color='black',
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
        
        # Style the chart
        ax_docs.set_title('Document Relevance and Certainty Scores', fontsize=16, fontweight='bold', pad=15)
        ax_docs.set_xlabel('Retrieved Documents', fontsize=14, fontweight='bold', labelpad=10)
        ax_docs.set_ylabel('Certainty Score', fontsize=14, fontweight='bold', labelpad=10)
        ax_docs.set_xticks(range(len(short_names)))
        ax_docs.set_xticklabels(short_names, rotation=45, ha='right', fontsize=10)
        ax_docs.set_ylim(0, 1.1)
        ax_docs.set_facecolor('#f8f9fa')
        
        # Add legend for relevance
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=TEAL_COLOR, edgecolor='w', label='Relevant'),
            Patch(facecolor=YELLOW_COLOR, edgecolor='w', label='Not Relevant')
        ]
        ax_docs.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # Add descriptive text box at the bottom of the figure
    textbox = (
        "Precision@K: Fraction of retrieved documents that are relevant\n"
        "Recall@K: Fraction of relevant documents that are retrieved\n"
        "F1 Score@K: Harmonic mean of Precision and Recall\n"
        "Certainty: Similarity score between query and document vectors"
    )
    fig.text(0.5, 0.01, textbox, ha="center", fontsize=12, 
             bbox={"facecolor":"white", "alpha":0.9, "pad":5, 
                   "edgecolor":"lightgray", "boxstyle":"round,pad=0.5"})
    
    # Final layout adjustments
    plt.tight_layout(pad=4, h_pad=4, w_pad=4)
    plt.subplots_adjust(bottom=0.08)
    
    # Save the figure with high quality
    plt.savefig('retrieval_metrics.png', dpi=300, bbox_inches='tight')
    
    # Return the figure for potential further customization
    return fig

def main():
    """Main function to demonstrate the enhanced metrics visualization."""
    # Check if collection exists in the local Weaviate instance
    if not check_collection_exists():
        print("Weaviate collection not found. Please run simplified_vectorizer.py first.")
        return
    
    # Get user query
    query = input("Enter your search query: ")
    
    # For demonstration purposes, we'll simulate relevant documents
    # In a real application, you would need ground truth data
    print("Since we don't have ground truth data, let's simulate some relevant documents.")
    print("For a real evaluation, you would need actual relevance judgments.")
    
    # Get sample results to use as "relevant" documents for demonstration
    sample_results = vector_search(query, limit=20)
    
    if not sample_results:
        print("No search results found. Try a different query.")
        return
    
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
    print("Relevant documents:")
    for i, doc in enumerate(relevant_docs):
        print(f"{i+1}. {doc[0]} (Page {doc[1]})")
    
    # Calculate metrics
    max_k = 10
    precision_at_k, recall_at_k, f1_at_k, retrieval_data = calculate_metrics_at_k(
        query, relevant_docs, max_k)
    
    # Display metrics in the console with enhanced formatting
    print("\n" + "="*60)
    print(f"{'K':^5}|{'Precision@K':^15}|{'Recall@K':^15}|{'F1 Score@K':^15}")
    print("="*60)
    for k in range(max_k):
        print(f"{k+1:^5}|{precision_at_k[k]:^15.4f}|{recall_at_k[k]:^15.4f}|{f1_at_k[k]:^15.4f}")
    print("-"*60)
    print(f"{'AVG':^5}|{np.mean(precision_at_k):^15.4f}|{np.mean(recall_at_k):^15.4f}|{np.mean(f1_at_k):^15.4f}")
    print(f"{'MAX':^5}|{max(precision_at_k):^15.4f}|{max(recall_at_k):^15.4f}|{max(f1_at_k):^15.4f}")
    print("="*60)
    
    # Plot the enhanced metrics visualizations
    fig = plot_metrics(precision_at_k, recall_at_k, f1_at_k, retrieval_data, max_k, query)
    print("\nEnhanced visualization saved as 'retrieval_metrics.png'")
    
    # Display some additional statistics
    auc_p = np.trapz(precision_at_k) / max_k
    auc_r = np.trapz(recall_at_k) / max_k
    auc_f1 = np.trapz(f1_at_k) / max_k
    
    print("\nAdditional Statistics:")
    print(f"Area Under Curve - Precision: {auc_p:.4f}")
    print(f"Area Under Curve - Recall: {auc_r:.4f}")
    print(f"Area Under Curve - F1 Score: {auc_f1:.4f}")
    print(f"Best K value based on F1 Score: {np.argmax(f1_at_k) + 1}")

if __name__ == "__main__":
    main() 