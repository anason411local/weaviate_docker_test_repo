from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Fixed model for cosine similarity analysis request
class CosineSimilarityRequest(BaseModel):
    # Make class_name optional since it's already in the path
    sample_size: int = Field(100, description="Number of objects to sample for analysis", ge=10, le=10000)

# The rest of the endpoint logic stays the same
def compute_cosine_similarities(documents):
    """Compute pairwise cosine similarities between all document vectors using sklearn."""
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
    
    # Second pass: only keep vectors with the correct dimension
    for doc in documents:
        vec = doc["_additional"]["vector"]
        if isinstance(vec, list) and len(vec) == correct_dim:
            vectors.append(vec)
    
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