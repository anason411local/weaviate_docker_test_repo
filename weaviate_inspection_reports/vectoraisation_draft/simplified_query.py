import os
import requests
import json
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

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

# Initialize the embedding model (same as in vectorizer script)
model_name = "BAAI/bge-m3"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embedding_model = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    cache_folder=None
)

def create_weaviate_headers():
    """Create headers for Weaviate API requests."""
    headers = {"Content-Type": "application/json"}
    if WEAVIATE_API_KEY:
        headers["Authorization"] = f"Bearer {WEAVIATE_API_KEY}"
    return headers

def check_collection_exists():
    """Check if the PDFDocuments collection exists."""
    headers = create_weaviate_headers()
    
    try:
        response = requests.get(f"{WEAVIATE_URL}/v1/schema", headers=headers)
        response.raise_for_status()
        schema = response.json()
        
        for cls in schema.get("classes", []):
            if cls.get("class") == "PDFDocuments":
                return True
                
        print("PDFDocuments collection not found! Run simplified_vectorizer.py first.")
        return False
    except Exception as e:
        print(f"Error checking schema: {e}")
        return False

def vector_search(query_text, limit=5):
    """Perform a vector search in the PDFDocuments collection."""
    # Generate embedding for the query
    query_vector = embedding_model.embed_query(query_text)
    
    # Prepare GraphQL query
    graphql_query = """
    {
      Get {
        PDFDocuments(
          nearVector: {
            vector: %s
            certainty: 0.7
          }
          limit: %d
        ) {
          text
          source
          filename
          page
          _additional {
            certainty
          }
        }
      }
    }
    """ % (json.dumps(query_vector), limit)
    
    # Execute query
    headers = create_weaviate_headers()
    
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
                
            return result.get("data", {}).get("Get", {}).get("PDFDocuments", [])
        else:
            print(f"Error in search: {response.status_code}")
            print(response.text)
            return []
    except Exception as e:
        print(f"Error in search: {e}")
        return []

def keyword_search(query_text, limit=5):
    """Perform a keyword search in the PDFDocuments collection."""
    graphql_query = """
    {
      Get {
        PDFDocuments(
          where: {
            operator: Like
            path: ["text"]
            valueText: "*%s*"
          }
          limit: %d
        ) {
          text
          source
          filename
          page
        }
      }
    }
    """ % (query_text.replace('"', '\\"'), limit)
    
    # Execute query
    headers = create_weaviate_headers()
    
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
                
            return result.get("data", {}).get("Get", {}).get("PDFDocuments", [])
        else:
            print(f"Error in search: {response.status_code}")
            print(response.text)
            return []
    except Exception as e:
        print(f"Error in search: {e}")
        return []

def main():
    """Main function to demonstrate searching."""
    # Check if collection exists
    if not check_collection_exists():
        return
    
    # Get user query
    query = input("Enter your search query: ")
    
    # Perform vector search
    print("\n===== VECTOR SEARCH RESULTS =====")
    vector_results = vector_search(query)
    
    if vector_results:
        for i, result in enumerate(vector_results):
            print(f"\n--- Result {i+1} ---")
            print(f"Source: {result['filename']} (Page {result['page']})")
            print(f"Text excerpt: {result['text'][:200]}...")
            if "_additional" in result and "certainty" in result["_additional"]:
                print(f"Certainty: {result['_additional']['certainty']:.4f}")
    else:
        print("No vector search results found.")
    
    # Perform keyword search
    print("\n===== KEYWORD SEARCH RESULTS =====")
    keyword_results = keyword_search(query)
    
    if keyword_results:
        for i, result in enumerate(keyword_results):
            print(f"\n--- Result {i+1} ---")
            print(f"Source: {result['filename']} (Page {result['page']})")
            print(f"Text excerpt: {result['text'][:800]}...")
    else:
        print("No keyword search results found.")

if __name__ == "__main__":
    main() 