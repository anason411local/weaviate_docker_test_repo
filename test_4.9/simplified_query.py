import os
import requests
import json
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

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

# Initialize the embedding model (same as in vectoraiser script)
model_name = "BAAI/bge-m3"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embedding_model = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    cache_folder=None
)

def check_collection_exists():
    """Check if the PDFDocuments collection exists."""
    headers = {
        "Authorization": f"Bearer {WEAVIATE_API_KEY}",
        "Content-Type": "application/json"
    }
    
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
    headers = {
        "Authorization": f"Bearer {WEAVIATE_API_KEY}",
        "Content-Type": "application/json"
    }
    
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
    headers = {
        "Authorization": f"Bearer {WEAVIATE_API_KEY}",
        "Content-Type": "application/json"
    }
    
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