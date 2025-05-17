import weaviate
import os 
from weaviate.collections.classes.config import CollectionConfig, Vectorizer # Added Vectorizer for type hint
from weaviate.collections.classes.types import Properties # For type hinting properties

# Weaviate Connection Settings
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "127.0.0.1")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8090"))
COLLECTION_NAME = "PDFDocuments" # The name of your collection

def check_collection_schema():
    """Connects to Weaviate and prints the schema of the specified collection."""
    client = None
    try:
        # Connect to Weaviate
        client = weaviate.connect_to_local(host=WEAVIATE_HOST, port=WEAVIATE_PORT)
        print(f"Attempting to connect to Weaviate at {WEAVIATE_HOST}:{WEAVIATE_PORT}...")

        if not client.is_ready():
            print("❌ Weaviate is not ready. Please check your connection.")
            return

        print("✅ Successfully connected to Weaviate.")

        # Check if the collection exists
        if not client.collections.exists(COLLECTION_NAME):
            print(f"Collection '{COLLECTION_NAME}' does not exist in Weaviate.")
            return

        print(f"\n--- Schema for Collection: '{COLLECTION_NAME}' ---")
        
        # Get the specific collection object first
        collection = client.collections.get(COLLECTION_NAME)
        
        # Then get its configuration
        collection_config: CollectionConfig = collection.config.get()
        
        print(f"Name: {collection_config.name}")
        if collection_config.description:
            print(f"Description: {collection_config.description}")
        
        # Print vectorizer configuration
        vec_config_display = "Not explicitly set or default"
        if collection_config.vectorizer_config is not None:
            # The vectorizer_config attribute holds a list of VectorizerConfig objects (or a single one)
            # For Configure.Vectorizer.none(), this might be an empty list or a specific object.
            current_vectorizer_config = collection_config.vectorizer_config
            
            if isinstance(current_vectorizer_config, list): # For named vectors setup
                if current_vectorizer_config:
                     # Displaying the first named vector config as an example
                    vec_config_display = f"Named Vectors: {[str(vc.vectorizer) for vc in current_vectorizer_config]}"
                else:
                    vec_config_display = "Vectorizer config is an empty list (likely 'none' or unconfigured)"
            elif isinstance(current_vectorizer_config, Vectorizer): # For a single, collection-wide vectorizer
                 vec_config_display = str(current_vectorizer_config.vectorizer)
            else: # Fallback for other structures, e.g. if it's directly Configure.Vectorizer.none()
                 vec_config_display = str(current_vectorizer_config)
        
        print(f"Vectorizer: {vec_config_display}") # Changed label for clarity

        print("\nProperties:")
        # collection_config.properties is a list of Property objects
        properties_list: list[Properties] = collection_config.properties # Added type hint
        if properties_list:
            for prop in properties_list:
                print(f"  -> Property Name: {prop.name}")
                print(f"     Data Type: {prop.data_type}") 
                if prop.description:
                    print(f"     Description: {prop.description}")
                if hasattr(prop, 'tokenization') and prop.tokenization: # Check for tokenization
                    print(f"     Tokenization: {prop.tokenization}")
                print("-" * 20)
        else:
            print("  No properties are defined for this collection.")

    except Exception as e:
        print(f"An error occurred while checking the schema: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if client and client.is_connected():
            client.close()
            print("\nWeaviate connection closed.")

if __name__ == "__main__":
    check_collection_schema()
