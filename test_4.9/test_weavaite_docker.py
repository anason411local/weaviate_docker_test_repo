import weaviate
import os

# --- Configuration ---
# Replace with your Weaviate instance URL
WEAVIATE_URL = "http://127.0.0.1:8090" # Or "http://your-server-ip:8080"

auth_config = None

# --- Connect to Weaviate ---
try:
    client = weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=auth_config # Set to None if no auth
        # timeout_config=(5, 15) # Optional: (connect_timeout, read_timeout) in seconds
    )
    print(f"Successfully connected to Weaviate at {WEAVIATE_URL}")
except Exception as e:
    print(f"Error connecting to Weaviate: {e}")
    exit()

# --- Define the Collection (Class) Schema ---
collection_name = "MyTestCollection"
collection_schema = {
    "class": collection_name,
    "description": "A simple test collection for my self-hosted Weaviate",
    "properties": [
        {
            "name": "name",
            "dataType": ["text"],
            "description": "The name of the item"
        },
        {
            "name": "description",
            "dataType": ["text"],
            "description": "A longer description of the item"
        }
    ],
    # Optional: Specify a vectorizer.
    # If you don't specify one and haven't set a default vectorizer module in Weaviate's config,
    # Weaviate will create a class without a vectorizer (meaning you'd need to provide vectors manually if needed).
    # For a simple start, you can often omit this, or use 'none'.
    # "vectorizer": "none", # Explicitly no vectorizer
    # Or, if you deployed Weaviate with a module like text2vec-transformers:
    # "vectorizer": "text2vec-transformers",
    # "moduleConfig": {
    #     "text2vec-transformers": {
    #         "vectorizeClassName": True # Whether to vectorize the class name
    #     }
    # }
}

# --- Create the Collection ---
try:
    # First, check if the class already exists to avoid errors if you run the script multiple times
    if not client.schema.exists(collection_name):
        client.schema.create_class(collection_schema)
        print(f"Collection '{collection_name}' created successfully!")
    else:
        print(f"Collection '{collection_name}' already exists. Skipping creation.")

    # --- Verify the Collection ---
    print(f"\nVerifying collection '{collection_name}':")
    retrieved_schema = client.schema.get(collection_name)
    print("Schema retrieved:")
    import json
    print(json.dumps(retrieved_schema, indent=2))

except Exception as e:
    print(f"Error during collection creation or verification: {e}")