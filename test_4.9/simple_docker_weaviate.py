#!/usr/bin/env python3

import weaviate
from weaviate.classes.config import Property, DataType
import time

# Weaviate settings for local Docker
WEAVIATE_HOST = "127.0.0.1"
WEAVIATE_PORT = 8090

def main():
    """Test connection to Weaviate and create a basic collection."""
    print("Testing connection to Weaviate...")
    
    try:
        # Connect to Weaviate
        client = weaviate.connect_to_local(
            host=WEAVIATE_HOST,
            port=WEAVIATE_PORT,
        )
        
        # Check if client is ready
        if not client.is_ready():
            print("❌ Weaviate is not ready. Please check your Docker container.")
            return
        
        print("✅ Connected to Weaviate successfully!")
        
        # Create a simple collection
        if client.collections.exists("SimpleTest"):
            print("Deleting existing SimpleTest collection...")
            client.collections.delete("SimpleTest")
        
        print("Creating SimpleTest collection...")
        collection = client.collections.create(
            name="SimpleTest",
            properties=[
                Property(name="title", data_type=DataType.TEXT),
                Property(name="content", data_type=DataType.TEXT),
            ]
        )
        
        print("✅ Collection created successfully!")
        
        # Insert a test object
        print("Inserting test object...")
        collection.data.insert({
            "title": "Test Document",
            "content": "This is a test document for Weaviate."
        })
        
        print("✅ Object inserted successfully!")
        
        # Wait a moment for the changes to propagate
        print("Waiting for changes to propagate...")
        time.sleep(3)
        
        # Query the object
        print("Querying the object...")
        result = collection.query.fetch_objects(
            limit=1
        )
        
        print("✅ Query successful!")
        print(f"Retrieved object: {result.objects[0].properties}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'client' in locals():
            client.close()

if __name__ == "__main__":
    main() 