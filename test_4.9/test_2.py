import weaviate
# Import DataType from its correct location
from weaviate.classes.config import DataType
# The alias 'wvc' might still be useful if other submodules like 'wvc.query' are used.
# import weaviate.classes as wvc # Keep if other wvc.submodule are used, otherwise can be removed if DataType is the only thing from it

# --- Configuration ---
WEAVIATE_HOST = "127.0.0.1"
WEAVIATE_PORT = 8090
OBJECT_LIMIT_PER_CLASS = 5 # How many objects to display per class

def describe_weaviate_instance():
    print(f"Attempting to connect to Weaviate at {WEAVIATE_HOST}:{WEAVIATE_PORT}...")
    try:
        # --- 1. Connect to Weaviate ---
        client = weaviate.connect_to_local(
            host=WEAVIATE_HOST,
            port=WEAVIATE_PORT
        )
        if client.is_ready():
            print(f"Successfully connected to Weaviate at {WEAVIATE_HOST}:{WEAVIATE_PORT}")
        else:
            print(f"Could not connect to Weaviate at {WEAVIATE_HOST}:{WEAVIATE_PORT}. Exiting.")
            return

        # --- 2. Get all Class Schemas ---
        print("\n--- Weaviate Instance Schemas ---")
        all_schemas = client.collections.list_all(simple=False) # simple=False gives detailed config

        if not all_schemas:
            print("No classes found in this Weaviate instance.")
            return

        for class_name, class_config in all_schemas.items():
            print(f"\n-----------------------------------------")
            print(f"Class Name: {class_config.name}")
            print(f"-----------------------------------------")

            if class_config.description:
                print(f"  Description: {class_config.description}")

            # --- 3. Display Class Properties ---
            print(f"\n  Properties for Class '{class_config.name}':")
            if class_config.properties:
                for prop in class_config.properties:
                    print(f"    - Name: {prop.name}")
                    # prop.data_type is an enum instance, printing it gives its string representation
                    print(f"      Data Type: {prop.data_type}")
                    if hasattr(prop, 'description') and prop.description:
                        print(f"      Description: {prop.description}")
                    
                    # Check if this is a reference property (no longer using DataType.CROSS_REFERENCE)
                    if hasattr(prop, 'target_collection') and prop.target_collection:
                        print(f"      Target Collections: {prop.target_collection if isinstance(prop.target_collection, list) else [prop.target_collection]}")
                    # Add more property details if needed (e.g., tokenization, indexing)
                    print("-" * 20)
            else:
                print("    No properties defined for this class.")

            # --- 4. Display Objects from the Class ---
            print(f"\n  Objects in Class '{class_config.name}' (limit {OBJECT_LIMIT_PER_CLASS}):")
            try:
                collection = client.collections.get(class_config.name)
                response = collection.query.fetch_objects(
                    limit=OBJECT_LIMIT_PER_CLASS,
                    # Example for fetching specific metadata (requires weaviate.classes.query.MetadataQuery)
                    # from weaviate.classes.query import MetadataQuery # if you need this
                    # return_metadata=MetadataQuery(uuid=True, last_update_time_unix=True)
                )

                if response.objects:
                    for i, obj in enumerate(response.objects):
                        print(f"\n    Object {i+1}:")
                        print(f"      Weaviate UUID: {obj.uuid}")
                        print(f"      Properties:")
                        for prop_name, prop_value in obj.properties.items():
                            display_value = prop_value
                            if isinstance(prop_value, str) and len(prop_value) > 70:
                                display_value = prop_value[:67] + "..."
                            elif isinstance(prop_value, list) and len(prop_value) > 5:
                                display_value = str(prop_value[:10]) + f"... (and {len(prop_value)-10} more)"
                            print(f"        {prop_name}: {display_value}")
                        # if obj.metadata: # If metadata was requested and returned
                        #     print(f"      Metadata:")
                        #     if obj.metadata.uuid: # Check if specific metadata fields exist
                        #         print(f"        UUID (from metadata): {obj.metadata.uuid}")
                        #     if hasattr(obj.metadata, 'last_update_time_unix') and obj.metadata.last_update_time_unix:
                        #          print(f"        Last Update: {obj.metadata.last_update_time_unix}")
                else:
                    print("    No objects found in this class (or none within the limit).")

            except Exception as e:
                print(f"    Error fetching objects for class '{class_config.name}': {e}")

        print(f"\n-----------------------------------------")

    except ConnectionRefusedError:
        print(f"ERROR: Connection refused. Is Weaviate running at {WEAVIATE_HOST}:{WEAVIATE_PORT}?")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'client' in locals() and client.is_connected():
            print("\nClosing Weaviate connection.")
            client.close()

if __name__ == "__main__":
    describe_weaviate_instance()