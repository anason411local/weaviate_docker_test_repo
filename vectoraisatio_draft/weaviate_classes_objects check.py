#!/usr/bin/env python3

import weaviate
import pandas as pd
from tabulate import tabulate
from typing import Dict, List, Any

# Connect to Weaviate
client = weaviate.Client(
    url="http://127.0.0.1:8090"
)

def get_schema() -> Dict[str, Any]:
    """Get the schema from Weaviate"""
    return client.schema.get()

def get_class_objects_count(class_name: str) -> int:
    """Get count of objects for a specific class"""
    try:
        result = client.query.aggregate(class_name).with_meta_count().do()
        return result["data"]["Aggregate"][class_name][0]["meta"]["count"]
    except Exception as e:
        print(f"Error getting count for class {class_name}: {e}")
        return 0

def get_class_properties(class_schema: Dict[str, Any]) -> List[str]:
    """Extract property names from a class schema"""
    if "properties" not in class_schema:
        return []
    return [prop["name"] for prop in class_schema["properties"]]

def get_class_details(schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract details for all classes in schema"""
    if "classes" not in schema:
        return []
    
    class_details = []
    for cls in schema["classes"]:
        class_name = cls["class"]
        objects_count = get_class_objects_count(class_name)
        properties = get_class_properties(cls)
        
        class_details.append({
            "Class Name": class_name,
            "Objects Count": objects_count,
            "Properties": ", ".join(properties),
            "Vector Index Type": cls.get("vectorIndexType", "None"),
            "Vector Dimension": cls.get("vectorIndexConfig", {}).get("dimension", "N/A")
        })
    
    return class_details

def main():
    print("Connecting to Weaviate at http://127.0.0.1:8090...")
    
    try:
        # Check if Weaviate is ready
        if not client.is_ready():
            print("Weaviate is not ready. Please check the connection.")
            return
        
        print("Weaviate connection successful!")
        
        # Get schema
        schema = get_schema()
        
        # Extract class details
        class_details = get_class_details(schema)
        
        if not class_details:
            print("No classes found in the Weaviate database.")
            return
        
        # Convert to DataFrame for tabulation
        df = pd.DataFrame(class_details)
        
        # Print tabulated output
        print("\n=== Weaviate Database Summary ===\n")
        print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))
        
        print(f"\nTotal Classes: {len(class_details)}")
        print(f"Total Objects: {sum(cls['Objects Count'] for cls in class_details)}")
        
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
