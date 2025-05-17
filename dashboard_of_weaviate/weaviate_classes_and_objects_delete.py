#!/usr/bin/env python3

import requests
import os
import json
from dotenv import load_dotenv
from tabulate import tabulate
import time
from datetime import datetime

# Load environment variables for API key if needed
load_dotenv()
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

# Colors for terminal output
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
ENDC = "\033[0m"
BOLD = "\033[1m"

def print_header(text):
    """Print a formatted header"""
    print(f"\n{BOLD}{BLUE}{'='*20} {text} {'='*20}{ENDC}\n")

def print_subheader(text):
    """Print a formatted subheader"""
    print(f"\n{BOLD}{GREEN}--- {text} ---{ENDC}\n")

def print_warning(text):
    """Print a warning message"""
    print(f"{YELLOW}⚠️  {text}{ENDC}")

def print_error(text):
    """Print an error message"""
    print(f"{RED}❌ {text}{ENDC}")

def print_success(text):
    """Print a success message"""
    print(f"{GREEN}✅ {text}{ENDC}")

def print_info(text):
    """Print an info message"""
    print(f"{CYAN}ℹ️ {text}{ENDC}")

def create_weaviate_headers():
    """Create headers for Weaviate API requests"""
    headers = {"Content-Type": "application/json"}
    if WEAVIATE_API_KEY:
        headers["Authorization"] = f"Bearer {WEAVIATE_API_KEY}"
    return headers

def check_weaviate_connection(base_url):
    """Check if Weaviate is accessible"""
    try:
        response = requests.get(f"{base_url}/v1/.well-known/ready", headers=create_weaviate_headers())
        if response.status_code == 200:
            print_success("Weaviate instance is ready")
            return True
        else:
            print_error(f"Weaviate instance not ready: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print_error(f"Error connecting to Weaviate: {e}")
        return False

def get_schema(base_url):
    """Get the schema from Weaviate"""
    try:
        response = requests.get(f"{base_url}/v1/schema", headers=create_weaviate_headers())
        if response.status_code == 200:
            return response.json()
        else:
            print_error(f"Error getting schema: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print_error(f"Error retrieving schema: {e}")
        return None

def count_objects(class_name, base_url):
    """Count objects in a class using GraphQL"""
    try:
        graphql_query = {
            "query": f"""
            {{
              Aggregate {{
                {class_name} {{
                  meta {{
                    count
                  }}
                }}
              }}
            }}
            """
        }
        
        response = requests.post(
            f"{base_url}/v1/graphql",
            headers=create_weaviate_headers(),
            json=graphql_query
        )
        
        if response.status_code == 200:
            result = response.json()
            if "errors" in result:
                print_warning(f"GraphQL errors for {class_name}: {result['errors']}")
                return 0
                
            # Extract count from response
            count = result.get("data", {}).get("Aggregate", {}).get(class_name, [{}])
            if count and len(count) > 0:
                return count[0].get("meta", {}).get("count", 0)
            return 0
        else:
            print_error(f"Error in GraphQL count query: {response.status_code} - {response.text}")
            return 0
    except Exception as e:
        print_error(f"Error counting objects for {class_name}: {e}")
        return 0

def list_classes_with_counts(base_url):
    """List all classes with their object counts"""
    schema = get_schema(base_url)
    if not schema:
        return []
    
    if "classes" not in schema or not schema["classes"]:
        print_info("No classes found in the Weaviate database")
        return []
    
    print_subheader("RETRIEVING CLASSES AND OBJECT COUNTS")
    
    classes_with_counts = []
    for i, class_info in enumerate(schema["classes"]):
        class_name = class_info.get("class", "Unknown")
        print(f"Checking class {i+1}/{len(schema['classes'])}: {class_name}")
        
        # Count objects
        object_count = count_objects(class_name, base_url)
        
        # Get vector config
        vector_config_type = class_info.get("vectorizer", "None")
        vector_dimension = "N/A"
        if "vectorIndexConfig" in class_info:
            vector_config = class_info.get("vectorIndexConfig", {})
            vector_dimension = vector_config.get("dimension", "N/A")
        
        # Get property count
        properties = class_info.get("properties", [])
        property_count = len(properties)
        
        classes_with_counts.append({
            "Class Name": class_name,
            "Objects Count": object_count,
            "Properties Count": property_count,
            "Vector Type": vector_config_type,
            "Dimension": vector_dimension
        })
    
    return classes_with_counts

def delete_class(class_name, base_url):
    """Delete a class from Weaviate"""
    try:
        print_warning(f"Attempting to delete class: {class_name}")
        response = requests.delete(
            f"{base_url}/v1/schema/{class_name}",
            headers=create_weaviate_headers()
        )
        
        if response.status_code in [200, 204]:
            print_success(f"Successfully deleted class: {class_name}")
            return True
        else:
            print_error(f"Error deleting class: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print_error(f"Error during class deletion: {e}")
        return False

def main():
    """Main function to list and delete Weaviate classes"""
    print_header("WEAVIATE CLASSES AND OBJECTS MANAGEMENT")
    print(f"{BOLD}Started at:{ENDC} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup Weaviate connection
    base_url = "http://localhost:8090"
    print(f"Connecting to Weaviate at: {base_url}")
    
    if WEAVIATE_API_KEY:
        print(f"Using API key: {'*' * len(WEAVIATE_API_KEY)}")
    else:
        print("No API key provided")
    
    # Check connection
    if not check_weaviate_connection(base_url):
        return
    
    # List classes with counts
    classes = list_classes_with_counts(base_url)
    
    if not classes:
        print_warning("No classes found or unable to retrieve classes")
        return
    
    # Display classes in a table
    print_subheader("CLASSES AND OBJECT COUNTS")
    headers = ["#", "Class Name", "Objects Count", "Properties Count", "Vector Type", "Dimension"]
    table_data = [
        [i+1, c["Class Name"], c["Objects Count"], c["Properties Count"], c["Vector Type"], c["Dimension"]]
        for i, c in enumerate(classes)
    ]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Menu for deletion
    while True:
        print_subheader("OPTIONS")
        print("1. Delete a specific class")
        print("2. Exit")
        
        choice = input(f"\n{BOLD}Enter your choice (1 or 2):{ENDC} ")
        
        if choice == "2":
            print_info("Exiting program...")
            break
            
        elif choice == "1":
            class_number = input(f"\n{BOLD}Enter the number (#) of the class to delete (or 'c' to cancel):{ENDC} ")
            
            if class_number.lower() == 'c':
                print_info("Operation cancelled")
                continue
                
            try:
                class_idx = int(class_number) - 1
                if 0 <= class_idx < len(classes):
                    class_to_delete = classes[class_idx]["Class Name"]
                    object_count = classes[class_idx]["Objects Count"]
                    
                    # Confirmation message with warnings
                    print_warning(f"You are about to delete class '{class_to_delete}' containing {object_count} objects")
                    print_warning("THIS ACTION CANNOT BE UNDONE!")
                    
                    confirmation = input(f"\n{BOLD}Type the class name '{class_to_delete}' to confirm deletion:{ENDC} ")
                    
                    if confirmation == class_to_delete:
                        if delete_class(class_to_delete, base_url):
                            # Remove from the list if deletion was successful
                            classes.pop(class_idx)
                            
                            # Display updated table
                            print_subheader("UPDATED CLASSES AND OBJECT COUNTS")
                            table_data = [
                                [i+1, c["Class Name"], c["Objects Count"], c["Properties Count"], c["Vector Type"], c["Dimension"]]
                                for i, c in enumerate(classes)
                            ]
                            print(tabulate(table_data, headers=headers, tablefmt="grid"))
                    else:
                        print_warning("Deletion cancelled - class name did not match")
                else:
                    print_error(f"Invalid class number. Please enter a number between 1 and {len(classes)}")
            except ValueError:
                print_error("Invalid input. Please enter a valid number")
        else:
            print_error("Invalid choice. Please enter 1 or 2")

if __name__ == "__main__":
    main()
