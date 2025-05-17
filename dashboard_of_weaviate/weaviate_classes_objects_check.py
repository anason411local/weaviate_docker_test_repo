#!/usr/bin/env python3

import weaviate
from weaviate.embedded import EmbeddedOptions
import pandas as pd
from tabulate import tabulate
from typing import Dict, List, Any
import requests
import os
from dotenv import load_dotenv
import json
import time
from datetime import datetime
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import uuid
import csv

# Load environment variables for API key if needed
load_dotenv()
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

# Colors for terminal output
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
ENDC = "\033[0m"
BOLD = "\033[1m"

# Output directory
OUTPUT_DIR = "weaviate_inspection_reports"

def ensure_output_dir():
    """Ensure output directory exists"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def get_weaviate_meta_info(base_url, headers):
    """Get Weaviate server metadata"""
    try:
        response = requests.get(f"{base_url}/v1/meta", headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            print_error(f"Failed to get metadata: {response.status_code} - {response.text}")
            return {}
    except Exception as e:
        print_error(f"Error getting metadata: {e}")
        return {}

def get_weaviate_health(base_url, headers):
    """Check Weaviate health status"""
    try:
        response = requests.get(f"{base_url}/v1/.well-known/health", headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            print_error(f"Failed to get health status: {response.status_code} - {response.text}")
            return {}
    except Exception as e:
        print_error(f"Error getting health status: {e}")
        return {}

def count_objects_via_graphql(class_name, base_url, headers):
    """Count objects in a class using GraphQL directly"""
    start_time = time.time()
    try:
        # GraphQL query to count objects
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
            headers=headers,
            json=graphql_query
        )
        
        if response.status_code == 200:
            result = response.json()
            if "errors" in result:
                print_warning(f"GraphQL errors for {class_name}: {result['errors']}")
                return 0, time.time() - start_time
                
            # Extract count from response
            count = result.get("data", {}).get("Aggregate", {}).get(class_name, [{}])
            if count and len(count) > 0:
                return count[0].get("meta", {}).get("count", 0), time.time() - start_time
            return 0, time.time() - start_time
        else:
            print_error(f"Error in GraphQL count query: {response.status_code} - {response.text}")
            return 0, time.time() - start_time
    except Exception as e:
        print_error(f"Error counting objects via GraphQL for {class_name}: {e}")
        return 0, time.time() - start_time

def get_object_by_id(class_name, object_id, base_url, headers):
    """Get a specific object by ID"""
    try:
        graphql_query = {
            "query": f"""
            {{
              Get {{
                {class_name}(where: {{
                  path: ["_additional", "id"]
                  operator: Equal
                  valueString: "{object_id}"
                }}) {{
                  _additional {{
                    id
                    vector
                    creationTimeUnix
                  }}
                }}
              }}
            }}
            """
        }
        
        response = requests.post(
            f"{base_url}/v1/graphql",
            headers=headers,
            json=graphql_query
        )
        
        if response.status_code == 200:
            result = response.json()
            if "errors" in result:
                print_warning(f"GraphQL errors getting object by ID: {result['errors']}")
                return None
                
            # Extract object from response
            objects = result.get("data", {}).get("Get", {}).get(class_name, [])
            if objects and len(objects) > 0:
                return objects[0]
            return None
        else:
            print_error(f"Error fetching object by ID: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print_error(f"Error fetching object by ID: {e}")
        return None

def run_benchmark_query(class_name, base_url, headers, num_trials=5):
    """Run benchmark queries and return performance metrics"""
    query_times = []
    
    for i in range(num_trials):
        # Simple count query
        start_time = time.time()
        _, query_time = count_objects_via_graphql(class_name, base_url, headers)
        query_times.append(query_time)
        
    avg_time = sum(query_times) / len(query_times)
    min_time = min(query_times)
    max_time = max(query_times)
    
    return {
        "avg_query_time": avg_time,
        "min_query_time": min_time,
        "max_query_time": max_time,
        "trials": num_trials
    }

def get_sample_objects(class_name, base_url, headers, limit=3):
    """Get sample objects with all properties"""
    start_time = time.time()
    try:
        # First get all property names
        schema_response = requests.get(f"{base_url}/v1/schema/{class_name}", headers=headers)
        if schema_response.status_code != 200:
            print_error(f"Error getting schema for {class_name}: {schema_response.status_code}")
            return [], 0
            
        schema = schema_response.json()
        properties = [prop.get("name") for prop in schema.get("properties", [])]
        
        # Create dynamic GraphQL query with all properties
        properties_query = " ".join(properties)
        
        graphql_query = {
            "query": f"""
            {{
              Get {{
                {class_name}(limit: {limit}) {{
                  {properties_query}
                  _additional {{
                    id
                    vector
                    creationTimeUnix
                  }}
                }}
              }}
            }}
            """
        }
        
        response = requests.post(
            f"{base_url}/v1/graphql",
            headers=headers,
            json=graphql_query
        )
        
        query_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            if "errors" in result:
                print_warning(f"GraphQL errors getting objects for {class_name}: {result['errors']}")
                return [], query_time
                
            # Extract objects from response
            objects = result.get("data", {}).get("Get", {}).get(class_name, [])
            return objects, query_time
        else:
            print_error(f"Error fetching objects: {response.status_code} - {response.text}")
            return [], query_time
    except Exception as e:
        print_error(f"Error fetching objects for {class_name}: {e}")
        return [], time.time() - start_time

def analyze_property_types(properties):
    """Analyze property types and counts"""
    type_counts = {}
    for prop in properties:
        data_type = prop.get("dataType", ["unknown"])[0]
        type_counts[data_type] = type_counts.get(data_type, 0) + 1
    
    return type_counts

def estimate_collection_size(collection_name, object_count, sample_objects, vector_dimension):
    """Estimate the storage size of a collection based on sample objects"""
    if not sample_objects or object_count == 0:
        return {
            "estimated_size_bytes": 0,
            "estimated_size_mb": 0,
            "per_object_bytes": 0
        }
    
    # Calculate size of vector data (assuming float32 - 4 bytes per dimension)
    vector_bytes_per_object = vector_dimension * 4 if vector_dimension and vector_dimension != "N/A" else 0
    
    # Calculate average size of properties based on samples
    total_prop_size = 0
    for obj in sample_objects:
        obj_size = 0
        for key, value in obj.items():
            if key != "_additional":
                # Rough estimate based on JSON serialization
                if isinstance(value, str):
                    obj_size += len(value.encode('utf-8'))
                elif isinstance(value, (int, float)):
                    obj_size += 8
                elif isinstance(value, bool):
                    obj_size += 1
                elif isinstance(value, (list, dict)):
                    obj_size += len(json.dumps(value).encode('utf-8'))
        total_prop_size += obj_size
    
    avg_prop_size = total_prop_size / len(sample_objects) if sample_objects else 0
    
    # Add overhead for indices, metadata, etc. (rough estimate: 20%)
    overhead_factor = 1.2
    
    per_object_bytes = (avg_prop_size + vector_bytes_per_object) * overhead_factor
    total_bytes = per_object_bytes * object_count
    
    return {
        "estimated_size_bytes": total_bytes,
        "estimated_size_mb": total_bytes / (1024 * 1024),
        "per_object_bytes": per_object_bytes
    }

def export_to_json(data, filename):
    """Export data to JSON file"""
    try:
        ensure_output_dir()
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        return filepath
    except Exception as e:
        print_error(f"Error exporting to JSON: {e}")
        return None

def export_to_csv(data, filename):
    """Export data to CSV file"""
    try:
        ensure_output_dir()
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        if isinstance(data, list) and len(data) > 0:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
        return filepath
    except Exception as e:
        print_error(f"Error exporting to CSV: {e}")
        return None

def format_size(size_bytes):
    """Format bytes to human-readable size"""
    if size_bytes == 0:
        return "0 B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

def main():
    """Main function to check Weaviate database and summarize classes and objects"""
    # Generate a unique ID for this inspection run
    inspection_id = str(uuid.uuid4())[:8]
    inspection_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_id = f"{inspection_timestamp}_{inspection_id}"
    
    print_header("WEAVIATE DATABASE INSPECTION TOOL")
    print(f"{BOLD}Report ID:{ENDC} {report_id}")
    print(f"{BOLD}Started at:{ENDC} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{BOLD}Connecting to Weaviate...{ENDC}")
    
    base_url = "http://localhost:8090"
    headers = {"Content-Type": "application/json"}
    
    if WEAVIATE_API_KEY:
        headers["Authorization"] = f"Bearer {WEAVIATE_API_KEY}"
        print(f"Using API key: {'*' * (len(WEAVIATE_API_KEY) if WEAVIATE_API_KEY else 0)}")
    else:
        print("No API key provided")
    
    # Summary data for final report
    inspection_report = {
        "id": report_id,
        "timestamp": datetime.now().isoformat(),
        "server_info": {},
        "health_check": {},
        "collections": [],
        "performance": {},
        "storage": {}
    }
    
    # First check if Weaviate is accessible via REST API
    try:
        start_time = time.time()
        response = requests.get(f"{base_url}/v1/.well-known/ready", headers=headers)
        response_time = time.time() - start_time
        
        if response.status_code != 200:
            print_error(f"Weaviate is not ready: {response.status_code} - {response.text}")
            return
        print_success(f"Weaviate is ready (response time: {response_time:.3f}s)")
        
        # Get Weaviate health status
        print_subheader("HEALTH CHECK")
        health_info = get_weaviate_health(base_url, headers)
        
        if health_info:
            status = "Healthy" if health_info.get("status", "") == "OK" else "Unhealthy"
            print(f"{BOLD}Status:{ENDC} {status}")
            inspection_report["health_check"] = health_info
        
        # Get Weaviate metadata
        print_subheader("WEAVIATE SERVER INFORMATION")
        meta_info = get_weaviate_meta_info(base_url, headers)
        
        if meta_info:
            version = meta_info.get("version", "Unknown")
            print(f"{BOLD}Version:{ENDC} {version}")
            
            if "modules" in meta_info:
                print(f"\n{BOLD}Modules:{ENDC}")
                for module_name, module_info in meta_info.get("modules", {}).items():
                    print(f"  - {module_name}: {module_info.get('version', 'Unknown')}")
            
            if "hostname" in meta_info:
                print(f"\n{BOLD}Hostname:{ENDC} {meta_info.get('hostname')}")
                
            inspection_report["server_info"] = meta_info
        
        # Get schema via REST API
        print_subheader("RETRIEVING SCHEMA")
        schema_response = requests.get(f"{base_url}/v1/schema", headers=headers)
        if schema_response.status_code != 200:
            print_error(f"Error getting schema: {schema_response.status_code} - {schema_response.text}")
            return
            
        schema = schema_response.json()
        
        # Process collections info from schema
        class_details = []
        total_objects = 0
        total_storage_bytes = 0
        benchmark_results = {}
        
        if "classes" not in schema or not schema["classes"]:
            print_error("No classes found in the Weaviate database.")
            return
            
        class_count = len(schema["classes"])
        print_success(f"Found {class_count} classes/collections")
        
        # Check for shards information
        try:
            shards_response = requests.get(f"{base_url}/v1/nodes", headers=headers)
            if shards_response.status_code == 200:
                shards_info = shards_response.json()
                if "nodes" in shards_info:
                    print_subheader("CLUSTER INFORMATION")
                    print(f"{BOLD}Cluster Size:{ENDC} {len(shards_info['nodes'])} nodes")
                    for i, node in enumerate(shards_info['nodes']):
                        print(f"Node {i+1}: {node.get('name', 'Unknown')} - Status: {node.get('status', 'Unknown')}")
                    
                    inspection_report["cluster"] = {
                        "size": len(shards_info['nodes']),
                        "nodes": shards_info['nodes']
                    }
        except Exception as e:
            print_warning(f"Could not retrieve cluster information: {e}")
        
        # Process each class
        print_subheader("COLLECTION DETAILS")
        
        # Create a collection summary dataframe
        collection_summary = []
        
        for class_info in schema["classes"]:
            class_name = class_info.get("class", "Unknown")
            print(f"\n{BOLD}{BLUE}Collection: {class_name}{ENDC}")
            
            collection_data = {
                "name": class_name,
                "properties": [],
                "objects": {},
                "vector_config": {},
                "storage": {},
                "performance": {}
            }
            
            # Run benchmark queries
            print(f"{BOLD}Running Performance Benchmark...{ENDC}")
            benchmark = run_benchmark_query(class_name, base_url, headers)
            print(f"  - Average Query Time: {benchmark['avg_query_time']:.6f}s")
            print(f"  - Min Query Time: {benchmark['min_query_time']:.6f}s")
            print(f"  - Max Query Time: {benchmark['max_query_time']:.6f}s")
            
            benchmark_results[class_name] = benchmark
            collection_data["performance"] = benchmark
            
            # Count objects using GraphQL
            object_count, count_time = count_objects_via_graphql(class_name, base_url, headers)
            total_objects += object_count
            print(f"{BOLD}Objects Count:{ENDC} {object_count} (query took {count_time:.6f}s)")
            
            collection_data["objects"]["count"] = object_count
            collection_data["objects"]["count_query_time"] = count_time
            
            # Get and analyze properties
            properties = class_info.get("properties", [])
            property_count = len(properties)
            print(f"{BOLD}Properties Count:{ENDC} {property_count}")
            
            # Store property details
            collection_data["properties"] = properties
            
            # Analyze property types
            property_types = analyze_property_types(properties)
            print(f"{BOLD}Property Types:{ENDC}")
            for prop_type, count in property_types.items():
                print(f"  - {prop_type}: {count}")
            
            collection_data["property_types"] = property_types
            
            # Get vector config
            vector_config_type = class_info.get("vectorizer", "None")
            vector_dimension = "N/A"
            if "vectorIndexConfig" in class_info:
                vector_config = class_info.get("vectorIndexConfig", {})
                vector_dimension = vector_config.get("dimension", "N/A")
                
            print(f"{BOLD}Vector Configuration:{ENDC}")
            print(f"  - Type: {vector_config_type}")
            print(f"  - Dimension: {vector_dimension}")
            
            collection_data["vector_config"] = {
                "type": vector_config_type,
                "dimension": vector_dimension
            }
            
            # Check for inverted index config
            inverted_config = class_info.get("invertedIndexConfig", {})
            if inverted_config:
                print(f"{BOLD}Inverted Index Configuration:{ENDC}")
                for key, value in inverted_config.items():
                    print(f"  - {key}: {value}")
                
                collection_data["inverted_index_config"] = inverted_config
            
            # Get a sample object if any exist
            if object_count > 0:
                print(f"\n{BOLD}Sample Objects:{ENDC}")
                sample_objects, sample_time = get_sample_objects(class_name, base_url, headers, limit=3)
                
                collection_data["objects"]["sample_query_time"] = sample_time
                
                if sample_objects:
                    collection_data["objects"]["samples"] = []
                    
                    for i, obj in enumerate(sample_objects):
                        print(f"\n  {BOLD}Object {i+1}:{ENDC}")
                        
                        obj_summary = {"properties": {}}
                        
                        # Get object ID
                        obj_id = obj.get("_additional", {}).get("id", "Unknown")
                        print(f"  - ID: {obj_id}")
                        obj_summary["id"] = obj_id
                        
                        # Get creation time
                        creation_time = obj.get("_additional", {}).get("creationTimeUnix", 0)
                        if creation_time:
                            try:
                                creation_date = datetime.fromtimestamp(creation_time/1000).strftime('%Y-%m-%d %H:%M:%S')
                                print(f"  - Created: {creation_date}")
                                obj_summary["created"] = creation_date
                            except:
                                print(f"  - Created: {creation_time}")
                                obj_summary["created"] = creation_time
                        
                        # Check if vector exists
                        vector = obj.get("_additional", {}).get("vector", [])
                        if vector:
                            print(f"  - Vector Length: {len(vector)}")
                            print(f"  - Vector Sample: [{', '.join(str(round(x, 4)) for x in vector[:3])}...]")
                            obj_summary["vector_length"] = len(vector)
                        
                        # Show a few key properties (text, source, etc)
                        key_props = ["text", "title", "name", "source", "filename"]
                        for prop in key_props:
                            if prop in obj and obj[prop]:
                                value = obj[prop]
                                if isinstance(value, str) and len(value) > 100:
                                    display_value = value[:100] + "..."
                                else:
                                    display_value = value
                                print(f"  - {prop}: {display_value}")
                                obj_summary["properties"][prop] = value
                        
                        # Show the number of properties
                        object_property_count = len([k for k in obj.keys() if k != "_additional"])
                        print(f"  - Properties: {object_property_count}")
                        
                        collection_data["objects"]["samples"].append(obj_summary)
                    
                    # Estimate storage size
                    if isinstance(vector_dimension, (int, float)) and vector_dimension > 0:
                        size_estimate = estimate_collection_size(class_name, object_count, sample_objects, vector_dimension)
                        total_storage_bytes += size_estimate["estimated_size_bytes"]
                        
                        print(f"\n{BOLD}Storage Estimation:{ENDC}")
                        print(f"  - Per Object: {format_size(size_estimate['per_object_bytes'])}")
                        print(f"  - Total Collection: {format_size(size_estimate['estimated_size_bytes'])}")
                        
                        collection_data["storage"] = size_estimate
            
            # Add this class to the summary
            collection_summary.append({
                "Collection": class_name,
                "Objects": object_count,
                "Properties": property_count,
                "Vector Type": vector_config_type,
                "Dimension": vector_dimension,
                "Data Types": ", ".join([f"{k}({v})" for k, v in property_types.items()]),
                "Avg Query Time": f"{benchmark['avg_query_time']:.4f}s"
            })
            
            # Add this class to the detailed output
            property_names = ", ".join([p.get("name", "Unknown") for p in properties[:5]])
            if len(properties) > 5:
                property_names += "..."
                
            class_details.append({
                "Class Name": class_name,
                "Objects Count": object_count,
                "Properties": property_names,
                "Property Count": property_count,
                "Vector Index Type": vector_config_type,
                "Vector Dimension": vector_dimension
            })
            
            # Add collection data to the report
            inspection_report["collections"].append(collection_data)
        
        # Create a properties table for all classes
        print_subheader("PROPERTY DETAILS BY COLLECTION")
        for class_info in schema["classes"]:
            class_name = class_info.get("class", "Unknown")
            properties = class_info.get("properties", [])
            
            if properties:
                print(f"\n{BOLD}{BLUE}Collection: {class_name} ({len(properties)} properties){ENDC}")
                
                # Create property details table
                property_details = []
                for prop in properties:
                    prop_name = prop.get("name", "Unknown")
                    prop_type = prop.get("dataType", ["unknown"])[0]
                    prop_description = prop.get("description", "")
                    prop_indexing = "Yes" if prop.get("indexInverted", True) else "No"
                    prop_tokenization = prop.get("tokenization", "word")
                    
                    property_details.append({
                        "Name": prop_name,
                        "Type": prop_type,
                        "Description": prop_description[:30] + "..." if len(prop_description) > 30 else prop_description,
                        "Indexed": prop_indexing,
                        "Tokenization": prop_tokenization
                    })
                
                # Print properties table
                props_df = pd.DataFrame(property_details)
                print(tabulate(props_df, headers="keys", tablefmt="grid", showindex=False))
                
        # Print summary table
        print_header("DATABASE SUMMARY")
        summary_df = pd.DataFrame(collection_summary)
        print(tabulate(summary_df, headers="keys", tablefmt="grid", showindex=False))
        
        # Performance statistics
        print_subheader("PERFORMANCE METRICS")
        if benchmark_results:
            print(f"{BOLD}Query Performance:{ENDC}")
            for class_name, result in benchmark_results.items():
                print(f"  - {class_name}: Avg {result['avg_query_time']:.6f}s, Min {result['min_query_time']:.6f}s, Max {result['max_query_time']:.6f}s")
        
        inspection_report["performance"]["benchmarks"] = benchmark_results
        
        # Storage statistics
        print_subheader("STORAGE ESTIMATION")
        print(f"{BOLD}Total Estimated Storage:{ENDC} {format_size(total_storage_bytes)}")
        inspection_report["storage"]["total_bytes"] = total_storage_bytes
        inspection_report["storage"]["total_formatted"] = format_size(total_storage_bytes)
        
        # Overall statistics
        print_subheader("STATISTICS")
        print(f"{BOLD}Total Collections:{ENDC} {class_count}")
        print(f"{BOLD}Total Objects:{ENDC} {total_objects}")
        print(f"{BOLD}Average Objects per Collection:{ENDC} {total_objects/class_count if class_count else 0:.2f}")
        
        total_properties = sum(len(class_info.get("properties", [])) for class_info in schema["classes"])
        print(f"{BOLD}Total Properties:{ENDC} {total_properties}")
        print(f"{BOLD}Average Properties per Collection:{ENDC} {total_properties/class_count if class_count else 0:.2f}")
        
        # Export report
        json_path = export_to_json(inspection_report, f"weaviate_inspection_{report_id}.json")
        csv_path = export_to_csv(collection_summary, f"weaviate_summary_{report_id}.csv")
        
        print_subheader("REPORT EXPORT")
        if json_path:
            print_success(f"Full inspection report exported to: {json_path}")
        if csv_path:
            print_success(f"Summary report exported to: {csv_path}")
        
        # Recommendations
        print_subheader("RECOMMENDATIONS")
        
        # Check for collections with no objects
        empty_collections = [c["name"] for c in inspection_report["collections"] if c["objects"].get("count", 0) == 0]
        if empty_collections:
            print_warning(f"Found {len(empty_collections)} empty collections: {', '.join(empty_collections)}")
            print_info("Consider removing unused collections to optimize your database")
        
        # Check for slow query performance
        slow_threshold = 0.1  # seconds
        slow_collections = [c["name"] for c in inspection_report["collections"] 
                           if c["performance"].get("avg_query_time", 0) > slow_threshold]
        if slow_collections:
            print_warning(f"Collections with slow query times (>{slow_threshold}s): {', '.join(slow_collections)}")
            print_info("Consider optimizing these collections or checking their configuration")
        
        # Check for large storage collections
        large_threshold = 100 * 1024 * 1024  # 100 MB
        large_collections = [c["name"] for c in inspection_report["collections"] 
                            if c.get("storage", {}).get("estimated_size_bytes", 0) > large_threshold]
        if large_collections:
            print_warning(f"Large collections (>{format_size(large_threshold)}): {', '.join(large_collections)}")
            print_info("Consider monitoring these collections for growth and potential sharding")
        
        print_subheader("INSPECTION COMPLETE")
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Inspection completed in {total_time:.2f} seconds")
        print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Report ID: {report_id}")
        
    except Exception as e:
        print_error(f"Error accessing Weaviate: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Make sure to run this in the test_env conda environment
    # using: conda activate test_env
    main() 