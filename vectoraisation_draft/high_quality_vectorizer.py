#!/usr/bin/env python3

import os
import requests
import json
import uuid
import re
import base64
import numpy as np
from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import time
import datetime
import unicodedata
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm

# Image processing imports
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
    print("✅ Pytesseract available for OCR")
except ImportError:
    PYTESSERACT_AVAILABLE = False
    print("⚠️ Pytesseract not available. OCR processing disabled.")

try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    from PIL.ExifTags import TAGS
    PIL_AVAILABLE = True
    print("✅ PIL available for image processing")
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️ PIL not available. Image processing disabled.")

# Load environment variables
load_dotenv()

# --- Configuration ---
# Weaviate connection
WEAVIATE_URL = "http://localhost:8087"
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
COLLECTION_NAME = "Area_expansion_Dep_Anas_V3"

# Set to True to recreate collection (delete existing)
RECREATE_COLLECTION = True

# Settings for improved vector quality
CHUNK_SIZE = 800  # Smaller chunks for better semantic focus
CHUNK_OVERLAP = 160  # Increased overlap for better context continuity
USE_HIERARCHICAL_EMBEDDINGS = True  # Blend document level context with chunks
HIERARCHICAL_WEIGHT = 0.20  # Increased weight for document-level context
VERBOSE_MODE = True  # Show detailed output

# Directory settings
PDF_DIR = "pdfs"  # Directory containing PDF files
TRANSCRIPT_DIR = "transcripts"  # Directory containing transcript files
IMAGE_DIR = "images"  # Directory containing image files

# Parallelization settings
NUM_PROCESSES = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU for system
BATCH_SIZE = 50  # Increased batch size for faster processing

# ----- Initialize required libraries -----
# Try to import optional libraries
try:
    import nltk
    NLTK_AVAILABLE = True
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    STOPWORDS = set(stopwords.words('english'))
    LEMMATIZER = WordNetLemmatizer()
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not available. Text processing will be limited.")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("Warning: PyMuPDF not available. PDF processing will be limited.")

# Initialize embedding model
print("Initializing embedding model...")
model_name = "BAAI/bge-m3"
model_kwargs = {'device': 'cpu', 'trust_remote_code': True}
encode_kwargs = {
    'normalize_embeddings': True,
    'batch_size': 50,  # Increased for better throughput
    'pooling_strategy': 'weighted_mean',  # Better quality than cls for BGE-M3
    'max_length': 8192,  # Increased context window
    'query_instruction_template': "Represent this text accurately for retrieval: {query}"  # Explicit instruction
}

embedding_model = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs)

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", "; ", ":", ", ", " ", ""]
)

# ----- Utility Functions -----
def clean_text(text):
    """Enhanced text cleaning for better vector quality"""
    if not text or not isinstance(text, str):
        return ""
    
    # Basic cleaning
    cleaned = re.sub(r'\s+', ' ', text).strip()
    
    # Remove common PDF artifacts
    artifacts = [
        r'(\n\s*\d+\s*\n)', r'^\s*\d+\s*$', r'©.*?reserved\.?',
        r'(Page|PAGE)(\s+\d+\s+of\s+\d+)', r'(http|https|www)\S+\s',
        r'\[\s*\d+\s*\]', r'(^|[^a-zA-Z0-9])\d{5,}([^a-zA-Z0-9]|$)',
        r'\\[a-zA-Z]+\{.*?\}', r'</?[a-z]+>',
    ]
    for pattern in artifacts:
        cleaned = re.sub(pattern, ' ', cleaned, flags=re.MULTILINE)
    
    # Fix spacing issues
    cleaned = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', cleaned)
    cleaned = re.sub(r'\s+([.,;:!?)])', r'\1', cleaned)
    cleaned = re.sub(r'([(])\s+', r'\1', cleaned)
    
    # Normalize quotes and apostrophes
    cleaned = re.sub(r'["""]', '"', cleaned)
    cleaned = re.sub(r"['']", "'", cleaned)
    
    # Unicode normalization
    cleaned = unicodedata.normalize('NFKC', cleaned)
    
    # Apply NLTK processing if available
    if NLTK_AVAILABLE:
        try:
            # Tokenize into sentences
            sentences = nltk.sent_tokenize(cleaned)
            normalized_sentences = []
            
            for sentence in sentences:
                if not sentence.strip():
                    continue
                
                # Fix capitalization
                if sentence and sentence[0].islower():
                    sentence = sentence[0].upper() + sentence[1:]
                
                # Ensure sentences end with punctuation
                if sentence and sentence[-1] not in ['.', '!', '?']:
                    sentence += '.'
                
                normalized_sentences.append(sentence)
            
            cleaned = ' '.join(normalized_sentences)
            
            # Optional lemmatization for better semantics
            try:
                words = nltk.word_tokenize(cleaned)
                lemmatized = [LEMMATIZER.lemmatize(word) for word in words if len(word) > 1]
                # Blend original with lemmatized for better embeddings
                cleaned = cleaned + " " + " ".join(lemmatized)
            except:
                pass  # Skip lemmatization if it fails
                
        except Exception as e:
            print(f"NLTK processing error: {e}")
    
    # Final cleaning
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    cleaned = re.sub(r'\[PAGE \d+\]\s*', '', cleaned)  # Remove page markers
    
    return cleaned if len(cleaned) >= 50 else ""

def extract_pdf_metadata(pdf_path):
    """Extract metadata from PDF files"""
    default_meta = {
        "title": "", "author": "", "subject": "", "keywords": "",
        "creation_date": "", "modification_date": "", "page_count": 0,
        "file_size": 0, "has_images": False, "image_count": 0
    }
    
    if not PYMUPDF_AVAILABLE or not os.path.exists(pdf_path):
        return default_meta
    
    try:
        doc = fitz.open(pdf_path)
        metadata = default_meta.copy()
        doc_meta = doc.metadata
        
        # Basic metadata
        metadata.update({
            "title": doc_meta.get("title", os.path.basename(pdf_path)),
            "author": doc_meta.get("author", ""),
            "subject": doc_meta.get("subject", ""),
            "keywords": doc_meta.get("keywords", ""),
            "creation_date": doc_meta.get("creationDate", ""),
            "modification_date": doc_meta.get("modDate", ""),
            "page_count": doc.page_count,
            "file_size": os.path.getsize(pdf_path) if os.path.exists(pdf_path) else 0
        })
        
        # Check for images
        for page_idx in range(min(3, doc.page_count)):
            page = doc[page_idx]
            images = page.get_images()
            if images:
                metadata["has_images"] = True
                metadata["image_count"] += len(images)
        
        doc.close()
        return metadata
    except Exception as e:
        print(f"Error extracting PDF metadata: {e}")
        return default_meta

def extract_json_metadata(pdf_path):
    """Get metadata from JSON file associated with PDF"""
    try:
        pdf_dir = os.path.dirname(pdf_path)
        pdf_name = os.path.basename(pdf_path)
        pdf_name_no_ext = os.path.splitext(pdf_name)[0]
        
        # Try to find matching JSON file
        json_path = os.path.join(pdf_dir, f"{pdf_name_no_ext}.json")
        
        if not os.path.exists(json_path):
            # Look for similar filenames
            json_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.json')]
            for json_file in json_files:
                json_name = os.path.splitext(json_file)[0]
                if pdf_name_no_ext.lower() in json_name.lower() or json_name.lower() in pdf_name_no_ext.lower():
                    json_path = os.path.join(pdf_dir, json_file)
                    print(f"Found matching JSON: {json_path} for PDF: {pdf_path}")
                    break
        
        if not os.path.exists(json_path):
            return {}
            
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # Process fields to match schema
        processed_data = {}
        for key, value in json_data.items():
            field_name = f"document_{key}" if not key.startswith("document_") else key
            
            if isinstance(value, (bool, int, float, str)):
                processed_data[field_name] = value
            else:
                try:
                    processed_data[field_name] = json.dumps(value)
                except:
                    processed_data[field_name] = str(value)
        
        return processed_data
    except Exception as e:
        print(f"Error extracting JSON metadata: {e}")
        return {}

def enhance_text_for_embedding(text, metadata=None):
    """Prepare text for optimal embeddings with enhanced instructions"""
    if not text:
        return text
    
    # Extract context from metadata
    title = metadata.get("title", "") if metadata else ""
    subject = metadata.get("subject", "") if metadata else ""
    keywords = metadata.get("keywords", "") if metadata else ""
    author = metadata.get("author", "") if metadata else ""
    
    # Optimized embedding instruction for BGE-M3 model
    instruction = "Represent this document accurately with high-quality semantic content for retrieval: "
    
    # Build rich context string with semantic boosting
    context_parts = []
    if title:
        context_parts.append(f"Title: {title}")
        # Add title twice for semantic emphasis
        context_parts.append(f"Document: {title}")
    if subject and len(subject) > 3:
        context_parts.append(f"Subject: {subject}")
        context_parts.append(f"Topic: {subject}")
    if keywords and len(keywords) > 3:
        context_parts.append(f"Keywords: {keywords}")
        # Emphasize keywords by repeating in different format
        context_parts.append(f"Tags: {keywords}")
    if author and len(author) > 2:
        context_parts.append(f"Author: {author}")
    
    # Add semantic focus markers
    context_parts.append(f"Content Type: Document")
    context_parts.append(f"Importance: High")
    
    context_str = ""
    if context_parts:
        context_str = " | ".join(context_parts) + "\n\n"
    
    # Combine for final embedding text with semantic emphasis
    enhanced_text = f"{instruction}{context_str}{text}"
    
    # Add trailing emphasis for recency bias in transformer models
    enhanced_text += "\n\nThis document contains important information for accurate retrieval and similarity matching."
    
    return enhanced_text

def create_document_summary(source_docs, title=""):
    """Create document-level summary for hierarchical embeddings"""
    if not source_docs:
        return ""
    
    summary_parts = []
    
    # Add title
    if title:
        summary_parts.append(f"Document: {title}")
    
    # Extract headings
    headings = []
    for doc in source_docs[:min(5, len(source_docs))]:
        for line in doc.page_content.split('\n'):
            line = line.strip()
            if 10 < len(line) < 100 and any(c.isupper() for c in line):
                headings.append(line)
                break
    
    if headings:
        summary_parts.append("Sections: " + " | ".join(headings[:5]))
    
    # Add first page content
    if source_docs:
        first_page = source_docs[0].page_content
        first_paragraph = first_page.split('\n\n')[0] if '\n\n' in first_page else first_page[:500]
        summary_parts.append(first_paragraph)
    
    # Combine parts
    summary = "\n\n".join(summary_parts)
    return summary[:1000] if len(summary) > 1000 else summary

def blend_hierarchical_embedding(chunk_vec, doc_vec, weight=HIERARCHICAL_WEIGHT):
    """Blend chunk and document embeddings for hierarchical representation"""
    if chunk_vec is None or doc_vec is None:
        return chunk_vec
    
    try:
        # Convert to numpy arrays if needed
        chunk_np = np.array(chunk_vec)
        doc_np = np.array(doc_vec)
        
        # Weighted blend
        blended = (1 - weight) * chunk_np + weight * doc_np
        
        # Normalize to unit length
        norm = np.linalg.norm(blended)
        if norm > 0:
            return (blended / norm).tolist()
    except Exception as e:
        print(f"Error blending embeddings: {e}")
    
    return chunk_vec

def create_weaviate_headers():
    """Create headers for Weaviate API requests"""
    headers = {"Content-Type": "application/json"}
    if WEAVIATE_API_KEY:
        headers["Authorization"] = f"Bearer {WEAVIATE_API_KEY}"
    return headers

def check_weaviate_connection():
    """Check if Weaviate is accessible"""
    try:
        response = requests.get(
            f"{WEAVIATE_URL}/v1/.well-known/ready",
            headers=create_weaviate_headers()
        )
        if response.status_code == 200:
            print("✅ Weaviate connection successful")
            return True
        else:
            print(f"❌ Weaviate connection error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Weaviate connection error: {e}")
        return False

def discover_json_fields():
    """Discover custom fields from JSON files"""
    fields = {}
    pdf_dir = "pdfs"
    
    if not os.path.exists(pdf_dir):
        return fields
    
    for root, _, files in os.walk(pdf_dir):
        for file in files:
            if file.lower().endswith('.json'):
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    for key, value in data.items():
                        field_name = f"document_{key}"
                        
                        if isinstance(value, bool):
                            field_type = "boolean"
                        elif isinstance(value, int):
                            field_type = "int"
                        elif isinstance(value, float):
                            field_type = "number"
                        else:
                            field_type = "text"
                        
                        fields[field_name] = field_type
                except:
                    pass
    
    print(f"Discovered {len(fields)} custom fields from JSON files")
    return fields

def setup_weaviate_collection():
    """Create or update Weaviate collection"""
    if not check_weaviate_connection():
        return False
    
    headers = create_weaviate_headers()
    collection_name = COLLECTION_NAME
    
    # Define schema with all needed properties
    schema = {
        "class": collection_name,
        "vectorizer": "none",  # We provide vectors manually
        "properties": [
            # Common properties
            {"name": "text", "dataType": ["text"]},
            {"name": "source", "dataType": ["text"]},
            {"name": "filename", "dataType": ["text"]},
            {"name": "folder", "dataType": ["text"]},
            {"name": "title", "dataType": ["text"]},
            {"name": "embedding_model", "dataType": ["text"]},
            {"name": "hierarchical_embedding", "dataType": ["boolean"]},
            {"name": "embedding_enhanced", "dataType": ["boolean"]},
            {"name": "chunk_size", "dataType": ["int"]},
            {"name": "chunk_overlap", "dataType": ["int"]},
            {"name": "file_size", "dataType": ["int"]},
            
            # PDF specific properties
            {"name": "page", "dataType": ["int"]},
            {"name": "author", "dataType": ["text"]},
            {"name": "subject", "dataType": ["text"]},
            {"name": "keywords", "dataType": ["text"]},
            {"name": "creation_date", "dataType": ["text"]},
            {"name": "modification_date", "dataType": ["text"]},
            {"name": "page_count", "dataType": ["int"]},
            {"name": "has_images", "dataType": ["boolean"]},
            {"name": "image_count", "dataType": ["int"]},
            {"name": "has_toc", "dataType": ["boolean"]},
            {"name": "has_links", "dataType": ["boolean"]},
            {"name": "has_forms", "dataType": ["boolean"]},
            
            # Transcript specific properties
            {"name": "content_type", "dataType": ["text"]},
            {"name": "source_type", "dataType": ["text"]},
            {"name": "speaker", "dataType": ["text"]},
            {"name": "speakers", "dataType": ["text"]},
            {"name": "duration", "dataType": ["text"]},
            {"name": "date_recorded", "dataType": ["text"]},
            {"name": "language", "dataType": ["text"]},
            {"name": "word_count", "dataType": ["int"]},
            {"name": "confidence_score", "dataType": ["number"]},
            {"name": "segment", "dataType": ["int"]},
            
            # Enhanced properties for transcripts
            {"name": "platform", "dataType": ["text"]},
            {"name": "version", "dataType": ["text"]},
            {"name": "transcript_source", "dataType": ["text"]},
            {"name": "transcript_quality", "dataType": ["text"]},
            {"name": "transcript_type", "dataType": ["text"]},
            {"name": "event_type", "dataType": ["text"]},
            {"name": "participants", "dataType": ["text"]},
            {"name": "main_topics", "dataType": ["text"]},
            {"name": "extracted_keywords", "dataType": ["text"]},
            {"name": "transcript_complete", "dataType": ["boolean"]},
            
            # Enhanced metadata for all content
            {"name": "processed_date", "dataType": ["text"]},
            {"name": "last_updated", "dataType": ["text"]},
            {"name": "content_category", "dataType": ["text"]},
            {"name": "content_tags", "dataType": ["text[]"]},
            {"name": "document_id", "dataType": ["text"]},
            
            # Image specific properties
            {"name": "width", "dataType": ["int"]},
            {"name": "height", "dataType": ["int"]},
            {"name": "format", "dataType": ["text"]},
            {"name": "mode", "dataType": ["text"]},
            {"name": "dpi", "dataType": ["text"]},
            {"name": "has_exif", "dataType": ["boolean"]},
            {"name": "camera_make", "dataType": ["text"]},
            {"name": "camera_model", "dataType": ["text"]},
            {"name": "datetime_taken", "dataType": ["text"]},
            {"name": "gps_coords", "dataType": ["text"]},
            {"name": "orientation", "dataType": ["text"]},
            {"name": "color_depth", "dataType": ["int"]},
            {"name": "image_quality", "dataType": ["text"]},
            {"name": "ocr_extracted", "dataType": ["boolean"]},
            
            # Google Drive metadata for images
            {"name": "image_id", "dataType": ["text"]},
            {"name": "image_name", "dataType": ["text"]},
            {"name": "image_mimeType", "dataType": ["text"]},
            {"name": "image_createdTime", "dataType": ["text"]},
            {"name": "image_modifiedTime", "dataType": ["text"]},
            {"name": "image_size", "dataType": ["text"]},
            {"name": "image_owner_name", "dataType": ["text"]},
            {"name": "image_owner_email", "dataType": ["text"]},
            {"name": "image_last_modifier", "dataType": ["text"]},
            {"name": "image_last_modifier_email", "dataType": ["text"]}
        ]
    }
    
    # Add custom fields from JSON files
    json_fields = discover_json_fields()
    for field_name, field_type in json_fields.items():
        if not any(p["name"] == field_name for p in schema["properties"]):
            schema["properties"].append({"name": field_name, "dataType": [field_type]})
    
    try:
        # Check if collection exists
        response = requests.get(
            f"{WEAVIATE_URL}/v1/schema/{collection_name}", 
            headers=headers
        )
        
        # Handle existing collection
        if response.status_code == 200:
            print(f"\nCollection '{collection_name}' already exists.")
            if RECREATE_COLLECTION:
                print("\nOptions:")
                print("1. Delete and recreate collection (all data will be lost)")
                print("2. Update existing collection (keep existing data)")
                print("3. Skip collection setup")
                
                while True:
                    choice = input("\nEnter your choice (1-3): ").strip()
                    if choice == "1":
                        confirm = input("WARNING: This will delete all existing data. Are you sure? (yes/no): ").strip().lower()
                        if confirm == "yes":
                            print(f"Deleting existing collection '{collection_name}'...")
                            delete_response = requests.delete(
                                f"{WEAVIATE_URL}/v1/schema/{collection_name}", 
                                headers=headers
                            )
                            if delete_response.status_code not in [200, 204, 404]:
                                print(f"Error deleting collection: {delete_response.status_code}")
                                return False
                            print(f"✅ Collection deleted successfully")
                            response.status_code = 404  # Mark as not existing
                            break
                        else:
                            print("Operation cancelled.")
                            return False
                    elif choice == "2":
                        print(f"Updating existing collection '{collection_name}'...")
                        # Check if schema needs updating
                        current_schema = response.json()
                        current_properties = {p.get("name"): True for p in current_schema.get("properties", [])}
                        
                        # Add any missing properties
                        for prop in schema["properties"]:
                            if prop["name"] not in current_properties:
                                print(f"Adding property {prop['name']}...")
                                requests.post(
                                    f"{WEAVIATE_URL}/v1/schema/{collection_name}/properties",
                                    headers=headers,
                                    json=prop
                                )
                        print("✅ Collection updated successfully")
                        return True
                    elif choice == "3":
                        print("Skipping collection setup.")
                        return True
                    else:
                        print("Invalid choice. Please enter 1, 2, or 3.")
            else:
                print(f"Using existing collection '{collection_name}'")
                return True
        
        # Create if doesn't exist
        if response.status_code != 200:
            print(f"Creating collection '{collection_name}'...")
            create_response = requests.post(
                f"{WEAVIATE_URL}/v1/schema", 
                headers=headers, 
                json=schema
            )
            if create_response.status_code != 200:
                print(f"Error creating collection: {create_response.status_code}")
                return False
            print(f"✅ Collection created successfully with {len(schema['properties'])} properties")
        
        return True
    except Exception as e:
        print(f"Error setting up collection: {e}")
        return False

def store_embedding(properties, vector):
    """Store a vector in Weaviate"""
    if not vector or not properties:
        return False
    
    # Generate ID based on content hash
    source = properties.get("source", "")
    page = properties.get("page", 0)
    text = properties.get("text", "")
    id_string = f"{source}-{page}-{text[:50]}"
    object_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, id_string))
    
    # Clean properties to ensure they match schema
    cleaned_properties = {}
    for key, value in properties.items():
        if value is not None and value != "":
            # Handle DPI tuple conversion
            if key == "dpi" and isinstance(value, tuple):
                cleaned_properties[key] = f"{value[0]},{value[1]}"
            else:
                cleaned_properties[key] = value
    
    # Prepare object data
    object_data = {
        "id": object_id,
        "class": COLLECTION_NAME,
        "properties": cleaned_properties,
        "vector": vector
    }
    
    # Send to Weaviate
    try:
        response = requests.post(
            f"{WEAVIATE_URL}/v1/objects", 
            headers=create_weaviate_headers(), 
            json=object_data
        )
        
        if response.status_code in [200, 201]:
            return True
        elif response.status_code == 422 and "already exists" in response.text:
            return "SKIPPED"
        else:
            if VERBOSE_MODE or response.status_code != 422:
                print(f"Error storing object: {response.status_code}")
                print(f"Response: {response.text}")
                print(f"Object keys: {list(cleaned_properties.keys())}")
            return False
    except Exception as e:
        print(f"Error in Weaviate request: {e}")
        return False

def extract_transcript_metadata(transcript_path):
    """Extract metadata from transcript files"""
    default_meta = {
        "title": "", "speaker": "", "duration": "", "date_recorded": "",
        "language": "", "content_type": "transcript", "source_type": "",
        "file_size": 0, "word_count": 0, "confidence_score": 0.0
    }
    
    if not os.path.exists(transcript_path):
        return default_meta, ""
    
    try:
        filename = os.path.basename(transcript_path)
        filepath = os.path.dirname(transcript_path)
        relative_path = os.path.relpath(filepath, TRANSCRIPT_DIR) if os.path.commonpath([filepath, TRANSCRIPT_DIR]) == TRANSCRIPT_DIR else ""
        
        metadata = default_meta.copy()
        metadata.update({
            "title": os.path.splitext(filename)[0],
            "file_size": os.path.getsize(transcript_path),
            "source_type": "audio" if "audio" in relative_path.lower() else 
                          "video" if "video" in relative_path.lower() else "unknown",
            "folder": relative_path
        })
        
        # Try to extract file format from filename
        if "_" in filename:
            parts = filename.split("_")
            for part in parts:
                if part.lower() in ["mp3", "mp4", "wav", "avi", "mov"]:
                    metadata["source_type"] = "audio" if part.lower() in ["mp3", "wav"] else "video"
                    break
        
        # Try to extract metadata from the file content
        with open(transcript_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Skip files with specific "no transcript" markers
        no_transcript_markers = [
            "No transcript generated or found",
            "No transcript available",
            "Transcript unavailable",
            "Failed to generate transcript"
        ]
        
        for marker in no_transcript_markers:
            if marker in content:
                print(f"Skipping {transcript_path}: No transcript content available")
                return metadata, ""
        
        # Handle empty or nearly empty files
        if len(content.strip()) < 20:
            print(f"Skipping {transcript_path}: Insufficient content (too short)")
            return metadata, ""
            
        # Look for metadata sections at the beginning of the file
        # Common format: Key: Value
        lines = content.split('\n')
        in_metadata_section = True
        metadata_lines = []
        content_start_line = 0
        
        # First pass: look for standard metadata format (Key: Value)
        for i, line in enumerate(lines[:min(30, len(lines))]):  # Check first 30 lines for metadata
            line = line.strip()
            if not line:
                if metadata_lines and not any(metadata_lines[-1].strip() == ''):
                    in_metadata_section = False
                continue
                
            if in_metadata_section:
                if ':' in line:
                    metadata_lines.append(line)
                    content_start_line = i + 1
                else:
                    # If we see a line without a colon after metadata, we're probably in content now
                    if metadata_lines:
                        in_metadata_section = False
            else:
                break
        
        # Process metadata lines
        for line in metadata_lines:
            parts = line.split(':', 1)
            if len(parts) == 2:
                key = parts[0].strip().lower().replace(' ', '_')
                value = parts[1].strip()
                
                # Map common metadata fields
                if key in ['title', 'speaker', 'speakers', 'duration', 'language']:
                    metadata[key] = value
                elif key in ['date', 'recorded', 'date_recorded', 'recording_date']:
                    metadata['date_recorded'] = value
                elif key in ['confidence', 'confidence_score']:
                    try:
                        metadata['confidence_score'] = float(value.rstrip('%')) / 100.0
                    except:
                        pass
                elif key not in ['content', 'transcript']:  # Skip content markers
                    metadata[f"transcript_{key}"] = value
        
        # Second pass: try to extract metadata from content if none found
        if not any(k in metadata for k in ['speaker', 'speakers', 'duration', 'date_recorded']):
            # Try to find speaker pattern at the beginning of lines
            speaker_pattern = r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*:'
            speakers = set()
            
            for line in lines[:min(100, len(lines))]:
                speaker_match = re.search(speaker_pattern, line)
                if speaker_match:
                    speaker = speaker_match.group(1)
                    if len(speaker) > 2 and len(speaker) < 30:
                        speakers.add(speaker)
            
            if speakers:
                metadata['speakers'] = ", ".join(speakers)
            
            # Try to find duration pattern
            duration_pattern = r'(Duration|Length|Time):\s*(\d+:?\d+:?\d*)'
            for line in lines[:min(30, len(lines))]:
                duration_match = re.search(duration_pattern, line, re.IGNORECASE)
                if duration_match:
                    metadata['duration'] = duration_match.group(2)
                    break
        
        # Try to extract transcript type
        transcript_types = ["automatic", "manual", "human", "ai", "machine", "auto-generated"]
        for t_type in transcript_types:
            if t_type in content.lower():
                metadata["transcript_type"] = t_type
                break
        
        # Calculate word count from content
        content_text = '\n'.join(lines[content_start_line:])
        metadata['word_count'] = len(content_text.split())
        
        # Extract platform info from filename
        if "_youtube_" in filename.lower():
            metadata["platform"] = "YouTube"
        elif "_gdrive_" in filename.lower():
            metadata["platform"] = "Google Drive"
        elif "zoom" in filename.lower() or "zoom" in content.lower():
            metadata["platform"] = "Zoom"
        
        # Extract additional metadata from filename
        file_parts = filename.split('_')
        if len(file_parts) > 1:
            # Look for version information
            version_pattern = r'version\s*(\d+)'
            for part in file_parts:
                version_match = re.search(version_pattern, part, re.IGNORECASE)
                if version_match:
                    metadata["version"] = f"Version {version_match.group(1)}"
                    break
            
            # Try to identify if it's a training video or other type
            content_categories = {
                "training": ["training", "tutorial", "guide", "how-to", "lesson", "course"],
                "presentation": ["presentation", "webinar", "seminar", "conference"],
                "interview": ["interview", "conversation", "discussion", "dialogue"],
                "podcast": ["podcast", "radio", "broadcast"]
            }
            
            # Check filename and first part of content for category hints
            text_to_check = (filename + " " + content[:min(1000, len(content))]).lower()
            
            for category, keywords in content_categories.items():
                for keyword in keywords:
                    if keyword in text_to_check:
                        metadata["content_category"] = category
                        break
                if "content_category" in metadata:
                    break
        
        # Mark transcript as complete if it seems to have a conclusion
        conclusion_markers = ["conclusion", "thank you", "thanks for watching", "the end", "goodbye", "bye"]
        last_paragraph = content[-min(500, len(content)):].lower()
        metadata["transcript_complete"] = any(marker in last_paragraph for marker in conclusion_markers)
        
        return metadata, content_text
        
    except Exception as e:
        print(f"Error extracting transcript metadata: {e}")
        return default_meta, ""

def clean_transcript_text(text):
    """Clean and format transcript text for better embeddings"""
    if not text or not isinstance(text, str):
        return ""
    
    # Check for empty or nearly empty content
    text = text.strip()
    if len(text) < 10:
        return ""
    
    # Replace common encoding issues
    text = text.replace("\u2019", "'")  # Replace right single quotation mark
    text = text.replace("\u201c", '"')  # Replace left double quotation mark
    text = text.replace("\u201d", '"')  # Replace right double quotation mark
    text = text.replace("\u2026", "...") # Replace ellipsis
    text = text.replace("\u2013", "-")  # Replace en dash
    text = text.replace("\u2014", "-")  # Replace em dash
    
    # Normalize whitespace initially but preserve paragraph breaks
    text = re.sub(r'[\t ]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)  # Preserve paragraph breaks
    
    # Extract and store speaker information before removing it
    speaker_pattern = r'(^|\n)([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)?)\s*:'
    speakers = set()
    for match in re.finditer(speaker_pattern, text):
        speaker = match.group(2)
        if speaker and speaker.strip() and not speaker.strip().isdigit():
            speakers.add(speaker.strip())
    
    # Remove timestamp patterns often found in transcripts
    timestamp_patterns = [
        r'\[\d{1,2}:\d{2}(:\d{2})?\]',
        r'\(\d{1,2}:\d{2}(:\d{2})?\)',
        r'\d{1,2}:\d{2}(:\d{2})?\s*-->\s*\d{1,2}:\d{2}(:\d{2})?',
        r'^\d{1,2}:\d{2}(:\d{2})?\s',
        r'\d{1,2}:\d{2}(:\d{2})?-\d{1,2}:\d{2}(:\d{2})?',  # Range format
        r'\[\d+:\d+\]',  # Simplified timestamp
        r'\[\d+\]'       # Numbered marker
    ]
    
    # Process line by line to handle timestamps while preserving structure
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Skip lines that are just timestamps or empty
        if not line.strip() or re.match(r'^\s*\d{1,2}:\d{2}(:\d{2})?\s*$', line):
            continue
            
        # Remove timestamps from this line
        cleaned_line = line
        for pattern in timestamp_patterns:
            cleaned_line = re.sub(pattern, '', cleaned_line)
            
        # Remove speaker labels but mark them as paragraph breaks
        cleaned_line = re.sub(speaker_pattern, '\n', cleaned_line)
        
        # Skip if line became empty after cleaning
        if cleaned_line.strip():
            cleaned_lines.append(cleaned_line.strip())
    
    # Combine lines with proper spacing
    cleaned = ' '.join(cleaned_lines)
    
    # Fix common transcript artifacts
    cleaned = re.sub(r'\.{3,}', '... ', cleaned)  # Fix multiple periods
    cleaned = re.sub(r'\s+', ' ', cleaned)        # Fix spacing
    
    # Fix common transcript issues
    cleaned = re.sub(r'(\w)- (\w)', r'\1\2', cleaned)  # Fix hyphenated words split across lines
    
    # Remove markers for inaudible content but preserve meaning
    inaudible_patterns = [
        r'\(inaudible\)',
        r'\(unintelligible\)',
        r'\(indiscernible\)',
        r'\(unclear\)',
        r'\(\?\)'
    ]
    
    for pattern in inaudible_patterns:
        cleaned = re.sub(pattern, ' [unclear] ', cleaned)
    
    # Clean up special markers
    cleaned = re.sub(r'\[unclear\]\s+\[unclear\]', ' [unclear] ', cleaned)
    
    # Apply unicode normalization
    cleaned = unicodedata.normalize('NFKC', cleaned)
    
    # Apply NLTK processing if available
    if NLTK_AVAILABLE:
        try:
            # Tokenize into sentences
            sentences = nltk.sent_tokenize(cleaned)
            normalized_sentences = []
            
            for sentence in sentences:
                if not sentence.strip():
                    continue
                
                # Fix capitalization
                if sentence and sentence[0].islower():
                    sentence = sentence[0].upper() + sentence[1:]
                
                # Ensure sentences end with punctuation
                if sentence and sentence[-1] not in ['.', '!', '?']:
                    sentence += '.'
                
                normalized_sentences.append(sentence)
            
            cleaned = ' '.join(normalized_sentences)
        except Exception as e:
            # If NLTK processing fails, just continue with the basic cleaned text
            pass
    
    # Final cleaning
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # For very short content, be more permissive
    if len(cleaned) < 50 and len(text) > 100:
        # If we lost too much content in cleaning, use a more basic cleaning
        basic_cleaned = re.sub(r'\s+', ' ', text).strip()
        return basic_cleaned
    
    return cleaned

def enhance_transcript_for_embedding(text, metadata=None):
    """Prepare transcript text for optimal embeddings with enhanced instructions"""
    if not text:
        return text
    
    # Extract context from metadata
    title = metadata.get("title", "") if metadata else ""
    speaker = metadata.get("speaker", "") if metadata else ""
    speakers = metadata.get("speakers", speaker) if metadata else speaker
    duration = metadata.get("duration", "") if metadata else ""
    date_recorded = metadata.get("date_recorded", "") if metadata else ""
    language = metadata.get("language", "") if metadata else ""
    source_type = metadata.get("source_type", "") if metadata else ""
    platform = metadata.get("platform", "") if metadata else ""
    version = metadata.get("version", "") if metadata else ""
    word_count = metadata.get("word_count", 0) if metadata else 0
    
    # Add current timestamp for processing date if not present
    processed_date = metadata.get("processed_date", "") if metadata else ""
    if not processed_date:
        processed_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if metadata:
            metadata["processed_date"] = processed_date
    
    # Try to extract main topics and keywords if not present
    if metadata and not metadata.get("main_topics") and text and len(text) > 100:
        try:
            # Simple keyword extraction - take most frequent words after removing stopwords
            if NLTK_AVAILABLE:
                words = [word.lower() for word in nltk.word_tokenize(text) 
                         if word.isalpha() and word.lower() not in STOPWORDS and len(word) > 3]
                
                # Get frequency distribution
                freq_dist = nltk.FreqDist(words)
                keywords = [word for word, _ in freq_dist.most_common(10)]
                
                if keywords:
                    metadata["extracted_keywords"] = ", ".join(keywords)
                    
                    # Try to determine main topics from keywords
                    if len(keywords) >= 3:
                        metadata["main_topics"] = ", ".join(keywords[:3])
        except:
            pass
    
    # Try to infer transcript quality if not present
    if metadata and not metadata.get("transcript_quality") and text:
        # Check for signs of poor quality
        poor_quality_markers = ["inaudible", "unintelligible", "(?)"]
        quality_score = 100
        
        for marker in poor_quality_markers:
            if marker in text.lower():
                quality_score -= 20
        
        # Check for complete sentences
        sentence_endings = re.findall(r'[.!?]', text)
        if len(sentence_endings) < 5 and len(text) > 200:
            quality_score -= 20
        
        if quality_score > 80:
            metadata["transcript_quality"] = "high"
        elif quality_score > 50:
            metadata["transcript_quality"] = "medium"
        else:
            metadata["transcript_quality"] = "low"
    
    # Optimize for embedding instruction
    instruction = "Represent this transcript accurately with high-quality semantic content for retrieval: "
    
    # Build rich context string with semantic boosting
    context_parts = []
    if title:
        context_parts.append(f"Title: {title}")
        # Add title twice for semantic emphasis
        context_parts.append(f"Transcript: {title}")
    if speakers:
        context_parts.append(f"Speakers: {speakers}")
    if source_type:
        context_parts.append(f"Source Type: {source_type}")
    if platform:
        context_parts.append(f"Platform: {platform}")
    if duration:
        context_parts.append(f"Duration: {duration}")
    if date_recorded:
        context_parts.append(f"Recorded: {date_recorded}")
    if language:
        context_parts.append(f"Language: {language}")
    if version:
        context_parts.append(f"Version: {version}")
    if metadata and metadata.get("main_topics"):
        context_parts.append(f"Topics: {metadata.get('main_topics')}")
    if metadata and metadata.get("extracted_keywords"):
        context_parts.append(f"Keywords: {metadata.get('extracted_keywords')}")
    if metadata and metadata.get("transcript_quality"):
        context_parts.append(f"Quality: {metadata.get('transcript_quality')}")
    
    # Add semantic focus markers
    context_parts.append(f"Content Type: Transcript")
    if source_type:
        context_parts.append(f"Media Type: {source_type.capitalize()}")
    context_parts.append(f"Importance: High")
    
    context_str = ""
    if context_parts:
        context_str = " | ".join(context_parts) + "\n\n"
    
    # Combine for final embedding text with semantic emphasis
    enhanced_text = f"{instruction}{context_str}{text}"
    
    # Add trailing emphasis for recency bias in transformer models
    enhanced_text += "\n\nThis transcript contains important information for accurate retrieval and similarity matching."
    
    return enhanced_text

def process_transcript_files():
    """Process transcript files and generate embeddings"""
    print("\nProcessing transcript files...")
    transcript_dir = TRANSCRIPT_DIR
    
    if not os.path.exists(transcript_dir):
        print(f"Error: Transcripts directory '{transcript_dir}' not found")
        return [], {}
    
    # Find all .txt files in transcript directory and subdirectories
    all_transcript_files = []
    for root, _, files in os.walk(transcript_dir):
        for file in files:
            if file.lower().endswith('.txt'):
                all_transcript_files.append(os.path.join(root, file))
    
    if not all_transcript_files:
        print("No transcript files found")
        return [], {}
    
    print(f"Found {len(all_transcript_files)} transcript files")
    
    # Process each transcript file
    all_chunks = []
    doc_embeddings = {}
    
    for transcript_file in tqdm(all_transcript_files, desc="Processing transcripts"):
        try:
            # Extract metadata and content
            metadata, content = extract_transcript_metadata(transcript_file)
            
            # Skip if explicitly marked as having no transcript
            if not content and "no transcript" in str(metadata.get("title", "")).lower():
                continue
                
            # Lower the minimum content threshold to capture more transcripts
            if not content or len(content.strip()) < 20:  # Only skip if truly empty or too short
                print(f"Skipping {transcript_file}: Insufficient content")
                continue
            
            # Create document embedding for hierarchical approach
            if USE_HIERARCHICAL_EMBEDDINGS:
                doc_title = metadata.get("title", os.path.basename(transcript_file))
                # Use the first part of content (or all if short)
                doc_text = enhance_transcript_for_embedding(content[:min(len(content), 1000)], metadata)
                try:
                    doc_embeddings[transcript_file] = embedding_model.embed_query(doc_text)
                except Exception as e:
                    print(f"Error creating document embedding: {e}")
            
            # Clean content
            cleaned_text = clean_transcript_text(content)
            
            # Process even if cleaning removed a lot of content
            if not cleaned_text or len(cleaned_text) < 50:
                # If cleaning removed too much, use original with basic cleaning
                cleaned_text = re.sub(r'\s+', ' ', content).strip()
                if len(cleaned_text) < 20:
                    continue
            
            # Split into chunks - with handling for short content
            if len(cleaned_text) < CHUNK_SIZE/2:
                # For short texts, don't split
                text_chunks = [cleaned_text]
            else:
                text_chunks = text_splitter.split_text(cleaned_text)
            
            if not text_chunks:
                # If splitting produced no chunks, use the whole text as one chunk
                if len(cleaned_text) > 20:
                    text_chunks = [cleaned_text]
                else:
                    continue
            
            # Process each chunk
            for i, chunk_text in enumerate(text_chunks):
                # Be more lenient with chunk size for transcripts
                if not chunk_text or len(chunk_text) < 20:
                    continue
                
                # Create chunk metadata
                chunk_meta = metadata.copy()
                chunk_meta["text"] = chunk_text
                chunk_meta["source"] = transcript_file
                chunk_meta["segment"] = i
                chunk_meta["embedding_model"] = model_name
                chunk_meta["hierarchical_embedding"] = USE_HIERARCHICAL_EMBEDDINGS
                chunk_meta["embedding_enhanced"] = True
                chunk_meta["chunk_size"] = CHUNK_SIZE
                chunk_meta["chunk_overlap"] = CHUNK_OVERLAP
                
                all_chunks.append({"metadata": chunk_meta, "source": transcript_file})
        
        except Exception as e:
            print(f"Error processing transcript {transcript_file}: {e}")
            continue  # Continue with next file even if this one fails
    
    print(f"Created {len(all_chunks)} chunks from transcript files")
    return all_chunks, doc_embeddings

def process_documents():
    """Process all documents with high quality vector enhancement"""
    print("Starting high quality vectorization process...")
    print(f"Using model: {model_name}")
    print(f"Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")
    print(f"Hierarchical embeddings: {'Enabled' if USE_HIERARCHICAL_EMBEDDINGS else 'Disabled'}")
    
    # Setup Weaviate
    if not setup_weaviate_collection():
        print("Failed to setup Weaviate collection")
        return
    
    all_chunks = []
    all_doc_embeddings = {}
    
    # Common metadata for processing session
    processing_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    processing_metadata = {
        "processed_date": processing_timestamp,
        "last_updated": processing_timestamp
    }
    
    # Process PDF files
    pdf_dir = PDF_DIR
    if os.path.exists(pdf_dir) and os.listdir(pdf_dir):
        print(f"\nProcessing PDFs from {pdf_dir}...")
        loader = DirectoryLoader(pdf_dir, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
        documents = loader.load()
        
        if documents:
            # Group by source
            docs_by_source = {}
            for doc in documents:
                source = doc.metadata.get("source", "")
                if source:
                    docs_by_source.setdefault(source, []).append(doc)
            
            # Sort by page number
            for source_docs in docs_by_source.values():
                source_docs.sort(key=lambda x: x.metadata.get("page", 0))
            
            print(f"Loaded {len(documents)} pages from {len(docs_by_source)} documents")
            
            # Process each document
            for source, source_docs in docs_by_source.items():
                print(f"Processing {os.path.basename(source)}...")
                
                # Extract metadata
                pdf_meta = extract_pdf_metadata(source)
                json_meta = extract_json_metadata(source)
                
                # Generate a unique document ID
                doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, source))
                
                combined_meta = {**pdf_meta, **json_meta, **processing_metadata}
                combined_meta["source"] = source
                combined_meta["filename"] = os.path.basename(source)
                combined_meta["content_type"] = "pdf"
                combined_meta["document_id"] = doc_id
                
                # Try to categorize content based on available metadata
                categories = []
                if "subject" in combined_meta and combined_meta["subject"]:
                    categories.append(combined_meta["subject"])
                
                if "keywords" in combined_meta and combined_meta["keywords"]:
                    # Extract tags from keywords
                    tags = [tag.strip() for tag in combined_meta["keywords"].split(",") if tag.strip()]
                    if tags:
                        combined_meta["content_tags"] = tags
                        # Add first two tags as categories if not too many categories
                        if len(categories) < 2 and tags:
                            categories.extend(tags[:2])
                
                if categories:
                    combined_meta["content_category"] = ", ".join(categories[:3])  # Limit to 3 categories
                
                # Create document embedding for hierarchical approach
                if USE_HIERARCHICAL_EMBEDDINGS:
                    doc_title = combined_meta.get("title", os.path.basename(source))
                    doc_summary = create_document_summary(source_docs, doc_title)
                    if doc_summary:
                        doc_text = enhance_text_for_embedding(doc_summary, {"title": doc_title})
                        try:
                            all_doc_embeddings[source] = embedding_model.embed_query(doc_text)
                        except Exception as e:
                            print(f"Error creating document embedding: {e}")
                
                # Combine all pages with markers
                full_text = ""
                for doc in source_docs:
                    page = doc.metadata.get("page", 0)
                    full_text += f"\n\n[PAGE {page+1}]\n{doc.page_content}"
                
                # Split into chunks
                text_chunks = text_splitter.split_text(full_text)
                
                # Process each chunk
                for chunk_text in text_chunks:
                    # Clean text
                    cleaned_text = clean_text(chunk_text)
                    if not cleaned_text or len(cleaned_text) < 100:
                        continue
                    
                    # Extract page number
                    page_match = re.search(r"\[PAGE (\d+)\]", chunk_text)
                    page_num = int(page_match.group(1))-1 if page_match else 0
                    
                    # Create chunk metadata
                    chunk_meta = combined_meta.copy()
                    chunk_meta["page"] = page_num
                    chunk_meta["text"] = cleaned_text
                    chunk_meta["embedding_model"] = model_name
                    chunk_meta["hierarchical_embedding"] = USE_HIERARCHICAL_EMBEDDINGS
                    chunk_meta["embedding_enhanced"] = True
                    chunk_meta["chunk_size"] = CHUNK_SIZE
                    chunk_meta["chunk_overlap"] = CHUNK_OVERLAP
                    
                    all_chunks.append({"metadata": chunk_meta, "source": source})
    else:
        print(f"PDF directory '{pdf_dir}' not found or empty")
    
    # Process transcript files
    transcript_chunks, transcript_embeddings = process_transcript_files()
    
    # Add common processing metadata to all transcript chunks
    for chunk in transcript_chunks:
        chunk["metadata"].update(processing_metadata)
        
        # Generate a unique document ID if not present
        if "document_id" not in chunk["metadata"]:
            source = chunk["metadata"].get("source", "")
            chunk["metadata"]["document_id"] = str(uuid.uuid5(uuid.NAMESPACE_DNS, source))
    
    all_chunks.extend(transcript_chunks)
    all_doc_embeddings.update(transcript_embeddings)
    
    # Process image files
    image_chunks, image_embeddings = process_image_files()
    
    # Add common processing metadata to all image chunks
    for chunk in image_chunks:
        chunk["metadata"].update(processing_metadata)
        
        # Generate a unique document ID if not present
        if "document_id" not in chunk["metadata"]:
            source = chunk["metadata"].get("source", "")
            chunk["metadata"]["document_id"] = str(uuid.uuid5(uuid.NAMESPACE_DNS, source))
    
    all_chunks.extend(image_chunks)
    all_doc_embeddings.update(image_embeddings)
    
    if not all_chunks:
        print("No documents processed")
        return
    
    print(f"\nTotal chunks to process: {len(all_chunks)}")
    
    # Process in batches
    batch_size = BATCH_SIZE
    success_count = 0
    skip_count = 0
    error_count = 0
    hierarchical_count = 0
    
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(all_chunks)-1)//batch_size + 1}")
        
        # Prepare texts for embedding
        texts = []
        for chunk in batch:
            content_type = chunk["metadata"].get("content_type", "")
            if content_type == "transcript":
                texts.append(enhance_transcript_for_embedding(chunk["metadata"]["text"], chunk["metadata"]))
            elif content_type == "image":
                texts.append(enhance_image_for_embedding(chunk["metadata"]["text"], chunk["metadata"]))
            else:  # Default to PDF enhancement
                texts.append(enhance_text_for_embedding(chunk["metadata"]["text"], chunk["metadata"]))
        
        try:
            # Generate embeddings
            embeddings = embedding_model.embed_documents(texts)
            
            # Store each chunk
            for chunk_data, vector in zip(batch, embeddings):
                source = chunk_data["source"]
                
                # Apply hierarchical blending if available
                if USE_HIERARCHICAL_EMBEDDINGS and source in all_doc_embeddings:
                    original_vector = vector
                    vector = blend_hierarchical_embedding(vector, all_doc_embeddings[source])
                    if vector != original_vector:
                        hierarchical_count += 1
                
                # Store in Weaviate
                result = store_embedding(chunk_data["metadata"], vector)
                
                if result == True:
                    success_count += 1
                elif result == "SKIPPED":
                    skip_count += 1
                else:
                    error_count += 1
            
            print(f"Batch complete: {success_count}/{len(all_chunks)} processed so far")
            
        except Exception as e:
            print(f"Error processing batch: {e}")
            error_count += len(batch)
    
    # Summary
    print("\n===== Vectorization Complete =====")
    print(f"Successfully stored: {success_count} chunks")
    print(f"Skipped: {skip_count} chunks")
    print(f"Errors: {error_count} chunks")
    print(f"Total processed: {len(all_chunks)} chunks")
    if USE_HIERARCHICAL_EMBEDDINGS:
        print(f"Hierarchical embeddings applied: {hierarchical_count} chunks")
    print("==================================")

# ----- Image Processing Functions -----
def extract_image_metadata(image_path):
    """Extract comprehensive metadata from image files"""
    default_meta = {
        "title": "", "width": 0, "height": 0, "format": "", "mode": "",
        "file_size": 0, "dpi": (0, 0), "has_exif": False, "camera_make": "",
        "camera_model": "", "datetime_taken": "", "gps_coords": "",
        "orientation": "", "color_depth": 0, "compression": "",
        "content_type": "image", "image_quality": "unknown"
    }
    
    if not PIL_AVAILABLE or not os.path.exists(image_path):
        return default_meta
    
    try:
        filename = os.path.basename(image_path)
        filepath = os.path.dirname(image_path)
        relative_path = os.path.relpath(filepath, IMAGE_DIR) if os.path.commonpath([filepath, IMAGE_DIR]) == IMAGE_DIR else ""
        
        metadata = default_meta.copy()
        metadata.update({
            "title": os.path.splitext(filename)[0],
            "file_size": os.path.getsize(image_path),
            "folder": relative_path,
            "filename": filename
        })
        
        # Open image and extract basic info
        with Image.open(image_path) as img:
            metadata.update({
                "width": img.width,
                "height": img.height,
                "format": img.format or "",
                "mode": img.mode or "",
                "dpi": getattr(img, 'dpi', (0, 0)) or (0, 0)
            })
            
            # Calculate color depth
            if img.mode == "1":
                metadata["color_depth"] = 1
            elif img.mode == "L":
                metadata["color_depth"] = 8
            elif img.mode == "P":
                metadata["color_depth"] = 8
            elif img.mode == "RGB":
                metadata["color_depth"] = 24
            elif img.mode == "RGBA":
                metadata["color_depth"] = 32
            elif img.mode == "CMYK":
                metadata["color_depth"] = 32
            else:
                metadata["color_depth"] = 0
            
            # Try to determine image quality based on resolution and file size
            total_pixels = img.width * img.height
            if total_pixels > 0:
                # Calculate bytes per pixel
                bytes_per_pixel = metadata["file_size"] / total_pixels
                
                if bytes_per_pixel > 3.0:
                    metadata["image_quality"] = "high"
                elif bytes_per_pixel > 1.5:
                    metadata["image_quality"] = "medium"
                else:
                    metadata["image_quality"] = "low"
            
            # Extract EXIF data if available
            try:
                exif_data = img._getexif()
                if exif_data:
                    metadata["has_exif"] = True
                    
                    for tag_id, value in exif_data.items():
                        tag = TAGS.get(tag_id, tag_id)
                        
                        if tag == "Make":
                            metadata["camera_make"] = str(value)
                        elif tag == "Model":
                            metadata["camera_model"] = str(value)
                        elif tag == "DateTime":
                            metadata["datetime_taken"] = str(value)
                        elif tag == "Orientation":
                            orientations = {
                                1: "Normal", 2: "Mirrored", 3: "Rotated 180°",
                                4: "Mirrored and rotated 180°", 5: "Mirrored and rotated 90° CCW",
                                6: "Rotated 90° CW", 7: "Mirrored and rotated 90° CW",
                                8: "Rotated 90° CCW"
                            }
                            metadata["orientation"] = orientations.get(value, f"Unknown ({value})")
                        elif tag == "GPSInfo" and isinstance(value, dict):
                            # Extract GPS coordinates if available
                            try:
                                lat = value.get(2)  # GPSLatitude
                                lat_ref = value.get(1)  # GPSLatitudeRef
                                lon = value.get(4)  # GPSLongitude
                                lon_ref = value.get(3)  # GPSLongitudeRef
                                
                                if lat and lon:
                                    # Convert DMS to decimal degrees
                                    def dms_to_decimal(dms, ref):
                                        degrees = float(dms[0])
                                        minutes = float(dms[1]) / 60.0
                                        seconds = float(dms[2]) / 3600.0
                                        decimal = degrees + minutes + seconds
                                        if ref in ['S', 'W']:
                                            decimal = -decimal
                                        return decimal
                                    
                                    lat_decimal = dms_to_decimal(lat, lat_ref)
                                    lon_decimal = dms_to_decimal(lon, lon_ref)
                                    metadata["gps_coords"] = f"{lat_decimal}, {lon_decimal}"
                            except:
                                pass
            except:
                pass  # EXIF not available or readable
        
        # Detect potential content category from filename and path
        content_categories = {
            "screenshot": ["screenshot", "screen", "capture", "snap"],
            "chart": ["chart", "graph", "plot", "diagram"],
            "document": ["document", "doc", "page", "scan"],
            "presentation": ["slide", "presentation", "ppt"],
            "success_story": ["success", "story", "case", "result"],
            "training": ["training", "tutorial", "guide", "how-to"],
            "marketing": ["marketing", "ad", "promo", "campaign"],
            "infographic": ["infographic", "info", "visual"]
        }
        
        text_to_check = (filename + " " + relative_path).lower()
        for category, keywords in content_categories.items():
            for keyword in keywords:
                if keyword in text_to_check:
                    metadata["content_category"] = category
                    break
            if "content_category" in metadata:
                break
        
        return metadata
        
    except Exception as e:
        print(f"Error extracting image metadata: {e}")
        return default_meta

def extract_image_json_metadata(image_path):
    """Get metadata from JSON file associated with image"""
    try:
        image_dir = os.path.dirname(image_path)
        image_name = os.path.basename(image_path)
        image_name_no_ext = os.path.splitext(image_name)[0]
        
        # Try to find matching JSON file
        possible_json_names = [
            f"{image_name_no_ext}_metadata.json",
            f"{image_name_no_ext}.json",
            f"{image_name}_metadata.json"
        ]
        
        json_path = None
        for json_name in possible_json_names:
            potential_path = os.path.join(image_dir, json_name)
            if os.path.exists(potential_path):
                json_path = potential_path
                break
        
        if not json_path:
            return {}
            
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # Process Google Drive metadata structure
        processed_data = {}
        for key, value in json_data.items():
            field_name = f"image_{key}" if not key.startswith("image_") else key
            
            if isinstance(value, (bool, int, float, str)):
                processed_data[field_name] = value
            elif isinstance(value, list) and len(value) > 0:
                # Handle arrays like owners
                if key == "owners" and len(value) > 0:
                    owner = value[0]
                    if isinstance(owner, dict):
                        processed_data["image_owner_name"] = owner.get("displayName", "")
                        processed_data["image_owner_email"] = owner.get("emailAddress", "")
                elif key == "lastModifyingUser" and isinstance(value, dict):
                    processed_data["image_last_modifier"] = value.get("displayName", "")
                else:
                    try:
                        processed_data[field_name] = json.dumps(value)
                    except:
                        processed_data[field_name] = str(value)
            elif isinstance(value, dict):
                # Handle nested objects
                if key == "lastModifyingUser":
                    processed_data["image_last_modifier"] = value.get("displayName", "")
                    processed_data["image_last_modifier_email"] = value.get("emailAddress", "")
                else:
                    try:
                        processed_data[field_name] = json.dumps(value)
                    except:
                        processed_data[field_name] = str(value)
            else:
                try:
                    processed_data[field_name] = json.dumps(value)
                except:
                    processed_data[field_name] = str(value)
        
        return processed_data
    except Exception as e:
        print(f"Error extracting JSON metadata for image: {e}")
        return {}

def preprocess_image_for_ocr(image_path):
    """Preprocess image to improve OCR quality"""
    if not PIL_AVAILABLE:
        return None
    
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert to grayscale for better OCR
            img = img.convert('L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(2.0)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.5)
            
            # Resize if too small (OCR works better on larger images)
            width, height = img.size
            if width < 1000 or height < 1000:
                scale_factor = max(1000 / width, 1000 / height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Apply median filter to reduce noise
            img = img.filter(ImageFilter.MedianFilter(size=3))
            
            return img
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def extract_text_from_image(image_path, metadata=None):
    """Extract text from image using OCR with enhanced preprocessing"""
    if not PYTESSERACT_AVAILABLE or not PIL_AVAILABLE:
        return ""
    
    try:
        # Preprocess image for better OCR
        processed_img = preprocess_image_for_ocr(image_path)
        if processed_img is None:
            return ""
        
        # Configure Tesseract for better results
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,!?@#$%&*()[]{}:;"\'+-=/<>|_~`'
        
        # Extract text
        extracted_text = pytesseract.image_to_string(processed_img, config=custom_config)
        
        # Try different PSM modes if first attempt yields poor results
        if len(extracted_text.strip()) < 20:
            # Try PSM 3 (fully automatic page segmentation)
            config_psm3 = r'--oem 3 --psm 3'
            extracted_text_psm3 = pytesseract.image_to_string(processed_img, config=config_psm3)
            
            if len(extracted_text_psm3.strip()) > len(extracted_text.strip()):
                extracted_text = extracted_text_psm3
        
        # If still poor results, try PSM 11 (sparse text)
        if len(extracted_text.strip()) < 20:
            config_psm11 = r'--oem 3 --psm 11'
            extracted_text_psm11 = pytesseract.image_to_string(processed_img, config=config_psm11)
            
            if len(extracted_text_psm11.strip()) > len(extracted_text.strip()):
                extracted_text = extracted_text_psm11
        
        # Clean up the extracted text
        cleaned_text = clean_ocr_text(extracted_text)
        
        # If we have metadata, try to enhance the text with context
        if metadata and cleaned_text:
            title = metadata.get("title", "")
            if title:
                # Add title context to help with embedding quality
                cleaned_text = f"Image: {title}\n\n{cleaned_text}"
        
        return cleaned_text
        
    except Exception as e:
        print(f"Error extracting text from image {image_path}: {e}")
        return ""

def clean_ocr_text(text):
    """Clean and enhance OCR extracted text"""
    if not text or not isinstance(text, str):
        return ""
    
    # Basic cleaning
    cleaned = text.strip()
    if len(cleaned) < 5:
        return ""
    
    # Fix common OCR errors
    ocr_corrections = {
        r'\s+': ' ',  # Multiple whitespace to single space
        r'([a-z])([A-Z])': r'\1 \2',  # Add space between lowercase and uppercase
        r'(\d)([A-Z])': r'\1 \2',  # Add space between digit and uppercase
        r'([a-z])(\d)': r'\1 \2',  # Add space between lowercase and digit
        r'\.{2,}': '.',  # Multiple dots to single dot
        r'-{2,}': '-',   # Multiple dashes to single dash
        r'_{2,}': '_',   # Multiple underscores to single underscore
        r'\|{2,}': '|',  # Multiple pipes to single pipe
        r'(\w)\|(\w)': r'\1 \2',  # Replace pipe between words with space
        r'(\w)_(\w)': r'\1 \2',   # Replace underscore between words with space
        r'([a-zA-Z])\s*\.\s*([a-zA-Z])': r'\1. \2',  # Fix spacing around periods
        r'([a-zA-Z])\s*,\s*([a-zA-Z])': r'\1, \2',   # Fix spacing around commas
        r'([a-zA-Z])\s*:\s*([a-zA-Z])': r'\1: \2',   # Fix spacing around colons
    }
    
    for pattern, replacement in ocr_corrections.items():
        cleaned = re.sub(pattern, replacement, cleaned)
    
    # Remove lines that are likely OCR artifacts
    lines = cleaned.split('\n')
    good_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Skip lines that are mostly symbols or single characters
        if len(line) < 3:
            continue
            
        # Skip lines that are mostly numbers without context
        if re.match(r'^[\d\s\-\.\,]+$', line) and len(line) < 10:
            continue
            
        # Skip lines that are mostly special characters
        special_char_ratio = sum(1 for c in line if not c.isalnum() and c != ' ') / len(line)
        if special_char_ratio > 0.7:
            continue
        
        good_lines.append(line)
    
    # Combine good lines
    cleaned = ' '.join(good_lines)
    
    # Apply unicode normalization
    cleaned = unicodedata.normalize('NFKC', cleaned)
    
    # Final cleanup
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned if len(cleaned) >= 10 else ""

def enhance_image_for_embedding(extracted_text, metadata=None):
    """Prepare image-derived text for optimal embeddings"""
    if not extracted_text:
        return extracted_text
    
    # Extract context from metadata
    title = metadata.get("title", "") if metadata else ""
    content_category = metadata.get("content_category", "") if metadata else ""
    width = metadata.get("width", 0) if metadata else 0
    height = metadata.get("height", 0) if metadata else 0
    format_type = metadata.get("format", "") if metadata else ""
    image_quality = metadata.get("image_quality", "") if metadata else ""
    owner_name = metadata.get("image_owner_name", "") if metadata else ""
    folder = metadata.get("folder", "") if metadata else ""
    
    # Optimized embedding instruction for image content
    instruction = "Represent this image content accurately with high-quality semantic information for retrieval: "
    
    # Build rich context string with semantic boosting
    context_parts = []
    if title:
        context_parts.append(f"Image Title: {title}")
        # Add title twice for semantic emphasis
        context_parts.append(f"Visual Content: {title}")
    if content_category:
        context_parts.append(f"Category: {content_category}")
        context_parts.append(f"Type: {content_category}")
    if format_type:
        context_parts.append(f"Format: {format_type}")
    if width and height:
        context_parts.append(f"Dimensions: {width}x{height}")
        # Classify by size
        if width * height > 2000000:
            context_parts.append("Resolution: High")
        elif width * height > 500000:
            context_parts.append("Resolution: Medium")
        else:
            context_parts.append("Resolution: Standard")
    if image_quality and image_quality != "unknown":
        context_parts.append(f"Quality: {image_quality}")
    if owner_name:
        context_parts.append(f"Creator: {owner_name}")
    if folder:
        context_parts.append(f"Source: {folder}")
    
    # Add semantic focus markers
    context_parts.append("Content Type: Image with Text")
    context_parts.append("Source: Visual Document")
    context_parts.append("Importance: High")
    
    context_str = ""
    if context_parts:
        context_str = " | ".join(context_parts) + "\n\n"
    
    # Combine for final embedding text with semantic emphasis
    enhanced_text = f"{instruction}{context_str}{extracted_text}"
    
    # Add trailing emphasis for recency bias in transformer models
    enhanced_text += "\n\nThis image contains important visual information and text for accurate retrieval and similarity matching."
    
    return enhanced_text

def process_image_files():
    """Process image files and generate embeddings"""
    print("\nProcessing image files...")
    image_dir = IMAGE_DIR
    
    if not os.path.exists(image_dir):
        print(f"Error: Images directory '{image_dir}' not found")
        return [], {}
    
    # Find all image files in directory and subdirectories
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.webp'}
    all_image_files = []
    
    for root, _, files in os.walk(image_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                all_image_files.append(os.path.join(root, file))
    
    if not all_image_files:
        print("No image files found")
        return [], {}
    
    print(f"Found {len(all_image_files)} image files")
    
    # Process each image file
    all_chunks = []
    doc_embeddings = {}
    
    for image_file in tqdm(all_image_files, desc="Processing images"):
        try:
            # Extract metadata
            image_meta = extract_image_metadata(image_file)
            json_meta = extract_image_json_metadata(image_file)
            
            # Combine metadata
            combined_meta = {**image_meta, **json_meta}
            combined_meta["source"] = image_file
            combined_meta["filename"] = os.path.basename(image_file)
            combined_meta["content_type"] = "image"
            
            # Extract text using OCR
            extracted_text = extract_text_from_image(image_file, combined_meta)
            
            # Skip if no text was extracted and image is very small
            if not extracted_text and combined_meta.get("width", 0) * combined_meta.get("height", 0) < 50000:
                print(f"Skipping {image_file}: No text extracted and image too small")
                continue
            
            # If no text, create a description based on metadata
            if not extracted_text:
                description_parts = []
                if combined_meta.get("title"):
                    description_parts.append(f"Image titled: {combined_meta['title']}")
                if combined_meta.get("content_category"):
                    description_parts.append(f"Category: {combined_meta['content_category']}")
                if combined_meta.get("width") and combined_meta.get("height"):
                    description_parts.append(f"Dimensions: {combined_meta['width']}x{combined_meta['height']}")
                if combined_meta.get("format"):
                    description_parts.append(f"Format: {combined_meta['format']}")
                
                if description_parts:
                    extracted_text = f"Visual content: {'. '.join(description_parts)}"
                else:
                    extracted_text = f"Image file: {combined_meta.get('filename', 'Unknown')}"
            
            # Create document embedding for hierarchical approach
            if USE_HIERARCHICAL_EMBEDDINGS:
                doc_title = combined_meta.get("title", os.path.basename(image_file))
                doc_text = enhance_image_for_embedding(extracted_text[:min(len(extracted_text), 1000)], combined_meta)
                try:
                    doc_embeddings[image_file] = embedding_model.embed_query(doc_text)
                except Exception as e:
                    print(f"Error creating document embedding: {e}")
            
            # For images, we typically don't need to split into chunks unless text is very long
            if len(extracted_text) > CHUNK_SIZE * 2:
                # Split very long extracted text
                text_chunks = text_splitter.split_text(extracted_text)
            else:
                # Use the whole extracted text as one chunk
                text_chunks = [extracted_text] if extracted_text else []
            
            if not text_chunks:
                continue
            
            # Process each chunk (usually just one for images)
            for i, chunk_text in enumerate(text_chunks):
                if not chunk_text or len(chunk_text) < 10:
                    continue
                
                # Create chunk metadata
                chunk_meta = combined_meta.copy()
                chunk_meta["text"] = chunk_text
                chunk_meta["segment"] = i
                chunk_meta["embedding_model"] = model_name
                chunk_meta["hierarchical_embedding"] = USE_HIERARCHICAL_EMBEDDINGS
                chunk_meta["embedding_enhanced"] = True
                chunk_meta["chunk_size"] = CHUNK_SIZE
                chunk_meta["chunk_overlap"] = CHUNK_OVERLAP
                chunk_meta["ocr_extracted"] = bool(PYTESSERACT_AVAILABLE and extracted_text)
                
                all_chunks.append({"metadata": chunk_meta, "source": image_file})
        
        except Exception as e:
            print(f"Error processing image {image_file}: {e}")
            continue  # Continue with next file even if this one fails
    
    print(f"Created {len(all_chunks)} chunks from image files")
    return all_chunks, doc_embeddings

if __name__ == "__main__":
    process_documents() 