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
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
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

# Load environment variables
load_dotenv()

# --- Configuration ---
# Weaviate connection
WEAVIATE_URL = "http://localhost:8090"
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
COLLECTION_NAME = "Area_expansion_Dep_Anas_V3"

# Set to True to recreate collection (delete existing)
RECREATE_COLLECTION = True

# Settings for improved vector quality
CHUNK_SIZE = 800# Smaller chunks for better semantic focus
CHUNK_OVERLAP = 170  # Increased overlap for better context continuity
USE_HIERARCHICAL_EMBEDDINGS = True  # Blend document level context with chunks
HIERARCHICAL_WEIGHT = 0.25  # Increased weight for document-level context
VERBOSE_MODE = True  # Show detailed output

# Parallelization settings
NUM_PROCESSES = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU for system
BATCH_SIZE = 16  # Increased batch size for faster processing

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
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available. Image processing disabled.")

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
    'batch_size': 16,  # Increased for better throughput
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
            {"name": "text", "dataType": ["text"]},
            {"name": "source", "dataType": ["text"]},
            {"name": "filename", "dataType": ["text"]},
            {"name": "folder", "dataType": ["text"]},
            {"name": "page", "dataType": ["int"]},
            {"name": "title", "dataType": ["text"]},
            {"name": "author", "dataType": ["text"]},
            {"name": "subject", "dataType": ["text"]},
            {"name": "keywords", "dataType": ["text"]},
            {"name": "creation_date", "dataType": ["text"]},
            {"name": "modification_date", "dataType": ["text"]},
            {"name": "page_count", "dataType": ["int"]},
            {"name": "file_size", "dataType": ["int"]},
            {"name": "has_images", "dataType": ["boolean"]},
            {"name": "image_count", "dataType": ["int"]},
            {"name": "has_toc", "dataType": ["boolean"]},
            {"name": "has_links", "dataType": ["boolean"]},
            {"name": "has_forms", "dataType": ["boolean"]},
            {"name": "embedding_model", "dataType": ["text"]},
            {"name": "hierarchical_embedding", "dataType": ["boolean"]},
            {"name": "embedding_enhanced", "dataType": ["boolean"]},
            {"name": "chunk_size", "dataType": ["int"]},
            {"name": "chunk_overlap", "dataType": ["int"]}
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
        
        # Handle recreation if requested
        if response.status_code == 200 and RECREATE_COLLECTION:
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
        else:
            print(f"Collection '{collection_name}' already exists")
            
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
    
    # Prepare object data
    object_data = {
        "id": object_id,
        "class": COLLECTION_NAME,
        "properties": properties,
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
            return False
    except Exception as e:
        print(f"Error in Weaviate request: {e}")
        return False

# ----- Main Process -----
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
    
    # Check PDF directory
    pdf_dir = "pdfs"
    if not os.path.exists(pdf_dir) or not os.listdir(pdf_dir):
        print(f"Error: PDFs directory '{pdf_dir}' not found or empty")
        return
    
    # Load documents
    print(f"Loading PDFs from {pdf_dir}...")
    loader = DirectoryLoader(pdf_dir, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
    documents = loader.load()
    
    if not documents:
        print("No documents loaded")
        return
    
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
    
    # Prepare for processing
    all_chunks = []
    doc_embeddings = {}
    
    # Process each document
    for source, source_docs in docs_by_source.items():
        print(f"Processing {os.path.basename(source)}...")
        
        # Extract metadata
        pdf_meta = extract_pdf_metadata(source)
        json_meta = extract_json_metadata(source)
        combined_meta = {**pdf_meta, **json_meta}
        combined_meta["source"] = source
        combined_meta["filename"] = os.path.basename(source)
        
        # Create document embedding for hierarchical approach
        if USE_HIERARCHICAL_EMBEDDINGS:
            doc_title = combined_meta.get("title", os.path.basename(source))
            doc_summary = create_document_summary(source_docs, doc_title)
            if doc_summary:
                doc_text = enhance_text_for_embedding(doc_summary, {"title": doc_title})
                try:
                    doc_embeddings[source] = embedding_model.embed_query(doc_text)
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
    
    print(f"Created {len(all_chunks)} optimized chunks")
    
    # Process in batches
    batch_size = 10  # Small batches for better handling
    success_count = 0
    skip_count = 0
    error_count = 0
    hierarchical_count = 0
    
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(all_chunks)-1)//batch_size + 1}")
        
        # Prepare texts for embedding
        texts = [enhance_text_for_embedding(chunk["metadata"]["text"], chunk["metadata"]) for chunk in batch]
        
        try:
            # Generate embeddings
            embeddings = embedding_model.embed_documents(texts)
            
            # Store each chunk
            for chunk_data, vector in zip(batch, embeddings):
                source = chunk_data["source"]
                
                # Apply hierarchical blending if available
                if USE_HIERARCHICAL_EMBEDDINGS and source in doc_embeddings:
                    original_vector = vector
                    vector = blend_hierarchical_embedding(vector, doc_embeddings[source])
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

if __name__ == "__main__":
    process_documents() 