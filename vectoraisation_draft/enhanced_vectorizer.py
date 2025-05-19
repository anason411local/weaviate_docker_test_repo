#!/usr/bin/env python3

import os
import requests
import json
import uuid
import re
import base64
import numpy as np
import time
import argparse
from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import fitz  # PyMuPDF
from tqdm import tqdm
import datetime
import unicodedata

# Attempt to import optional dependencies
try:
    import nltk
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
    # Download resources
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not available. Some text processing features disabled.")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available. Image processing disabled.")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Memory usage optimization disabled.")

# Load environment variables
load_dotenv()

# --- Configuration ---
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8090")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
COLLECTION_NAME = "Area_expansion_Dep_Anas"

# --- Advanced Vector Configuration ---
# Embedding model
MODEL_NAME = "BAAI/bge-m3"

# Cloud-like similarity adjustment
APPLY_CLOUD_CALIBRATION = True
SELF_HOST_MEAN = 0.52
CLOUD_MEAN = 0.64
SELF_HOST_STD = 0.07
CLOUD_STD = 0.07

# Chunking parameters - more intelligent defaults
DEFAULT_CHUNK_SIZE = 800      # Smaller chunks for more precise embedding
DEFAULT_CHUNK_OVERLAP = 150   # Sufficient overlap to maintain context
MIN_CHUNK_SIZE = 100          # Don't create chunks smaller than this

# Advanced text processing
REMOVE_STOPWORDS = False      # Can be enabled for denser semantic content
REMOVE_NUMBERS = False        # Can be enabled to reduce noise
ADD_TITLE_TO_CHUNKS = True    # Include document title in chunk for context

# Hierarchical embedding boost
USE_HIERARCHICAL_EMBEDDINGS = True
PARENT_EMBEDDING_WEIGHT = 0.15  # How much document-level context to blend in

# --- End of Configuration ---

class VectorQualityOptimizer:
    def __init__(self, args):
        self.args = args
        self.chunk_size = args.chunk_size
        self.chunk_overlap = args.chunk_overlap
        self.apply_calibration = not args.disable_calibration
        self.cloud_mean = args.cloud_mean
        self.use_hierarchical = not args.disable_hierarchical
        self.verbose = args.verbose
        
        # Initialize embedding model
        self.embedding_model = self._create_embedding_model()
        
        # Set up text splitter with advanced parameters
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize statistics counters
        self.total_docs = 0
        self.normalized_count = 0
        self.calibrated_count = 0
        self.hierarchical_count = 0
        self.success_count = 0
        self.error_count = 0
        self.skipped_count = 0

    def _create_embedding_model(self):
        """Create and configure the embedding model with optimal parameters."""
        print(f"Initializing embedding model: {MODEL_NAME}")
        
        # Enhanced model configuration
        model_kwargs = {
            'device': 'cpu',  # Change to 'cuda' if GPU available
            'trust_remote_code': True
        }
        
        # Improved encode settings
        encode_kwargs = {
            'normalize_embeddings': True,  # Always normalize
            'batch_size': 8,  # Balance between memory usage and speed
            'show_progress_bar': self.verbose
        }
        
        return HuggingFaceBgeEmbeddings(
            model_name=MODEL_NAME,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            cache_folder=None
        )

    def create_weaviate_headers(self):
        """Create headers for Weaviate API requests."""
        headers = {"Content-Type": "application/json"}
        if WEAVIATE_API_KEY:
            headers["Authorization"] = f"Bearer {WEAVIATE_API_KEY}"
        return headers

    def check_weaviate_connection(self):
        """Check if Weaviate is accessible."""
        try:
            response = requests.get(
                f"{WEAVIATE_URL}/v1/.well-known/ready", 
                headers=self.create_weaviate_headers()
            )
            
            if response.status_code == 200:
                print("✅ Weaviate connection successful")
                return True
            else:
                print(f"❌ Weaviate error: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"❌ Weaviate connection error: {e}")
            return False

    def detect_calibration_status(self):
        """Analyze existing vectors to detect calibration status."""
        if not self.check_weaviate_connection():
            return False, 0, 0
            
        print("Checking existing vectors for calibration status...")
        headers = self.create_weaviate_headers()
        
        # Get a sample of vectors
        sample_size = 100
        graphql_query = f"""
        {{
          Get {{
            {COLLECTION_NAME}(
              limit: {sample_size}
            ) {{
              _additional {{
                id
                vector
              }}
            }}
          }}
        }}
        """
        
        try:
            response = requests.post(
                f"{WEAVIATE_URL}/v1/graphql",
                headers=headers,
                json={"query": graphql_query}
            )
            
            if response.status_code != 200:
                print("Could not check vectors - continuing without calibration detection.")
                return False, 0, 0
                
            result = response.json()
            if "errors" in result:
                print(f"GraphQL errors: {result['errors']}")
                return False, 0, 0
                
            documents = result.get("data", {}).get("Get", {}).get(COLLECTION_NAME, [])
            if not documents:
                print("No existing vectors found.")
                return False, 0, 0
                
            vectors = []
            for doc in documents:
                if "_additional" in doc and "vector" in doc["_additional"]:
                    vectors.append(np.array(doc["_additional"]["vector"]))
                    
            if len(vectors) < 5:
                return False, 0, 0
                
            # Calculate similarities to detect calibration
            similarity_matrix = []
            for i in range(min(20, len(vectors))):
                for j in range(i+1, min(20, len(vectors))):
                    sim = np.dot(vectors[i], vectors[j])
                    similarity_matrix.append(sim)
                    
            if not similarity_matrix:
                return False, 0, 0
                
            mean_sim = np.mean(similarity_matrix)
            if self.verbose:
                print(f"Detected average similarity between vectors: {mean_sim:.4f}")
            
            # Check if vectors appear to be already calibrated
            calibrated = abs(mean_sim - CLOUD_MEAN) < 0.05
            uncalibrated = abs(mean_sim - SELF_HOST_MEAN) < 0.05
            
            return calibrated or uncalibrated, mean_sim, len(vectors)
        except Exception as e:
            print(f"Error checking calibration status: {e}")
            return False, 0, 0

    def create_collection_if_not_exists(self):
        """Create or update the Weaviate collection schema."""
        if not self.check_weaviate_connection():
            return False
            
        headers = self.create_weaviate_headers()
        collection_name = COLLECTION_NAME
        schema_url = f"{WEAVIATE_URL}/v1/schema/{collection_name}"
        
        # Define schema with proper data types
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
                {"name": "languages", "dataType": ["text"]},
                {"name": "has_toc", "dataType": ["boolean"]},
                {"name": "has_links", "dataType": ["boolean"]},
                {"name": "has_forms", "dataType": ["boolean"]},
                {"name": "has_annotations", "dataType": ["boolean"]},
                {"name": "is_encrypted", "dataType": ["boolean"]},
                {"name": "is_reflowable", "dataType": ["boolean"]},
                {"name": "has_embedded_files", "dataType": ["boolean"]},
                {"name": "embedding_model", "dataType": ["text"]},
                {"name": "chunk_size", "dataType": ["int"]},
                {"name": "chunk_overlap", "dataType": ["int"]},
                {"name": "hierarchical", "dataType": ["boolean"]},
                {"name": "calibrated", "dataType": ["boolean"]}
            ]
        }
        
        try:
            # Check if collection exists
            response = requests.get(schema_url, headers=headers)
            
            if response.status_code == 200:
                print(f"Collection '{collection_name}' already exists.")
                return True
            else:
                # Collection doesn't exist, create it
                print(f"Creating collection '{collection_name}'...")
                create_response = requests.post(
                    f"{WEAVIATE_URL}/v1/schema", 
                    headers=headers, 
                    json=schema
                )
                
                if create_response.status_code == 200:
                    print(f"✅ Created collection '{collection_name}' successfully")
                    return True
                else:
                    print(f"❌ Error creating schema: {create_response.status_code} - {create_response.text}")
                    return False
        except Exception as e:
            print(f"Error during collection setup: {e}")
            return False

    def load_and_process_pdfs(self, directory_path):
        """Load PDFs and process them with advanced chunking and enrichment."""
        print(f"Loading PDFs from {directory_path}...")
        
        # Use LangChain loader for consistent handling
        loader = DirectoryLoader(
            directory_path, 
            glob="**/*.pdf", 
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        
        try:
            documents = loader.load()
        except Exception as e:
            print(f"Error loading documents: {e}")
            return []
            
        print(f"Loaded {len(documents)} document pages initially")
        if not documents:
            return []

        # Group documents by source
        docs_by_source = {}
        for doc in documents:
            source = doc.metadata.get("source", "unknown_source")
            docs_by_source.setdefault(source, []).append(doc)
        
        print(f"Found {len(docs_by_source)} unique PDF documents")
        
        # Process each document
        final_chunks = []
        
        for source, source_docs in tqdm(docs_by_source.items(), desc="Processing documents"):
            # Sort pages
            source_docs.sort(key=lambda x: x.metadata.get("page", 0))
            
            # Extract document metadata from the first page
            doc_metadata = self.extract_pdf_metadata(source)
            title = doc_metadata.get("title", os.path.basename(source))
            
            # Combine all text with page markers for better chunking
            full_text = ""
            for page_doc in source_docs:
                page_num = page_doc.metadata.get("page", 0) + 1
                full_text += f"[PAGE {page_num}]\n{page_doc.page_content}\n\n"
            
            # Create document-level embedding for hierarchical approach
            if self.use_hierarchical:
                doc_summary = self.create_document_summary(full_text, title)
                doc_embedding = self.embed_text(doc_summary, is_doc_level=True)
            else:
                doc_embedding = None
            
            # Split into chunks
            chunks = self.text_splitter.split_text(full_text)
            
            # Process each chunk with advanced text cleaning and embedding
            for i, chunk_text in enumerate(chunks):
                # Clean and enhance text
                cleaned_text = self.clean_text(chunk_text)
                if not cleaned_text or len(cleaned_text) < MIN_CHUNK_SIZE:
                    continue
                    
                # Add title if enabled
                if ADD_TITLE_TO_CHUNKS and title:
                    enhanced_text = f"Title: {title}\n\n{cleaned_text}"
                else:
                    enhanced_text = cleaned_text
                
                # Extract page numbers from chunk
                page_nums = re.findall(r'\[PAGE (\d+)\]', chunk_text)
                chunk_page = int(page_nums[0]) - 1 if page_nums else 0
                
                # Create chunk metadata
                chunk_metadata = {
                    "text": enhanced_text,
                    "source": source,
                    "filename": os.path.basename(source),
                    "folder": os.path.dirname(source),
                    "page": chunk_page,
                    "embedding_model": MODEL_NAME,
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "hierarchical": self.use_hierarchical,
                    "calibrated": self.apply_calibration
                }
                
                # Add document metadata
                chunk_metadata.update(doc_metadata)
                
                final_chunks.append({
                    "text": enhanced_text,
                    "metadata": chunk_metadata,
                    "doc_embedding": doc_embedding
                })
        
        print(f"Created {len(final_chunks)} optimized text chunks")
        return final_chunks

    def extract_pdf_metadata(self, pdf_path):
        """Extract rich metadata from a PDF file."""
        default_meta = {
            "title": "", "author": "", "subject": "", "keywords": "",
            "creation_date": "", "modification_date": "", "page_count": 0,
            "file_size": 0, "has_images": False, "image_count": 0
        }
        
        try:
            doc = fitz.open(pdf_path)
            metadata = default_meta.copy()
            
            # Extract basic metadata
            doc_meta = doc.metadata
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
            for page_num in range(min(3, doc.page_count)):  # Check first 3 pages
                page = doc[page_num]
                if page.get_images():
                    metadata["has_images"] = True
                    metadata["image_count"] = len(doc.get_page_images(page_num))
                    break
            
            doc.close()
            return metadata
        except Exception as e:
            print(f"Error extracting PDF metadata from {pdf_path}: {e}")
            return default_meta

    def create_document_summary(self, text, title=""):
        """Create a condensed summary of document for hierarchical embedding."""
        # Extract potential headers (text after PAGE markers)
        headers = re.findall(r'\[PAGE \d+\]\s*([^\n\[\]]{5,100})', text)
        
        # Get first few paragraphs
        paragraphs = text.split('\n\n')[:3]
        
        # Combine title, headers and paragraphs
        if title:
            summary_parts = [f"Document: {title}"]
        else:
            summary_parts = []
            
        if headers:
            summary_parts.append("Sections: " + " | ".join(headers[:5]))
            
        summary_parts.extend(paragraphs)
        
        # Limit to reasonable length
        summary = "\n\n".join(summary_parts)
        if len(summary) > 2000:
            summary = summary[:2000]
            
        return summary

    def clean_text(self, text):
        """Apply advanced text cleaning for better embedding quality."""
        if not text:
            return ""
            
        # Basic cleaning
        cleaned = re.sub(r'\s+', ' ', text).strip()
        
        # Remove artifacts
        artifacts = [
            r'(\n\s*\d+\s*\n)', r'^\s*\d+\s*$', r'©.*?reserved\.?',
            r'(Page|PAGE)(\s+\d+\s+of\s+\d+)', r'(http|https|www)\S+\s',
            r'\[\s*\d+\s*\]', r'(^|[^a-zA-Z0-9])\d{5,}([^a-zA-Z0-9]|$)',
            r'\\[a-zA-Z]+\{.*?\}', r'</?[a-z]+>',
        ]
        for pattern in artifacts:
            cleaned = re.sub(pattern, ' ', cleaned, flags=re.MULTILINE)
        
        # Fix hyphenation
        cleaned = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', cleaned)
        
        # Fix spacing around punctuation
        cleaned = re.sub(r'\s+([.,;:!?)])', r'\1', cleaned)
        cleaned = re.sub(r'([(])\s+', r'\1', cleaned)
        
        # Normalize quotes and apostrophes
        cleaned = re.sub(r'["""]', '"', cleaned)
        cleaned = re.sub(r"['']", "'", cleaned)
        
        # Unicode normalization
        cleaned = unicodedata.normalize('NFKC', cleaned)
        
        # Enhanced sentence structure with NLTK if available
        if NLTK_AVAILABLE:
            try:
                sentences = nltk.sent_tokenize(cleaned)
                for i in range(len(sentences)):
                    if sentences[i] and sentences[i][0].islower():
                        sentences[i] = sentences[i][0].upper() + sentences[i][1:]
                    if sentences[i] and sentences[i][-1] not in ['.', '!', '?']:
                        sentences[i] += '.'
                cleaned = ' '.join(sentences)
                
                # Optional stopword removal (disabled by default)
                if REMOVE_STOPWORDS:
                    stop_words = set(stopwords.words('english'))
                    words = cleaned.split()
                    cleaned = ' '.join([w for w in words if w.lower() not in stop_words])
            except:
                pass  # Fallback if NLTK processing fails
        
        # Final cleaning
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Remove [PAGE X] markers for the final version
        cleaned = re.sub(r'\[PAGE \d+\]\s*', '', cleaned)
        
        # Length check
        if len(cleaned) < MIN_CHUNK_SIZE:
            return ""
            
        return cleaned

    def embed_text(self, text, is_doc_level=False):
        """Generate optimal embeddings with proper instruction formatting."""
        if not text:
            return None
            
        # Add specialized instruction prefix for BGE model
        if is_doc_level:
            instruction = "Represent this document for retrieval: "
        else:
            instruction = "Represent this passage for retrieval: "
            
        # Add instruction to text
        text_to_embed = f"{instruction}{text}"
        
        # Generate embedding
        try:
            result = self.embedding_model.embed_query(text_to_embed)
            return result
        except Exception as e:
            print(f"Error generating embedding: {e}")
            print(f"Problematic text: {text[:100]}...")
            return None

    def normalize_vector(self, vector):
        """Normalize vector to unit length for consistent similarity measures."""
        if not vector:
            return None
            
        vec_np = np.array(vector)
        norm = np.linalg.norm(vec_np)
        
        if norm > 0:
            normalized = vec_np / norm
            self.normalized_count += 1
            return normalized.tolist()
        
        return vector

    def cloud_calibrate_vector(self, vector):
        """Apply cloud-like calibration to vector."""
        if not self.apply_calibration or not vector:
            return vector
            
        vec_np = np.array(vector)
        
        # Calculate scaling factor
        scaling_factor = self.cloud_mean / SELF_HOST_MEAN
        
        # Apply a small boost that affects cosine similarity
        adjustment = (scaling_factor - 1.0) * 0.2
        
        # Create primary direction vector
        dim = len(vec_np)
        primary_direction = np.ones(dim) / np.sqrt(dim)
        
        # Blend the vector with the primary direction
        calibrated_vec = vec_np + (adjustment * primary_direction)
        
        # Re-normalize
        norm = np.linalg.norm(calibrated_vec)
        if norm > 0:
            calibrated_vec = calibrated_vec / norm
            self.calibrated_count += 1
            
        return calibrated_vec.tolist()

    def create_hierarchical_embedding(self, chunk_embedding, doc_embedding):
        """Blend chunk and document embeddings for hierarchical representation."""
        if not self.use_hierarchical or not chunk_embedding or not doc_embedding:
            return chunk_embedding
            
        # Convert to numpy arrays
        chunk_vec = np.array(chunk_embedding)
        doc_vec = np.array(doc_embedding)
        
        # Weighted blend
        blended_vec = (1 - PARENT_EMBEDDING_WEIGHT) * chunk_vec + PARENT_EMBEDDING_WEIGHT * doc_vec
        
        # Normalize
        norm = np.linalg.norm(blended_vec)
        if norm > 0:
            hierarchical_vec = blended_vec / norm
            self.hierarchical_count += 1
            return hierarchical_vec.tolist()
            
        return chunk_embedding

    def check_object_exists(self, object_id):
        """Check if an object already exists in Weaviate."""
        headers = self.create_weaviate_headers()
        try:
            response = requests.get(
                f"{WEAVIATE_URL}/v1/objects/{COLLECTION_NAME}/{object_id}", 
                headers=headers
            )
            return response.status_code == 200
        except:
            return False

    def store_chunk_in_weaviate(self, chunk_data):
        """Store a processed chunk in Weaviate with its vector."""
        text = chunk_data.get("text", "")
        metadata = chunk_data.get("metadata", {})
        doc_embedding = chunk_data.get("doc_embedding")
        
        # Skip if no text
        if not text:
            return "EMPTY"
            
        # Generate ID based on content to aid deduplication
        unique_str = f"{metadata.get('source', '')}-{metadata.get('page', 0)}-{text[:50]}"
        object_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_str))
        
        # Check if object already exists
        if self.check_object_exists(object_id):
            self.skipped_count += 1
            return "SKIPPED"
            
        # Generate embedding for this chunk
        chunk_embedding = self.embed_text(text)
        if not chunk_embedding:
            return "EMBEDDING_FAILED"
            
        # Normalize the embedding
        chunk_embedding = self.normalize_vector(chunk_embedding)
        
        # Apply hierarchical blending if available
        if doc_embedding and self.use_hierarchical:
            chunk_embedding = self.create_hierarchical_embedding(chunk_embedding, doc_embedding)
            
        # Apply cloud calibration if enabled
        if self.apply_calibration:
            chunk_embedding = self.cloud_calibrate_vector(chunk_embedding)
            
        # Prepare object data
        object_data = {
            "id": object_id,
            "class": COLLECTION_NAME,
            "properties": metadata,
            "vector": chunk_embedding
        }
        
        # Store in Weaviate
        headers = self.create_weaviate_headers()
        try:
            response = requests.post(
                f"{WEAVIATE_URL}/v1/objects", 
                headers=headers, 
                json=object_data
            )
            
            if response.status_code in [200, 201]:
                self.success_count += 1
                return "SUCCESS"
            else:
                self.error_count += 1
                if self.verbose:
                    print(f"Error creating object: {response.status_code} - {response.text}")
                return "ERROR"
        except Exception as e:
            self.error_count += 1
            if self.verbose:
                print(f"Exception storing chunk: {e}")
            return "EXCEPTION"

    def process_in_batches(self, chunks):
        """Process chunks in optimized batches."""
        total = len(chunks)
        
        # Determine batch size based on available memory and dataset size
        batch_size = self.calculate_batch_size(chunks)
        
        # Process in batches
        for i in range(0, total, batch_size):
            batch = chunks[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(total-1)//batch_size + 1} ({len(batch)} chunks)")
            
            # Process each chunk in the batch
            with tqdm(total=len(batch), desc="Storing chunks") as pbar:
                for chunk in batch:
                    result = self.store_chunk_in_weaviate(chunk)
                    pbar.update(1)
                    if result == "ERROR" and self.verbose:
                        pbar.write("Error storing chunk")

    def calculate_batch_size(self, chunks):
        """Calculate optimal batch size based on available resources."""
        default_batch = 20
        
        if not PSUTIL_AVAILABLE or not chunks:
            return default_batch
            
        try:
            # Check available memory
            available_memory_gb = psutil.virtual_memory().available / (1024 * 1024 * 1024)
            
            # Calculate average chunk size (rough estimate)
            avg_text_len = sum(len(chunk.get("text", "")) for chunk in chunks[:min(100, len(chunks))]) / min(100, len(chunks))
            
            # Adjust batch size based on memory and chunk size
            if available_memory_gb < 2:  # Very low memory
                return max(5, default_batch // 2)
            elif available_memory_gb > 8 and avg_text_len < 2000:  # High memory, small chunks
                return min(50, default_batch * 2)
                
            return default_batch
        except:
            return default_batch

    def print_summary(self):
        """Print summary of processing results."""
        print("\n" + "="*60)
        print(f"🏁 VECTORIZATION COMPLETE")
        print("="*60)
        print(f"Successfully stored: {self.success_count} chunks")
        print(f"Skipped (duplicates): {self.skipped_count} chunks")
        print(f"Errors: {self.error_count} chunks")
        print(f"Total processed: {self.success_count + self.skipped_count + self.error_count} chunks")
        print("-"*60)
        print(f"Embedding model: {MODEL_NAME}")
        print(f"Chunk size: {self.chunk_size}, Overlap: {self.chunk_overlap}")
        if self.apply_calibration:
            print(f"Cloud calibration: APPLIED (target mean: {self.cloud_mean})")
            print(f"Calibrated vectors: {self.calibrated_count}")
            print(f"Expected similarity boost: ~{((self.cloud_mean/SELF_HOST_MEAN - 1)*100):.1f}%")
        else:
            print("Cloud calibration: DISABLED")
        if self.use_hierarchical:
            print(f"Hierarchical embeddings: ENABLED (parent weight: {PARENT_EMBEDDING_WEIGHT})")
            print(f"Hierarchical vectors: {self.hierarchical_count}")
        else:
            print("Hierarchical embeddings: DISABLED")
        print(f"Normalized vectors: {self.normalized_count}")
        print("="*60)
        
        if self.success_count == 0:
            print("\n⚠️ No documents were successfully imported. Please check the errors.")
            print("Possible issues:")
            print("  - Weaviate connection problems")
            print("  - Missing PDF files")
            print("  - Embedding generation errors")

    def run(self):
        """Run the complete vectorization process."""
        print("\n" + "="*60)
        print("🚀 ENHANCED VECTOR QUALITY OPTIMIZER")
        print("="*60)
        print(f"Target collection: {COLLECTION_NAME}")
        
        # Check Weaviate connection
        if not self.check_weaviate_connection():
            print("❌ Cannot connect to Weaviate. Exiting.")
            return False
            
        # Check for existing calibration
        if self.apply_calibration:
            print(f"Cloud calibration ENABLED (target mean: {self.cloud_mean})")
            calibration_detected, mean_sim, count = self.detect_calibration_status()
            if calibration_detected and count > 0:
                if abs(mean_sim - self.cloud_mean) < 0.05:
                    print(f"⚠️ Your database contains {count} vectors that appear ALREADY CALIBRATED")
                    print(f"⚠️ Adding more calibrated vectors is appropriate.")
                elif abs(mean_sim - SELF_HOST_MEAN) < 0.05:
                    print(f"⚠️ Your database has {count} UNCALIBRATED vectors (mean: {mean_sim:.4f})")
                    print(f"⚠️ Adding calibrated vectors will create MIXED calibration.")
                    proceed = input("Continue with calibration? (y/n): ")
                    if proceed.lower() != 'y':
                        print("Disabling calibration to match existing vectors.")
                        self.apply_calibration = False
                else:
                    print(f"⚠️ Your database has unusual calibration (mean: {mean_sim:.4f})")
        else:
            print("Cloud calibration DISABLED")
            
        # Check hierarchical status
        if self.use_hierarchical:
            print(f"Hierarchical embeddings ENABLED (parent weight: {PARENT_EMBEDDING_WEIGHT})")
            print("This will blend document-level context with chunk-level embeddings")
        else:
            print("Hierarchical embeddings DISABLED")
            
        # Create or check collection
        if not self.create_collection_if_not_exists():
            print("❌ Failed to setup Weaviate collection. Exiting.")
            return False
            
        # Process PDFs from directory
        pdfs_directory = self.args.pdfs_dir
        if not os.path.isdir(pdfs_directory):
            print(f"❌ PDFs directory '{pdfs_directory}' not found. Exiting.")
            return False
            
        # Load and process documents
        chunks = self.load_and_process_pdfs(pdfs_directory)
        if not chunks:
            print("❌ No document chunks generated. Exiting.")
            return False
            
        # Store chunks in Weaviate
        self.process_in_batches(chunks)
        
        # Print summary
        self.print_summary()
        
        return self.success_count > 0

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Enhanced Vector Quality Optimizer for Weaviate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Main parameters
    parser.add_argument("--pdfs-dir", type=str, default="pdfs",
                       help="Directory containing PDF files")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                       help="Size of text chunks for embedding")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
                       help="Overlap between chunks")
    
    # Advanced options
    parser.add_argument("--disable-calibration", action="store_true",
                       help="Disable cloud-like similarity calibration")
    parser.add_argument("--cloud-mean", type=float, default=CLOUD_MEAN,
                       help="Target cloud similarity mean")
    parser.add_argument("--disable-hierarchical", action="store_true",
                       help="Disable hierarchical embeddings")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Create and run the optimizer
    optimizer = VectorQualityOptimizer(args)
    optimizer.run()

if __name__ == "__main__":
    main() 