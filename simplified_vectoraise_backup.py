import os
import requests
import json
import uuid
import re
import base64
from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import fitz  # PyMuPDF for enhanced metadata extraction
import datetime
import unicodedata
import time
import numpy as np

# Attempt to import nltk and PIL, and set flags for their availability
try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: nltk library not found. Text cleaning and sentence tokenization will be limited. Please install with 'pip install nltk'")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: Pillow (PIL) library not found. Cover image extraction will be disabled. Please install with 'pip install Pillow'")


# Download NLTK resources if NLTK is available
if NLTK_AVAILABLE:
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    except Exception as e:
        print(f"Warning: Failed to download NLTK resources (punkt, stopwords). This might affect text processing. Error: {e}")
        # You might want to set NLTK_AVAILABLE to False here if these resources are critical
        # NLTK_AVAILABLE = False


# Load environment variables
load_dotenv()

# --- Configuration for Weaviate ---
# Using localhost:8090 for your local self-hosted Weaviate instance
WEAVIATE_URL = "http://localhost:8090"  # Changed from 127.0.0.1 to localhost
# For Docker containers, you might need to use 'host.docker.internal' instead of 'localhost'
# WEAVIATE_URL = "http://host.docker.internal:8090"  # Uncomment if needed
# WEAVIATE_API_KEY will be loaded from .env.
# If your local Weaviate does not use an API key, WEAVIATE_API_KEY can be empty or not set in .env.
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

# Collection name - centralized here for easy changing
COLLECTION_NAME = "Area_expansion_Dep_Anas"

print(f"Attempting to connect to Weaviate at: {WEAVIATE_URL}")

if not WEAVIATE_API_KEY:
    print("Warning: WEAVIATE_API_KEY is not set in the .env file. Proceeding without API key.")
    print("If your self-hosted Weaviate instance requires an API key, requests will likely fail.")
else:
    print("WEAVIATE_API_KEY found in .env file.")

# The script previously had logic to prepend https:// if no scheme was found.
# Since we are hardcoding http:// for your local instance, this is less critical,
# but good to keep in mind if the URL were to be configured differently.
if not WEAVIATE_URL.startswith(("http://", "https://")):
    print(f"Warning: WEAVIATE_URL '{WEAVIATE_URL}' does not have a scheme. Defaulting to http://.")
    WEAVIATE_URL = f"http://{WEAVIATE_URL}"
# --- End of Weaviate Configuration ---


# Define the embedding model configuration
model_name = "BAAI/bge-m3"
model_kwargs = {
    'device': 'cpu', # Consider changing to 'cuda' if you have a GPU and PyTorch with CUDA support
    'trust_remote_code': True
}
encode_kwargs = {
    'normalize_embeddings': True,
    'batch_size': 8,
    'show_progress_bar': True,
    # Advanced settings for better quality embeddings
    'pooling_strategy': 'cls',  # Use CLS token pooling for optimal quality
    'max_length': 1024  # Increase from default for more context
}
embedding_model = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    cache_folder=None # Set a path here if you want to cache the model, e.g., "./model_cache"
)

# Text splitting parameters - optimized for quality
CHUNK_SIZE = 800  # Smaller chunk size for more focused embeddings
CHUNK_OVERLAP = 200  # Significant overlap to maintain context

# Initialize enhanced text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    # Optimized separators for semantic chunking in priority order
    separators=[
        "\n\n\n",  # Triple linebreak (highest priority)
        "\n\n",    # Double linebreak
        "\n",      # Single linebreak
        ".",       # End of sentence
        "!",       # Exclamation
        "?",       # Question
        ";",       # Semicolon 
        ":",       # Colon
        ",",       # Comma
        " ",       # Space (lowest priority)
        ""         # Character level (fallback)
    ]
)

# Add near the top with other globals
USE_HIERARCHICAL_EMBEDDINGS = True
PARENT_EMBEDDING_WEIGHT = 0.15

def clean_text(text):
    """Enhanced text cleaning to improve embedding quality"""
    if not text:
        return ""
    cleaned = re.sub(r'\s+', ' ', text).strip()
    artifacts = [
        r'(\n\s*\d+\s*\n)', r'^\s*\d+\s*$', r'©.*?reserved\.?',
        r'(Page|PAGE)(\s+\d+\s+of\s+\d+)', r'(http|https|www)\S+\s',
        r'\[\s*\d+\s*\]', r'(^|[^a-zA-Z0-9])\d{5,}([^a-zA-Z0-9]|$)',
        r'\\[a-zA-Z]+\{.*?\}', r'</?[a-z]+>',
    ]
    for pattern in artifacts:
        cleaned = re.sub(pattern, ' ', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', cleaned)
    cleaned = re.sub(r'\s+([.,;:!?)])', r'\1', cleaned)
    cleaned = re.sub(r'([(])\s+', r'\1', cleaned)
    cleaned = re.sub(r'["""]', '"', cleaned)
    cleaned = re.sub(r"['']", "'", cleaned)
    cleaned = unicodedata.normalize('NFKC', cleaned)
    
    # Advanced OCR error correction for better semantics
    ocr_fixes = [
        (r'l\b', 'i'), (r'\bII\b', 'H'), (r'\b0\b', 'O'), (r'rn\b', 'm'),
        (r'rnm\b', 'mm'), (r'\blJ\b', 'U'), (r'\bl\b', '1'), (r'\blo\b', '10'),
        (r'\ba\s+(\w)', r'a \1'),
    ]
    for old, new in ocr_fixes:
        cleaned = re.sub(old, new, cleaned)
    
    # Obfuscate emails and URLs for better semantic focus
    cleaned = re.sub(r'(https?:\/\/[^\s]+)', lambda m: m.group(1).replace('.', ' dot '), cleaned)
    pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    def email_replacer(match):
        email = match.group(0)
        return email.replace('@', ' at ').replace('.', ' dot ')
    cleaned = re.sub(pattern, email_replacer, cleaned)

    if NLTK_AVAILABLE: # Only use nltk if it was successfully imported
        try:
            # Enhanced sentence normalization
            sentences = nltk.sent_tokenize(cleaned)
            normalized_sentences = []
            
            for sentence in sentences:
                if not sentence.strip():
                    continue
                # Fix capitalization for better semantics
                if sentence and sentence[0].islower():
                    sentence = sentence[0].upper() + sentence[1:]
                # Ensure proper sentence endings
                if sentence and sentence[-1] not in ['.', '!', '?']:
                    sentence += '.'
                    
                normalized_sentences.append(sentence)
                
            cleaned = ' '.join(normalized_sentences)
            
            # Optional lemmatization for better semantic matching
            if 'wordnet' in nltk.data.path:  # Check if WordNet is available
                try:
                    from nltk.stem import WordNetLemmatizer
                    lemmatizer = WordNetLemmatizer()
                    words = nltk.word_tokenize(cleaned)
                    lemmatized_words = [lemmatizer.lemmatize(word) for word in words 
                                        if len(word) > 1 or word.isalnum()]
                    # Keep original but blend with lemmatized for better embedding
                    cleaned = cleaned + " " + " ".join(lemmatized_words)
                except:
                    pass  # Fallback if lemmatization fails
                    
        except Exception as e: # Catch potential errors during nltk processing
            print(f"Warning: Error during NLTK processing: {e}. Proceeding with un-tokenized text for this chunk.")

    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if len(cleaned) < 50:
        return ""
    return cleaned

def extract_json_metadata(pdf_path):
    """Extract metadata from JSON file associated with a PDF.
    Handles all JSON fields with document_ prefix to match schema."""
    try:
        pdf_base_name = os.path.basename(pdf_path)
        pdf_name_no_ext = os.path.splitext(pdf_base_name)[0]
        pdf_dir = os.path.dirname(pdf_path)
        json_path_try1 = str(pdf_path).replace('.pdf', '.json').replace('.PDF', '.json')
        json_path = None

        if os.path.exists(json_path_try1):
            json_path = json_path_try1
        else:
            json_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.json')]
            for json_file_candidate in json_files:
                json_name_no_ext = os.path.splitext(json_file_candidate)[0]
                if json_name_no_ext.lower().startswith(pdf_name_no_ext.lower()) or pdf_name_no_ext.lower() in json_name_no_ext.lower():
                    json_path = os.path.join(pdf_dir, json_file_candidate)
                    print(f"Found matching JSON: {json_path} for PDF: {pdf_path}")
                    break
        if not json_path or not os.path.exists(json_path):
            return {}
            
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            
        if 'id' in json_data: 
            del json_data['id']
            
        processed_data = {}
        for key, value in json_data.items():
            field_name = f"document_{key}"
            
            # Preserve primitive data types for schema compatibility
            if isinstance(value, (bool, int, float, str)):
                processed_data[field_name] = value
            else:
                # Convert complex objects to JSON strings
                try:
                    processed_data[field_name] = json.dumps(value)
                except (TypeError, OverflowError):
                    # If JSON serialization fails, convert to string representation
                    processed_data[field_name] = str(value)
                    
        return processed_data
    except Exception as e:
        print(f"Error extracting JSON metadata from {pdf_path}: {e}")
        return {}

def extract_pdf_metadata(pdf_path):
    default_meta = {
        "title": "", "author": "", "subject": "", "keywords": "", "creator": "",
        "producer": "", "creation_date": "", "modification_date": "", "page_count": 0,
        "file_size": 0, "pdf_version": "unknown", "toc": "", "has_toc": False, "has_links": False,
        "has_forms": False, "has_annotations": False, "has_images": False, "image_count": 0,
        "has_tables": False, "table_count": 0, "total_chars": 0, "total_words": 0,
        "languages": "unknown", "is_encrypted": False, "encryption_method": "",
        "permissions": "", "is_reflowable": False, "has_embedded_files": False,
        "embedded_file_names": "[]", "embedded_files_info": "[]", "fonts_used": "[]", # Ensure JSON valid strings
        "cover_image": ""
    }
    try:
        doc = fitz.open(pdf_path)
        metadata = default_meta.copy() # Start with defaults
        doc_meta = doc.metadata
        metadata.update({
            "title": doc_meta.get("title", ""), "author": doc_meta.get("author", ""),
            "subject": doc_meta.get("subject", ""), "keywords": doc_meta.get("keywords", ""),
            "creator": doc_meta.get("creator", ""), "producer": doc_meta.get("producer", ""),
            "creation_date": doc_meta.get("creationDate", ""), "modification_date": doc_meta.get("modDate", ""),
            "page_count": doc.page_count, "file_size": os.path.getsize(pdf_path) if os.path.exists(pdf_path) else 0,
            "pdf_version": "unknown", # Set a default value
            "has_forms": bool(doc.is_form_pdf), "is_encrypted": doc.is_encrypted,
            "encryption_method": str(doc.encryption_method()) if doc.is_encrypted else "", # Ensure it's a string
            "permissions": json.dumps(doc.permissions) if hasattr(doc, 'permissions') else "{}",
            "is_reflowable": bool(doc.is_reflowable),
            "has_embedded_files": len(doc.embfile_names()) > 0,
            "embedded_file_names": json.dumps(doc.embfile_names()),
        })
        
        # Handle PDF version - fixed the issue with missing version attribute
        try:
            if hasattr(doc, 'version') and doc.version:
                metadata["pdf_version"] = f"{doc.version[0]}.{doc.version[1]}" if isinstance(doc.version, tuple) and len(doc.version) > 1 else "unknown"
        except:
            metadata["pdf_version"] = "unknown"

        toc_data = doc.get_toc()
        if toc_data:
            metadata["has_toc"] = True
            metadata["toc"] = json.dumps([{"level": level, "title": title, "page": page} for level, title, page in toc_data])

        for date_field in ["creation_date", "modification_date"]:
            date_str = metadata[date_field]
            if date_str and date_str.startswith("D:"):
                try:
                    parsed_date = fitz.utils. późniejszy(date_str) # Use fitz utility if available, or manual parse
                    metadata[date_field] = datetime.datetime(parsed_date.year, parsed_date.month, parsed_date.day, parsed_date.hour, parsed_date.minute, parsed_date.second).isoformat()
                except: # Fallback for manual parsing if fitz utility fails or not suitable
                    try:
                        date_str_core = date_str[2:].split('Z')[0].split('+')[0].split('-')[0] #Basic cleanup
                        year, month, day = int(date_str_core[0:4]), int(date_str_core[4:6]), int(date_str_core[6:8])
                        hour, minute, second = (int(date_str_core[8:10]) if len(date_str_core) > 8 else 0), \
                                             (int(date_str_core[10:12]) if len(date_str_core) > 10 else 0), \
                                             (int(date_str_core[12:14]) if len(date_str_core) > 12 else 0)
                        metadata[date_field] = datetime.datetime(year, month, day, hour, minute, second).isoformat()
                    except: pass # Keep original if parsing fails

        fonts_used = set()
        detected_langs_page = {}

        for page_num, page_obj in enumerate(doc):
            text = page_obj.get_text("text")
            metadata["total_chars"] += len(text)
            metadata["total_words"] += len(re.findall(r'\w+', text))
            if page_obj.get_images(): metadata["has_images"] = True; metadata["image_count"] += len(page_obj.get_images())
            if page_obj.get_links(): metadata["has_links"] = True
            if list(page_obj.annots()): metadata["has_annotations"] = True # Check if generator is non-empty
            
            blocks = page_obj.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_LIGATURES & ~fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
            for block in blocks:
                if block["type"] == 0: # text block
                    for line in block["lines"]:
                        for span in line["spans"]:
                            fonts_used.add(span["font"])
                            # Basic language detection per span (can be noisy)
                            # lang = span.get("lang", "unknown") # fitz often provides this
                            # if lang != "unknown": detected_langs_page[lang] = detected_langs_page.get(lang,0)+1

            if "   " in text and ("|" in text or "\t" in text): metadata["has_tables"] = True; metadata["table_count"] += text.count("\n\n") // 3


        metadata["fonts_used"] = json.dumps(list(fonts_used))
        # Consolidate page languages (if any useful were detected)
        # if detected_langs_page: metadata["languages"] = ",".join(sorted(detected_langs_page, key=detected_langs_page.get, reverse=True))


        if metadata["has_embedded_files"]:
            infos = []
            for i, name in enumerate(doc.embfile_names()):
                try: infos.append({"name": name, "size": len(doc.embfile_get(i))}) # embfile_get by index
                except: infos.append({"name": name, "size": -1}) # Error getting size
            metadata["embedded_files_info"] = json.dumps(infos)

        if PIL_AVAILABLE and metadata["has_images"] and doc.page_count > 0:
            try:
                pix = doc[0].get_pixmap(alpha=False) # Get pixmap of first page
                if pix.width > 0 and pix.height > 0:
                    img_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    img_pil.thumbnail((300, 300))
                    buffer = BytesIO()
                    img_pil.save(buffer, format="JPEG")
                    metadata["cover_image"] = f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
            except Exception as e: print(f"Error extracting cover image: {e}")

        json_meta = extract_json_metadata(str(pdf_path)) # Ensure pdf_path is string for extract_json_metadata
        metadata.update(json_meta)
        doc.close()
        return metadata
    except Exception as e:
        print(f"Error extracting PDF metadata from {pdf_path}: {e}")
        # Attempt to get JSON metadata even if PDF processing fails
        json_meta = extract_json_metadata(str(pdf_path))
        final_meta = default_meta.copy()
        final_meta.update(json_meta)
        if os.path.exists(pdf_path): # Get file size if possible
            final_meta["file_size"] = os.path.getsize(pdf_path)
        return final_meta


def extract_content_features(pdf_path):
    default_features = {
        "page_features": [], "document_structure": {"heading_levels": [], "section_counts": 0, "avg_section_length": 0,},
        "layout_info": {"multi_column": False, "has_header_footer": False, "page_dimensions": []},
        "readability_stats": {"avg_sentence_length": 0, "avg_word_length": 0, "complex_word_percentage": 0,}
    }
    try:
        doc = fitz.open(pdf_path)
        content_features = default_features.copy()
        total_sentences_doc, total_words_doc, total_chars_doc, complex_words_doc = 0, 0, 0, 0

        for page_num, page_obj in enumerate(doc):
            page_rect = page_obj.rect
            content_features["layout_info"]["page_dimensions"].append({"width": page_rect.width, "height": page_rect.height})
            text_content = page_obj.get_text("text")
            blocks = page_obj.get_text("blocks", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_LIGATURES & ~fitz.TEXT_PRESERVE_WHITESPACE)
            
            if len(blocks) > 3: # Check for multi-column
                x_positions = [block[0] for block in blocks if len(block) > 4 and block[4].strip()] # x0 of text blocks
                if len(set(round(x/10) for x in x_positions)) > 2: # Heuristic: significant variation in x positions
                     content_features["layout_info"]["multi_column"] = True

            for block in blocks: # Check for header/footer
                if len(block)>4 and (block[1] < 50 or block[3] > page_rect.height - 50): # y0 or y1
                    content_features["layout_info"]["has_header_footer"] = True; break

            current_page_words = re.findall(r'\b\w+\b', text_content.lower())
            total_words_doc += len(current_page_words)
            for word in current_page_words: total_chars_doc += len(word); complex_words_doc += (1 if len(word) > 8 else 0)
            
            if NLTK_AVAILABLE:
                try: current_page_sentences = nltk.sent_tokenize(text_content); total_sentences_doc += len(current_page_sentences)
                except: pass # Ignore sentence tokenization error for this page

            # Simplified heading extraction (font size based)
            page_text_dict = page_obj.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_LIGATURES & ~fitz.TEXT_PRESERVE_WHITESPACE)
            for blk in page_text_dict.get("blocks", []):
                if blk["type"] == 0: # text block
                    for ln in blk.get("lines", []):
                        for sp in ln.get("spans", []):
                            font_size = sp.get("size", 0)
                            txt = sp.get("text", "").strip()
                            if txt and font_size > 14 and len(txt.split()) < 10: # Heuristic for headings
                                content_features["document_structure"]["heading_levels"].append({"text": txt, "font_size": font_size, "page": page_num + 1})
            
            content_features["page_features"].append({
                "page_number": page_num + 1, "text_blocks": len([b for b in blocks if len(b)>4 and b[4].strip()]),
                "word_count": len(current_page_words), "char_count": len(text_content),
                "image_count": len(page_obj.get_images())
            })

        if total_sentences_doc > 0: content_features["readability_stats"]["avg_sentence_length"] = total_words_doc / total_sentences_doc
        if total_words_doc > 0:
            content_features["readability_stats"]["avg_word_length"] = total_chars_doc / total_words_doc
            content_features["readability_stats"]["complex_word_percentage"] = (complex_words_doc / total_words_doc) * 100

        headings = sorted(content_features["document_structure"]["heading_levels"], key=lambda x: (x["page"], -x["font_size"]))
        if headings:
            # Basic section counting based on top-level headings (largest font sizes)
            top_font_size = headings[0]['font_size']
            main_headings = [h for h in headings if h['font_size'] >= top_font_size * 0.9] # Allow slight variation for top level
            content_features["document_structure"]["section_counts"] = len(main_headings)
            # Further refine heading levels to a summary
            content_features["document_structure"]["heading_levels"] = [{"text": h["text"][:100], "level": int(h["font_size"]), "page":h["page"]} for h in headings[:20]]


        doc.close()
        return content_features
    except Exception as e:
        print(f"Error extracting content features from {pdf_path}: {e}")
        return default_features


def load_and_split_pdfs(directory_path: str):
    print(f"Loading PDFs from {directory_path}...")
    loader = DirectoryLoader(
        directory_path, glob="**/*.pdf", loader_cls=PyPDFLoader,
        show_progress=True, use_multithreading=False, # Multithreading can be problematic with some PDF libs
        loader_kwargs={"extract_images": False} # PyPDFLoader specific, set to True if images needed as separate docs
    )
    try:
        documents = loader.load()
    except Exception as e:
        print(f"Error loading documents: {e}. Please check PDF files and directory.")
        return []
        
    print(f"Loaded {len(documents)} document pages initially.")
    if not documents: return []

    docs_by_source = {}
    for doc in documents:
        source = doc.metadata.get("source", "unknown_source")
        docs_by_source.setdefault(source, []).append(doc)

    final_split_docs_content = [] # Store as list of strings first for easier splitting

    # Using RecursiveCharacterTextSplitter as the primary robust splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len,
        separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""] # More granular separators
    )

    print("Applying text splitting...")
    for source, source_docs in docs_by_source.items():
        source_docs.sort(key=lambda x: x.metadata.get("page", 0))
        full_text_for_source = ""
        for page_doc in source_docs:
            # Add a page marker. Using a less Markdown-like marker to avoid issues with Markdown splitter if used later
            full_text_for_source += f"[PAGE_BREAK: {page_doc.metadata.get('page', 0) + 1}]\n{page_doc.page_content}\n\n"
        
        chunks = text_splitter.split_text(full_text_for_source)
        for chunk_text in chunks:
            # Create a representative metadata for the chunk
            # For simplicity, we'll associate the chunk with the source and the first page metadata
            # More sophisticated logic could try to map chunks back to original pages
            chunk_metadata = source_docs[0].metadata.copy() if source_docs else {"source": source}
            # Store as dicts with text and metadata
            final_split_docs_content.append({"page_content": chunk_text, "metadata": chunk_metadata})

    print(f"Split into {len(final_split_docs_content)} text chunks.")

    # Deduplication (simple version based on exact content match after stripping)
    print("Deduplicating chunks...")
    unique_chunks_dict = {} # Use dict to store unique chunks by content
    for chunk_data in final_split_docs_content:
        content_key = chunk_data["page_content"].strip()
        if content_key and content_key not in unique_chunks_dict:
            unique_chunks_dict[content_key] = chunk_data
    
    unique_chunks_list = list(unique_chunks_dict.values())
    print(f"After deduplication: {len(unique_chunks_list)} unique chunks.")

    # Post-processing and metadata enrichment
    enriched_docs_for_weaviate = []
    processed_files_metadata_cache = {} # Cache for PDF metadata

    for chunk_data in unique_chunks_list:
        cleaned_text = clean_text(chunk_data["page_content"])
        if len(cleaned_text) < 100: # Skip very short chunks after cleaning
            continue

        source_path_str = str(chunk_data["metadata"].get("source", "unknown_source"))
        
        if source_path_str not in processed_files_metadata_cache:
            pdf_meta = extract_pdf_metadata(source_path_str)
            content_feats = extract_content_features(source_path_str)
            processed_files_metadata_cache[source_path_str] = {"pdf_meta": pdf_meta, "content_features": content_feats}
        
        cached_data = processed_files_metadata_cache[source_path_str]
        pdf_meta = cached_data["pdf_meta"]
        content_feats = cached_data["content_features"]

        # Try to find original page number if [PAGE_BREAK: X] exists in chunk
        page_from_chunk = chunk_data["metadata"].get("page", 0) # Default from Langchain doc page
        page_match = re.search(r"\[PAGE_BREAK: (\d+)\]", cleaned_text)
        if page_match:
            try: page_from_chunk = int(page_match.group(1)) -1 # Store 0-indexed
            except: pass


        doc_for_weaviate = {
            "text": cleaned_text,
            "source": source_path_str,
            "filename": os.path.basename(source_path_str),
            "folder": os.path.dirname(source_path_str),
            "page": page_from_chunk, # Best guess for page
            **pdf_meta, # Spread the PDF metadata
        }
        # Add content features, prefixing to avoid clashes
        for k, v in content_feats.items():
            if isinstance(v, dict): # like document_structure, layout_info, readability_stats
                 for sub_k, sub_v in v.items():
                    doc_for_weaviate[f"feature_{k}_{sub_k}"] = json.dumps(sub_v) if isinstance(sub_v, (list,dict)) else sub_v
            elif isinstance(v, list): # like page_features
                doc_for_weaviate[f"feature_{k}"] = json.dumps(v)
            else:
                doc_for_weaviate[f"feature_{k}"] = v
        
        # Remove any problematic None values, replace with empty string or default
        for key, value in doc_for_weaviate.items():
            if value is None:
                if "count" in key or "page" in key or "size" in key: doc_for_weaviate[key] = 0
                elif "has_" in key or "is_" in key: doc_for_weaviate[key] = False
                else: doc_for_weaviate[key] = ""
            # Ensure boolean fields are actual booleans for Weaviate if not already
            if isinstance(value, str) and key in pdf_meta and isinstance(pdf_meta[key], bool): # check original type
                if value.lower() == 'true': doc_for_weaviate[key] = True
                elif value.lower() == 'false': doc_for_weaviate[key] = False


        enriched_docs_for_weaviate.append(doc_for_weaviate)

    print(f"Prepared {len(enriched_docs_for_weaviate)} final documents for Weaviate.")
    return enriched_docs_for_weaviate


def create_weaviate_headers():
    headers = {"Content-Type": "application/json"}
    if WEAVIATE_API_KEY:
        headers["Authorization"] = f"Bearer {WEAVIATE_API_KEY}"
    return headers

def check_weaviate_connection():
    try:
        response = requests.get(f"{WEAVIATE_URL}/v1/.well-known/ready", headers=create_weaviate_headers())
        if response.status_code == 200:
            print("✅ Weaviate instance is ready.")
            return True
        else:
            print(f"❌ Weaviate instance not ready or error: {response.status_code} - {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection Error: Could not connect to Weaviate at {WEAVIATE_URL}. Is Weaviate running?")
        print(f"Details: {e}")
        return False

def discover_json_fields():
    """
    Scan the pdfs directory for JSON files and extract potential schema fields with document_ prefix.
    
    Returns:
        dict: Map of discovered field names to their data types
    """
    potential_fields = {}
    pdf_dir = "pdfs"
    
    try:
        if not os.path.exists(pdf_dir):
            print(f"PDFs directory '{pdf_dir}' not found, skipping JSON field discovery.")
            return potential_fields
            
        # Find all JSON files in the pdfs directory
        json_files = []
        for root, _, files in os.walk(pdf_dir):
            json_files.extend([os.path.join(root, f) for f in files if f.lower().endswith('.json')])
        
        print(f"Found {len(json_files)} JSON files for schema analysis")
        
        # Process each JSON file to identify fields
        for json_path in json_files:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract fields
                for key, value in data.items():
                    field_name = f"document_{key}"
                    
                    # Determine data type
                    if isinstance(value, bool):
                        field_type = "boolean"
                    elif isinstance(value, int):
                        field_type = "int"
                    elif isinstance(value, float):
                        field_type = "number"
                    elif isinstance(value, (dict, list)):
                        field_type = "text"  # Complex objects stored as JSON strings
                    else:
                        field_type = "text"  # Default to text
                    
                    potential_fields[field_name] = field_type
            except Exception as e:
                print(f"Error processing JSON file {json_path}: {e}")
                continue
                
        print(f"Discovered {len(potential_fields)} potential document fields from JSON files")
    except Exception as e:
        print(f"Error during JSON field discovery: {e}")
    
    return potential_fields

def create_collection_if_not_exists():
    if not check_weaviate_connection(): return False
    headers = create_weaviate_headers()
    collection_name = COLLECTION_NAME
    schema_url = f"{WEAVIATE_URL}/v1/schema/{collection_name}"
    
    # Discover JSON fields that should be included in the schema
    json_fields = discover_json_fields()
    
    # Schema definition with corrected data types
    schema = {
        "class": collection_name,
        "vectorizer": "none", # We provide vectors manually
        "properties": [
            {"name": "text", "dataType": ["text"]}, {"name": "source", "dataType": ["text"]},
            {"name": "filename", "dataType": ["text"]}, {"name": "folder", "dataType": ["text"]},
            {"name": "page", "dataType": ["int"]}, {"name": "title", "dataType": ["text"]},
            {"name": "author", "dataType": ["text"]}, {"name": "subject", "dataType": ["text"]},
            {"name": "keywords", "dataType": ["text"]}, # Keep as regular text (not array)
            {"name": "creation_date", "dataType": ["text"]}, # Keep as text, will be formatted properly
            {"name": "modification_date", "dataType": ["text"]}, # Keep as text, will be formatted
            {"name": "page_count", "dataType": ["int"]}, {"name": "file_size", "dataType": ["int"]},
            {"name": "has_images", "dataType": ["boolean"]}, {"name": "image_count", "dataType": ["int"]},
            {"name": "languages", "dataType": ["text"]}, # Keep as regular text (not array)
            {"name": "feature_layout_info_multi_column", "dataType": ["boolean"]},
            {"name": "feature_readability_stats_avg_sentence_length", "dataType": ["number"]},
        ]
    }
    # Add all boolean fields from default_meta as boolean properties
    default_meta_keys_for_schema = {
        "has_toc": "boolean", "has_links": "boolean", "has_forms": "boolean", 
        "has_annotations": "boolean", "is_encrypted": "boolean", 
        "is_reflowable": "boolean", "has_embedded_files": "boolean"
    }
    for prop_name, prop_type in default_meta_keys_for_schema.items():
        if not any(p["name"] == prop_name for p in schema["properties"]):
            schema["properties"].append({"name": prop_name, "dataType": [prop_type]})
    
    # Add other varying types from default_meta
    other_meta_keys = {
        "producer": "text", "pdf_version": "text", "toc": "text", "table_count":"int",
        "total_chars": "int", "total_words": "int", "encryption_method":"text",
        "permissions":"text", "embedded_file_names":"text", "embedded_files_info":"text",
        "fonts_used":"text", "cover_image":"text" # cover_image is long, consider blob if performance issue
    }
    for prop_name, prop_type in other_meta_keys.items():
         if not any(p["name"] == prop_name for p in schema["properties"]):
            schema["properties"].append({"name": prop_name, "dataType": [prop_type]})
    
    # Add discovered JSON fields
    for field_name, field_type in json_fields.items():
        if not any(p["name"] == field_name for p in schema["properties"]):
            schema["properties"].append({"name": field_name, "dataType": [field_type]})
    
    try:
        # Check if collection exists
        response = requests.get(schema_url, headers=headers)
        
        if response.status_code == 200:
            print(f"✅ Collection '{collection_name}' already exists.")
            
            # Check if schema needs updating with new fields
            current_schema = response.json()
            current_properties = {prop.get("name"): prop.get("dataType", ["text"])[0] 
                               for prop in current_schema.get("properties", [])}
            
            # Find new properties needed
            new_properties = []
            for prop in schema["properties"]:
                prop_name = prop["name"]
                if prop_name not in current_properties:
                    new_properties.append(prop)
            
            if new_properties:
                print(f"Adding {len(new_properties)} new properties to existing schema:")
                for prop in new_properties:
                    print(f"  - {prop['name']} ({prop['dataType'][0]})")
                
                # Ask user what to do
                print("\nOptions for existing collection:")
                print("1. Add new fields to schema (preserves existing data)")
                print("2. Delete collection and recreate with updated schema")
                print("3. Continue with existing schema (may lose JSON metadata)")
                
                user_choice = input("Enter your choice (1, 2, or 3): ").strip()
                
                if user_choice == "1":
                    # Update schema with new properties
                    for prop in new_properties:
                        try:
                            prop_response = requests.post(
                                f"{schema_url}/properties",
                                headers=headers,
                                json=prop
                            )
                            if prop_response.status_code == 200:
                                print(f"  ✅ Added property: {prop['name']}")
                            else:
                                print(f"  ❌ Failed to add property {prop['name']}: {prop_response.status_code}")
                        except Exception as e:
                            print(f"  ❌ Error adding property {prop['name']}: {e}")
                    
                    global CHECK_EXISTING_OBJECTS
                    CHECK_EXISTING_OBJECTS = True
                    return True
                
                elif user_choice == "2":
                    print(f"Deleting collection '{collection_name}' to recreate with updated schema...")
                    delete_response = requests.delete(schema_url, headers=headers)
                    if delete_response.status_code not in [200, 204, 404]:
                        print(f"❌ Error deleting collection: {delete_response.status_code} - {delete_response.text}")
                        return False
                    print(f"✅ Successfully deleted collection '{collection_name}'.")
                    
                    # Create new collection with full schema
                    create_response = requests.post(f"{WEAVIATE_URL}/v1/schema", headers=headers, json=schema)
                    if create_response.status_code == 200:
                        print(f"✅ Created new collection '{collection_name}' with updated schema including JSON fields")
                        global RECREATED_COLLECTION
                        RECREATED_COLLECTION = True
                        return True
                    else:
                        print(f"❌ Error creating schema: {create_response.status_code} - {create_response.text}")
                        return False
                else:
                    # Default to option 3: continue with existing schema
                    print(f"Continuing with existing schema for '{collection_name}' (some JSON metadata may not be stored)")
                    CHECK_EXISTING_OBJECTS = True
                    return True
            else:
                print("Existing schema contains all needed properties. Will add new objects only.")
                
                # Ask user what to do about existing objects
                print("\nOptions for handling objects:")
                print("1. Skip existing objects (objects with same UUID)")
                print("2. Delete collection and recreate with all objects")
                
                user_choice = input("Enter your choice (1 or 2): ").strip()
                
                if user_choice == "2":
                    print(f"Deleting collection '{collection_name}' to recreate with new objects...")
                    delete_response = requests.delete(schema_url, headers=headers)
                    if delete_response.status_code not in [200, 204, 404]:
                        print(f"❌ Error deleting collection: {delete_response.status_code} - {delete_response.text}")
                        return False
                    print(f"✅ Successfully deleted collection '{collection_name}'.")
                    
                    # Create new collection
                    create_response = requests.post(f"{WEAVIATE_URL}/v1/schema", headers=headers, json=schema)
                    if create_response.status_code == 200:
                        print(f"✅ Created new collection '{collection_name}' successfully.")
                        RECREATED_COLLECTION = True
                        return True
                    else:
                        print(f"❌ Error creating schema: {create_response.status_code} - {create_response.text}")
                        return False
                else:
                    # Default to option 1: keep collection, add new objects
                    print(f"Keeping existing collection '{collection_name}'. Will add new objects only.")
                    CHECK_EXISTING_OBJECTS = True
                    return True
        else:
            # Collection doesn't exist, create it
            print(f"Collection '{collection_name}' does not exist. Creating with schema...")
            create_response = requests.post(f"{WEAVIATE_URL}/v1/schema", headers=headers, json=schema)
            if create_response.status_code == 200:
                print(f"✅ Created collection '{collection_name}' with {len(schema['properties'])} properties including JSON fields")
                return True
            else:
                print(f"❌ Error creating schema: {create_response.status_code} - {create_response.text}")
                return False
    except requests.exceptions.RequestException as e:
        print(f"Error during collection management: {e}")
        return False

# Global flags for collection management
CHECK_EXISTING_OBJECTS = False
RECREATED_COLLECTION = False

def check_object_exists(object_id):
    """Check if an object with the given ID already exists in Weaviate."""
    if not CHECK_EXISTING_OBJECTS:
        return False  # Skip check if not needed (collection was recreated)
        
    headers = create_weaviate_headers()
    try:
        response = requests.get(f"{WEAVIATE_URL}/v1/objects/{COLLECTION_NAME}/{object_id}", headers=headers)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False  # If error checking, assume object doesn't exist

def create_weaviate_object(doc_properties, vector):
    headers = create_weaviate_headers()
    # Generate UUID based on content to aid deduplication if re-run
    unique_str_for_id = f"{doc_properties.get('source', '')}-{doc_properties.get('page', 0)}-{doc_properties.get('text', '')[:50]}"
    object_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_str_for_id))
    
    # Check if object already exists and skip if needed
    if check_object_exists(object_id):
        print(f"  Skipping existing object {object_id}")
        return "SKIPPED"  # Return special status for skipped objects
    
    # Sanitize property dictionary
    sanitized_properties = {}
    for k, v in doc_properties.items():
        # Skip empty values for date fields
        if k in ['creation_date', 'modification_date'] and (not v or v == ''):
            continue
            
        # Handle document_ fields specifically
        if k.startswith('document_'):
            if isinstance(v, (dict, list)):
                # Convert complex types to JSON string for storage
                sanitized_properties[k] = json.dumps(v)
            else:
                sanitized_properties[k] = v
            continue
            
        # Handle boolean values that might be strings
        if isinstance(v, str):
            if v.lower() == "true": 
                sanitized_properties[k] = True
            elif v.lower() == "false": 
                sanitized_properties[k] = False
            else: 
                # Make sure languages and keywords are regular strings, not arrays
                if k in ['languages', 'keywords']:
                    sanitized_properties[k] = str(v)
                else:
                    sanitized_properties[k] = v
        else:
            # Make sure languages and keywords are strings if they're lists
            if k in ['languages', 'keywords'] and isinstance(v, list):
                sanitized_properties[k] = ','.join(str(x) for x in v)
            else:
                sanitized_properties[k] = v
            
    # Make sure version attribute exists
    sanitized_properties['pdf_version'] = sanitized_properties.get('pdf_version', 'unknown')
            
    object_data = {
        "id": object_id,
        "class": COLLECTION_NAME, # Must match collection name
        "properties": sanitized_properties,
        "vector": vector
    }
    
    try:
        response = requests.post(f"{WEAVIATE_URL}/v1/objects", headers=headers, json=object_data)
        if response.status_code in [200, 201]: # 200 for OK, 201 for Created
            return True
        else:
            print(f"❌ Error creating object {object_id}: {response.status_code} - {response.text}")
            # If the error is due to property type mismatch, log details to help debugging
            if response.status_code == 422:
                try:
                    error_details = response.json()
                    print(f"  Details: {error_details}")
                    # Try to identify problematic properties
                    if 'error' in error_details and 'message' in error_details['error']:
                        error_msg = error_details['error']['message']
                        if 'property' in error_msg.lower():
                            prop_match = re.search(r'property\s+([a-zA-Z0-9_]+)', error_msg.lower())
                            if prop_match:
                                problem_prop = prop_match.group(1)
                                value = sanitized_properties.get(problem_prop, "unknown")
                                print(f"  Problem property: {problem_prop}, Value: {value}, Type: {type(value).__name__}")
                except:
                    pass  # If we can't parse the error details, just continue
            return False
    except requests.exceptions.RequestException as e:
        print(f"Error creating object post: {e}")
        return False

def enhance_text_for_embedding(text_content, doc_meta=None):
    """Enhance text with context for optimal embedding quality."""
    if not text_content.strip(): 
        return text_content
    
    # Extract context from metadata for semantic enrichment
    title = doc_meta.get("title", "") if doc_meta else ""
    author = doc_meta.get("author", "") if doc_meta else ""
    subject = doc_meta.get("subject", "") if doc_meta else ""
    keywords = doc_meta.get("keywords", "") if doc_meta else ""
    
    # Optimized instruction prefix (BGE models respond well to this format)
    instruction = "Represent this document for retrieval: "
    
    # Build enhanced context parts
    context_parts = []
    
    # Add key metadata if available (strengthens topic representation)
    if title:
        context_parts.append(f"Title: {title}")
    if subject and len(subject) > 3:
        context_parts.append(f"Subject: {subject}")
    if keywords and len(keywords) > 3:
        context_parts.append(f"Keywords: {keywords}")
    
    # Create context string with semantic delimiters
    context_str = ""
    if context_parts:
        context_str = " | ".join(context_parts) + "\n\n"
    
    # Combine instruction, context and content optimally
    enhanced_text = f"{instruction}{context_str}{text_content}"
    
    return enhanced_text

def calculate_optimal_batch_size(documents):
    """
    Calculate optimal batch size based on number of documents, average document length,
    and available system resources.
    
    Args:
        documents: List of document dictionaries to process
        
    Returns:
        int: Recommended batch size
    """
    # Base batch size calculations
    doc_count = len(documents)
    
    # Calculate average document length
    avg_length = 0
    if documents:
        total_length = sum(len(doc.get("text", "")) for doc in documents)
        avg_length = total_length / doc_count
    
    # Adjust base batch size based on document count
    if doc_count < 50:
        base_batch = 5  # Small dataset
    elif doc_count < 200:
        base_batch = 10  # Medium dataset
    else:
        base_batch = 20  # Large dataset
    
    # Adjust further based on average document length
    length_factor = 1.0
    if avg_length > 5000:
        length_factor = 0.5  # Very long documents
    elif avg_length > 2000:
        length_factor = 0.7  # Long documents
    elif avg_length < 500:
        length_factor = 1.5  # Short documents
    
    # Consider system resources - attempt to detect available memory
    # This is an approximation and may not work on all systems
    try:
        import psutil
        available_memory_gb = psutil.virtual_memory().available / (1024 * 1024 * 1024)
        memory_factor = 1.0
        
        # Adjust batch size based on available memory
        if available_memory_gb < 2:  # Less than 2GB available
            memory_factor = 0.5
        elif available_memory_gb > 8:  # More than 8GB available
            memory_factor = 1.5
            
        print(f"System has {available_memory_gb:.1f} GB available memory, using memory factor: {memory_factor}")
        
    except ImportError:
        # If psutil not available, skip memory-based adjustment
        memory_factor = 1.0
        print("psutil not available, skipping memory-based batch size adjustment")
    
    # Calculate final batch size (min 3, max 30)
    batch_size = max(3, min(30, int(base_batch * length_factor * memory_factor)))
    
    print(f"Calculated optimal batch size: {batch_size} (based on {doc_count} documents with avg length {avg_length:.1f} chars)")
    
    # Allow user to override the batch size
    try:
        user_input = input(f"Use calculated batch size {batch_size}? Enter a different number to override, or press Enter to accept: ").strip()
        if user_input and user_input.isdigit():
            user_batch_size = int(user_input)
            if user_batch_size > 0:
                print(f"Using user-specified batch size: {user_batch_size}")
                return user_batch_size
    except (KeyboardInterrupt, EOFError):
        # If user sends interrupt or EOF, just use calculated value
        pass
    
    return batch_size

def create_document_level_embedding(source_docs, doc_title):
    """Create a document-level embedding for hierarchical embeddings"""
    if not USE_HIERARCHICAL_EMBEDDINGS:
        return None
        
    # Create a representative document summary
    summary_parts = []
    
    # Add title
    if doc_title:
        summary_parts.append(f"Document: {doc_title}")
    
    # Extract headings or first sentences from pages to create a summary
    extracted_headings = []
    for doc in source_docs[:min(5, len(source_docs))]:  # Use first 5 pages max
        page_content = doc.page_content
        sentences = page_content.split('. ')[:3]  # Get first 3 sentences
        
        # Look for potential headings (short, capitalized text with no period)
        for sentence in sentences:
            if 10 < len(sentence) < 100 and not sentence.endswith('.'):
                if any(c.isupper() for c in sentence):
                    extracted_headings.append(sentence.strip())
                    break
    
    # Add extracted headings
    if extracted_headings:
        summary_parts.append("Sections: " + " | ".join(extracted_headings[:5]))
    
    # Add first page intro
    if source_docs:
        first_page = source_docs[0].page_content
        first_paragraph = first_page.split('\n\n')[0] if '\n\n' in first_page else first_page[:500]
        summary_parts.append(first_paragraph)
    
    # Combine parts and limit length
    document_summary = "\n\n".join(summary_parts)
    if len(document_summary) > 1000:
        document_summary = document_summary[:1000]
    
    # Create the embedding for the document summary
    if document_summary:
        try:
            doc_text_for_embedding = enhance_text_for_embedding(document_summary)
            doc_embedding = embedding_model.embed_documents([doc_text_for_embedding])[0]
            return doc_embedding
        except Exception as e:
            print(f"Warning: Failed to create document-level embedding: {e}")
    
    return None

def create_blended_hierarchical_embedding(chunk_embedding, doc_embedding):
    """Create hierarchical embedding by blending chunk and document embeddings"""
    if not USE_HIERARCHICAL_EMBEDDINGS or doc_embedding is None:
        return chunk_embedding
    
    try:
        # Convert to numpy arrays
        chunk_vec = np.array(chunk_embedding)
        doc_vec = np.array(doc_embedding)
        
        # Blend the embeddings with weighting
        blended_vec = (1 - PARENT_EMBEDDING_WEIGHT) * chunk_vec + PARENT_EMBEDDING_WEIGHT * doc_vec
        
        # Renormalize to unit length
        norm = np.linalg.norm(blended_vec)
        if norm > 0:
            return (blended_vec / norm).tolist()
    except Exception as e:
        print(f"Warning: Error in hierarchical embedding blend: {e}")
    
    return chunk_embedding

def main():
    print("Starting PDF vectorization process...")
    print(f"Using embedding model: {model_name}")
    print(f"Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")
    print(f"Hierarchical embeddings: {'Enabled' if USE_HIERARCHICAL_EMBEDDINGS else 'Disabled'}")

    pdfs_directory = "pdfs"
    if not os.path.isdir(pdfs_directory):
        print(f"ERROR: PDFs directory '{pdfs_directory}' not found. Please create it and add PDF files.")
        return
    if not os.listdir(pdfs_directory): # Check if directory is empty
        print(f"Warning: PDFs directory '{pdfs_directory}' is empty. No files to process.")
        return

    # Check Weaviate connection
    print("Testing Weaviate connection...")
    if not check_weaviate_connection():
        print("❌ Cannot connect to Weaviate. Please check if Weaviate is running and the URL is correct.")
        print(f"Current URL: {WEAVIATE_URL}")
        return

    # Create or check collection
    if not create_collection_if_not_exists():
        print("Failed to create or verify Weaviate collection. Aborting.")
        return

    # Load and process documents
    try:
        print("Loading and processing documents...")
        # First, load raw documents grouped by source
        loader = DirectoryLoader(pdfs_directory, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
        raw_docs = loader.load()
        
        # Group documents by source
        docs_by_source = {}
        for doc in raw_docs:
            source = doc.metadata.get("source", "unknown_source")
            docs_by_source.setdefault(source, []).append(doc)
            
        # Sort each source's documents by page number
        for source, docs in docs_by_source.items():
            docs.sort(key=lambda x: x.metadata.get("page", 0))
        
        print(f"Loaded {len(raw_docs)} pages from {len(docs_by_source)} documents")
        
        # Process each document with advanced chunking and hierarchical embeddings
        all_chunks = []
        doc_embeddings = {}  # Store document embeddings for hierarchical approach
        
        for source, source_docs in docs_by_source.items():
            print(f"Processing {os.path.basename(source)}...")
            
            # Extract metadata
            pdf_meta = extract_pdf_metadata(source)
            json_meta = extract_json_metadata(source)
            combined_meta = {**pdf_meta, **json_meta}
            combined_meta["source"] = source
            
            # Create document-level embedding if using hierarchical approach
            if USE_HIERARCHICAL_EMBEDDINGS:
                doc_title = combined_meta.get("title", os.path.basename(source))
                doc_embeddings[source] = create_document_level_embedding(source_docs, doc_title)
            
            # Combine all text with page markers
            full_text = ""
            for doc in source_docs:
                page_num = doc.metadata.get("page", 0) + 1
                full_text += f"\n\n[PAGE {page_num}]\n{doc.page_content}"
            
            # Split into semantic chunks
            chunks = text_splitter.split_text(full_text)
            
            # Process each chunk
            for chunk in chunks:
                # Clean and prepare text
                cleaned_text = clean_text(chunk)
                if not cleaned_text or len(cleaned_text) < 100:  # Skip very short chunks
                    continue
                
                # Extract page number if present
                page_match = re.search(r"\[PAGE (\d+)\]", chunk)
                page_num = int(page_match.group(1)) - 1 if page_match else 0
                
                # Prepare metadata for this chunk
                chunk_meta = combined_meta.copy()
                chunk_meta["page"] = page_num
                
                all_chunks.append({
                    "text": cleaned_text,
                    "metadata": chunk_meta,
                    "source": source
                })
        
        print(f"Created {len(all_chunks)} optimized chunks")
        
        # Calculate batch size
        batch_size = calculate_optimal_batch_size(all_chunks)
        print(f"Using batch size: {batch_size}")
        
        # Process chunks in batches
        success_count = 0
        error_count = 0
        skipped_count = 0
        hierarchical_count = 0
        
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(all_chunks)-1)//batch_size + 1} ({len(batch)} chunks)")
            
            # Prepare texts for embedding with enhanced context
            texts_to_embed = [
                enhance_text_for_embedding(chunk["text"], chunk["metadata"]) 
                for chunk in batch
            ]
            
            # Generate embeddings
            try:
                print(f"  Generating {len(texts_to_embed)} embeddings...")
                embeddings = embedding_model.embed_documents(texts_to_embed)
                
                # Store chunks with their embeddings
                for chunk_data, vector in zip(batch, embeddings):
                    source = chunk_data["source"]
                    
                    # Apply hierarchical blending if available
                    if USE_HIERARCHICAL_EMBEDDINGS and source in doc_embeddings and doc_embeddings[source] is not None:
                        original_vector = vector
                        vector = create_blended_hierarchical_embedding(vector, doc_embeddings[source])
                        if vector != original_vector:
                            hierarchical_count += 1
                    
                    # Create Weaviate object
                    result = create_weaviate_object(chunk_data["metadata"], vector)
                    
                    if result == True:
                        success_count += 1
                    elif result == "SKIPPED":
                        skipped_count += 1
                    else:
                        error_count += 1
                
                print(f"  Batch complete: {success_count}/{len(all_chunks)} stored so far")
                
            except Exception as e:
                print(f"❌ Error processing batch: {e}")
                error_count += len(batch)
        
        # Print final stats
        print("\n=== Vectorization Complete ===")
        print(f"Successfully stored: {success_count} chunks")
        print(f"Skipped (already existed): {skipped_count} chunks")
        print(f"Failed: {error_count} chunks")
        print(f"Total processed: {success_count + skipped_count + error_count} chunks")
        print(f"Using hierarchical embeddings: {USE_HIERARCHICAL_EMBEDDINGS}")
        if USE_HIERARCHICAL_EMBEDDINGS:
            print(f"Hierarchical enhancement applied: {hierarchical_count} chunks")
        
    except Exception as e:
        print(f"Error during document processing: {e}")
        return

if __name__ == "__main__":
    print("--------------------------------------------------------------------------")
    print("Please ensure you have installed all required libraries:")
    print("  pip install nltk Pillow langchain-text-splitters langchain-community sentence-transformers PyMuPDF python-dotenv requests fitz psutil")
    print("--------------------------------------------------------------------------\n")
    main()