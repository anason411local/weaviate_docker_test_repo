import os
import requests
import json
import uuid
import re
import base64
import nltk
from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import fitz  # PyMuPDF for enhanced metadata extraction
import datetime
import unicodedata
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Download NLTK resources (will only download if not already present)
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    NLTK_AVAILABLE = True
except:
    NLTK_AVAILABLE = False

# Load environment variables
load_dotenv()

# Get Weaviate credentials
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

# Check if credentials are set
if not WEAVIATE_URL or not WEAVIATE_API_KEY:
    raise ValueError("WEAVIATE_URL and WEAVIATE_API_KEY must be set in .env file")

# Ensure URL has proper scheme
if WEAVIATE_URL and not WEAVIATE_URL.startswith(("http://", "https://")):
    WEAVIATE_URL = f"https://{WEAVIATE_URL}"
    print(f"Added https:// prefix to Weaviate URL: {WEAVIATE_URL}")

# Define the embedding model configuration
model_name = "BAAI/bge-m3"
model_kwargs = {
    'device': 'cpu',
    'trust_remote_code': True
}
encode_kwargs = {
    'normalize_embeddings': True,
    'batch_size': 8,
    'show_progress_bar': True
}
embedding_model = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    cache_folder=None
)

# Text splitting parameters - adjusted for better semantic coherence
CHUNK_SIZE = 1000  # Reduced from 2500 for more focused chunks
CHUNK_OVERLAP = 300  # Increased overlap percentage for better context preservation

def clean_text(text):
    """Clean and normalize text to improve embedding quality."""
    if not text:
        return ""
        
    # Remove excessive whitespace
    cleaned = re.sub(r'\s+', ' ', text).strip()
    
    # Remove PDF artifacts and noise
    artifacts = [
        r'(\n\s*\d+\s*\n)',          # Page numbers
        r'^\s*\d+\s*$',              # Standalone numbers
        r'©.*?reserved\.?',          # Copyright notices
        r'(Page|PAGE)(\s+\d+\s+of\s+\d+)', # Page X of Y headers
        r'(http|https|www)\S+\s',    # URLs (better handled separately)
        r'\[\s*\d+\s*\]',            # Citation markers like [1] [2]
        r'(^|[^a-zA-Z0-9])\d{5,}([^a-zA-Z0-9]|$)', # Long numbers (likely noise)
        r'\\[a-zA-Z]+\{.*?\}',       # LaTeX commands
        r'</?[a-z]+>',               # Simple HTML tags
    ]
    
    for pattern in artifacts:
        cleaned = re.sub(pattern, ' ', cleaned, flags=re.MULTILINE)
    
    # Fix hyphenated words that span lines (common in PDFs)
    cleaned = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', cleaned)
    
    # Fix spacing around punctuation
    cleaned = re.sub(r'\s+([.,;:!?)])', r'\1', cleaned)
    cleaned = re.sub(r'([(])\s+', r'\1', cleaned)
    
    # Standardize quotes and apostrophes
    cleaned = re.sub(r'["""]', '"', cleaned)
    cleaned = re.sub(r"['']", "'", cleaned)
    
    # Normalize unicode characters
    cleaned = unicodedata.normalize('NFKC', cleaned)
    
    # Fix common OCR/PDF extraction errors
    ocr_fixes = [
        (r'l\b', 'i'),           # Replace lone 'l' with 'i'
        (r'\bII\b', 'H'),        # Replace 'II' with 'H'
        (r'\b0\b', 'O'),         # Replace '0' with 'O'
        (r'rn\b', 'm'),          # Replace 'rn' with 'm'
        (r'rnm\b', 'mm'),        # Replace 'rnm' with 'mm'
        (r'\blJ\b', 'U'),        # Replace 'lJ' with 'U'
        (r'\bl\b', '1'),         # Replace standalone 'l' with '1'
        (r'\blo\b', '10'),       # Replace 'lo' with '10'
        (r'\ba\s+(\w)', r'a \1'),  # Fix spacing after 'a'
    ]
    
    for old, new in ocr_fixes:
        cleaned = re.sub(old, new, cleaned)
    
    # Clean URLs and email addresses for better parsing
    cleaned = re.sub(r'(https?:\/\/[^\s]+)', lambda m: m.group(1).replace('.', ' dot '), cleaned)
    
    # Process email addresses
    pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    def email_replacer(match):
        email = match.group(0)
        return email.replace('@', ' at ').replace('.', ' dot ')
    cleaned = re.sub(pattern, email_replacer, cleaned)
    
    # Normalize sentences if NLTK is available (ensures proper sentence structure)
    if NLTK_AVAILABLE:
        try:
            sentences = nltk.sent_tokenize(cleaned)
            # Fix sentence boundaries
            for i in range(len(sentences)):
                # Ensure sentences start with capital letter
                if sentences[i] and sentences[i][0].islower():
                    sentences[i] = sentences[i][0].upper() + sentences[i][1:]
                # Ensure sentences end with proper punctuation
                if sentences[i] and sentences[i][-1] not in ['.', '!', '?']:
                    sentences[i] += '.'
            cleaned = ' '.join(sentences)
        except:
            pass
    
    # Normalize whitespace again after all transformations
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            
    # Ensure the text is not too short
    if len(cleaned) < 50:  # Increased from 20 to ensure more meaningful chunks
        return ""
        
    return cleaned

def extract_json_metadata(pdf_path):
    """Extract metadata from a JSON file associated with a PDF."""
    try:
        # Get the PDF filename without extension
        pdf_base_name = os.path.basename(pdf_path)
        pdf_name_no_ext = os.path.splitext(pdf_base_name)[0]
        pdf_dir = os.path.dirname(pdf_path)
        
        # First try exact name match (replacing .pdf with .json)
        json_path = str(pdf_path).replace('.pdf', '.json')
        
        # If exact match doesn't exist, search for any JSON file that starts with the PDF name
        if not os.path.exists(json_path):
            # Look in the same directory as the PDF
            json_files = [f for f in os.listdir(pdf_dir) if f.endswith('.json')]
            matching_json = None
            
            # Try to find any JSON file that starts with the PDF name
            for json_file in json_files:
                json_name_no_ext = os.path.splitext(json_file)[0]
                if json_name_no_ext.startswith(pdf_name_no_ext) or pdf_name_no_ext in json_name_no_ext:
                    matching_json = json_file
                    break
            
            if matching_json:
                json_path = os.path.join(pdf_dir, matching_json)
                print(f"Found matching JSON: {json_path} for PDF: {pdf_path}")
            else:
                return {}
        
        # Check if the JSON file exists
        if not os.path.exists(json_path):
            return {}
        
        # Read and parse the JSON file
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # Remove the 'id' field if it exists
        if 'id' in json_data:
            del json_data['id']
            
        # Convert complex objects to JSON strings
        processed_data = {}
        for key, value in json_data.items():
            if isinstance(value, (dict, list)):
                processed_data[f"document_{key}"] = json.dumps(value)
            else:
                processed_data[f"document_{key}"] = value
        
        return processed_data
    except Exception as e:
        print(f"Error extracting JSON metadata from {pdf_path}: {e}")
        return {}

def extract_pdf_metadata(pdf_path):
    """Extract detailed metadata from a PDF file using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        metadata = {
            # Basic metadata
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "keywords": doc.metadata.get("keywords", ""),
            "creator": doc.metadata.get("creator", ""),
            "producer": doc.metadata.get("producer", ""),
            "creation_date": doc.metadata.get("creationDate", ""),
            "modification_date": doc.metadata.get("modDate", ""),
            "page_count": doc.page_count,
            "file_size": os.path.getsize(pdf_path),
            "pdf_version": f"{doc.version}",
            
            # Advanced structure analysis
            "has_toc": len(doc.get_toc()) > 0,
            "has_links": False,  # Will be set during page analysis
            "has_forms": bool(doc.is_form_pdf),
            "has_annotations": False,  # Will be set during page analysis
            
            # Content type analysis
            "has_images": False,  # Will be set during page analysis
            "image_count": 0,
            "has_tables": False,  # Will be determined during page analysis
            "table_count": 0,
            
            # Text statistics
            "total_chars": 0,
            "total_words": 0,
            "languages": "",  # Will attempt to detect languages
            
            # Security and encryption
            "is_encrypted": doc.is_encrypted,
            "encryption_method": doc.encryption_method if doc.is_encrypted else "",
            "permissions": json.dumps(doc.permissions) if hasattr(doc, 'permissions') else "",
            
            # Format details
            "is_reflowable": bool(doc.is_reflowable),
            "has_embedded_files": len(doc.embfile_names()) > 0,
            "embedded_file_names": json.dumps(doc.embfile_names()),
        }
        
        # Extract table of contents if available
        toc = doc.get_toc()
        if toc:
            # Convert TOC to a simpler format
            metadata["toc"] = json.dumps([{"level": level, "title": title, "page": page} 
                                          for level, title, page in toc])
        else:
            metadata["toc"] = ""
            
        # Clean up date formats if they exist
        for date_field in ["creation_date", "modification_date"]:
            if metadata[date_field]:
                try:
                    # Convert PDF date format to ISO format
                    date_str = metadata[date_field]
                    if date_str.startswith("D:"):
                        # PDF date format: D:YYYYMMDDHHmmSSOHH'mm'
                        date_str = date_str[2:]  # Remove D:
                        year = int(date_str[0:4])
                        month = int(date_str[4:6])
                        day = int(date_str[6:8])
                        hour = int(date_str[8:10]) if len(date_str) > 8 else 0
                        minute = int(date_str[10:12]) if len(date_str) > 10 else 0
                        second = int(date_str[12:14]) if len(date_str) > 12 else 0
                        date_obj = datetime.datetime(year, month, day, hour, minute, second)
                        metadata[date_field] = date_obj.isoformat()
                except:
                    # If parsing fails, keep the original value
                    pass
        
        # Analyze each page for more detailed metadata
        fonts_used = set()
        for page_num, page in enumerate(doc):
            # Count text characters and words
            text = page.get_text()
            metadata["total_chars"] += len(text)
            words = re.findall(r'\w+', text)
            metadata["total_words"] += len(words)
            
            # Check for images
            image_list = page.get_images()
            if image_list:
                metadata["has_images"] = True
                metadata["image_count"] += len(image_list)
            
            # Check for links
            links = page.get_links()
            if links:
                metadata["has_links"] = True
            
            # Check for annotations
            annots = page.annots()
            if annots:
                metadata["has_annotations"] = True
            
            # Extract font information
            for block in page.get_text("dict")["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        if "spans" in line:
                            for span in line["spans"]:
                                if "font" in span:
                                    fonts_used.add(span["font"])
            
            # Table detection (simple heuristic)
            # Look for grid-like structures with whitespace alignment
            if "   " in text and "|" in text or "\t" in text:
                metadata["has_tables"] = True
                # Count tables (very approximate)
                metadata["table_count"] += text.count("\n\n") // 3  # Rough estimate

        # Add font information
        metadata["fonts_used"] = json.dumps(list(fonts_used))
        
        # Simple language detection based on common words (very basic)
        try:
            all_text = "".join([page.get_text() for page in doc])
            # Very simple language detection
            languages = []
            # English markers
            if re.search(r'\b(the|and|is|in|to|of)\b', all_text, re.IGNORECASE):
                languages.append("en")
            # Spanish markers
            if re.search(r'\b(el|la|los|las|y|en|de)\b', all_text, re.IGNORECASE):
                languages.append("es")
            # French markers
            if re.search(r'\b(le|la|les|et|en|de)\b', all_text, re.IGNORECASE):
                languages.append("fr")
            # German markers
            if re.search(r'\b(der|die|das|und|in|von)\b', all_text, re.IGNORECASE):
                languages.append("de")
            
            metadata["languages"] = ",".join(languages) if languages else "unknown"
        except:
            metadata["languages"] = "unknown"

        # Extract embedded files if available
        if metadata["has_embedded_files"]:
            embedded_files = []
            for name in doc.embfile_names():
                embedded_files.append({
                    "name": name,
                    "size": len(doc.extract_embedfile(name)[1])
                })
            metadata["embedded_files_info"] = json.dumps(embedded_files)
        else:
            metadata["embedded_files_info"] = ""
        
        # Extract cover image if available
        if PIL_AVAILABLE and metadata["has_images"] and doc.page_count > 0:
            try:
                first_page = doc[0]
                image_list = first_page.get_images(full=True)
                if image_list:
                    # Get the largest image on the first page (likely the cover)
                    largest_img = None
                    max_size = 0
                    for img_index, img in enumerate(image_list):
                        xref = img[0]
                        if xref:
                            base_image = doc.extract_image(xref)
                            if base_image:
                                img_bytes = base_image["image"]
                                img_size = len(img_bytes)
                                if img_size > max_size:
                                    max_size = img_size
                                    largest_img = base_image
                    
                    if largest_img:
                        # Store the image data
                        img_data = largest_img["image"]
                        img_ext = largest_img["ext"]
                        
                        # Create a thumbnail version to save space
                        img = Image.open(BytesIO(img_data))
                        img.thumbnail((300, 300))
                        buffer = BytesIO()
                        img.save(buffer, format="JPEG")
                        img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        metadata["cover_image"] = f"data:image/jpeg;base64,{img_b64}"
            except Exception as e:
                print(f"Error extracting cover image: {e}")
                metadata["cover_image"] = ""
        else:
            metadata["cover_image"] = ""
        
        # Get associated JSON metadata
        json_metadata = extract_json_metadata(pdf_path)
        
        # Merge PDF metadata with JSON metadata
        metadata.update(json_metadata)
        
        doc.close()
        return metadata
    except Exception as e:
        print(f"Error extracting metadata from {pdf_path}: {e}")
        
        # Get associated JSON metadata even if PDF metadata extraction fails
        json_metadata = extract_json_metadata(pdf_path)
        
        # Create default metadata and merge with JSON metadata
        default_metadata = {
            "title": "",
            "author": "",
            "subject": "",
            "keywords": "",
            "creator": "",
            "producer": "",
            "creation_date": "",
            "modification_date": "",
            "page_count": 0,
            "file_size": 0,
            "pdf_version": "",
            "toc": "",
            "has_toc": False,
            "has_links": False,
            "has_forms": False,
            "has_annotations": False,
            "has_images": False,
            "image_count": 0,
            "has_tables": False,
            "table_count": 0,
            "total_chars": 0, 
            "total_words": 0,
            "languages": "unknown",
            "is_encrypted": False,
            "encryption_method": "",
            "permissions": "",
            "is_reflowable": False,
            "has_embedded_files": False,
            "embedded_file_names": "",
            "embedded_files_info": "",
            "fonts_used": "[]",
            "cover_image": ""
        }
        default_metadata.update(json_metadata)
        return default_metadata

def extract_content_features(pdf_path):
    """Extract detailed content features from a PDF."""
    try:
        doc = fitz.open(pdf_path)
        
        content_features = {
            "page_features": [],
            "document_structure": {
                "heading_levels": [],
                "section_counts": 0,
                "avg_section_length": 0,
            },
            "layout_info": {
                "multi_column": False,
                "has_header_footer": False,
                "page_dimensions": []
            },
            "readability_stats": {
                "avg_sentence_length": 0,
                "avg_word_length": 0,
                "complex_word_percentage": 0,
            }
        }
        
        # Extract content features for each page
        total_blocks = 0
        total_text_blocks = 0
        total_sentences = 0
        total_words = 0
        total_chars = 0
        complex_words = 0  # Words with 3+ syllables (approximated by length > 8)
        
        for page_num, page in enumerate(doc):
            # Get page dimensions
            content_features["layout_info"]["page_dimensions"].append({
                "width": page.rect.width,
                "height": page.rect.height
            })
            
            # Get page text and analyze
            text = page.get_text()
            
            # Extract layout information
            blocks = page.get_text("blocks")
            total_blocks += len(blocks)
            
            # Check for multi-column layout
            if len(blocks) > 3:
                x_positions = [block[0] for block in blocks if len(block) > 4]
                if len(x_positions) > 3:
                    # If x positions cluster in different areas, likely multi-column
                    left_count = sum(1 for x in x_positions if x < page.rect.width / 3)
                    middle_count = sum(1 for x in x_positions if page.rect.width / 3 <= x < 2 * page.rect.width / 3)
                    right_count = sum(1 for x in x_positions if x >= 2 * page.rect.width / 3)
                    
                    if min(left_count, middle_count, right_count) > 2:
                        content_features["layout_info"]["multi_column"] = True
            
            # Check for headers and footers
            for block in blocks:
                if isinstance(block, tuple) and len(block) > 4:
                    # Check if any text blocks appear at the very top or bottom of pages consistently
                    if block[1] < 50 or block[3] > page.rect.height - 50:
                        content_features["layout_info"]["has_header_footer"] = True
            
            # Count text blocks
            text_blocks = [b for b in blocks if isinstance(b, tuple) and len(b) > 4 and b[4].strip()]
            total_text_blocks += len(text_blocks)
            
            # Analyze readability
            sentences = re.split(r'[.!?]', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            total_sentences += len(sentences)
            
            words = re.findall(r'\b\w+\b', text.lower())
            total_words += len(words)
            
            for word in words:
                total_chars += len(word)
                if len(word) > 8:  # Approximation for complex words
                    complex_words += 1
            
            # Extract potential heading structure
            dict_text = page.get_text("dict")
            for block in dict_text.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        font_size = span.get("size", 0)
                        text = span.get("text", "").strip()
                        
                        # Headings are typically in larger font and short
                        if text and font_size > 12 and len(text) < 100:
                            content_features["document_structure"]["heading_levels"].append({
                                "text": text,
                                "font_size": font_size,
                                "page": page_num + 1
                            })
            
            # Extract page-specific features
            page_feature = {
                "page_number": page_num + 1,
                "text_blocks": len(text_blocks),
                "word_count": len(words),
                "char_count": len(text),
                "image_count": len(page.get_images())
            }
            content_features["page_features"].append(page_feature)
        
        # Calculate aggregate statistics
        if total_sentences > 0:
            content_features["readability_stats"]["avg_sentence_length"] = total_words / total_sentences
        
        if total_words > 0:
            content_features["readability_stats"]["avg_word_length"] = total_chars / total_words
            content_features["readability_stats"]["complex_word_percentage"] = (complex_words / total_words) * 100
        
        # Identify document sections (approximate)
        headings = sorted(content_features["document_structure"]["heading_levels"], 
                          key=lambda x: (x["page"], -x["font_size"]))
        
        if headings:
            # Group headings by size to determine hierarchy
            font_sizes = sorted(set(h["font_size"] for h in headings), reverse=True)
            
            # Consider the top 3 largest sizes as headings
            heading_sizes = font_sizes[:min(3, len(font_sizes))]
            
            # Count sections (level 1 headings)
            if heading_sizes:
                main_headings = [h for h in headings if h["font_size"] == heading_sizes[0]]
                content_features["document_structure"]["section_counts"] = len(main_headings)
                
                if len(main_headings) > 1:
                    # Calculate average section length in pages
                    section_lengths = []
                    for i in range(len(main_headings) - 1):
                        section_lengths.append(main_headings[i + 1]["page"] - main_headings[i]["page"])
                    section_lengths.append(doc.page_count - main_headings[-1]["page"] + 1)
                    content_features["document_structure"]["avg_section_length"] = sum(section_lengths) / len(section_lengths)
        
        # Clean up heading levels to not include all text
        heading_summary = []
        for h in headings[:20]:  # Limit to first 20 headings
            heading_summary.append({
                "text": h["text"][:100],
                "level": font_sizes.index(h["font_size"]) + 1 if h["font_size"] in font_sizes else 9,
                "page": h["page"]
            })
        content_features["document_structure"]["heading_levels"] = heading_summary
        
        doc.close()
        return content_features
    except Exception as e:
        print(f"Error extracting content features from {pdf_path}: {e}")
        return {
            "page_features": [],
            "document_structure": {
                "heading_levels": [],
                "section_counts": 0,
                "avg_section_length": 0,
            },
            "layout_info": {
                "multi_column": False,
                "has_header_footer": False,
                "page_dimensions": []
            },
            "readability_stats": {
                "avg_sentence_length": 0,
                "avg_word_length": 0,
                "complex_word_percentage": 0,
            }
        }

def load_and_split_pdfs(directory_path: str):
    """Load PDFs from a directory, split them, and prepare for embedding."""
    print(f"Loading PDFs from {directory_path}...")
    
    # Set up document loader
    loader = DirectoryLoader(
        directory_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    
    # Load documents
    documents = loader.load()
    print(f"Loaded {len(documents)} document pages")
    
    # Organize documents by source to enable better chunking
    docs_by_source = {}
    for doc in documents:
        source = doc.metadata.get("source", "")
        if source not in docs_by_source:
            docs_by_source[source] = []
        docs_by_source[source].append(doc)
    
    # Process documents by source to maintain document-level context
    split_docs = []
    
    # Define semantic splitters with intelligent boundaries
    semantic_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Reduced for more focused chunks
        chunk_overlap=250,  # Balanced overlap
        length_function=len,
        separators=[
            # Semantic boundaries in descending priority
            "\n## ", "\n### ", "\n#### ",  # Headers
            "\n\n",                        # Paragraphs
            "\n",                          # Line breaks
            ". ",                          # Sentences
            "! ", "? ",                    # Exclamations/questions
            ";", ":",                      # Semicolons/colons
            ",",                           # Commas
            " ",                           # Words
            ""                             # Characters
        ]
    )
    
    # Define header splitter
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "header1"),
            ("##", "header2"),
            ("###", "header3"),
        ]
    )
    
    print("Applying multi-stage semantic chunking...")
    for source, source_docs in docs_by_source.items():
        # Sort pages by page number
        source_docs.sort(key=lambda x: x.metadata.get("page", 0))
        
        # Combine all pages from the same document for better chunking
        combined_text = ""
        for doc in source_docs:
            page_num = doc.metadata.get("page", 0)
            # Add page header to help with context
            page_text = f"\n\n## Page {page_num + 1}\n\n{doc.page_content}"
            combined_text += page_text
        
        # Create a single document with the combined text
        combined_doc = source_docs[0].model_copy()  # Use model_copy instead of copy
        combined_doc.page_content = combined_text
        
        # First detect section boundaries
        try:
            # Try header-based splitting first for better semantic chunks
            header_split_docs = header_splitter.split_text(combined_text)
            
            # Convert to LangChain document format
            header_docs = []
            for header_doc in header_split_docs:
                doc = combined_doc.model_copy()  # Use model_copy instead of copy
                doc.page_content = header_doc.get("text", "")
                # Add header metadata
                for header_key, header_value in header_doc.items():
                    if header_key != "text":
                        doc.metadata[header_key] = header_value
                header_docs.append(doc)
            
            if header_docs:
                # Apply character-based splitting if chunks are still too large
                large_chunks = [doc for doc in header_docs if len(doc.page_content) > 1000]
                small_chunks = [doc for doc in header_docs if len(doc.page_content) <= 1000]
                
                if large_chunks:
                    # Further split large chunks with semantic boundaries
                    additional_splits = semantic_splitter.split_documents(large_chunks)
                    split_docs.extend(additional_splits)
                
                # Add smaller chunks directly
                split_docs.extend(small_chunks)
            else:
                # Fall back to semantic splitting if no headers found
                raise ValueError("No header-based splits found")
        except Exception as e:
            # Fall back to semantic splitting
            print(f"Using semantic splitting for {source}: {str(e)}")
            char_split_docs = semantic_splitter.split_text(combined_text)
            
            # Convert to LangChain document format
            for text_chunk in char_split_docs:
                doc = combined_doc.model_copy()  # Use model_copy instead of copy
                doc.page_content = text_chunk
                split_docs.append(doc)
    
    print(f"Split into {len(split_docs)} semantic chunks")
    
    # Deduplicate nearly identical chunks
    print("Deduplicating chunks...")
    unique_chunks = []
    content_hashes = set()
    
    for doc in split_docs:
        # Create a simplified hash of the content for deduplication
        content = doc.page_content.strip()
        if not content:
            continue
            
        # Create a hash based on first and last 50 chars plus length
        content_hash = f"{content[:50]}{len(content)}{content[-50:]}"
        
        if content_hash not in content_hashes:
            content_hashes.add(content_hash)
            unique_chunks.append(doc)
    
    print(f"After deduplication: {len(unique_chunks)} unique chunks")
    
    # Apply post-processing to ensure high-quality chunks
    enhanced_docs = []
    for doc in unique_chunks:
        # Skip empty documents
        if not doc.page_content.strip():
            continue
            
        # Clean and normalize text
        cleaned_text = clean_text(doc.page_content)
        
        # Skip if cleaned text is too short
        if len(cleaned_text) < 100:
            continue
            
        # Update document with cleaned text
        doc.page_content = cleaned_text
        enhanced_docs.append(doc)
    
    print(f"After cleaning: {len(enhanced_docs)} quality chunks")
    
    # Track processed files to avoid extracting metadata multiple times
    processed_files = {}
    
    # Process document chunks
    processed_docs = []
    for doc in enhanced_docs:
        source_path = Path(doc.metadata.get("source", ""))
        
        # Extract metadata only once per file
        if str(source_path) not in processed_files:
            pdf_metadata = extract_pdf_metadata(str(source_path))
            content_features = extract_content_features(str(source_path))
            processed_files[str(source_path)] = {
                "metadata": pdf_metadata,
                "content_features": content_features
            }
        else:
            pdf_metadata = processed_files[str(source_path)]["metadata"]
            content_features = processed_files[str(source_path)]["content_features"]
        
        # Get page-specific content features if available
        page_num = doc.metadata.get("page", 0)
        page_features = None
        for pf in content_features["page_features"]:
            if pf["page_number"] == page_num + 1:  # Convert 0-indexed to 1-indexed
                page_features = pf
                break
        
        if not page_features and content_features["page_features"]:
            # Use default if specific page not found
            page_features = content_features["page_features"][0]
        
        # Create processed document with enhanced metadata
        processed_doc = {
            # Basic content
            "text": doc.page_content,
            "source": str(source_path),
            "filename": source_path.name,
            "folder": str(source_path.parent),
            "page": doc.metadata.get("page", 0),
            
            # Basic metadata
            "title": pdf_metadata["title"],
            "author": pdf_metadata["author"],
            "subject": pdf_metadata["subject"],
            "keywords": pdf_metadata["keywords"],
            "creator": pdf_metadata["creator"],
            "producer": pdf_metadata["producer"],
            "creation_date": pdf_metadata["creation_date"],
            "modification_date": pdf_metadata["modification_date"],
            "page_count": pdf_metadata["page_count"],
            "file_size": pdf_metadata["file_size"],
            "pdf_version": pdf_metadata["pdf_version"],
            "toc": pdf_metadata["toc"],
            
            # Enhanced metadata
            "has_toc": pdf_metadata["has_toc"],
            "has_links": pdf_metadata["has_links"],
            "has_forms": pdf_metadata["has_forms"],
            "has_annotations": pdf_metadata["has_annotations"],
            "has_images": pdf_metadata["has_images"],
            "image_count": pdf_metadata["image_count"],
            "has_tables": pdf_metadata["has_tables"],
            "table_count": pdf_metadata["table_count"],
            "total_chars": pdf_metadata["total_chars"],
            "total_words": pdf_metadata["total_words"],
            "languages": pdf_metadata["languages"],
            "is_encrypted": pdf_metadata["is_encrypted"],
            "encryption_method": pdf_metadata["encryption_method"],
            "permissions": pdf_metadata["permissions"],
            "is_reflowable": pdf_metadata["is_reflowable"],
            "has_embedded_files": pdf_metadata["has_embedded_files"],
            "embedded_file_names": pdf_metadata["embedded_file_names"],
            "embedded_files_info": pdf_metadata["embedded_files_info"],
            "fonts_used": pdf_metadata["fonts_used"],
            "cover_image": pdf_metadata["cover_image"],
            
            # Content structure features
            "document_structure": json.dumps(content_features["document_structure"]),
            "layout_multi_column": content_features["layout_info"]["multi_column"],
            "layout_has_header_footer": content_features["layout_info"]["has_header_footer"],
            "readability_avg_sentence_length": content_features["readability_stats"]["avg_sentence_length"],
            "readability_avg_word_length": content_features["readability_stats"]["avg_word_length"],
            "readability_complex_word_percentage": content_features["readability_stats"]["complex_word_percentage"],
        }
        
        # Add page-specific features if available
        if page_features:
            processed_doc.update({
                "page_text_blocks": page_features.get("text_blocks", 0),
                "page_word_count": page_features.get("word_count", 0),
                "page_char_count": page_features.get("char_count", 0),
                "page_image_count": page_features.get("image_count", 0),
            })
        
        # Add document metadata fields
        for key, value in {k: v for k, v in pdf_metadata.items() if k.startswith("document_")}.items():
            processed_doc[key] = value
        
        processed_docs.append(processed_doc)
    
    return processed_docs

def create_collection():
    """Create a collection with advanced schema directly via REST API."""
    headers = {
        "Authorization": f"Bearer {WEAVIATE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Check if collection exists first
    try:
        response = requests.get(f"{WEAVIATE_URL}/v1/schema/PDFDocuments", headers=headers)
        if response.status_code == 200:
            # Collection exists, don't delete it - just return True to continue adding documents
            print("✅ PDFDocuments collection already exists - will add new documents to existing collection")
            return True
    except Exception as e:
        print(f"Error checking if collection exists: {e}")
        # Continue anyway as the collection might not exist
    
    # Define an enhanced schema with more metadata properties
    schema = {
        "class": "PDFDocuments",
        "vectorizer": "none",  # No automatic vectorization
        "properties": [
            # Basic content properties
            {"name": "text", "dataType": ["text"]},
            {"name": "source", "dataType": ["text"]},
            {"name": "filename", "dataType": ["text"]},
            {"name": "folder", "dataType": ["text"]},
            {"name": "page", "dataType": ["int"]},
            
            # Basic metadata properties
            {"name": "title", "dataType": ["text"]},
            {"name": "author", "dataType": ["text"]},
            {"name": "subject", "dataType": ["text"]},
            {"name": "keywords", "dataType": ["text"]},
            {"name": "creator", "dataType": ["text"]},
            {"name": "producer", "dataType": ["text"]},
            {"name": "creation_date", "dataType": ["text"]},
            {"name": "modification_date", "dataType": ["text"]},
            {"name": "page_count", "dataType": ["int"]},
            {"name": "file_size", "dataType": ["int"]},
            {"name": "pdf_version", "dataType": ["text"]},
            {"name": "toc", "dataType": ["text"]},
            
            # Enhanced metadata properties
            {"name": "has_toc", "dataType": ["boolean"]},
            {"name": "has_links", "dataType": ["boolean"]},
            {"name": "has_forms", "dataType": ["boolean"]},
            {"name": "has_annotations", "dataType": ["boolean"]},
            {"name": "has_images", "dataType": ["boolean"]},
            {"name": "image_count", "dataType": ["int"]},
            {"name": "has_tables", "dataType": ["boolean"]},
            {"name": "table_count", "dataType": ["int"]},
            {"name": "total_chars", "dataType": ["int"]},
            {"name": "total_words", "dataType": ["int"]},
            {"name": "languages", "dataType": ["text"]},
            {"name": "is_encrypted", "dataType": ["boolean"]},
            {"name": "encryption_method", "dataType": ["text"]},
            {"name": "permissions", "dataType": ["text"]},
            {"name": "is_reflowable", "dataType": ["boolean"]},
            {"name": "has_embedded_files", "dataType": ["boolean"]},
            {"name": "embedded_file_names", "dataType": ["text"]},
            {"name": "embedded_files_info", "dataType": ["text"]},
            {"name": "fonts_used", "dataType": ["text"]},
            {"name": "cover_image", "dataType": ["text"]},
            
            # Content structure features
            {"name": "document_structure", "dataType": ["text"]},
            {"name": "layout_multi_column", "dataType": ["boolean"]},
            {"name": "layout_has_header_footer", "dataType": ["boolean"]},
            {"name": "readability_avg_sentence_length", "dataType": ["number"]},
            {"name": "readability_avg_word_length", "dataType": ["number"]},
            {"name": "readability_complex_word_percentage", "dataType": ["number"]},
            
            # Page-specific features
            {"name": "page_text_blocks", "dataType": ["int"]},
            {"name": "page_word_count", "dataType": ["int"]},
            {"name": "page_char_count", "dataType": ["int"]},
            {"name": "page_image_count", "dataType": ["int"]},
            
            # Document metadata properties
            {"name": "document_name", "dataType": ["text"]},
            {"name": "document_mimeType", "dataType": ["text"]},
            {"name": "document_createdTime", "dataType": ["text"]},
            {"name": "document_modifiedTime", "dataType": ["text"]},
            {"name": "document_owners", "dataType": ["text"]},
            {"name": "document_lastModifyingUser", "dataType": ["text"]},
            {"name": "document_size", "dataType": ["text"]},
        ]
    }
    
    # Create schema
    try:
        response = requests.post(
            f"{WEAVIATE_URL}/v1/schema",
            headers=headers,
            json=schema
        )
        
        if response.status_code == 200:
            print("✅ Created PDFDocuments collection with comprehensive metadata and content features")
            return True
        else:
            print(f"Error creating schema: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"Error creating schema: {e}")
        return False

def create_object(doc, vector):
    """Create a single object in the database."""
    headers = {
        "Authorization": f"Bearer {WEAVIATE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Add a UUID based on content to avoid duplicates
    object_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc['filename']}-{doc['page']}-{doc['text'][:100]}"))
    
    # Create object with all metadata
    object_data = {
        "id": object_id,
        "class": "PDFDocuments",
        "properties": doc,
        "vector": vector
    }
    
    try:
        response = requests.post(
            f"{WEAVIATE_URL}/v1/objects",
            headers=headers,
            json=object_data
        )
        
        if response.status_code in [200, 201]:
            return True
        else:
            print(f"Error creating object: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"Error creating object: {e}")
        return False

def main():
    """Main function to process PDFs and store them in Weaviate."""
    print("Starting PDF vectorization process...")
    print(f"Using model: {model_name}")
    print(f"Chunk size: 800, Overlap: 250")  # Updated values
    
    # Create collection
    if not create_collection():
        print("Failed to create collection, aborting")
        return
    
    # Load and split PDFs
    documents = load_and_split_pdfs("pdfs")
    print(f"Processing {len(documents)} document chunks...")
    
    # Process in small batches
    batch_size = 5  # Reduced batch size due to larger document objects
    success_count = 0
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:min(i+batch_size, len(documents))]
        print(f"Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} ({len(batch)} documents)")
        
        # Get embeddings with content-aware prompting
        texts = []
        for doc in batch:
            # Collect metadata for enhanced embeddings
            metadata = {
                "title": doc["title"] if "title" in doc else "",
                "subject": doc["subject"] if "subject" in doc else "",
                "keywords": doc["keywords"] if "keywords" in doc else "",
                "page": doc["page"] if "page" in doc else None
            }
            
            # Enhance embedding quality with structured prompt
            text_to_embed = enhance_text_for_embedding(doc["text"], metadata)
            texts.append(text_to_embed)
        
        try:
            # Generate embeddings with enhanced prompting
            print(f"  Generating embeddings for batch with instruction-based prompting...")
            embeddings = embedding_model.embed_documents(texts)
            
            # Create objects
            for j, (doc, embedding) in enumerate(zip(batch, embeddings)):
                if create_object(doc, embedding):
                    success_count += 1
                    
                if (j+1) % 2 == 0:  # Changed from 5 to 2 due to more complex objects
                    print(f"  Processed {j+1}/{len(batch)} documents in current batch")
                    
        except Exception as e:
            print(f"Error processing batch: {e}")
    
    print(f"Completed! Successfully imported {success_count}/{len(documents)} documents with comprehensive metadata and content features")

def enhance_text_for_embedding(text, doc_metadata=None):
    """Enhance text with structural prompts and metadata to improve embedding quality.
    
    This function applies embedding enhancement using the "instruction-aware" format
    that many embedding models (including BGE-M3) respond well to.
    """
    # Skip if text is empty
    if not text.strip():
        return text
    
    # Extract relevant metadata if available
    title = ""
    subject = ""
    keywords = ""
    page_num = None
    
    if doc_metadata:
        title = doc_metadata.get("title", "")
        subject = doc_metadata.get("subject", "")
        keywords = doc_metadata.get("keywords", "")
        page_num = doc_metadata.get("page")
    
    # Determine document context from available metadata
    context_parts = []
    if title:
        context_parts.append(f"Title: {title}")
    if subject:
        context_parts.append(f"Subject: {subject}")
    if keywords:
        context_parts.append(f"Keywords: {keywords}")
    if page_num is not None:
        context_parts.append(f"Page: {page_num + 1}")
    
    context = "\n".join(context_parts)
    
    # Create prompt template with instruction and metadata context
    instruction = "Represent this document chunk for semantic retrieval:"
    
    if context:
        template = f"{instruction}\n\nDocument Information:\n{context}\n\nContent:\n{text}"
    else:
        template = f"{instruction}\n\nContent:\n{text}"
    
    return template

if __name__ == "__main__":
    main() 