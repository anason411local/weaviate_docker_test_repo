import os
import requests # Keep for potential future use, though not directly used now
import json
import uuid
import re
import base64
import nltk
from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import fitz  # PyMuPDF
import datetime
import unicodedata
import weaviate
from weaviate.classes.config import Property, DataType, Configure 
from weaviate.classes.data import DataObject 
from weaviate.auth import AuthApiKey # If you were to use API key auth

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL (Pillow) is not installed. Cover image extraction will be disabled.")

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True) 
    NLTK_AVAILABLE = True
except Exception as e:
    NLTK_AVAILABLE = False
    print(f"Warning: NLTK resources could not be downloaded ({e}). Some text cleaning features might be limited.")

# Load environment variables
load_dotenv()

# Weaviate Connection Settings
# HTTP connection details (primary for schema, discovery)
WEAVIATE_HTTP_HOST = os.getenv("WEAVIATE_HTTP_HOST", "127.0.0.1")
WEAVIATE_HTTP_PORT = int(os.getenv("WEAVIATE_HTTP_PORT", "8090")) # Updated default
WEAVIATE_HTTP_SECURE = os.getenv("WEAVIATE_HTTP_SECURE", "False").lower() == "true"

# gRPC connection details (used by client for performance if available)
WEAVIATE_GRPC_HOST = os.getenv("WEAVIATE_GRPC_HOST", "127.0.0.1")
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50061")) # Updated default
WEAVIATE_GRPC_SECURE = os.getenv("WEAVIATE_GRPC_SECURE", "False").lower() == "true"

# Optional: API Key for authenticated Weaviate (e.g., WCS)
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")


# Text splitting parameters
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 300))

# Embedding model configuration
MODEL_NAME = os.getenv("MODEL_NAME", "BAAI/bge-m3")
MODEL_KWARGS = json.loads(os.getenv("MODEL_KWARGS", '{"device": "cpu", "trust_remote_code": true}'))
ENCODE_KWARGS = json.loads(os.getenv("ENCODE_KWARGS", '{"normalize_embeddings": true}'))

embedding_model = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs=MODEL_KWARGS,
    encode_kwargs=ENCODE_KWARGS,
    show_progress=True
)

def get_weaviate_client():
    """Creates and returns a Weaviate client instance using custom connection."""
    connection_params = {
        "http_host": WEAVIATE_HTTP_HOST,
        "http_port": WEAVIATE_HTTP_PORT,
        "http_secure": WEAVIATE_HTTP_SECURE,
        "grpc_host": WEAVIATE_GRPC_HOST,
        "grpc_port": WEAVIATE_GRPC_PORT,
        "grpc_secure": WEAVIATE_GRPC_SECURE,
        # Additional headers can be passed if needed, e.g. for OIDC
        # "headers": {"X-My-Custom-Header": "value"}
    }
    
    if WEAVIATE_API_KEY:
        auth_config = AuthApiKey(api_key=WEAVIATE_API_KEY)
        connection_params["auth_client_secret"] = auth_config
    
    client = weaviate.connect_to_custom(**connection_params)
    return client

def clean_text(text):
    """Clean and normalize text to improve embedding quality."""
    if not text:
        return ""
    cleaned = re.sub(r'\s+', ' ', text).strip()
    artifacts = [
        r'(\n\s*\d+\s*\n)', r'^\s*\d+\s*$', r'©.*?reserved\.?',
        r'(Page|PAGE)(\s+\d+\s+of\s+\d+)', r'\[\s*\d+\s*\]',
        r'(^|[^a-zA-Z0-9])\d{6,}([^a-zA-Z0-9]|$)', 
        r'\\[a-zA-Z]+\{.*?\}', r'</?[a-zA-Z0-9]+.*?>' 
    ]
    for pattern in artifacts:
        cleaned = re.sub(pattern, ' ', cleaned, flags=re.MULTILINE | re.IGNORECASE)
    cleaned = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', cleaned) 
    cleaned = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', cleaned)
    cleaned = re.sub(r'\s+([.,;:!?)])', r'\1', cleaned)
    cleaned = re.sub(r'([(])\s+', r'\1', cleaned)
    cleaned = re.sub(r'["“„”]', '"', cleaned)
    cleaned = re.sub(r"['‘’`]", "'", cleaned)
    try:
        cleaned = unicodedata.normalize('NFKC', cleaned)
    except Exception: 
        cleaned = unicodedata.normalize('NFC', cleaned)


    cleaned = re.sub(r'(https?:\/\/[^\s]+)', '[URL]', cleaned)
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    cleaned = re.sub(email_pattern, '[EMAIL]', cleaned)
        
    if NLTK_AVAILABLE:
        try:
            sentences = nltk.sent_tokenize(cleaned)
            processed_sentences = []
            for s in sentences:
                s_stripped = s.strip()
                if not s_stripped: continue
                if s_stripped[0].islower():
                    s_stripped = s_stripped[0].upper() + s_stripped[1:]
                if s_stripped[-1] not in ['.', '!', '?']:
                    s_stripped += '.'
                processed_sentences.append(s_stripped)
            cleaned = ' '.join(processed_sentences)
        except Exception: 
            pass 
    
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned if len(cleaned) >= 50 else ""


def extract_json_metadata(pdf_path_str: str):
    """Extract metadata from a JSON file associated with a PDF."""
    json_path = Path(pdf_path_str).with_suffix('.json')
    if not json_path.exists():
        return {}
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        if 'id' in json_data: del json_data['id']
        return {f"document_{k}": json.dumps(v) if isinstance(v, (dict, list)) else v 
                for k, v in json_data.items()}
    except Exception as e:
        print(f"Error extracting JSON metadata from {json_path}: {e}")
        return {}

def extract_pdf_metadata(pdf_path_str: str):
    """Extract detailed metadata from a PDF file using PyMuPDF."""
    pdf_path = Path(pdf_path_str)
    metadata = {
        "title": "", "author": "", "subject": "", "keywords": [], "creator": "", 
        "producer": "", "creation_date": None, "modification_date": None, "page_count": 0,
        "file_size": 0.0, "pdf_version": "", "has_toc": False, "has_links": False, 
        "has_forms": False, "has_annotations": False, "has_images": False, "image_count": 0,
        "has_tables": False, "table_count": 0, "total_chars": 0, "total_words": 0,
        "languages": [], "is_encrypted": False, "encryption_method": "", 
        "permissions": "{}", "is_reflowable": False, "has_embedded_files": False,
        "embedded_file_names": "[]", "toc_json": "[]", "embedded_files_info": "[]",
        "fonts_used_json": "[]", "cover_image_b64": ""
    }

    try:
        doc = fitz.open(pdf_path)
        doc_meta_raw = doc.metadata 
        
        raw_keywords = doc_meta_raw.get("keywords", "")
        if isinstance(raw_keywords, str) and raw_keywords.strip():
            metadata["keywords"] = [kw.strip() for kw in raw_keywords.split(',') if kw.strip()]
        elif isinstance(raw_keywords, list): 
             metadata["keywords"] = [str(kw).strip() for kw in raw_keywords if str(kw).strip()]
        else:
            metadata["keywords"] = []


        metadata.update({
            "title": doc_meta_raw.get("title", ""),
            "author": doc_meta_raw.get("author", ""),
            "subject": doc_meta_raw.get("subject", ""),
            "creator": doc_meta_raw.get("creator", ""),
            "producer": doc_meta_raw.get("producer", ""),
            "page_count": doc.page_count,
            "file_size": float(pdf_path.stat().st_size) if pdf_path.exists() else 0.0,
            "pdf_version": doc_meta_raw.get("format", "").replace("PDF-", ""), 
            "has_toc": len(doc.get_toc(simple=False)) > 0,
            "has_forms": bool(doc.is_form_pdf),
            "is_encrypted": doc.is_encrypted,
            "encryption_method": str(doc.encryption_method()) if doc.is_encrypted else "",
            "permissions": json.dumps(doc.permissions) if hasattr(doc, 'permissions') else "{}",
            "has_embedded_files": len(doc.embfile_names()) > 0,
            "embedded_file_names": json.dumps(doc.embfile_names()),
        })
        
        metadata["is_reflowable"] = False 

        toc_list_raw = doc.get_toc(simple=False)
        if toc_list_raw:
            metadata["toc_json"] = json.dumps([
                {"level": level, "title": title, "page": page, 
                 "dest_page": dest.get("to", (0,0))[1] if dest and dest.get("kind") == fitz.LINK_GOTO else None} 
                for level, title, page, dest in toc_list_raw
            ])

        for date_field_key in ["creation_date", "modification_date"]:
            raw_date_val = doc_meta_raw.get("creationDate" if date_field_key == "creation_date" else "modDate", "")
            if raw_date_val and isinstance(raw_date_val, str) and raw_date_val.startswith("D:"):
                try:
                    date_str_cleaned = raw_date_val[2:].split('Z')[0].split('+')[0].split('-')[0].split(' ')[0]
                    year = int(date_str_cleaned[0:4])
                    month = int(date_str_cleaned[4:6])
                    day = int(date_str_cleaned[6:8])
                    hour = int(date_str_cleaned[8:10]) if len(date_str_cleaned) > 8 else 0
                    minute = int(date_str_cleaned[10:12]) if len(date_str_cleaned) > 10 else 0
                    second = int(date_str_cleaned[12:14]) if len(date_str_cleaned) > 12 else 0
                    if not (1 <= month <= 12): month = 1 
                    if not (1 <= day <= 31): day = 1 
                    
                    dt_obj = datetime.datetime(year, month, day, hour, minute, second, tzinfo=datetime.timezone.utc)
                    metadata[date_field_key] = dt_obj.isoformat()
                except (ValueError, TypeError) as e_date:
                    metadata[date_field_key] = None 
        
        fonts_collected = set()
        detected_languages_set = set() 

        for page_idx in range(doc.page_count):
            page = doc.load_page(page_idx)
            text_on_page = page.get_text("text")
            metadata["total_chars"] += len(text_on_page)
            metadata["total_words"] += len(re.findall(r'\b\w+\b', text_on_page))
            
            page_images = page.get_images(full=True)
            if page_images: metadata["has_images"] = True; metadata["image_count"] += len(page_images)
            if page.get_links(): metadata["has_links"] = True
            if list(page.annots()): metadata["has_annotations"] = True
            
            for font_detail in page.get_fonts(full=True): fonts_collected.add(font_detail[3]) 
            
            try: 
                page_tables = page.find_tables()
                if page_tables.tables: metadata["has_tables"] = True; metadata["table_count"] += len(page_tables.tables)
            except Exception: pass 

            if page_idx < 3: 
                 if re.search(r'\b(the|and|is|of)\b', text_on_page, re.I): detected_languages_set.add("en")

        metadata["fonts_used_json"] = json.dumps(list(fonts_collected))
        metadata["languages"] = list(detected_languages_set) if detected_languages_set else []


        if metadata["has_embedded_files"]:
            emb_files_info_list = []
            for i in range(doc.embfile_count()):
                try: info = doc.embfile_info(i); emb_files_info_list.append({"name": info["filename"], "size": info["len"]})
                except Exception: pass 
            metadata["embedded_files_info"] = json.dumps(emb_files_info_list)

        if PIL_AVAILABLE and metadata["has_images"] and doc.page_count > 0:
            try:
                for page_idx in range(min(doc.page_count, 1)): 
                    page_for_cover = doc.load_page(page_idx)
                    images_on_page = page_for_cover.get_images(full=True)
                    if images_on_page:
                        largest_img_xref = None; max_img_area = 0
                        for img_info in images_on_page:
                            pix = fitz.Pixmap(doc, img_info[0])
                            if pix.width * pix.height > max_img_area:
                                max_img_area = pix.width * pix.height
                                largest_img_xref = img_info[0]
                        
                        if largest_img_xref:
                            pix_data = fitz.Pixmap(doc, largest_img_xref)
                            img_bytes = pix_data.tobytes("jpeg") 
                            img_pil = Image.open(BytesIO(img_bytes))
                            img_pil.thumbnail((300, 300)) 
                            buffer = BytesIO()
                            img_pil.save(buffer, format="JPEG")
                            metadata["cover_image_b64"] = f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
                        break 
            except Exception as e_cover: print(f"Warning: Error extracting cover image from {pdf_path}: {e_cover}")
        
        doc.close()
    except Exception as e_main_meta:
        print(f"Error extracting PDF metadata from {pdf_path}: {e_main_meta}")
    
    json_sidecar_meta = extract_json_metadata(pdf_path_str)
    metadata.update(json_sidecar_meta) 
    return metadata


def extract_content_features(pdf_path_str: str):
    """Extract detailed content features from a PDF using PyMuPDF."""
    pdf_path = Path(pdf_path_str)
    content_features = {
        "page_features_list": [], 
        "doc_structure_details": {"heading_levels_list": [], "section_counts": 0, "avg_section_length_pages": 0.0},
        "layout_details": {"is_multi_column": False, "has_header_or_footer": False, "page_dimensions_list": []},
        "readability_details": {"avg_sentence_len": 0.0, "avg_word_len_chars": 0.0, "complex_word_percent": 0.0}
    }
    try:
        doc = fitz.open(pdf_path)
        agg_total_sentences, agg_total_words_readability, agg_total_chars_in_words, agg_complex_words = 0, 0, 0, 0
        all_detected_headings = []

        for page_idx in range(doc.page_count):
            page = doc.load_page(page_idx)
            page_rect = page.rect
            content_features["layout_details"]["page_dimensions_list"].append({"width": float(page_rect.width), "height": float(page_rect.height)})
            
            text_content_page = page.get_text("text")
            blocks_on_page = page.get_text("blocks", sort=True) 

            if not content_features["layout_details"]["is_multi_column"] and len(blocks_on_page) > 2:
                text_block_x_coords = sorted([b[0] for b in blocks_on_page if b[6] == 0 and (b[2]-b[0]) > 10]) 
                if len(text_block_x_coords) > 1:
                    gaps = [text_block_x_coords[i+1] - text_block_x_coords[i] for i in range(len(text_block_x_coords)-1)]
                    if any(g > page_rect.width * 0.05 for g in gaps): 
                        content_features["layout_details"]["is_multi_column"] = True
            
            if not content_features["layout_details"]["has_header_or_footer"]:
                for _, y0, _, y1, text_block_content, _, block_type in blocks_on_page:
                    if block_type == 0 and text_block_content.strip() and \
                       (y0 < page_rect.height * 0.1 or y1 > page_rect.height * 0.9) and \
                       len(text_block_content.split()) < 15: 
                        content_features["layout_details"]["has_header_or_footer"] = True; break
            
            page_sentences = nltk.sent_tokenize(text_content_page) if NLTK_AVAILABLE else re.split(r'[.!?]+', text_content_page)
            page_sentences = [s.strip() for s in page_sentences if s.strip()]
            agg_total_sentences += len(page_sentences)
            
            page_words_list = re.findall(r'\b\w+\b', text_content_page.lower())
            agg_total_words_readability += len(page_words_list)
            for word_item in page_words_list:
                agg_total_chars_in_words += len(word_item)
                if len(word_item) >= 7: agg_complex_words += 1 
            
            page_text_dict = page.get_text("dict", sort=True)
            for block_dict in page_text_dict.get("blocks", []):
                if block_dict.get("type") == 0: 
                    for line_dict in block_dict.get("lines", []):
                        line_text_combined = "".join([span.get("text","") for span in line_dict.get("spans",[])]).strip()
                        if line_text_combined and len(line_text_combined.split()) < 12: 
                            span_sizes = [s.get("size",0) for s in line_dict.get("spans",[])]
                            span_flags = [s.get("flags",0) for s in line_dict.get("spans",[])]
                            if span_sizes:
                                avg_size = sum(span_sizes) / len(span_sizes)
                                is_bold_line = any(f & 2 for f in span_flags) 
                                if avg_size > 13.5 and is_bold_line: 
                                    all_detected_headings.append({
                                        "text": line_text_combined[:200], "font_size": round(avg_size,1), 
                                        "page": page_idx + 1, "bold": True
                                    })
            
            content_features["page_features_list"].append({
                "page_number": page_idx + 1,
                "text_block_count": len([b for b in blocks_on_page if b[6]==0]),
                "word_count_on_page": len(page_words_list),
                "char_count_on_page": len(text_content_page),
                "image_count_on_page": len(page.get_images(full=True))
            })

        if agg_total_sentences > 0: content_features["readability_details"]["avg_sentence_len"] = round(agg_total_words_readability / agg_total_sentences, 2)
        if agg_total_words_readability > 0:
            content_features["readability_details"]["avg_word_len_chars"] = round(agg_total_chars_in_words / agg_total_words_readability, 2)
            content_features["readability_details"]["complex_word_percent"] = round((agg_complex_words / agg_total_words_readability) * 100, 2)

        unique_headings_sorted = sorted(list(set((h['text'], h['font_size'], h['page']) for h in all_detected_headings)), key=lambda x: (x[2], -x[1], x[0]))
        content_features["doc_structure_details"]["heading_levels_list"] = [{"text": t, "font_size": s, "page": p} for t,s,p in unique_headings_sorted[:30]] 
        
        top_level_headings = [h for h in unique_headings_sorted if h[1] > 15] 
        content_features["doc_structure_details"]["section_counts"] = len(top_level_headings)
        if len(top_level_headings) > 1:
            section_page_lengths = [top_level_headings[i+1][2] - top_level_headings[i][2] for i in range(len(top_level_headings)-1)]
            if section_page_lengths: content_features["doc_structure_details"]["avg_section_length_pages"] = round(sum(section_page_lengths) / len(section_page_lengths), 1)
        elif len(top_level_headings) == 1 and doc.page_count > 0:
             content_features["doc_structure_details"]["avg_section_length_pages"] = float(doc.page_count)
        doc.close()
    except Exception as e_content:
        print(f"Error extracting content features from {pdf_path}: {e_content}")
    return content_features


def load_and_split_pdfs(directory_path: str):
    """Load PDFs, extract text per page, combine, and then split into chunks."""
    print(f"Loading PDFs from {directory_path}...")
    loader = DirectoryLoader(
        directory_path, glob="**/*.pdf", loader_cls=PyPDFLoader,
        show_progress=True, use_multithreading=True, silent_errors=True
    )
    try:
        langchain_pages = loader.load() 
    except Exception as e:
        print(f"Error loading documents with DirectoryLoader: {e}"); langchain_pages = []

    if not langchain_pages: print("No document pages loaded."); return []
    print(f"Loaded {len(langchain_pages)} document pages initially.")
    
    docs_content_by_source = {}
    for lc_page in langchain_pages:
        source_path = lc_page.metadata.get("source", "unknown_source_file")
        if source_path not in docs_content_by_source:
            docs_content_by_source[source_path] = []
        docs_content_by_source[source_path].append(lc_page)
    
    final_chunked_documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len,
        separators=["\n\n\n", "\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""], keep_separator=False
    )
    
    print("Applying semantic chunking per document...")
    for source_pdf_path, lc_pages_for_pdf in docs_content_by_source.items():
        lc_pages_for_pdf.sort(key=lambda p: p.metadata.get("page", 0)) 
        
        full_pdf_text_with_markers = ""
        for i, page_doc in enumerate(lc_pages_for_pdf):
            page_num_from_meta = page_doc.metadata.get("page", i) 
            full_pdf_text_with_markers += f"\n\n<<<PDF_PAGE_MARKER:{page_num_from_meta + 1}>>>\n\n{page_doc.page_content}"

        if not lc_pages_for_pdf: continue
        combined_doc_for_splitting = Document(page_content=full_pdf_text_with_markers, metadata=lc_pages_for_pdf[0].metadata.copy())
        
        text_chunks_from_pdf = text_splitter.split_documents([combined_doc_for_splitting])

        for text_chunk_doc in text_chunks_from_pdf: 
            page_marker_matches = re.findall(r"<<<PDF_PAGE_MARKER:(\d+)>>>", text_chunk_doc.page_content)
            chunk_original_page_num = int(page_marker_matches[0]) - 1 if page_marker_matches else lc_pages_for_pdf[0].metadata.get("page", 0)
            
            content_no_markers = re.sub(r"\n\n<<<PDF_PAGE_MARKER:\d+>>>\n\n", "\n\n", text_chunk_doc.page_content).strip()
            final_cleaned_content = clean_text(content_no_markers)

            if len(final_cleaned_content) >= 100: 
                text_chunk_doc.page_content = final_cleaned_content
                text_chunk_doc.metadata["source"] = source_pdf_path 
                text_chunk_doc.metadata["page"] = chunk_original_page_num 
                final_chunked_documents.append(text_chunk_doc)
            
    print(f"Split into {len(final_chunked_documents)} semantic chunks.")
    
    print("Deduplicating chunks...")
    unique_final_chunks = []
    hashes_of_seen_content = set()
    for chunk_to_check in final_chunked_documents:
        sample_for_hash = chunk_to_check.page_content[:150] + chunk_to_check.page_content[-150:]
        current_hash = hash(sample_for_hash)
        if current_hash not in hashes_of_seen_content:
            hashes_of_seen_content.add(current_hash)
            unique_final_chunks.append(chunk_to_check)
    print(f"After deduplication: {len(unique_final_chunks)} unique chunks.")
    
    docs_for_weaviate_ingestion = []
    cached_metadata_per_file = {}

    for final_lc_chunk in unique_final_chunks: 
        pdf_file_path = final_lc_chunk.metadata.get("source", "unknown_pdf_file")
        
        if pdf_file_path not in cached_metadata_per_file:
            cached_metadata_per_file[pdf_file_path] = {
                "pdf_level_meta": extract_pdf_metadata(pdf_file_path),
                "content_features_meta": extract_content_features(pdf_file_path)
            }
        
        doc_level_metadata = cached_metadata_per_file[pdf_file_path]["pdf_level_meta"]
        doc_level_content_features = cached_metadata_per_file[pdf_file_path]["content_features_meta"]

        chunk_page_0_indexed = final_lc_chunk.metadata.get("page", 0)
        
        page_specific_content_fts = {}
        for p_fts in doc_level_content_features.get("page_features_list", []):
            if p_fts.get("page_number") == chunk_page_0_indexed + 1: 
                page_specific_content_fts = {
                    "page_text_blocks": p_fts.get("text_block_count", 0),
                    "page_word_count": p_fts.get("word_count_on_page", 0),
                    "page_char_count": p_fts.get("char_count_on_page", 0),
                    "page_image_count": p_fts.get("image_count_on_page", 0),
                }
                break
        
        weaviate_data_object = {
            "text": final_lc_chunk.page_content,
            "filepath": pdf_file_path,
            "filename": Path(pdf_file_path).name,
            "page": chunk_page_0_indexed, 
            
            "title": doc_level_metadata.get("title", ""),
            "author": doc_level_metadata.get("author", ""),
            "subject": doc_level_metadata.get("subject", ""),
            "keywords": doc_level_metadata.get("keywords", []), 
            "creation_date": doc_level_metadata.get("creation_date"), 
            "modification_date": doc_level_metadata.get("modification_date"), 
            "total_pages": doc_level_metadata.get("page_count", 0),
            "creator": doc_level_metadata.get("creator", ""),
            "producer": doc_level_metadata.get("producer", ""),
            "file_size": doc_level_metadata.get("file_size", 0.0),
            "pdf_version": doc_level_metadata.get("pdf_version", ""),
            "toc_json": doc_level_metadata.get("toc_json", "[]"),
            "has_toc": doc_level_metadata.get("has_toc", False),
            "has_links": doc_level_metadata.get("has_links", False),
            "has_forms": doc_level_metadata.get("has_forms", False),
            "has_annotations": doc_level_metadata.get("has_annotations", False),
            "has_images": doc_level_metadata.get("has_images", False),
            "image_count": doc_level_metadata.get("image_count", 0), 
            "has_tables": doc_level_metadata.get("has_tables", False),
            "table_count": doc_level_metadata.get("table_count", 0),
            "total_chars": doc_level_metadata.get("total_chars", 0), 
            "total_words": doc_level_metadata.get("total_words", 0), 
            "languages": doc_level_metadata.get("languages", []), 
            "is_encrypted": doc_level_metadata.get("is_encrypted", False),
            "permissions_json": doc_level_metadata.get("permissions", "{}"),
            "fonts_used_json": doc_level_metadata.get("fonts_used_json", "[]"),
            "cover_image_b64": doc_level_metadata.get("cover_image_b64", ""),
            
            "doc_structure_json": json.dumps(doc_level_content_features.get("doc_structure_details", {})),
            "layout_multi_column": doc_level_content_features.get("layout_details", {}).get("is_multi_column", False),
            "layout_has_header_footer": doc_level_content_features.get("layout_details", {}).get("has_header_or_footer", False),
            "readability_avg_sentence_length": doc_level_content_features.get("readability_details", {}).get("avg_sentence_len", 0.0),
            "readability_avg_word_length": doc_level_content_features.get("readability_details", {}).get("avg_word_len_chars", 0.0),
            "readability_complex_word_percentage": doc_level_content_features.get("readability_details", {}).get("complex_word_percent", 0.0),
        }
        weaviate_data_object.update(page_specific_content_fts) 

        for key, value in doc_level_metadata.items():
            if key.startswith("document_"): 
                 weaviate_data_object[key] = value 

        docs_for_weaviate_ingestion.append(weaviate_data_object)
        
    return docs_for_weaviate_ingestion


def create_collection_if_not_exists(client: weaviate.WeaviateClient, collection_name: str):
    """Creates the Weaviate collection with a defined schema if it doesn't exist."""
    if client.collections.exists(collection_name):
        print(f"Collection '{collection_name}' already exists. Deleting and recreating for a clean run.")
        client.collections.delete(collection_name)

    print(f"Creating '{collection_name}' collection...")
    try:
        properties_list = [
            Property(name="text", data_type=DataType.TEXT), 
            Property(name="filepath", data_type=DataType.TEXT),
            Property(name="filename", data_type=DataType.TEXT), 
            Property(name="page", data_type=DataType.INT),
            Property(name="title", data_type=DataType.TEXT), 
            Property(name="author", data_type=DataType.TEXT),
            Property(name="subject", data_type=DataType.TEXT), 
            Property(name="keywords", data_type=DataType.TEXT_ARRAY, description="Keywords from PDF metadata."), 
            Property(name="creation_date", data_type=DataType.DATE), 
            Property(name="modification_date", data_type=DataType.DATE),
            Property(name="total_pages", data_type=DataType.INT), 
            Property(name="creator", data_type=DataType.TEXT),
            Property(name="producer", data_type=DataType.TEXT), 
            Property(name="file_size", data_type=DataType.NUMBER),
            Property(name="pdf_version", data_type=DataType.TEXT), 
            Property(name="toc_json", data_type=DataType.TEXT, description="Table of Contents as JSON string."),
            Property(name="has_toc", data_type=DataType.BOOL), 
            Property(name="has_links", data_type=DataType.BOOL),
            Property(name="has_forms", data_type=DataType.BOOL), 
            Property(name="has_annotations", data_type=DataType.BOOL),
            Property(name="has_images", data_type=DataType.BOOL), 
            Property(name="image_count", data_type=DataType.INT),
            Property(name="has_tables", data_type=DataType.BOOL), 
            Property(name="table_count", data_type=DataType.INT),
            Property(name="total_chars", data_type=DataType.INT), 
            Property(name="total_words", data_type=DataType.INT),
            Property(name="languages", data_type=DataType.TEXT_ARRAY, description="Detected languages in the document."), 
            Property(name="is_encrypted", data_type=DataType.BOOL), 
            Property(name="permissions_json", data_type=DataType.TEXT, description="Permissions as JSON string."),
            Property(name="fonts_used_json", data_type=DataType.TEXT, description="Fonts used as JSON string."), 
            Property(name="cover_image_b64", data_type=DataType.TEXT, description="Base64 encoded cover image thumbnail."),
            Property(name="doc_structure_json", data_type=DataType.TEXT, description="Document structure features as JSON string."),
            Property(name="layout_multi_column", data_type=DataType.BOOL), 
            Property(name="layout_has_header_footer", data_type=DataType.BOOL),
            Property(name="readability_avg_sentence_length", data_type=DataType.NUMBER),
            Property(name="readability_avg_word_length", data_type=DataType.NUMBER),
            Property(name="readability_complex_word_percentage", data_type=DataType.NUMBER),
            Property(name="page_text_blocks", data_type=DataType.INT, description="Number of text blocks on the chunk's original page."),
            Property(name="page_word_count", data_type=DataType.INT, description="Word count on the chunk's original page."),
            Property(name="page_char_count", data_type=DataType.INT, description="Character count of the chunk's original page."),
            Property(name="page_image_count", data_type=DataType.INT, description="Image count on the chunk's original page.")
        ]
        
        client.collections.create(
            name=collection_name,
            properties=properties_list,
            vectorizer_config=Configure.Vectorizer.none() 
        )
        print(f"✅ Created '{collection_name}' collection.")
        return True
    except Exception as e:
        print(f"Error creating collection '{collection_name}': {e}")
        import traceback; traceback.print_exc()
        return False

def main():
    """Main function to process PDFs and store them in Weaviate."""
    print("Starting PDF vectorization process...")
    print(f"Using model: {MODEL_NAME}")
    print(f"Weaviate HTTP: {WEAVIATE_HTTP_HOST}:{WEAVIATE_HTTP_PORT}, Secure: {WEAVIATE_HTTP_SECURE}")
    print(f"Weaviate gRPC: {WEAVIATE_GRPC_HOST}:{WEAVIATE_GRPC_PORT}, Secure: {WEAVIATE_GRPC_SECURE}")
    
    client_init_test = None
    try:
        client_init_test = get_weaviate_client() 
        if client_init_test.is_ready():
            print("✅ Connected to Weaviate successfully for initial check.")
            print(f"Weaviate client version: {weaviate.__version__}")
        else:
            print("❌ Weaviate is not ready. Please check your Docker container or Weaviate instance."); return
    except Exception as e:
        print(f"❌ Error connecting to Weaviate for initial check: {e}"); return
    finally:
        if client_init_test and client_init_test.is_connected(): client_init_test.close()

    collection_name = "PDFDocuments" 
    client_for_schema = None
    try:
        client_for_schema = get_weaviate_client() 
        if not create_collection_if_not_exists(client_for_schema, collection_name):
            print("Failed to ensure collection exists, aborting."); return
    finally:
        if client_for_schema and client_for_schema.is_connected(): client_for_schema.close()
    
    pdf_dir = "pdfs"
    if not os.path.isdir(pdf_dir): os.makedirs(pdf_dir, exist_ok=True); print(f"'{pdf_dir}' directory created. Please add PDF files."); return
    if not os.listdir(pdf_dir): print(f"'{pdf_dir}' directory is empty. Please add PDF files."); return
        
    docs_to_ingest = load_and_split_pdfs(pdf_dir)
    if not docs_to_ingest: print("No document chunks to process after loading and splitting. Exiting."); return
    print(f"Prepared {len(docs_to_ingest)} document chunks for Weaviate ingestion...")
    
    client_for_ingest = None
    try:
        client_for_ingest = get_weaviate_client() 
        if not client_for_ingest.is_ready(): print("❌ Weaviate connection lost before batch import."); return

        pdf_docs_collection = client_for_ingest.collections.get(collection_name)
        
        data_objects_to_insert = []
        print("Generating embeddings and preparing data objects for batch insertion...")
        for i, doc_properties_original in enumerate(docs_to_ingest):
            doc_properties = doc_properties_original.copy()

            text_for_embedding = enhance_text_for_embedding(
                doc_properties.get("text", ""), 
                {"title": doc_properties.get("title"), "subject": doc_properties.get("subject"), 
                 "keywords": doc_properties.get("keywords"), "page": doc_properties.get("page")}
            )
            if not text_for_embedding: 
                print(f"Skipping object {i+1} due to empty text for embedding: {doc_properties.get('filename','N/A')}")
                continue

            try:
                for date_k in ["creation_date", "modification_date"]:
                    date_val = doc_properties.get(date_k)
                    if isinstance(date_val, str):
                        try:
                            if not re.match(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{3,6})?(?:[+-]\d{2}:\d{2}|Z)?$', date_val):
                                dt = datetime.datetime.fromisoformat(date_val.replace("Z", "+00:00"))
                                doc_properties[date_k] = dt.isoformat()
                        except (ValueError, TypeError):
                            doc_properties[date_k] = None
                    elif date_val is not None and not isinstance(date_val, str): 
                         doc_properties[date_k] = None

                for num_k in ["file_size", "readability_avg_sentence_length", "readability_avg_word_length", "readability_complex_word_percentage"]:
                    if num_k in doc_properties and doc_properties[num_k] is not None:
                        try: doc_properties[num_k] = float(doc_properties[num_k])
                        except (ValueError, TypeError): doc_properties[num_k] = 0.0 

                int_ks = ["page", "total_pages", "image_count", "total_chars", "total_words", "table_count", 
                            "page_text_blocks", "page_word_count", "page_char_count", "page_image_count"]
                for int_k in int_ks:
                    if int_k in doc_properties and doc_properties[int_k] is not None:
                        try: doc_properties[int_k] = int(doc_properties[int_k])
                        except (ValueError, TypeError): doc_properties[int_k] = 0 
                
                for arr_k in ["keywords", "languages"]:
                    current_val = doc_properties.get(arr_k)
                    processed_list = []
                    if isinstance(current_val, str):
                        if "," in current_val: 
                            processed_list = [item.strip() for item in current_val.split(',') if item.strip()]
                        elif current_val.strip(): 
                            processed_list = [current_val.strip()]
                    elif isinstance(current_val, list):
                        processed_list = [str(item).strip() for item in current_val if item is not None and str(item).strip()]
                    
                    if not processed_list: 
                        if arr_k in doc_properties:
                            del doc_properties[arr_k] 
                    else:
                        doc_properties[arr_k] = processed_list
                
                final_props_for_weaviate = {k: v for k, v in doc_properties.items() if v is not None}
                
                obj_uuid_str = str(uuid.uuid5(uuid.NAMESPACE_DNS, 
                                         f"{final_props_for_weaviate.get('filename', '')}-{final_props_for_weaviate.get('page', 0)}-{text_for_embedding[:60]}"))

                embedding_vector = embedding_model.embed_query(text_for_embedding)
                
                data_objects_to_insert.append(
                    DataObject(properties=final_props_for_weaviate, uuid=obj_uuid_str, vector=embedding_vector)
                )
            except Exception as e_obj_prep:
                print(f"Error preparing object {i+1} ({doc_properties.get('filename','N/A')}): {e_obj_prep}")
                import traceback; traceback.print_exc()
        
        if data_objects_to_insert:
            print(f"\nAttempting to insert {len(data_objects_to_insert)} objects in batch...")
            batch_return = pdf_docs_collection.data.insert_many(data_objects_to_insert)
            
            print(f"Batch import attempt finished.")
            if batch_return.has_errors:
                print(f"❌ Batch import encountered {len(batch_return.errors)} errors:")
                for obj_idx, error_obj in batch_return.errors.items(): 
                    failed_obj_props = data_objects_to_insert[obj_idx].properties if obj_idx < len(data_objects_to_insert) else {}
                    failed_filename = failed_obj_props.get('filename', 'N/A')
                    print(f"  - Object at original index {obj_idx} (File: {failed_filename}): {error_obj.message}") 
            else:
                num_errors = len(batch_return.errors) if batch_return.errors else 0
                print(f"✅ Successfully imported {len(data_objects_to_insert) - num_errors} objects.")
            print(f"Total successful UUIDs reported by insert_many: {len(batch_return.uuids)}")

        else:
            print("No data objects were prepared for batch insertion.")
            
    except Exception as e_batch:
        print(f"An error occurred during the main batch import process: {e_batch}")
        import traceback; traceback.print_exc()
    finally:
        if client_for_ingest and client_for_ingest.is_connected(): client_for_ingest.close()

    print(f"\n--- Process Completed ---")


def enhance_text_for_embedding(text, doc_metadata=None):
    """Enhance text with structural prompts and metadata to improve embedding quality."""
    if not text or not text.strip(): return "" 
    
    context_parts = []
    if doc_metadata: 
        title = doc_metadata.get("title", "")
        subject = doc_metadata.get("subject", "")
        keywords_data = doc_metadata.get("keywords") 
        keywords_str = ""
        if isinstance(keywords_data, list): keywords_str = ", ".join(keywords_data)
        elif isinstance(keywords_data, str): keywords_str = keywords_data 

        page_num = doc_metadata.get("page") 

        if title: context_parts.append(f"Title: {title}")
        if subject: context_parts.append(f"Subject: {subject}")
        if keywords_str: context_parts.append(f"Keywords: {keywords_str}")
        if page_num is not None: context_parts.append(f"Page: {page_num + 1}") 
    
    context_str_final = "\n".join(context_parts)
    instruction = "Represent this document chunk for semantic retrieval:"
    
    return f"{instruction}\n\nDocument Information:\n{context_str_final}\n\nContent:\n{text}" if context_str_final else f"{instruction}\n\nContent:\n{text}"

if __name__ == "__main__":
    main()
