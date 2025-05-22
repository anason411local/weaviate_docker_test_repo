# High-Quality Vector Embeddings Generator

This script processes PDFs and transcript files to generate high-quality vector embeddings for semantic search and retrieval using Weaviate.

## Features

- **Multi-format support**: Process both PDF documents and transcript files (.txt)
- **Hierarchical embeddings**: Blend document-level context with chunk-level embeddings for better relevance
- **Enhanced metadata extraction**: Extract rich metadata from both PDFs and transcripts
- **Advanced text cleaning**: Clean and normalize text for optimal embedding quality
- **Parallelized processing**: Efficient multi-batch processing of large document collections

## Prerequisites

1. Python 3.7+
2. Weaviate instance running (default: http://localhost:8087)
3. Required packages (install via `pip install -r requirements.txt`):
   - langchain
   - langchain_text_splitters
   - langchain_community
   - huggingface_hub
   - python-dotenv
   - requests
   - numpy
   - tqdm
   - PyMuPDF (optional, for enhanced PDF processing)
   - nltk (optional, for enhanced text processing)
   - PIL (optional, for image processing)

## Directory Structure

```
.
├── pdfs/                 # Directory for PDF files to process
│   ├── file1.pdf
│   ├── file1.json        # Optional metadata in JSON format
│   └── ...
├── transcripts/          # Directory for transcript files
│   ├── audio/            # Optional subdirectory structure
│   │   ├── file1.txt
│   │   └── ...
│   ├── video/
│   │   ├── file2.txt
│   │   └── ...
│   └── ...
└── high_quality_vectorizer.py  # Main script
```

## Transcript File Format

The script supports transcript files (.txt) with optional metadata headers. The recommended format is:

```
Title: Example Transcript
Speaker: John Doe
Date Recorded: 2023-05-15
Duration: 00:45:30
Language: English

[Transcript content starts here...]
```

Any metadata at the beginning of the file with the format `Key: Value` will be extracted and stored as properties in Weaviate.

## Configuration

Edit the configuration variables at the top of the script:

```python
# Weaviate connection
WEAVIATE_URL = "http://localhost:8087"
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
COLLECTION_NAME = "Area_expansion_Dep_Anas_V3"

# Set to True to recreate collection (delete existing)
RECREATE_COLLECTION = True

# Directory settings
PDF_DIR = "pdfs"  # Directory containing PDF files
TRANSCRIPT_DIR = "transcripts"  # Directory containing transcript files

# Settings for improved vector quality
CHUNK_SIZE = 800  # Smaller chunks for better semantic focus
CHUNK_OVERLAP = 160  # Increased overlap for better context continuity
USE_HIERARCHICAL_EMBEDDINGS = True  # Blend document level context with chunks
HIERARCHICAL_WEIGHT = 0.20  # Increased weight for document-level context
```

## Usage

1. Place your PDF files in the `pdfs/` directory
2. Place your transcript files in the `transcripts/` directory
3. Run the script:

```bash
python high_quality_vectorizer.py
```

## Advanced Features

### Custom JSON Metadata

You can provide additional metadata for PDFs by creating a JSON file with the same name as the PDF file. For example, if you have `document.pdf`, create `document.json` in the same directory with custom metadata fields.

### Hierarchical Embeddings

The script creates document-level embeddings and blends them with chunk-level embeddings to provide better context awareness. This can be configured with:

```python
USE_HIERARCHICAL_EMBEDDINGS = True  # Enable/disable
HIERARCHICAL_WEIGHT = 0.20  # Weight of document-level embedding (0-1)
```

### Enhanced Metadata Extraction

- For PDFs: Extracts title, author, subject, keywords, creation date, page count, etc.
- For Transcripts: Extracts title, speaker, duration, recording date, language, etc.

## Output

The script creates embeddings in Weaviate with a rich set of properties including:

- Common properties: text, source, filename, title, etc.
- PDF-specific properties: page, author, subject, has_images, etc.
- Transcript-specific properties: speaker, duration, date_recorded, language, etc.

Each document is split into optimally-sized chunks for retrieval, with document-level context blended in for better relevance. 