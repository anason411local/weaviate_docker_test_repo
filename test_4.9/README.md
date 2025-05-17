# PDF Vectorizer with Weaviate v4

This tool processes PDF documents, extracts text and metadata, and stores them in a Weaviate vector database for semantic search and retrieval.

## Requirements

- Python 3.8+
- Miniconda with `test_env` environment
- Docker for running Weaviate

## Weaviate Docker Setup

This script is configured to connect to a local Weaviate instance running on Docker at:
- Host: 127.0.0.1
- Port: 8090

To start Weaviate with Docker, run:

```bash
docker run -d --name weaviate-v4 \
  -p 8090:8080 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH="/var/lib/weaviate" \
  -e DEFAULT_VECTORIZER_MODULE="none" \
  -e ENABLE_MODULES="" \
  -e CLUSTER_HOSTNAME="node1" \
  weaviate/weaviate:1.23.7
```

## Usage

1. Place your PDF files in the `pdfs` directory
2. Make sure the `test_env` conda environment is set up
3. Run the vectorizer with:

```bash
./run_vectorizer.sh
```

## Key Features

- Uses BGE-M3 embedding model for high-quality semantic vectors
- Extracts comprehensive PDF metadata and content features
- Instruction-enhanced embeddings for better semantic search
- Uses Weaviate v4 Python client for vector storage
- Processes documents in batches to manage memory usage 