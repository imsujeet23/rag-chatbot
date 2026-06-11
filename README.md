# RAG Chatbot System

A complete Retrieval-Augmented Generation (RAG) system built with FAISS, HuggingFace embeddings, and HuggingFace LLM.

## Features

- 📄 **Multi-format Support**: Load PDF and TXT documents
- 🔍 **Smart Chunking**: Overlapping chunks for better context preservation
- 🧠 **HuggingFace Embeddings**: Using `sentence-transformers/all-MiniLM-L6-v2`
- 🗂️ **FAISS Vector Store**: Efficient similarity search
- 🤖 **LLM Integration**: TinyLlama for generation (can be swapped)
- 🚀 **REST API**: FastAPI-based endpoints for easy integration
- 💾 **Persistence**: Save and load indexes

## Project Structure

```
rag-chatbot/
├── app/
│   ├── api.py           # FastAPI REST endpoints
│   ├── chunker.py       # Document chunking logic
│   ├── embedder.py      # Embedding generation
│   ├── llm.py           # LLM integration
│   ├── loader.py        # Document loading
│   ├── retreiver.py     # FAISS retrieval
│   ├── utils.py         # Utility functions
│   └── __init__.py      # Package initialization
├── models/              # Model storage
├── uploads/             # Document uploads
├── vector_store/        # FAISS index storage
├── tests/               # Test suite
├── requirements.txt     # Dependencies
├── example.py           # Example usage script
└── README.md           # This file
```

## Installation

1. **Clone or navigate to the project**:
   ```bash
   cd /Users/sujeetkumar/Documents/Coding/rag-chatbot
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Option 1: Using the REST API

1. **Start the server**:
   ```bash
   uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
   ```

2. **API will be available at**: `http://localhost:8000`

3. **Interactive API docs**: `http://localhost:8000/docs`

### Option 2: Using the Example Script

1. **Add documents to `uploads/` folder** (PDF or TXT files)

2. **Run the example**:
   ```bash
   python example.py
   ```

## API Endpoints

### GET `/`
Get API information and available endpoints.

### GET `/health`
Health check endpoint. Returns status of all components.

### POST `/upload`
Upload a document (PDF or TXT) and index it.

**Request**:
```bash
curl -F "file=@document.pdf" http://localhost:8000/upload
```

**Response**:
```json
{
  "status": "success",
  "filename": "document.pdf",
  "chunks_created": 42,
  "message": "Document 'document' uploaded and indexed successfully"
}
```

### POST `/query`
Query the vector store for similar documents.

**Request Body**:
```json
{
  "query": "What are the main topics?",
  "top_k": 5
}
```

**Response**:
```json
{
  "query": "What are the main topics?",
  "results": [
    {
      "content": "...",
      "document": "document_name",
      "chunk_index": 0,
      "distance": 0.234,
      "similarity_score": 0.81
    }
  ],
  "total_results": 5
}
```

### POST `/chat`
Chat with RAG - retrieves context and generates answer.

**Request Body**:
```json
{
  "question": "What are the main topics?",
  "top_k": 5,
  "max_length": 512
}
```

**Response**:
```json
{
  "question": "What are the main topics?",
  "context": "...",
  "answer": "Based on the provided context...",
  "source_chunks": [...]
}
```

### GET `/index-stats`
Get statistics about the current index.

**Response**:
```json
{
  "total_chunks": 150,
  "embedding_dim": 384,
  "index_size": 150
}
```

### POST `/clear-index`
Clear the index and start fresh.

## Component Details

### DocumentChunker
- Splits documents into overlapping chunks
- Default chunk size: 1000 characters
- Default overlap: 200 characters
- Configurable in `api.py` startup

### HuggingFaceEmbedder
- Uses `sentence-transformers/all-MiniLM-L6-v2`
- Embedding dimension: 384
- Batch processing support

### FAISSRetriever
- IndexFlatL2 (L2 distance)
- Persistent storage (save/load)
- Batch retrieval support
- Distance to similarity conversion

### HuggingFaceLLM
- Uses TinyLlama 1.1B by default (fast, lightweight)
- Supports larger models (swap in `api.py`)
- RAG-aware prompt formatting
- Configurable generation parameters

## Usage Examples

### Python Script

```python
from app.loader import load_document
from app.chunker import DocumentChunker
from app.embedder import HuggingFaceEmbedder
from app.retreiver import FAISSRetriever
from app.llm import HuggingFaceLLM

# Initialize components
embedder = HuggingFaceEmbedder()
retriever = FAISSRetriever(embedding_dim=384)
llm = HuggingFaceLLM()

# Load and process document
content, name = load_document("document.pdf")
chunker = DocumentChunker()
chunks = chunker.chunk_text(content, name)
chunks = embedder.embed_chunks(chunks)
retriever.add_chunks(chunks)

# Query
query = "What is the main topic?"
query_emb = embedder.embed_text(query)
results = retriever.retrieve(query_emb, top_k=3)

# Generate answer
context = "\n".join([r['content'] for r in results])
answer = llm.generate_with_context(context, query)
print(answer)
```

### cURL Commands

**Upload a document**:
```bash
curl -X POST -F "file=@document.pdf" http://localhost:8000/upload
```

**Query the system**:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is AI?", "top_k": 3}'
```

**Chat with RAG**:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is AI?", "top_k": 5}'
```

## Configuration

Edit `app/api.py` to customize:

```python
# Embedding model
embedder = HuggingFaceEmbedder(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # Change this
)

# Chunking parameters
chunker = DocumentChunker(
    chunk_size=1000,    # Adjust chunk size
    overlap=200         # Adjust overlap
)

# LLM model
llm = HuggingFaceLLM(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Change this
)
```

## Performance Tips

1. **Batch Processing**: Use batch methods for multiple documents/queries
2. **Device Selection**: Ensure GPU is available for faster inference
3. **Model Selection**: Smaller models (TinyLlama) for quick responses, larger models for better quality
4. **Chunking**: Adjust chunk size based on your domain:
   - Technical docs: 500-1000 characters
   - Long-form content: 1000-2000 characters
   - Short snippets: 200-500 characters

## Troubleshooting

### Out of Memory Errors
- Reduce chunk size
- Use smaller LLM model
- Process documents in batches

### Slow Inference
- Use a smaller model
- Check GPU availability
- Reduce max_length for generation

### No Results Retrieved
- Check if documents are uploaded
- Verify `/index-stats` shows chunks
- Try different top_k values
- Check query similarity with `/query` endpoint

## Future Enhancements

- [ ] Support for more file formats (DOCX, PPTX, HTML)
- [ ] Hybrid search (BM25 + semantic)
- [ ] Query expansion and reranking
- [ ] Multi-GPU support
- [ ] Caching layer for faster responses
- [ ] Fine-tuned domain-specific models
- [ ] Web UI dashboard
- [ ] Authentication and rate limiting

## License

This project is open source and available under the MIT License.

## Support

For issues or questions, please check the logs in the terminal where the API is running.
