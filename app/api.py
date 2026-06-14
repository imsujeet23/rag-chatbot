from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import logging
import os
from pathlib import Path
from typing import List, Optional
import asyncio

# Import components
from app.loader import load_document
from app.chunker import DocumentChunker
from app.embedder import HuggingFaceEmbedder
from app.retreiver import FAISSRetriever
from app.llm import HuggingFaceLLM
from app.utils import get_upload_dir, get_vector_store_dir, sanitize_filename

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="Retrieval Augmented Generation Chatbot",
    version="1.0.0"
)
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Global variables for components
embedder: Optional[HuggingFaceEmbedder] = None
retriever: Optional[FAISSRetriever] = None
llm: Optional[HuggingFaceLLM] = None
chunker: Optional[DocumentChunker] = None


# Pydantic models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    max_length: int = 512


class ChatRequest(BaseModel):
    question: str
    top_k: int = 5
    max_length: int = 512


class DocumentInfo(BaseModel):
    filename: str
    size_mb: float


class IndexStats(BaseModel):
    total_chunks: int
    embedding_dim: int
    index_size: int


class QueryResponse(BaseModel):
    query: str
    results: List[dict]
    total_results: int


class ChatResponse(BaseModel):
    question: str
    context: str
    answer: str
    source_chunks: List[dict]


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global embedder, retriever, llm, chunker
    
    logger.info("Initializing RAG components...")
    
    try:
        # Initialize embedder
        embedder = HuggingFaceEmbedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize retriever
        retriever = FAISSRetriever(embedding_dim=embedder.get_embedding_dimension())
        
        # Initialize chunker
        chunker = DocumentChunker(chunk_size=1000, overlap=200)
        
        # Try to load existing index
        vector_store_path = get_vector_store_dir() / "faiss_index"
        if (vector_store_path.parent / (vector_store_path.name + ".index")).exists():
            try:
                retriever.load(str(vector_store_path))
                logger.info("Loaded existing FAISS index")
            except Exception as e:
                logger.warning(f"Could not load existing index: {str(e)}")
        
        # Initialize LLM (this may take time on first load)
        logger.info("Loading LLM model (this may take a moment)...")
        llm = HuggingFaceLLM(
            model_name="distilgpt2"
        )
        
        logger.info("RAG components initialized successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Save index on shutdown."""
    global retriever
    
    if retriever:
        try:
            vector_store_path = get_vector_store_dir() / "faiss_index"
            retriever.save(str(vector_store_path))
            logger.info("Saved FAISS index on shutdown")
        except Exception as e:
            logger.warning(f"Error saving index: {str(e)}")


# API Endpoints

@app.get("/")
async def root():
    """Serve the web application."""
    return FileResponse(static_dir / "index.html")


@app.get("/api")
async def api_info():
    """Root endpoint."""
    return {
        "message": "RAG Chatbot API",
        "endpoints": {
            "health": "/health",
            "upload": "/upload",
            "query": "/query",
            "chat": "/chat",
            "index_stats": "/index-stats",
            "clear_index": "/clear-index"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "components": {
            "embedder": embedder is not None,
            "retriever": retriever is not None,
            "llm": llm is not None,
            "chunker": chunker is not None
        }
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and index a document.
    Supports PDF and TXT files.
    """
    global retriever, embedder, chunker
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ['.pdf', '.txt']:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported"
        )
    
    try:
        # Save file
        upload_dir = get_upload_dir()
        safe_filename = sanitize_filename(file.filename)
        filepath = upload_dir / safe_filename
        
        contents = await file.read()
        with open(filepath, 'wb') as f:
            f.write(contents)
        
        logger.info(f"Uploaded file: {safe_filename}")
        
        # Load document
        content, doc_name = load_document(str(filepath))
        
        # Chunk document
        chunks = chunker.chunk_text(content, doc_name)
        
        # Embed chunks
        chunks = embedder.embed_chunks(chunks)
        
        # Add to retriever
        retriever.add_chunks(chunks)
        
        return {
            "status": "success",
            "filename": safe_filename,
            "chunks_created": len(chunks),
            "message": f"Document '{doc_name}' uploaded and indexed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the vector store for similar documents.
    """
    global retriever, embedder
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Generate query embedding
        query_embedding = embedder.embed_text(request.query)
        
        # Retrieve similar chunks
        results = retriever.retrieve(query_embedding, top_k=request.top_k)
        
        # Remove embeddings from results for JSON serialization
        for result in results:
            if 'embedding' in result:
                del result['embedding']
        
        return QueryResponse(
            query=request.query,
            results=results,
            total_results=len(results)
        )
        
    except Exception as e:
        logger.error(f"Error querying documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint with RAG.
    Retrieves context and generates answer using LLM.
    """
    global retriever, embedder, llm
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        # Generate query embedding
        query_embedding = embedder.embed_text(request.question)
        
        # Retrieve similar chunks
        retrieved_chunks = retriever.retrieve(query_embedding, top_k=request.top_k)
        
        # Prepare context
        context = "\n".join([chunk['content'] for chunk in retrieved_chunks])
        
        # Generate answer
        answer = llm.generate_with_context(
            context=context,
            query=request.question,
            max_length=request.max_length
        )
        
        # Remove embeddings from results
        source_chunks = []
        for chunk in retrieved_chunks:
            chunk_copy = chunk.copy()
            if 'embedding' in chunk_copy:
                del chunk_copy['embedding']
            source_chunks.append(chunk_copy)
        
        return ChatResponse(
            question=request.question,
            context=context,
            answer=answer,
            source_chunks=source_chunks
        )
        
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/index-stats", response_model=IndexStats)
async def get_index_stats():
    """Get statistics about the current index."""
    global retriever
    
    if not retriever:
        raise HTTPException(status_code=500, detail="Retriever not initialized")
    
    stats = retriever.get_index_stats()
    return IndexStats(**stats)


@app.post("/clear-index")
async def clear_index():
    """Clear the index and start fresh."""
    global retriever, embedder
    
    try:
        if retriever and embedder:
            retriever = FAISSRetriever(embedding_dim=embedder.get_embedding_dimension())
            return {"status": "success", "message": "Index cleared"}
        else:
            raise HTTPException(status_code=500, detail="Components not initialized")
    except Exception as e:
        logger.error(f"Error clearing index: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
