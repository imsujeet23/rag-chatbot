"""
Example script demonstrating how to use the RAG system locally
(without the FastAPI server).
"""

from app.loader import load_documents_from_directory
from app.chunker import DocumentChunker
from app.embedder import HuggingFaceEmbedder
from app.retreiver import FAISSRetriever
from app.llm import HuggingFaceLLM
from app.utils import get_upload_dir, get_vector_store_dir
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run example RAG pipeline."""
    
    logger.info("Starting RAG System Example...")
    
    # 1. Load documents
    logger.info("\n1. Loading documents...")
    upload_dir = get_upload_dir()
    documents = load_documents_from_directory(str(upload_dir))
    
    if not documents:
        logger.warning(f"No documents found in {upload_dir}")
        logger.info(f"Please add PDF or TXT files to {upload_dir}")
        return
    
    logger.info(f"Loaded {len(documents)} documents")
    
    # 2. Initialize chunker
    logger.info("\n2. Initializing chunker...")
    chunker = DocumentChunker(chunk_size=1000, overlap=200)
    chunks = chunker.chunk_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")
    
    # 3. Initialize embedder
    logger.info("\n3. Initializing embedder...")
    embedder = HuggingFaceEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # 4. Generate embeddings
    logger.info("\n4. Generating embeddings...")
    chunks = embedder.embed_chunks(chunks)
    logger.info(f"Generated embeddings for {len(chunks)} chunks")
    
    # 5. Initialize retriever and add chunks
    logger.info("\n5. Initializing FAISS retriever...")
    retriever = FAISSRetriever(embedding_dim=embedder.get_embedding_dimension())
    retriever.add_chunks(chunks)
    
    # 6. Save index
    logger.info("\n6. Saving index...")
    vector_store_path = get_vector_store_dir() / "faiss_index"
    retriever.save(str(vector_store_path))
    logger.info(f"Saved index to {vector_store_path}")
    
    # 7. Test retrieval
    logger.info("\n7. Testing retrieval...")
    test_query = "What are the main topics covered?"
    query_embedding = embedder.embed_text(test_query)
    results = retriever.retrieve(query_embedding, top_k=3)
    
    logger.info(f"\nQuery: {test_query}")
    logger.info(f"Retrieved {len(results)} chunks:")
    for i, result in enumerate(results, 1):
        logger.info(f"\n  Result {i}:")
        logger.info(f"    Document: {result['document']}")
        logger.info(f"    Content: {result['content'][:100]}...")
        logger.info(f"    Similarity: {result['similarity_score']:.3f}")
    
    # 8. Initialize LLM and test generation
    logger.info("\n8. Initializing LLM...")
    llm = HuggingFaceLLM(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    
    # 9. Test RAG chat
    logger.info("\n9. Testing RAG chat...")
    context = "\n".join([chunk['content'] for chunk in results])
    question = test_query
    
    logger.info(f"Generating answer based on context...")
    answer = llm.generate_with_context(
        context=context,
        query=question,
        max_length=256
    )
    
    logger.info(f"\nQuestion: {question}")
    logger.info(f"Answer: {answer}")
    
    logger.info("\n✅ Example completed successfully!")
    logger.info("\nTo run the API server, execute:")
    logger.info("  uvicorn app.api:app --reload")


if __name__ == "__main__":
    main()
