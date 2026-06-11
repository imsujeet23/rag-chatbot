import numpy as np
import faiss
import pickle
import logging
from pathlib import Path
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class FAISSRetriever:
    """FAISS-based vector retriever for RAG."""
    
    def __init__(self, embedding_dim: int):
        """
        Initialize FAISS retriever.
        
        Args:
            embedding_dim: Dimension of embeddings
        """
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.chunks = []
        logger.info(f"Initialized FAISS retriever with embedding dimension: {embedding_dim}")
    
    def add_chunks(self, chunks: List[dict]) -> None:
        """
        Add chunks with embeddings to the index.
        
        Args:
            chunks: List of chunk dictionaries with 'embedding' key
        """
        embeddings = np.array([chunk['embedding'] for chunk in chunks]).astype(np.float32)
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store chunk metadata
        self.chunks.extend(chunks)
        
        logger.info(f"Added {len(chunks)} chunks to FAISS index. Total: {len(self.chunks)}")
    
    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5) -> List[dict]:
        """
        Retrieve top-k similar chunks for a query embedding.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to retrieve
            
        Returns:
            List of retrieved chunks with similarity scores
        """
        if len(self.chunks) == 0:
            logger.warning("No chunks in index. Returning empty results.")
            return []
        
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            chunk = self.chunks[int(idx)].copy()
            chunk['distance'] = float(distance)
            chunk['similarity_score'] = 1.0 / (1.0 + float(distance))  # Convert distance to similarity
            results.append(chunk)
        
        logger.info(f"Retrieved {len(results)} chunks for query")
        return results
    
    def retrieve_batch(self, query_embeddings: np.ndarray, top_k: int = 5) -> List[List[dict]]:
        """
        Retrieve top-k similar chunks for multiple query embeddings.
        
        Args:
            query_embeddings: Array of query embeddings (n_queries, embedding_dim)
            top_k: Number of top results per query
            
        Returns:
            List of result lists, one per query
        """
        query_embeddings = query_embeddings.astype(np.float32)
        distances, indices = self.index.search(query_embeddings, min(top_k, len(self.chunks)))
        
        all_results = []
        for dist_row, idx_row in zip(distances, indices):
            results = []
            for idx, distance in zip(idx_row, dist_row):
                chunk = self.chunks[int(idx)].copy()
                chunk['distance'] = float(distance)
                chunk['similarity_score'] = 1.0 / (1.0 + float(distance))
                results.append(chunk)
            all_results.append(results)
        
        return all_results
    
    def save(self, filepath: str) -> None:
        """
        Save the index and chunks to disk.
        
        Args:
            filepath: Path to save the index (without extension)
        """
        try:
            filepath = str(filepath)
            # Save FAISS index
            faiss.write_index(self.index, filepath + '.index')
            
            # Save chunks metadata
            with open(filepath + '.chunks', 'wb') as f:
                pickle.dump(self.chunks, f)
            
            logger.info(f"Saved FAISS index and chunks to {filepath}")
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            raise
    
    def load(self, filepath: str) -> None:
        """
        Load index and chunks from disk.
        
        Args:
            filepath: Path to load the index from (without extension)
        """
        try:
            filepath = str(filepath)
            # Load FAISS index
            self.index = faiss.read_index(filepath + '.index')
            
            # Load chunks metadata
            with open(filepath + '.chunks', 'rb') as f:
                self.chunks = pickle.load(f)
            
            logger.info(f"Loaded FAISS index and chunks from {filepath}")
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            raise
    
    def get_index_stats(self) -> dict:
        """Get statistics about the current index."""
        return {
            'total_chunks': len(self.chunks),
            'embedding_dim': self.embedding_dim,
            'index_size': self.index.ntotal
        }
