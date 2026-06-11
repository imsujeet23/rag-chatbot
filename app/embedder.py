import numpy as np
from typing import List, Union
import logging
import os

logger = logging.getLogger(__name__)


class HuggingFaceEmbedder:
    """Generate embeddings using HuggingFace sentence-transformers."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedder with a HuggingFace model.
        
        Args:
            model_name: Name of the HuggingFace sentence-transformer model
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model_name = model_name
            
            # Set cache directory
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache/huggingface/sentence-transformers")
            os.makedirs(cache_dir, exist_ok=True)
            
            self.model = SentenceTransformer(model_name, cache_folder=cache_dir)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded embedding model: {model_name} (dim: {self.embedding_dim})")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Error embedding text: {str(e)}")
            raise
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            Array of embeddings (n_texts, embedding_dim)
        """
        try:
            embeddings = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"Error embedding texts: {str(e)}")
            raise
    
    def embed_chunks(self, chunks: List[dict]) -> List[dict]:
        """
        Generate embeddings for document chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'content' key
            
        Returns:
            List of chunks with added 'embedding' key
        """
        texts = [chunk['content'] for chunk in chunks]
        embeddings = self.embed_texts(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding
        
        logger.info(f"Generated embeddings for {len(chunks)} chunks")
        return chunks
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return self.embedding_dim
