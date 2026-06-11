"""
Test suite for RAG Chatbot components.
Run with: python -m pytest tests/
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import os

# Import RAG components
from app.chunker import DocumentChunker
from app.embedder import HuggingFaceEmbedder
from app.retreiver import FAISSRetriever


class TestDocumentChunker:
    """Tests for DocumentChunker."""
    
    @pytest.fixture
    def chunker(self):
        return DocumentChunker(chunk_size=100, overlap=20)
    
    def test_chunk_text_basic(self, chunker):
        """Test basic text chunking."""
        text = "This is a test. " * 20  # Create longer text
        chunks = chunker.chunk_text(text, "test_doc")
        
        assert len(chunks) > 0
        assert all('content' in chunk for chunk in chunks)
        assert all('document' in chunk for chunk in chunks)
        assert all(chunk['document'] == 'test_doc' for chunk in chunks)
    
    def test_chunk_metadata(self, chunker):
        """Test that chunks contain proper metadata."""
        text = "Test sentence. " * 30
        chunks = chunker.chunk_text(text, "doc")
        
        assert all('chunk_index' in chunk for chunk in chunks)
        assert all('chunk_size' in chunk for chunk in chunks)
        assert all(isinstance(chunk['chunk_index'], int) for chunk in chunks)
    
    def test_empty_text(self, chunker):
        """Test chunking empty text."""
        chunks = chunker.chunk_text("", "empty")
        assert len(chunks) == 0
    
    def test_chunk_documents(self, chunker):
        """Test chunking multiple documents."""
        docs = [
            ("First document text. " * 10, "doc1"),
            ("Second document text. " * 10, "doc2")
        ]
        chunks = chunker.chunk_documents(docs)
        
        assert len(chunks) > 0
        doc_names = {chunk['document'] for chunk in chunks}
        assert 'doc1' in doc_names
        assert 'doc2' in doc_names


class TestHuggingFaceEmbedder:
    """Tests for HuggingFaceEmbedder."""
    
    @pytest.fixture
    def embedder(self):
        return HuggingFaceEmbedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    
    def test_embed_text(self, embedder):
        """Test single text embedding."""
        text = "This is a test sentence."
        embedding = embedder.embed_text(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (embedder.embedding_dim,)
        assert embedding.dtype == np.float32
    
    def test_embed_texts(self, embedder):
        """Test batch text embedding."""
        texts = [
            "First sentence.",
            "Second sentence.",
            "Third sentence."
        ]
        embeddings = embedder.embed_texts(texts)
        
        assert embeddings.shape == (3, embedder.embedding_dim)
        assert embeddings.dtype == np.float32
    
    def test_embed_chunks(self, embedder):
        """Test embedding document chunks."""
        chunks = [
            {'content': 'Chunk 1', 'document': 'doc1', 'chunk_index': 0},
            {'content': 'Chunk 2', 'document': 'doc1', 'chunk_index': 1},
            {'content': 'Chunk 3', 'document': 'doc2', 'chunk_index': 0},
        ]
        embedded_chunks = embedder.embed_chunks(chunks)
        
        assert len(embedded_chunks) == 3
        assert all('embedding' in chunk for chunk in embedded_chunks)
        assert all(chunk['embedding'].shape == (embedder.embedding_dim,) 
                  for chunk in embedded_chunks)
    
    def test_embedding_dimension(self, embedder):
        """Test that embedding dimension is consistent."""
        dim = embedder.get_embedding_dimension()
        assert isinstance(dim, int)
        assert dim > 0


class TestFAISSRetriever:
    """Tests for FAISSRetriever."""
    
    @pytest.fixture
    def retriever(self):
        return FAISSRetriever(embedding_dim=384)
    
    def test_add_chunks(self, retriever):
        """Test adding chunks to retriever."""
        embeddings = np.random.rand(5, 384).astype(np.float32)
        chunks = [
            {
                'content': f'Chunk {i}',
                'document': 'test_doc',
                'chunk_index': i,
                'embedding': emb
            }
            for i, emb in enumerate(embeddings)
        ]
        
        retriever.add_chunks(chunks)
        assert len(retriever.chunks) == 5
        assert retriever.index.ntotal == 5
    
    def test_retrieve(self, retriever):
        """Test chunk retrieval."""
        embeddings = np.random.rand(10, 384).astype(np.float32)
        chunks = [
            {
                'content': f'Chunk {i}',
                'document': 'doc',
                'chunk_index': i,
                'embedding': emb
            }
            for i, emb in enumerate(embeddings)
        ]
        
        retriever.add_chunks(chunks)
        
        # Query with first embedding
        results = retriever.retrieve(embeddings[0], top_k=3)
        
        assert len(results) == 3
        assert all('similarity_score' in result for result in results)
        assert all(0 <= result['similarity_score'] <= 1 for result in results)
    
    def test_save_load(self, retriever):
        """Test saving and loading index."""
        embeddings = np.random.rand(5, 384).astype(np.float32)
        chunks = [
            {
                'content': f'Chunk {i}',
                'document': 'doc',
                'chunk_index': i,
                'embedding': emb
            }
            for i, emb in enumerate(embeddings)
        ]
        
        retriever.add_chunks(chunks)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_index"
            
            # Save
            retriever.save(str(save_path))
            assert (Path(tmpdir) / "test_index.index").exists()
            assert (Path(tmpdir) / "test_index.chunks").exists()
            
            # Load
            new_retriever = FAISSRetriever(embedding_dim=384)
            new_retriever.load(str(save_path))
            
            assert len(new_retriever.chunks) == 5
            assert new_retriever.index.ntotal == 5
    
    def test_get_index_stats(self, retriever):
        """Test index statistics."""
        embeddings = np.random.rand(3, 384).astype(np.float32)
        chunks = [
            {
                'content': f'Chunk {i}',
                'document': 'doc',
                'chunk_index': i,
                'embedding': emb
            }
            for i, emb in enumerate(embeddings)
        ]
        
        retriever.add_chunks(chunks)
        stats = retriever.get_index_stats()
        
        assert stats['total_chunks'] == 3
        assert stats['embedding_dim'] == 384
        assert stats['index_size'] == 3


class TestIntegration:
    """Integration tests for RAG pipeline."""
    
    def test_end_to_end_pipeline(self):
        """Test complete RAG pipeline."""
        # Initialize components
        chunker = DocumentChunker(chunk_size=50, overlap=10)
        embedder = HuggingFaceEmbedder()
        retriever = FAISSRetriever(
            embedding_dim=embedder.get_embedding_dimension()
        )
        
        # Process document
        doc_content = "This is a test document. " * 20
        chunks = chunker.chunk_text(doc_content, "test_doc")
        chunks = embedder.embed_chunks(chunks)
        retriever.add_chunks(chunks)
        
        # Query
        query = "test document"
        query_emb = embedder.embed_text(query)
        results = retriever.retrieve(query_emb, top_k=3)
        
        assert len(results) > 0
        assert len(results) <= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
