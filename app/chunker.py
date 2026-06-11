from typing import List
import logging

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Document chunking utility for RAG systems."""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Maximum characters per chunk
            overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, document_name: str = "") -> List[dict]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            document_name: Name of the document for metadata
            
        Returns:
            List of chunk dictionaries with content and metadata
        """
        chunks = []
        text = text.strip()
        
        # Split by sentences first (rough split)
        sentences = text.replace('\n', ' ').split('.')
        
        current_chunk = ""
        chunk_num = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Add period back
            sentence_with_period = sentence + "."
            
            # Check if adding this sentence exceeds chunk size
            if len(current_chunk) + len(sentence_with_period) > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    'content': current_chunk.strip(),
                    'document': document_name,
                    'chunk_index': chunk_num,
                    'chunk_size': len(current_chunk)
                })
                chunk_num += 1
                
                # Keep overlap
                overlap_text = current_chunk[-self.overlap:] if len(current_chunk) > self.overlap else current_chunk
                current_chunk = overlap_text + " " + sentence_with_period
            else:
                current_chunk += " " + sentence_with_period if current_chunk else sentence_with_period
        
        # Add last chunk
        if current_chunk.strip():
            chunks.append({
                'content': current_chunk.strip(),
                'document': document_name,
                'chunk_index': chunk_num,
                'chunk_size': len(current_chunk)
            })
        
        logger.info(f"Created {len(chunks)} chunks from document '{document_name}'")
        return chunks
    
    def chunk_documents(self, documents: List[tuple]) -> List[dict]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of tuples (content, document_name)
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        for content, doc_name in documents:
            chunks = self.chunk_text(content, doc_name)
            all_chunks.extend(chunks)
        
        logger.info(f"Chunked {len(documents)} documents into {len(all_chunks)} total chunks")
        return all_chunks
