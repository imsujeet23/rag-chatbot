import os
from pathlib import Path
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


def load_pdf(filepath: str) -> str:
    """
    Load text from PDF file.
    
    Args:
        filepath: Path to the PDF file
        
    Returns:
        Extracted text from the PDF
    """
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        logger.info(f"Loaded PDF: {filepath} ({len(text)} characters)")
        return text
    except Exception as e:
        logger.error(f"Error loading PDF {filepath}: {str(e)}")
        raise


def load_text(filepath: str) -> str:
    """
    Load text from TXT file.
    
    Args:
        filepath: Path to the text file
        
    Returns:
        Content of the text file
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        logger.info(f"Loaded TXT: {filepath} ({len(text)} characters)")
        return text
    except Exception as e:
        logger.error(f"Error loading TXT {filepath}: {str(e)}")
        raise


def load_document(filepath: str) -> Tuple[str, str]:
    """
    Load document from file. Supports PDF and TXT formats.
    
    Args:
        filepath: Path to the document
        
    Returns:
        Tuple of (document_content, document_name)
    """
    filepath = str(filepath)
    file_extension = Path(filepath).suffix.lower()
    
    if file_extension == '.pdf':
        content = load_pdf(filepath)
    elif file_extension == '.txt':
        content = load_text(filepath)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}. Supported: .pdf, .txt")
    
    document_name = Path(filepath).stem
    return content, document_name


def load_documents_from_directory(directory: str) -> List[Tuple[str, str]]:
    """
    Load all documents from a directory.
    
    Args:
        directory: Path to directory containing documents
        
    Returns:
        List of tuples (document_content, document_name)
    """
    documents = []
    dir_path = Path(directory)
    
    # Load PDF files
    for pdf_file in dir_path.glob('*.pdf'):
        try:
            content, name = load_document(str(pdf_file))
            documents.append((content, name))
        except Exception as e:
            logger.warning(f"Skipped {pdf_file}: {str(e)}")
    
    # Load text files
    for txt_file in dir_path.glob('*.txt'):
        try:
            content, name = load_document(str(txt_file))
            documents.append((content, name))
        except Exception as e:
            logger.warning(f"Skipped {txt_file}: {str(e)}")
    
    logger.info(f"Loaded {len(documents)} documents from {directory}")
    return documents
