import os
import logging
from pathlib import Path
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_upload_dir() -> Path:
    """Get the upload directory path."""
    upload_dir = Path(__file__).parent.parent / "uploads"
    upload_dir.mkdir(exist_ok=True)
    return upload_dir


def get_vector_store_dir() -> Path:
    """Get the vector store directory path."""
    vector_dir = Path(__file__).parent.parent / "vector_store"
    vector_dir.mkdir(exist_ok=True)
    return vector_dir


def get_models_dir() -> Path:
    """Get the models directory path."""
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    return models_dir


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to remove unsafe characters."""
    import re
    return re.sub(r'[^a-zA-Z0-9._-]', '_', filename)


def get_file_size_mb(filepath: str) -> float:
    """Get file size in MB."""
    return os.path.getsize(filepath) / (1024 * 1024)
