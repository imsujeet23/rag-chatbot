"""
Configuration file for RAG Chatbot System.
Centralized settings for all components.
"""

import os
from pathlib import Path

# Environment
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_WORKERS = int(os.getenv("API_WORKERS", 1))

# Model Configuration
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

LLM_DEVICE = os.getenv("LLM_DEVICE", "auto")  # 'cpu', 'cuda', 'auto'

# Chunking Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# Retrieval Configuration
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 5))
MAX_TOP_K = int(os.getenv("MAX_TOP_K", 20))

# LLM Generation Configuration
DEFAULT_MAX_LENGTH = int(os.getenv("DEFAULT_MAX_LENGTH", 512))
MAX_MAX_LENGTH = int(os.getenv("MAX_MAX_LENGTH", 1024))
GENERATION_TEMPERATURE = float(os.getenv("GENERATION_TEMPERATURE", 0.7))
GENERATION_TOP_P = float(os.getenv("GENERATION_TOP_P", 0.9))
GENERATION_TOP_K = int(os.getenv("GENERATION_TOP_K", 50))

# File Upload Configuration
ALLOWED_EXTENSIONS = {".pdf", ".txt"}
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 100))

# Directory Configuration
BASE_DIR = Path(__file__).parent.parent
UPLOADS_DIR = BASE_DIR / "uploads"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
MODELS_DIR = BASE_DIR / "models"
TESTS_DIR = BASE_DIR / "tests"

# Ensure directories exist
UPLOADS_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
TESTS_DIR.mkdir(exist_ok=True)

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# FAISS Configuration
FAISS_INDEX_TYPE = "IndexFlatL2"  # L2 distance metric
FAISS_SAVE_PATH = VECTOR_STORE_DIR / "faiss_index"
