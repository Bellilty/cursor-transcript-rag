"""
Configuration management for the conversation RAG system.

Loads settings from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional


@dataclass
class Config:
    """Configuration for the conversation RAG system."""
    
    # Embedding settings
    embedding_provider: str = "sentence-transformers"
    embedding_model: str = "BAAI/bge-m3"
    
    # Vector store settings
    vector_store: str = "faiss"
    faiss_index_type: str = "flat"  # flat | ivf
    
    # Storage settings
    data_dir: Path = Path("./data")
    sqlite_db_path: Optional[Path] = None
    
    # Chunking settings
    chunk_size: int = 256  # Reduced for more focused chunks
    chunk_overlap: int = 64  # Reduced proportionally
    
    # Retrieval settings
    default_top_k: int = 5
    min_similarity_threshold: float = 0.5
    
    # Indexing policy (source type filtering)
    index_include_requirement_prompts: bool = False
    index_include_docs: bool = False
    index_include_progress_chatter: bool = False
    index_include_setup: bool = False
    index_include_status_reports: bool = False
    index_include_limitations: bool = False
    
    # Cursor transcripts
    cursor_transcripts_dir: Path = Path.home() / ".cursor/projects"
    auto_ingest_on_startup: bool = False
    
    def __post_init__(self):
        """Load configuration from environment variables."""
        self.embedding_provider = os.getenv(
            "EMBEDDING_PROVIDER", self.embedding_provider
        )
        self.embedding_model = os.getenv(
            "EMBEDDING_MODEL", self.embedding_model
        )
        self.vector_store = os.getenv("VECTOR_STORE", self.vector_store)
        self.faiss_index_type = os.getenv(
            "FAISS_INDEX_TYPE", self.faiss_index_type
        )
        
        data_dir_env = os.getenv("DATA_DIR")
        if data_dir_env:
            self.data_dir = Path(data_dir_env)
        
        sqlite_path_env = os.getenv("SQLITE_DB_PATH")
        if sqlite_path_env:
            self.sqlite_db_path = Path(sqlite_path_env)
        else:
            self.sqlite_db_path = self.data_dir / "conversations.db"
        
        chunk_size_env = os.getenv("CHUNK_SIZE")
        if chunk_size_env:
            self.chunk_size = int(chunk_size_env)
        
        chunk_overlap_env = os.getenv("CHUNK_OVERLAP")
        if chunk_overlap_env:
            self.chunk_overlap = int(chunk_overlap_env)
        
        top_k_env = os.getenv("DEFAULT_TOP_K")
        if top_k_env:
            self.default_top_k = int(top_k_env)
        
        threshold_env = os.getenv("MIN_SIMILARITY_THRESHOLD")
        if threshold_env:
            self.min_similarity_threshold = float(threshold_env)
        
        # Indexing policy
        self.index_include_requirement_prompts = os.getenv(
            "INDEX_INCLUDE_REQUIREMENT_PROMPTS", "false"
        ).lower() in ("true", "1", "yes")
        
        self.index_include_docs = os.getenv(
            "INDEX_INCLUDE_DOCS", "false"
        ).lower() in ("true", "1", "yes")
        
        self.index_include_progress_chatter = os.getenv(
            "INDEX_INCLUDE_PROGRESS_CHATTER", "false"
        ).lower() in ("true", "1", "yes")
        
        self.index_include_setup = os.getenv(
            "INDEX_INCLUDE_SETUP", "false"
        ).lower() in ("true", "1", "yes")
        
        self.index_include_status_reports = os.getenv(
            "INDEX_INCLUDE_STATUS_REPORTS", "false"
        ).lower() in ("true", "1", "yes")
        
        self.index_include_limitations = os.getenv(
            "INDEX_INCLUDE_LIMITATIONS", "false"
        ).lower() in ("true", "1", "yes")
        
        cursor_dir_env = os.getenv("CURSOR_TRANSCRIPTS_DIR")
        if cursor_dir_env:
            self.cursor_transcripts_dir = Path(cursor_dir_env).expanduser()
        
        auto_ingest_env = os.getenv("AUTO_INGEST_ON_STARTUP")
        if auto_ingest_env:
            self.auto_ingest_on_startup = auto_ingest_env.lower() in ("true", "1", "yes")
        
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "faiss").mkdir(exist_ok=True)
    
    @property
    def faiss_index_path(self) -> Path:
        """Path to FAISS index file."""
        return self.data_dir / "faiss" / "index.faiss"
    
    @property
    def faiss_id_mapping_path(self) -> Path:
        """Path to FAISS ID mapping file."""
        return self.data_dir / "faiss" / "id_mapping.json"
    
    @property
    def faiss_metadata_path(self) -> Path:
        """Path to FAISS metadata file."""
        return self.data_dir / "faiss" / "metadata.json"
