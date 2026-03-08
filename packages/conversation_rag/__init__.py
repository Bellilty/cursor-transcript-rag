"""
Portable Conversation History RAG System

A reusable, local-first conversation-history RAG system optimized for 
multilingual retrieval (Hebrew, French, English).
"""

__version__ = "0.1.0"

from .types import Message, Chunk, RetrievalResult, EmbeddingProvider, VectorStore
from .config import Config

__all__ = [
    "Message",
    "Chunk",
    "RetrievalResult",
    "EmbeddingProvider",
    "VectorStore",
    "Config",
]
