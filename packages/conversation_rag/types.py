"""
Core types and interfaces for the conversation RAG system.

These types are completely portable and framework-agnostic.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, Tuple
import numpy as np
import numpy.typing as npt


@dataclass
class Message:
    """Represents a single message in a conversation."""
    
    id: str
    conversation_id: str
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.role not in ("user", "assistant", "system"):
            raise ValueError(f"Invalid role: {self.role}")


@dataclass
class Chunk:
    """Represents a text chunk with optional embedding."""
    
    id: str
    message_id: str
    conversation_id: str
    content: str
    chunk_index: int
    embedding: Optional[npt.NDArray[np.float32]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.chunk_index < 0:
            raise ValueError(f"Invalid chunk_index: {self.chunk_index}")


@dataclass
class RetrievalResult:
    """Result from a similarity search."""
    
    chunk: Chunk
    message: Message
    similarity_score: float
    rank: int
    
    def __post_init__(self):
        if not 0.0 <= self.similarity_score <= 1.0:
            raise ValueError(f"Invalid similarity_score: {self.similarity_score}")
        if self.rank < 1:
            raise ValueError(f"Invalid rank: {self.rank}")


class EmbeddingProvider(Protocol):
    """Interface for embedding providers."""
    
    def embed(self, texts: List[str]) -> npt.NDArray[np.float32]:
        """
        Embed a batch of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            2D numpy array of shape (len(texts), embedding_dim)
        """
        ...
    
    def embed_query(self, query: str) -> npt.NDArray[np.float32]:
        """
        Embed a single query.
        
        Args:
            query: Query string to embed
            
        Returns:
            1D numpy array of shape (embedding_dim,)
        """
        ...
    
    @property
    def embedding_dim(self) -> int:
        """Dimension of embeddings produced by this provider."""
        ...


class VectorStore(Protocol):
    """Interface for vector stores."""
    
    def add(self, vectors: npt.NDArray[np.float32], ids: List[str]) -> None:
        """
        Add vectors to the store.
        
        Args:
            vectors: 2D numpy array of shape (n, embedding_dim)
            ids: List of chunk IDs corresponding to each vector
        """
        ...
    
    def search(
        self, 
        query_vector: npt.NDArray[np.float32], 
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: 1D numpy array of shape (embedding_dim,)
            top_k: Number of results to return
            
        Returns:
            List of (chunk_id, similarity_score) tuples
        """
        ...
    
    def save(self, path: str) -> None:
        """Save the vector store to disk."""
        ...
    
    def load(self, path: str) -> None:
        """Load the vector store from disk."""
        ...
    
    @property
    def size(self) -> int:
        """Number of vectors in the store."""
        ...


class MessageSource(Protocol):
    """Interface for message sources/adapters."""
    
    def read_messages(self) -> List[Message]:
        """
        Read messages from the source.
        
        Returns:
            List of Message objects
        """
        ...
