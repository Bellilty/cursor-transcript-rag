"""
SentenceTransformer-based embedding provider.

Supports multiple multilingual models optimized for Hebrew, French, and English.
Default: BAAI/bge-m3 for state-of-the-art multilingual retrieval.
"""

from typing import List
import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer


class SentenceTransformerProvider:
    """Embedding provider using sentence-transformers library."""
    
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu"):
        """
        Initialize the embedding provider.
        
        Args:
            model_name: Name of the sentence-transformer model
                - BAAI/bge-m3: SOTA multilingual (1024-dim, Hebrew/French/English)
                - paraphrase-multilingual-mpnet-base-v2: Alternative (768-dim)
                - all-MiniLM-L6-v2: Lightweight fallback (384-dim)
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._embedding_dim = None
    
    def _ensure_model_loaded(self):
        """Lazy load the model on first use."""
        if self._model is None:
            print(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            test_embedding = self._model.encode(["test"], convert_to_numpy=True)
            self._embedding_dim = test_embedding.shape[1]
            print(f"Model loaded. Embedding dimension: {self._embedding_dim}")
    
    def embed(self, texts: List[str]) -> npt.NDArray[np.float32]:
        """
        Embed a batch of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            2D numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.embedding_dim)
        
        self._ensure_model_loaded()
        
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Normalize for cosine similarity
            show_progress_bar=len(texts) > 10,
            batch_size=32,
        )
        
        return embeddings.astype(np.float32)
    
    def embed_query(self, query: str) -> npt.NDArray[np.float32]:
        """
        Embed a single query.
        
        Args:
            query: Query string to embed
            
        Returns:
            1D numpy array of shape (embedding_dim,)
        """
        self._ensure_model_loaded()
        
        embedding = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        
        return embedding[0].astype(np.float32)
    
    @property
    def embedding_dim(self) -> int:
        """Dimension of embeddings produced by this provider."""
        if self._embedding_dim is None:
            self._ensure_model_loaded()
        return self._embedding_dim
