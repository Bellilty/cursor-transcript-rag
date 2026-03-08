"""
FAISS-based vector store implementation.

Provides efficient similarity search with persistence support.
"""

import json
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import numpy.typing as npt
import faiss


class FAISSVectorStore:
    """Vector store using FAISS for similarity search."""
    
    def __init__(
        self,
        embedding_dim: int,
        index_type: str = "flat",
        index_path: Optional[Path] = None,
        id_mapping_path: Optional[Path] = None,
    ):
        """
        Initialize FAISS vector store.
        
        Args:
            embedding_dim: Dimension of embeddings
            index_type: Type of FAISS index ('flat' or 'ivf')
            index_path: Path to save/load index
            id_mapping_path: Path to save/load ID mapping
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index_path = index_path
        self.id_mapping_path = id_mapping_path
        
        self._index = None
        self._id_to_idx: dict = {}
        self._idx_to_id: dict = {}
        self._next_idx = 0
        
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize or load FAISS index."""
        if self.index_type == "flat":
            self._index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine similarity)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self._index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
    
    def add(self, vectors: npt.NDArray[np.float32], ids: List[str]) -> None:
        """
        Add vectors to the store.
        
        Args:
            vectors: 2D numpy array of shape (n, embedding_dim)
            ids: List of chunk IDs corresponding to each vector
        """
        if len(ids) != len(vectors):
            raise ValueError(f"Mismatch: {len(ids)} ids but {len(vectors)} vectors")
        
        if vectors.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} doesn't match index dimension {self.embedding_dim}"
            )
        
        # Train index if needed (for IVF)
        if self.index_type == "ivf" and not self._index.is_trained:
            self._index.train(vectors)
        
        # Add vectors
        start_idx = self._next_idx
        self._index.add(vectors)
        
        # Update mappings
        for i, chunk_id in enumerate(ids):
            idx = start_idx + i
            self._id_to_idx[chunk_id] = idx
            self._idx_to_id[idx] = chunk_id
        
        self._next_idx += len(ids)
    
    def search(
        self,
        query_vector: npt.NDArray[np.float32],
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: 1D numpy array of shape (embedding_dim,)
            top_k: Number of results to return
            
        Returns:
            List of (chunk_id, similarity_score) tuples, sorted by similarity
        """
        if self._index.ntotal == 0:
            return []
        
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Search
        k = min(top_k, self._index.ntotal)
        distances, indices = self._index.search(query_vector, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty results
                continue
            
            chunk_id = self._idx_to_id.get(int(idx))
            if chunk_id:
                # Convert inner product distance to similarity score (0-1)
                similarity = float(dist)
                results.append((chunk_id, similarity))
        
        return results
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save the vector store to disk.
        
        Args:
            path: Optional override for index path
        """
        index_path = Path(path) if path else self.index_path
        if not index_path:
            raise ValueError("No index path specified")
        
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self._index, str(index_path))
        
        if self.id_mapping_path:
            mapping_data = {
                "id_to_idx": self._id_to_idx,
                "idx_to_id": {str(k): v for k, v in self._idx_to_id.items()},
                "next_idx": self._next_idx,
            }
            with open(self.id_mapping_path, "w") as f:
                json.dump(mapping_data, f, indent=2)
    
    def load(self, path: Optional[str] = None) -> None:
        """
        Load the vector store from disk.
        
        Args:
            path: Optional override for index path
        """
        index_path = Path(path) if path else self.index_path
        if not index_path or not index_path.exists():
            raise ValueError(f"Index file not found: {index_path}")
        
        self._index = faiss.read_index(str(index_path))
        
        if self.id_mapping_path and self.id_mapping_path.exists():
            with open(self.id_mapping_path, "r") as f:
                mapping_data = json.load(f)
            
            self._id_to_idx = mapping_data["id_to_idx"]
            self._idx_to_id = {int(k): v for k, v in mapping_data["idx_to_id"].items()}
            self._next_idx = mapping_data["next_idx"]
        else:
            print("Warning: ID mapping not found, creating new mapping")
            self._id_to_idx = {}
            self._idx_to_id = {}
            self._next_idx = self._index.ntotal
    
    @property
    def size(self) -> int:
        """Number of vectors in the store."""
        return self._index.ntotal if self._index else 0
