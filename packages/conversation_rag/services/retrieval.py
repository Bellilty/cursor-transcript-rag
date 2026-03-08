"""
Retrieval service for searching conversation history.

Pipeline: Query → Normalize → Embed → Search → Rerank → Format
"""

from typing import List, Optional
from ..types import RetrievalResult
from ..storage import ConversationRepository
from ..embedding.sentence_transformer import SentenceTransformerProvider
from ..vector_store import FAISSVectorStore
from .normalization import normalize_text
from .reranker import RerankerService


class RetrievalService:
    """Service for retrieving relevant conversation history."""
    
    def __init__(
        self,
        repository: ConversationRepository,
        embedding_provider: SentenceTransformerProvider,
        primary_vector_store: FAISSVectorStore,
        secondary_vector_store: FAISSVectorStore = None,
        min_similarity_threshold: float = 0.5,
        reranker: Optional[RerankerService] = None,
        use_fallback: bool = True,
    ):
        """
        Initialize retrieval service.
        
        Args:
            repository: Storage repository
            embedding_provider: Embedding provider
            primary_vector_store: Primary vector store (implementation-focused)
            secondary_vector_store: Optional secondary vector store (other content)
            min_similarity_threshold: Minimum similarity score for results
            reranker: Optional reranking service
            use_fallback: Whether to fallback to secondary if primary has few results
        """
        self.repository = repository
        self.embedding_provider = embedding_provider
        self.primary_vector_store = primary_vector_store
        self.secondary_vector_store = secondary_vector_store
        self.min_similarity_threshold = min_similarity_threshold
        self.reranker = reranker or RerankerService()
        self.use_fallback = use_fallback
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        conversation_id: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """
        Search for relevant conversation history.
        
        Searches PRIMARY index first, falls back to SECONDARY if needed.
        
        Args:
            query: Search query
            top_k: Number of results to return
            conversation_id: Optional filter by conversation ID
            
        Returns:
            List of retrieval results, sorted by reranked score
        """
        if not query or not query.strip():
            return []
        
        normalized_query = normalize_text(query)
        query_embedding = self.embedding_provider.embed_query(normalized_query)
        
        # Search PRIMARY first
        candidate_k = top_k * 3
        primary_results = self.primary_vector_store.search(query_embedding, top_k=candidate_k)
        
        # Build results from primary
        results = self._build_results(primary_results, conversation_id, "primary")
        
        # If primary has weak results and fallback enabled, try secondary
        if self.use_fallback and self.secondary_vector_store:
            strong_primary = [r for r in results if r.similarity_score > 0.6]
            if len(strong_primary) < top_k // 2:
                secondary_results = self.secondary_vector_store.search(
                    query_embedding, 
                    top_k=candidate_k // 2
                )
                secondary_built = self._build_results(secondary_results, conversation_id, "secondary")
                results.extend(secondary_built)
        
        if not results:
            return []
        
        # Rerank using quality and lexical signals
        reranked_results = self.reranker.rerank(results, query, top_k)
        
        return reranked_results
    
    def _build_results(
        self, 
        search_results: List[Tuple[str, float]],
        conversation_id: Optional[str],
        index_source: str
    ) -> List[RetrievalResult]:
        """Build RetrievalResult objects from search results."""
        if not search_results:
            return []
        
        chunk_ids = [chunk_id for chunk_id, _ in search_results]
        chunks = self.repository.get_chunks_by_ids(chunk_ids)
        
        chunk_map = {chunk.id: chunk for chunk in chunks}
        
        message_ids = list(set(chunk.message_id for chunk in chunks))
        messages = {
            msg.id: msg
            for msg in [self.repository.get_message(msg_id) for msg_id in message_ids]
            if msg is not None
        }
        
        results = []
        for rank, (chunk_id, similarity) in enumerate(search_results, 1):
            if similarity < self.min_similarity_threshold:
                continue
            
            chunk = chunk_map.get(chunk_id)
            if not chunk:
                continue
            
            if conversation_id and chunk.conversation_id != conversation_id:
                continue
            
            message = messages.get(chunk.message_id)
            if not message:
                continue
            
            # Store which index this came from
            chunk.metadata["index_source"] = index_source
            
            result = RetrievalResult(
                chunk=chunk,
                message=message,
                similarity_score=float(similarity),
                rank=rank,
            )
            results.append(result)
        
        return results
    
    def format_for_agent(self, results: List[RetrievalResult]) -> str:
        """
        Format retrieval results for agent consumption.
        
        Args:
            results: List of retrieval results
            
        Returns:
            Formatted markdown string
        """
        if not results:
            return "No relevant conversation history found."
        
        output = [f"## Retrieved Context (top {len(results)} results)\n"]
        
        for result in results:
            output.append(f"### Result {result.rank} (similarity: {result.similarity_score:.2f})")
            output.append(f"**Conversation:** {result.chunk.conversation_id}")
            output.append(f"**Timestamp:** {result.message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            output.append(f"**Role:** {result.message.role}")
            output.append("**Content:**")
            output.append(result.chunk.content)
            output.append("\n---\n")
        
        return "\n".join(output)
