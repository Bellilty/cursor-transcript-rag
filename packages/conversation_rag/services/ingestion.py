"""
Ingestion service for processing and storing conversation history.

Pipeline: Message → Classify → Filter → Normalize → Chunk → Quality Score → Embed → Store
"""

import hashlib
from typing import List, Dict
from ..types import Message, Chunk
from ..storage import ConversationRepository
from ..embedding.sentence_transformer import SentenceTransformerProvider
from ..vector_store import FAISSVectorStore
from .chunking import ChunkingService
from .normalization import normalize_text
from .quality_filter import QualityFilter
from .message_classifier import MessageClassifier


class IngestionService:
    """Service for ingesting messages and building the vector index."""
    
    def __init__(
        self,
        repository: ConversationRepository,
        embedding_provider: SentenceTransformerProvider,
        primary_vector_store: FAISSVectorStore,
        secondary_vector_store: FAISSVectorStore = None,
        chunking_service: ChunkingService = None,
        quality_filter: QualityFilter = None,
        message_classifier: MessageClassifier = None,
        index_policy: Dict[str, bool] = None,
    ):
        """
        Initialize ingestion service.
        
        Args:
            repository: Storage repository
            embedding_provider: Embedding provider
            primary_vector_store: Primary index for implementation content
            secondary_vector_store: Optional secondary index for other content
            chunking_service: Text chunking service
            quality_filter: Optional quality filter for content
            message_classifier: Optional message classifier
            index_policy: Policy for which source types to index
        """
        self.repository = repository
        self.embedding_provider = embedding_provider
        self.primary_vector_store = primary_vector_store
        self.secondary_vector_store = secondary_vector_store
        self.chunking_service = chunking_service or ChunkingService()
        self.quality_filter = quality_filter or QualityFilter()
        self.message_classifier = message_classifier or MessageClassifier()
        self.index_policy = index_policy or {}
    
    def ingest_message(self, message: Message) -> List[Chunk]:
        """
        Ingest a single message.
        
        Args:
            message: Message to ingest
            
        Returns:
            List of created chunks
        """
        # Classify message source type
        source_type = self.message_classifier.classify_message(message)
        index_namespace = self.message_classifier.get_index_namespace(source_type)
        
        # Check if should index this source type
        if not self._should_index_source_type(source_type):
            return []
        
        # Quality filter: skip low-value messages
        if self.quality_filter.should_skip_message(message):
            return []
        
        self.repository.save_conversation(message.conversation_id, {})
        
        # Store message with source_type and index_namespace metadata
        message.metadata["source_type"] = source_type
        message.metadata["index_namespace"] = index_namespace
        self.repository.save_message(message)
        
        chunk_texts = self.chunking_service.chunk_text(message.content)
        
        if not chunk_texts:
            return []
        
        chunks = []
        for idx, chunk_text in enumerate(chunk_texts):
            # Quality filter: skip low-value chunks
            if self.quality_filter.should_skip_chunk(chunk_text):
                continue
            
            normalized_text = normalize_text(chunk_text)
            
            chunk_id = self._generate_chunk_id(message.id, idx)
            
            # Calculate quality score and metadata for reranking
            quality_data = self.quality_filter.score_chunk_quality(
                chunk_text, message.role
            )
            
            chunk = Chunk(
                id=chunk_id,
                message_id=message.id,
                conversation_id=message.conversation_id,
                content=chunk_text,
                chunk_index=idx,
                metadata={
                    "normalized_content": normalized_text,
                    "message_role": message.role,
                    "message_timestamp": message.timestamp.isoformat(),
                    "source_type": source_type,
                    "index_namespace": index_namespace,  # Add namespace
                    # Quality and type flags
                    "quality_score": quality_data["quality_score"],
                    "is_setup_like": quality_data.get("is_setup_like", False),
                    "is_install_like": quality_data.get("is_install_like", False),
                    "is_docs_summary_like": quality_data.get("is_docs_summary_like", False),
                    "is_project_structure_like": quality_data.get("is_project_structure_like", False),
                    "is_meta_report_like": quality_data.get("is_meta_report_like", False),
                    "is_limitations_like": quality_data.get("is_limitations_like", False),
                    "is_requirement_blob": quality_data.get("is_requirement_blob", False),
                    "is_doc_like": quality_data.get("is_doc_like", False),
                    "is_checklist_like": quality_data.get("is_checklist_like", False),
                    "is_implementation_like": quality_data.get("is_implementation_like", False),
                    "is_decision_like": quality_data.get("is_decision_like", False),
                    "is_explanation_like": quality_data.get("is_explanation_like", False),
                    "is_markdown_heavy": quality_data.get("is_markdown_heavy", False),
                    "is_code_heavy": quality_data.get("is_code_heavy", False),
                    # Diagnostics
                    "content_length": quality_data.get("content_length", len(chunk_text)),
                    "heading_count": quality_data.get("heading_count", 0),
                    "bullet_count": quality_data.get("bullet_count", 0),
                    "code_block_count": quality_data.get("code_block_count", 0),
                    "file_path_reference_count": quality_data.get("file_path_reference_count", 0),
                    "technical_term_count": quality_data.get("technical_term_count", 0),
                    "setup_term_count": quality_data.get("setup_term_count", 0),
                    "doc_term_count": quality_data.get("doc_term_count", 0),
                },
            )
            chunks.append(chunk)
            self.repository.save_chunk(chunk)
        
        if not chunks:
            return []
        
        normalized_texts = [c.metadata["normalized_content"] for c in chunks]
        embeddings = self.embedding_provider.embed(normalized_texts)
        
        chunk_ids = [c.id for c in chunks]
        
        # Add to appropriate index based on namespace
        if index_namespace == "primary":
            self.primary_vector_store.add(embeddings, chunk_ids)
        elif self.secondary_vector_store:
            self.secondary_vector_store.add(embeddings, chunk_ids)
        
        for i, chunk in enumerate(chunks):
            chunk.metadata["vector_index"] = i
        
        return chunks
    
    def _should_index_source_type(self, source_type: str) -> bool:
        """Check if source type should be indexed based on policy."""
        # Map source types to policy keys
        policy_map = {
            "user_requirement_prompt": "index_include_requirement_prompts",
            "setup_install_doc": "index_include_setup",
            "generated_project_doc": "index_include_docs",
            "progress_chatter": "index_include_progress_chatter",
            "status_report": "index_include_status_reports",
            "limitations_meta": "index_include_limitations",
        }
        
        policy_key = policy_map.get(source_type)
        if policy_key:
            # Check policy (default False for these types)
            return self.index_policy.get(policy_key, False)
        
        # High-signal types always indexed
        return True
    
    def ingest_conversation(self, messages: List[Message]) -> Dict:
        """
        Ingest multiple messages from a conversation.
        
        Args:
            messages: List of messages to ingest
            
        Returns:
            Dict with statistics including source type and namespace breakdown
        """
        stats = {
            "total_messages": len(messages),
            "by_source_type": {},
            "indexed": {},
            "skipped": {},
            "total_chunks": 0,
            "primary_chunks": 0,
            "secondary_chunks": 0,
        }
        
        for message in messages:
            # Classify first
            source_type = self.message_classifier.classify_message(message)
            index_namespace = self.message_classifier.get_index_namespace(source_type)
            
            # Track by source type
            stats["by_source_type"][source_type] = stats["by_source_type"].get(source_type, 0) + 1
            
            # Try to ingest
            chunks = self.ingest_message(message)
            
            if chunks:
                stats["indexed"][source_type] = stats["indexed"].get(source_type, 0) + 1
                stats["total_chunks"] += len(chunks)
                
                # Track by namespace
                if index_namespace == "primary":
                    stats["primary_chunks"] += len(chunks)
                else:
                    stats["secondary_chunks"] += len(chunks)
            else:
                stats["skipped"][source_type] = stats["skipped"].get(source_type, 0) + 1
        
        return stats
    
    def rebuild_index(self) -> int:
        """
        Rebuild the vector index from all stored chunks.
        
        Returns:
            Number of chunks indexed
        """
        print("Rebuilding vector index from database...")
        
        chunks = self.repository.get_all_chunks()
        
        if not chunks:
            print("No chunks found in database")
            return 0
        
        normalized_texts = []
        chunk_ids = []
        
        for chunk in chunks:
            normalized_text = chunk.metadata.get("normalized_content")
            if not normalized_text:
                normalized_text = normalize_text(chunk.content)
                chunk.metadata["normalized_content"] = normalized_text
            
            normalized_texts.append(normalized_text)
            chunk_ids.append(chunk.id)
        
        print(f"Embedding {len(chunks)} chunks...")
        embeddings = self.embedding_provider.embed(normalized_texts)
        
        print("Adding to vector store...")
        self.vector_store.add(embeddings, chunk_ids)
        
        print(f"Index rebuilt with {len(chunks)} chunks")
        return len(chunks)
    
    def _generate_chunk_id(self, message_id: str, chunk_index: int) -> str:
        """Generate unique chunk ID."""
        content = f"{message_id}:{chunk_index}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
