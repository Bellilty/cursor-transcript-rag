#!/usr/bin/env python3
"""
Rebuild FAISS index from existing database.

Useful when changing embedding models or index settings.
"""

from conversation_rag.config import Config
from conversation_rag.storage import ConversationRepository
from conversation_rag.embedding.sentence_transformer import SentenceTransformerProvider
from conversation_rag.vector_store import FAISSVectorStore
from conversation_rag.services import ChunkingService, IngestionService, QualityFilter


def main():
    """Rebuild index."""
    print("=" * 60)
    print("Rebuild FAISS Index")
    print("=" * 60)
    
    config = Config()
    
    print("\nInitializing components...")
    repository = ConversationRepository(config.sqlite_db_path)
    
    message_count = repository.count_messages()
    chunk_count = repository.count_chunks()
    
    print(f"\nDatabase stats:")
    print(f"  - Messages: {message_count}")
    print(f"  - Chunks: {chunk_count}")
    
    if chunk_count == 0:
        print("\nNo chunks in database. Run ingestion first:")
        print("  python scripts/ingest_transcripts.py")
        return
    
    response = input(f"\nRebuild index for {chunk_count} chunks? (y/N): ")
    if response.lower() != 'y':
        print("Cancelled")
        return
    
    print("\nLoading embedding model...")
    embedding_provider = SentenceTransformerProvider(
        model_name=config.embedding_model,
        device="cpu"
    )
    
    print("Creating new vector store...")
    vector_store = FAISSVectorStore(
        embedding_dim=embedding_provider.embedding_dim,
        index_type=config.faiss_index_type,
        index_path=config.faiss_index_path,
        id_mapping_path=config.faiss_id_mapping_path,
    )
    
    chunking_service = ChunkingService(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    
    quality_filter = QualityFilter()
    
    ingestion_service = IngestionService(
        repository=repository,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        chunking_service=chunking_service,
        quality_filter=quality_filter,
    )
    
    print("\nRebuilding index...")
    indexed_count = ingestion_service.rebuild_index()
    
    print("\nSaving index...")
    vector_store.save()
    
    print("\n" + "=" * 60)
    print("✓ Index rebuilt successfully!")
    print("=" * 60)
    print(f"\nIndexed {indexed_count} chunks")
    print(f"Saved to: {config.faiss_index_path}")
    print()


if __name__ == "__main__":
    main()
