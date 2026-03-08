#!/usr/bin/env python3
"""
Ingest Cursor transcript files into the RAG system.

Scans ~/.cursor/projects/.../agent-transcripts/ and indexes all conversations.
"""

import sys
from pathlib import Path

# Add adapters to path (not installed as package)
sys.path.insert(0, str(Path(__file__).parent.parent / "adapters"))

from conversation_rag.config import Config
from conversation_rag.storage import ConversationRepository
from conversation_rag.embedding.sentence_transformer import SentenceTransformerProvider
from conversation_rag.vector_store import FAISSVectorStore
from conversation_rag.services import ChunkingService, IngestionService, QualityFilter, MessageClassifier

from cursor_transcripts import CursorTranscriptScanner


def main():
    """Run ingestion."""
    print("=" * 60)
    print("Cursor Transcript Ingestion (Source-Level Filtering)")
    print("=" * 60)
    
    config = Config()
    
    print(f"\nIndexing Policy:")
    print(f"  - Requirement prompts: {config.index_include_requirement_prompts}")
    print(f"  - Docs/summaries: {config.index_include_docs}")
    print(f"  - Setup/install: {config.index_include_setup}")
    print(f"  - Progress chatter: {config.index_include_progress_chatter}")
    print(f"  - Status reports: {config.index_include_status_reports}")
    print(f"  - Limitations: {config.index_include_limitations}")
    
    print(f"\nScanning: {config.cursor_transcripts_dir}")
    scanner = CursorTranscriptScanner(config.cursor_transcripts_dir)
    
    transcript_files = scanner.find_transcript_files()
    print(f"Found {len(transcript_files)} transcript files")
    
    if not transcript_files:
        print("\nNo transcript files found.")
        print(f"Searched in: {config.cursor_transcripts_dir}")
        print("Expected structure:")
        print("  - agent-transcripts/*.jsonl")
        print("  - agent-transcripts/<conversation_id>/<conversation_id>.jsonl")
        print("Tip: Check if transcripts exist:")
        print(f"  find {config.cursor_transcripts_dir} -name '*.jsonl' 2>/dev/null")
        return
    
    print("\nInitializing RAG system...")
    repository = ConversationRepository(config.sqlite_db_path)
    
    embedding_provider = SentenceTransformerProvider(
        model_name=config.embedding_model,
        device="cpu"
    )
    
    # Create PRIMARY index (implementation-focused)
    primary_vector_store = FAISSVectorStore(
        embedding_dim=embedding_provider.embedding_dim,
        index_type=config.faiss_index_type,
        index_path=config.data_dir / "faiss" / "primary_index.faiss",
        id_mapping_path=config.data_dir / "faiss" / "primary_id_mapping.json",
    )
    
    # Create SECONDARY index (other content)
    secondary_vector_store = FAISSVectorStore(
        embedding_dim=embedding_provider.embedding_dim,
        index_type=config.faiss_index_type,
        index_path=config.data_dir / "faiss" / "secondary_index.faiss",
        id_mapping_path=config.data_dir / "faiss" / "secondary_id_mapping.json",
    )
    
    if (config.data_dir / "faiss" / "primary_index.faiss").exists():
        print("Loading existing PRIMARY index...")
        try:
            primary_vector_store.load()
            print(f"Loaded {primary_vector_store.size} vectors from PRIMARY")
        except Exception as e:
            print(f"Warning: Could not load PRIMARY index: {e}")
    
    if (config.data_dir / "faiss" / "secondary_index.faiss").exists():
        print("Loading existing SECONDARY index...")
        try:
            secondary_vector_store.load()
            print(f"Loaded {secondary_vector_store.size} vectors from SECONDARY")
        except Exception as e:
            print(f"Warning: Could not load SECONDARY index: {e}")
    
    chunking_service = ChunkingService(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    
    quality_filter = QualityFilter()
    message_classifier = MessageClassifier()
    
    # Build index policy from config
    index_policy = {
        "index_include_requirement_prompts": config.index_include_requirement_prompts,
        "index_include_docs": config.index_include_docs,
        "index_include_progress_chatter": config.index_include_progress_chatter,
        "index_include_setup": config.index_include_setup,
        "index_include_status_reports": config.index_include_status_reports,
        "index_include_limitations": config.index_include_limitations,
    }
    
    ingestion_service = IngestionService(
        repository=repository,
        embedding_provider=embedding_provider,
        primary_vector_store=primary_vector_store,
        secondary_vector_store=secondary_vector_store,
        chunking_service=chunking_service,
        quality_filter=quality_filter,
        message_classifier=message_classifier,
        index_policy=index_policy,
    )
    
    print("\nIngesting transcripts...")
    total_stats = {
        "by_source_type": {},
        "indexed": {},
        "skipped": {},
        "total_chunks": 0,
        "primary_chunks": 0,
        "secondary_chunks": 0,
    }
    
    for i, adapter in enumerate(scanner.iter_adapters(), 1):
        print(f"\n[{i}/{len(transcript_files)}] {adapter.transcript_path.name}")
        
        try:
            messages = adapter.read_messages()
            
            if not messages:
                print("  No messages found, skipping")
                continue
            
            stats = ingestion_service.ingest_conversation(messages)
            
            # Aggregate stats
            for source_type, count in stats["by_source_type"].items():
                total_stats["by_source_type"][source_type] = \
                    total_stats["by_source_type"].get(source_type, 0) + count
            
            for source_type, count in stats["indexed"].items():
                total_stats["indexed"][source_type] = \
                    total_stats["indexed"].get(source_type, 0) + count
            
            for source_type, count in stats["skipped"].items():
                total_stats["skipped"][source_type] = \
                    total_stats["skipped"].get(source_type, 0) + count
            
            total_stats["total_chunks"] += stats["total_chunks"]
            total_stats["primary_chunks"] += stats["primary_chunks"]
            total_stats["secondary_chunks"] += stats["secondary_chunks"]
            
            print(f"  ✓ {stats['total_messages']} messages → {stats['total_chunks']} chunks")
            print(f"    PRIMARY: {stats['primary_chunks']}, SECONDARY: {stats['secondary_chunks']}")
            if stats["skipped"]:
                skipped_types = ", ".join(f"{k}:{v}" for k, v in stats["skipped"].items())
                print(f"    Skipped: {skipped_types}")
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    print("\nSaving indexes...")
    primary_vector_store.save()
    secondary_vector_store.save()
    
    print("\n" + "=" * 60)
    print("✓ Ingestion complete!")
    print("=" * 60)
    
    print(f"\nSource Type Distribution:")
    for source_type, count in sorted(total_stats["by_source_type"].items()):
        indexed = total_stats["indexed"].get(source_type, 0)
        skipped = total_stats["skipped"].get(source_type, 0)
        print(f"  {source_type}:")
        print(f"    Total: {count}, Indexed: {indexed}, Skipped: {skipped}")
    
    print(f"\nIndex Statistics:")
    print(f"  - PRIMARY chunks: {total_stats['primary_chunks']}")
    print(f"  - PRIMARY size: {primary_vector_store.size} vectors")
    print(f"  - SECONDARY chunks: {total_stats['secondary_chunks']}")
    print(f"  - SECONDARY size: {secondary_vector_store.size} vectors")
    print(f"  - Total chunks indexed: {total_stats['total_chunks']}")
    print(f"\nIndexes saved to: {config.data_dir / 'faiss'}")
    print()


if __name__ == "__main__":
    main()
