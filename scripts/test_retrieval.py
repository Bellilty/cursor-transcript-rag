#!/usr/bin/env python3
"""
Test end-to-end retrieval quality.

Runs sample queries and evaluates retrieval results with reranking details.
"""

from conversation_rag.config import Config
from conversation_rag.storage import ConversationRepository
from conversation_rag.embedding.sentence_transformer import SentenceTransformerProvider
from conversation_rag.vector_store import FAISSVectorStore
from conversation_rag.services import RetrievalService


def main():
    """Test retrieval."""
    print("=" * 60)
    print("Retrieval Quality Test (with Reranking)")
    print("=" * 60)
    
    config = Config()
    
    print("\nInitializing RAG system...")
    repository = ConversationRepository(config.sqlite_db_path)
    
    embedding_provider = SentenceTransformerProvider(
        model_name=config.embedding_model,
        device="cpu"
    )
    
    # Load PRIMARY index
    primary_vector_store = FAISSVectorStore(
        embedding_dim=embedding_provider.embedding_dim,
        index_type=config.faiss_index_type,
        index_path=config.data_dir / "faiss" / "primary_index.faiss",
        id_mapping_path=config.data_dir / "faiss" / "primary_id_mapping.json",
    )
    
    # Load SECONDARY index
    secondary_vector_store = FAISSVectorStore(
        embedding_dim=embedding_provider.embedding_dim,
        index_type=config.faiss_index_type,
        index_path=config.data_dir / "faiss" / "secondary_index.faiss",
        id_mapping_path=config.data_dir / "faiss" / "secondary_id_mapping.json",
    )
    
    primary_path = config.data_dir / "faiss" / "primary_index.faiss"
    secondary_path = config.data_dir / "faiss" / "secondary_index.faiss"
    
    if not primary_path.exists():
        print("\nError: No PRIMARY index found. Run ingestion first:")
        print("  python scripts/ingest_transcripts.py")
        return
    
    print("Loading PRIMARY index...")
    primary_vector_store.load()
    print(f"Loaded {primary_vector_store.size} vectors from PRIMARY")
    
    has_secondary = False
    if secondary_path.exists():
        print("Loading SECONDARY index...")
        secondary_vector_store.load()
        print(f"Loaded {secondary_vector_store.size} vectors from SECONDARY")
        has_secondary = True
    else:
        print("No SECONDARY index found (using PRIMARY only)")
        secondary_vector_store = None
    
    retrieval_service = RetrievalService(
        repository=repository,
        embedding_provider=embedding_provider,
        primary_vector_store=primary_vector_store,
        secondary_vector_store=secondary_vector_store,
        min_similarity_threshold=config.min_similarity_threshold,
    )
    
    test_queries = [
        "How did we implement the embedding provider?",
        "What database schema did we design?",
        "Show me discussions about vector stores",
        "Comment avons-nous implémenté le chunking?",  # French
        "איך יישמנו את מערכת ה-RAG?",  # Hebrew
    ]
    
    print("\n" + "=" * 60)
    print("Running test queries:")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 60)
        
        results = retrieval_service.search(query, top_k=3)
        
        if not results:
            print("No results found")
            continue
        
        for result in results:
            # Extract reranking scores from metadata
            meta = result.chunk.metadata
            semantic = meta.get("rerank_semantic", 0.0)
            quality = meta.get("rerank_quality", 0.0)
            lexical = meta.get("rerank_lexical", 0.0)
            final = meta.get("rerank_final", result.similarity_score)
            query_type = meta.get("query_type", "general")
            source_type = meta.get("source_type", "unknown")
            index_source = meta.get("index_source", "unknown")  # PRIMARY or SECONDARY
            
            # Content type flags
            flags = []
            if meta.get("is_setup_like"):
                flags.append("SETUP")
            if meta.get("is_install_like"):
                flags.append("INSTALL")
            if meta.get("is_docs_summary_like"):
                flags.append("DOCS_SUMMARY")
            if meta.get("is_project_structure_like"):
                flags.append("PROJECT_STRUCT")
            if meta.get("is_meta_report_like"):
                flags.append("META_REPORT")
            if meta.get("is_limitations_like"):
                flags.append("LIMITATIONS")
            if meta.get("is_requirement_blob"):
                flags.append("REQ_BLOB")
            if meta.get("is_doc_like"):
                flags.append("DOC")
            if meta.get("is_implementation_like"):
                flags.append("IMPL")
            if meta.get("is_decision_like"):
                flags.append("DECISION")
            if meta.get("is_checklist_like"):
                flags.append("CHECKLIST")
            
            # Diagnostics
            content_len = meta.get("content_length", len(result.chunk.content))
            heading_count = meta.get("heading_count", 0)
            bullet_count = meta.get("bullet_count", 0)
            
            print(f"\n  [{result.rank}] Final Score: {final:.3f}")
            print(f"      ├─ Semantic (FAISS): {semantic:.3f}")
            print(f"      ├─ Quality: {quality:.3f}")
            print(f"      └─ Lexical: {lexical:.3f}")
            
            if flags:
                print(f"      Flags: {', '.join(flags)}")
            
            print(f"  Index: {index_source.upper()}")
            print(f"  Source Type: {source_type}")
            print(f"  Query Type: {query_type}")
            print(f"  Diagnostics: {content_len}chars, {heading_count}h, {bullet_count}b")
            print(f"  Conversation: {result.chunk.conversation_id}")
            print(f"  Timestamp: {result.message.timestamp.strftime('%Y-%m-%d %H:%M')}")
            print(f"  Role: {result.message.role}")
            print(f"  Content preview: {result.chunk.content[:120]}...")
    
    print("\n" + "=" * 60)
    print("✓ Retrieval test complete!")
    print("=" * 60)
    print("\nReranking Legend:")
    print("  - Semantic: BGE-M3 embedding similarity (from FAISS)")
    print("  - Quality: Content quality score (filters noise/chatter)")
    print("  - Lexical: Term overlap with query (boosts relevance)")
    print("  - Final: Weighted combination of all three")
    print()


if __name__ == "__main__":
    main()
