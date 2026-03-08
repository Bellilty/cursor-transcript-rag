#!/usr/bin/env python3
"""
Setup script for conversation RAG system.

Initializes database, downloads models, and verifies installation.
"""

from pathlib import Path

from conversation_rag.config import Config
from conversation_rag.storage import ConversationRepository
from conversation_rag.embedding.sentence_transformer import SentenceTransformerProvider


def main():
    """Run setup."""
    print("=" * 60)
    print("Conversation RAG Setup")
    print("=" * 60)
    
    config = Config()
    
    print(f"\n✓ Configuration loaded:")
    print(f"  - Data directory: {config.data_dir}")
    print(f"  - Database: {config.sqlite_db_path}")
    print(f"  - Embedding model: {config.embedding_model}")
    print(f"  - Vector store: {config.vector_store}")
    
    print(f"\n✓ Creating directories...")
    config.data_dir.mkdir(parents=True, exist_ok=True)
    (config.data_dir / "faiss").mkdir(exist_ok=True)
    
    print(f"\n✓ Initializing database...")
    repository = ConversationRepository(config.sqlite_db_path)
    print(f"  - Messages: {repository.count_messages()}")
    print(f"  - Chunks: {repository.count_chunks()}")
    
    print(f"\n✓ Downloading embedding model: {config.embedding_model}")
    print("  (This may take a few minutes on first run...)")
    
    embedding_provider = SentenceTransformerProvider(
        model_name=config.embedding_model,
        device="cpu"
    )
    
    print(f"  - Model loaded successfully")
    print(f"  - Embedding dimension: {embedding_provider.embedding_dim}")
    
    print(f"\n✓ Testing embeddings...")
    test_texts = [
        "Hello, this is a test in English.",
        "Bonjour, ceci est un test en français.",
        "שלום, זהו מבחן בעברית.",
    ]
    
    embeddings = embedding_provider.embed(test_texts)
    print(f"  - Embedded {len(test_texts)} test sentences")
    print(f"  - Shape: {embeddings.shape}")
    
    print("\n" + "=" * 60)
    print("✓ Setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Ingest conversation history:")
    print("   python scripts/ingest_transcripts.py")
    print("\n2. Start MCP server:")
    print("   python mcp-server/server.py")
    print("\n3. Configure Cursor MCP settings (see README.md)")
    print()


if __name__ == "__main__":
    main()
