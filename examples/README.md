# Examples

This directory contains sample data and usage examples for the Conversation RAG system.

## Sample Files

### sample_transcript.jsonl

Example Cursor agent transcript file showing the `.jsonl` format:

- Multiple message types (user_message, assistant_message)
- Timestamps in ISO format
- Multilingual content (English, French, Hebrew)

**Usage:**
```bash
# Create a test transcript adapter
from adapters.cursor_transcripts import CursorTranscriptAdapter
from pathlib import Path

adapter = CursorTranscriptAdapter(Path("examples/sample_transcript.jsonl"))
messages = adapter.read_messages()
print(f"Read {len(messages)} messages")
```

### sample_manual_import.json

Example manual import format for non-Cursor conversations:

- Structured JSON with conversation and messages
- Metadata for tracking source and language
- Demonstrates multilingual content

**Usage:**
```python
import json
from pathlib import Path
from datetime import datetime
from packages.conversation_rag.types import Message

# Load sample data
with open("examples/sample_manual_import.json") as f:
    data = json.load(f)

# Convert to Message objects
messages = []
for msg in data["messages"]:
    message = Message(
        id=msg["id"],
        conversation_id=msg["conversation_id"],
        role=msg["role"],
        content=msg["content"],
        timestamp=datetime.fromisoformat(msg["timestamp"]),
        metadata=msg.get("metadata", {})
    )
    messages.append(message)

# Now ingest with your IngestionService
```

## Example Scripts

### Quick Test with Sample Data

```python
#!/usr/bin/env python3
"""Quick test using sample data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "packages"))
sys.path.insert(0, str(Path(__file__).parent.parent / "adapters"))

from conversation_rag.config import Config
from conversation_rag.storage import ConversationRepository
from conversation_rag.embedding.sentence_transformer import SentenceTransformerProvider
from conversation_rag.vector_store import FAISSVectorStore
from conversation_rag.services import ChunkingService, IngestionService, RetrievalService
from cursor_transcripts import CursorTranscriptAdapter

def main():
    # Initialize
    config = Config()
    repository = ConversationRepository(config.sqlite_db_path)
    
    embedding_provider = SentenceTransformerProvider(
        model_name=config.embedding_model,
        device="cpu"
    )
    
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
    
    ingestion_service = IngestionService(
        repository=repository,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        chunking_service=chunking_service,
    )
    
    # Ingest sample transcript
    adapter = CursorTranscriptAdapter(Path("examples/sample_transcript.jsonl"))
    messages = adapter.read_messages()
    print(f"Ingesting {len(messages)} messages...")
    
    chunks = ingestion_service.ingest_conversation(messages)
    print(f"Created {chunks} chunks")
    
    # Save
    vector_store.save()
    
    # Test retrieval
    retrieval_service = RetrievalService(
        repository=repository,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        min_similarity_threshold=config.min_similarity_threshold,
    )
    
    test_queries = [
        "How do embeddings work?",
        "Comment améliorer la recherche?",
        "איך משפרים חיפוש?",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = retrieval_service.search(query, top_k=2)
        for r in results:
            print(f"  - [{r.similarity_score:.3f}] {r.chunk.content[:100]}...")

if __name__ == "__main__":
    main()
```

## Multilingual Test Queries

### English
- "What is a vector database?"
- "How do I implement RAG?"
- "Explain embedding models"
- "Show me conversations about FAISS"

### French
- "Qu'est-ce qu'une base de données vectorielle?"
- "Comment implémenter RAG?"
- "Expliquez les modèles d'embedding"
- "Montrez-moi les conversations sur FAISS"

### Hebrew
- "מהו מסד נתונים וקטורי?"
- "איך ליישם RAG?"
- "הסבר מודלים של אימבדינג"
- "הראה לי שיחות על FAISS"

### Cross-lingual
- Query in English, find French/Hebrew results
- Query in Hebrew, find English/French results
- Mixed language queries

## Integration Examples

See [PORTING.md](../PORTING.md) for more integration examples including:
- Slack bots
- REST APIs
- Custom adapters
- Database integrations
