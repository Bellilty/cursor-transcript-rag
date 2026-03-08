# Porting Guide: Reusing This RAG System

This guide explains how to port the Conversation History RAG system to another Cursor project or codebase.

## Overview

The system is designed with **portability as the primary goal**. The reusable core is completely decoupled from Cursor-specific logic.

## What's Portable vs. What's Not

### ✅ Fully Portable (Copy As-Is)

These components work in any project:

- `packages/conversation-rag/` - Core RAG system
- `mcp-server/` - MCP server implementation
- `cursor-integration/rules/` - Cursor rules
- `cursor-integration/commands/` - Command documentation
- `scripts/setup.py` - Setup script
- `scripts/test_*.py` - Testing scripts
- `.env.example` - Configuration template

### 🔧 Adapter Layer (Needs Customization)

These are specific to Cursor transcripts:

- `adapters/cursor-transcripts/` - Reads Cursor `.jsonl` files
- `scripts/ingest_transcripts.py` - Uses Cursor adapter

For other message sources, you'll write a new adapter (see below).

## Step-by-Step Porting

### 1. Copy Core Files

```bash
# In your new project
mkdir -p conversation-rag-system
cd conversation-rag-system

# Copy reusable components
cp -r /path/to/cursor-rag-dev/packages .
cp -r /path/to/cursor-rag-dev/mcp-server .
cp -r /path/to/cursor-rag-dev/cursor-integration .
cp -r /path/to/cursor-rag-dev/scripts/setup.py scripts/
cp -r /path/to/cursor-rag-dev/scripts/test_*.py scripts/
cp /path/to/cursor-rag-dev/.env.example .
cp /path/to/cursor-rag-dev/pyproject.toml .
```

### 2. Set Up Environment

```bash
# Copy and customize config
cp .env.example .env

# Edit paths for new project
vim .env
```

Update these variables:

```bash
DATA_DIR=./data  # Or your preferred location
SQLITE_DB_PATH=./data/conversations.db
CURSOR_TRANSCRIPTS_DIR=~/.cursor/projects  # If using Cursor transcripts
```

### 3. Install Dependencies

```bash
pip install sentence-transformers faiss-cpu numpy tiktoken python-dotenv mcp
```

### 4. Run Setup

```bash
python scripts/setup.py
```

This initializes database, downloads models, and verifies installation.

### 5. Create Message Source Adapter

If you're not using Cursor transcripts, create a custom adapter.

#### Example: JSON File Adapter

Create `adapters/json-files/json_adapter.py`:

```python
"""Adapter for reading JSON conversation files."""

import json
from pathlib import Path
from datetime import datetime
from typing import List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages"))
from conversation_rag.types import Message


class JSONFileAdapter:
    """Adapter for JSON conversation files."""
    
    def __init__(self, json_path: Path):
        self.json_path = json_path
    
    def read_messages(self) -> List[Message]:
        """Read messages from JSON file."""
        with open(self.json_path) as f:
            data = json.load(f)
        
        messages = []
        for msg in data.get("messages", []):
            message = Message(
                id=msg["id"],
                conversation_id=msg.get("conversation_id", "default"),
                role=msg["role"],
                content=msg["content"],
                timestamp=datetime.fromisoformat(msg["timestamp"]),
                metadata=msg.get("metadata", {}),
            )
            messages.append(message)
        
        return messages
```

#### Example: Database Adapter

Create `adapters/database/db_adapter.py`:

```python
"""Adapter for reading from existing database."""

import sqlite3
from datetime import datetime
from typing import List
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages"))
from conversation_rag.types import Message


class DatabaseAdapter:
    """Adapter for existing database."""
    
    def __init__(self, db_path: Path, query: str):
        self.db_path = db_path
        self.query = query
    
    def read_messages(self) -> List[Message]:
        """Read messages from database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(self.query)
            rows = cursor.fetchall()
        
        messages = []
        for row in rows:
            message = Message(
                id=str(row["id"]),
                conversation_id=str(row.get("conversation_id", "default")),
                role=row["role"],
                content=row["content"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                metadata={},
            )
            messages.append(message)
        
        return messages
```

### 6. Create Custom Ingestion Script

Create `scripts/ingest_custom.py`:

```python
#!/usr/bin/env python3
"""Ingest custom message source."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "packages"))
sys.path.insert(0, str(Path(__file__).parent.parent / "adapters"))

from conversation_rag.config import Config
from conversation_rag.storage import ConversationRepository
from conversation_rag.embedding.sentence_transformer import SentenceTransformerProvider
from conversation_rag.vector_store import FAISSVectorStore
from conversation_rag.services import ChunkingService, IngestionService

# Import your custom adapter
from json_files import JSONFileAdapter  # Or your adapter

def main():
    config = Config()
    
    # Initialize RAG system
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
    
    # Use your adapter
    adapter = JSONFileAdapter(Path("data/conversations.json"))
    messages = adapter.read_messages()
    
    # Ingest
    chunks = ingestion_service.ingest_conversation(messages)
    print(f"Ingested {len(messages)} messages → {chunks} chunks")
    
    # Save
    vector_store.save()
    print("Index saved")

if __name__ == "__main__":
    main()
```

### 7. Configure MCP Server

Update Cursor MCP settings with the new project path:

```json
{
  "mcpServers": {
    "conversation-rag": {
      "command": "python3",
      "args": ["/path/to/new-project/mcp-server/server.py"],
      "env": {
        "DATA_DIR": "/path/to/new-project/data"
      }
    }
  }
}
```

### 8. Install Cursor Rule

```bash
# Project-specific
mkdir -p .cursor/rules
cp cursor-integration/rules/conversation-rag.mdrule .cursor/rules/
```

### 9. Test the System

```bash
# Test embeddings
python scripts/test_embeddings.py

# Ingest your data
python scripts/ingest_custom.py

# Test retrieval
python scripts/test_retrieval.py
```

### 10. Start Using

In Cursor:

```
/rag what is our API architecture?
```

## Porting Checklist

Use this checklist when porting:

- [ ] Copy core packages
- [ ] Copy MCP server
- [ ] Copy Cursor integration files
- [ ] Copy scripts (setup, test)
- [ ] Create `.env` file
- [ ] Install Python dependencies
- [ ] Run `setup.py`
- [ ] Create adapter for your message source
- [ ] Create ingestion script
- [ ] Test embeddings
- [ ] Ingest data
- [ ] Test retrieval
- [ ] Configure MCP in Cursor
- [ ] Install Cursor rule
- [ ] Test `/rag` command

## Adaptation Time Estimate

- **Basic port** (using existing adapter): ~15 minutes
- **Custom adapter** (simple source): ~30 minutes
- **Complex adapter** (multiple sources, transformations): ~1-2 hours

## Common Adaptations

### Using Different Embedding Model

In `.env`:

```bash
EMBEDDING_MODEL=paraphrase-multilingual-mpnet-base-v2
```

Then rebuild:

```bash
python scripts/rebuild_index.py
```

### Using PostgreSQL Instead of SQLite

1. Install `psycopg2`:

```bash
pip install psycopg2-binary
```

2. Create `packages/conversation-rag/storage/postgres_repository.py`:

```python
"""PostgreSQL repository implementation."""

import json
import psycopg2
from datetime import datetime
from typing import List, Optional
from ..types import Message, Chunk


class PostgreSQLRepository:
    """Repository using PostgreSQL."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self._init_db()
    
    def _init_db(self):
        """Initialize database with schema."""
        schema_path = Path(__file__).parent / "schema.sql"
        with open(schema_path) as f:
            schema_sql = f.read()
        
        # Replace TEXT with JSONB for PostgreSQL
        schema_sql = schema_sql.replace("metadata TEXT", "metadata JSONB")
        
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cursor:
                cursor.execute(schema_sql)
            conn.commit()
    
    # Implement same methods as ConversationRepository
    # ...
```

3. Update config to use PostgreSQL

### Adding Reranking

1. Install reranker:

```bash
pip install sentence-transformers
```

2. Add reranking to retrieval service:

```python
# In packages/conversation-rag/services/retrieval.py

class RetrievalService:
    def __init__(self, ..., reranker_model: str = None):
        # ...
        if reranker_model:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(reranker_model)
        else:
            self.reranker = None
    
    def search(self, query: str, top_k: int = 5, **kwargs):
        # Initial retrieval
        results = self._vector_search(query, top_k * 2)
        
        # Rerank if available
        if self.reranker:
            pairs = [[query, r.chunk.content] for r in results]
            scores = self.reranker.predict(pairs)
            results = sorted(
                zip(results, scores),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]
            results = [r for r, _ in results]
        
        return results[:top_k]
```

## Integration Examples

### Slack Bot Integration

```python
from slack_bolt import App
from conversation_rag.services import RetrievalService

app = App(token=os.environ["SLACK_BOT_TOKEN"])

@app.message("search")
def handle_search(message, say):
    query = message["text"].replace("search", "").strip()
    results = retrieval_service.search(query, top_k=3)
    formatted = retrieval_service.format_for_agent(results)
    say(formatted)
```

### REST API Integration

```python
from flask import Flask, request, jsonify
from conversation_rag.services import RetrievalService

app = Flask(__name__)

@app.route("/search", methods=["POST"])
def search():
    query = request.json.get("query")
    top_k = request.json.get("top_k", 5)
    
    results = retrieval_service.search(query, top_k)
    
    return jsonify({
        "results": [
            {
                "content": r.chunk.content,
                "similarity": r.similarity_score,
                "timestamp": r.message.timestamp.isoformat(),
            }
            for r in results
        ]
    })
```

## Future: Packaging as pip Package

To make this even more portable, consider packaging:

```bash
# In your fork
pip install build
python -m build

# Install in other projects
pip install conversation-rag
```

Then usage becomes:

```python
from conversation_rag import Config, IngestionService, RetrievalService
# ...
```

## Support

For questions or issues with porting:

1. Check the main [README.md](README.md)
2. Review example adapters in `adapters/`
3. Check script examples in `scripts/`
4. File an issue in your fork if needed

## Summary

The system is designed for **maximum portability**:

- ✅ Core logic is framework-agnostic
- ✅ Clean interfaces for adapters
- ✅ All Cursor-specific code isolated
- ✅ Easy to swap components (embedding, vector store)
- ✅ Minimal dependencies

**Expected porting time: 15-60 minutes** depending on customization needs.
