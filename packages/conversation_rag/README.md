# Conversation RAG - Core Module

This is the **portable, reusable core** of the conversation history RAG system. It contains zero business logic and can be dropped into any project.

## Design Philosophy

1. **Interface-driven** - All components implement clean protocols
2. **Dependency injection** - Services accept interfaces, not concrete implementations
3. **Framework-agnostic** - No coupling to Cursor, Flask, or any framework
4. **Type-safe** - Full type hints with dataclasses and Protocols
5. **Testable** - Easy to mock and test in isolation

## Module Structure

```
conversation-rag/
├── __init__.py              # Public API
├── config.py                # Environment-based configuration
├── types.py                 # Core types and interfaces
├── embedding/
│   ├── __init__.py
│   ├── provider.py          # EmbeddingProvider protocol (future)
│   └── sentence_transformer.py  # Default implementation
├── vector_store/
│   ├── __init__.py
│   ├── store.py             # VectorStore protocol (future)
│   └── faiss_store.py       # FAISS implementation
├── storage/
│   ├── __init__.py
│   ├── schema.sql           # Database schema
│   └── repository.py        # SQLite/PostgreSQL repository
└── services/
    ├── __init__.py
    ├── chunking.py          # Text chunking
    ├── normalization.py     # Text normalization
    ├── ingestion.py         # Ingestion pipeline
    └── retrieval.py         # Retrieval pipeline
```

## Core Types

### Message

Represents a conversation message:

```python
@dataclass
class Message:
    id: str
    conversation_id: str
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: datetime
    metadata: Dict[str, Any]
```

### Chunk

Represents a text chunk with optional embedding:

```python
@dataclass
class Chunk:
    id: str
    message_id: str
    conversation_id: str
    content: str
    chunk_index: int
    embedding: Optional[np.ndarray]
    metadata: Dict[str, Any]
```

### RetrievalResult

Result from similarity search:

```python
@dataclass
class RetrievalResult:
    chunk: Chunk
    message: Message
    similarity_score: float
    rank: int
```

## Interfaces

### EmbeddingProvider

Protocol for embedding providers:

```python
class EmbeddingProvider(Protocol):
    def embed(self, texts: List[str]) -> np.ndarray: ...
    def embed_query(self, query: str) -> np.ndarray: ...
    
    @property
    def embedding_dim(self) -> int: ...
```

### VectorStore

Protocol for vector stores:

```python
class VectorStore(Protocol):
    def add(self, vectors: np.ndarray, ids: List[str]) -> None: ...
    def search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[str, float]]: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
    
    @property
    def size(self) -> int: ...
```

### MessageSource

Protocol for message sources/adapters:

```python
class MessageSource(Protocol):
    def read_messages(self) -> List[Message]: ...
```

## Services

### ChunkingService

Splits text into overlapping chunks:

```python
chunking_service = ChunkingService(
    chunk_size=512,
    chunk_overlap=128
)

chunks = chunking_service.chunk_text(text)
```

### IngestionService

Complete ingestion pipeline:

```python
ingestion_service = IngestionService(
    repository=repository,
    embedding_provider=embedding_provider,
    vector_store=vector_store,
    chunking_service=chunking_service,
)

# Ingest single message
chunks = ingestion_service.ingest_message(message)

# Ingest conversation
total_chunks = ingestion_service.ingest_conversation(messages)

# Rebuild index from database
count = ingestion_service.rebuild_index()
```

### RetrievalService

Semantic search and formatting:

```python
retrieval_service = RetrievalService(
    repository=repository,
    embedding_provider=embedding_provider,
    vector_store=vector_store,
    min_similarity_threshold=0.5,
)

# Search
results = retrieval_service.search(
    query="How do embeddings work?",
    top_k=5,
    conversation_id=None  # Optional filter
)

# Format for agent
formatted = retrieval_service.format_for_agent(results)
```

## Storage

### ConversationRepository

SQLite repository with PostgreSQL-compatible schema:

```python
repository = ConversationRepository(db_path)

# Save
repository.save_conversation(conversation_id, metadata)
repository.save_message(message)
repository.save_chunk(chunk)

# Retrieve
message = repository.get_message(message_id)
chunk = repository.get_chunk(chunk_id)
chunks = repository.get_chunks_by_ids(chunk_ids)

# Stats
message_count = repository.count_messages()
chunk_count = repository.count_chunks()
```

## Embedding Providers

### SentenceTransformerProvider

Default embedding provider using sentence-transformers:

```python
embedding_provider = SentenceTransformerProvider(
    model_name="BAAI/bge-m3",  # or other model
    device="cpu"  # or "cuda"
)

# Batch embed
embeddings = embedding_provider.embed(texts)  # (n, dim)

# Single query
query_embedding = embedding_provider.embed_query(query)  # (dim,)

# Check dimension
dim = embedding_provider.embedding_dim  # e.g., 1024
```

## Vector Stores

### FAISSVectorStore

FAISS-based vector store:

```python
vector_store = FAISSVectorStore(
    embedding_dim=1024,
    index_type="flat",  # or "ivf"
    index_path=Path("data/faiss/index.faiss"),
    id_mapping_path=Path("data/faiss/id_mapping.json"),
)

# Add vectors
vector_store.add(vectors, ids)

# Search
results = vector_store.search(query_vector, top_k=5)
# Returns: List[(chunk_id, similarity_score)]

# Persist
vector_store.save()

# Load
vector_store.load()

# Check size
count = vector_store.size
```

## Configuration

Configuration from environment variables with defaults:

```python
config = Config()

# Embedding settings
config.embedding_provider  # "sentence-transformers"
config.embedding_model     # "BAAI/bge-m3"

# Vector store settings
config.vector_store        # "faiss"
config.faiss_index_type    # "flat"

# Storage
config.data_dir            # Path("./data")
config.sqlite_db_path      # Path("./data/conversations.db")

# Chunking
config.chunk_size          # 512
config.chunk_overlap       # 128

# Retrieval
config.default_top_k       # 5
config.min_similarity_threshold  # 0.5

# Cursor transcripts
config.cursor_transcripts_dir  # Path("~/.cursor/projects")
```

## Usage Example

```python
from conversation_rag import Config
from conversation_rag.storage import ConversationRepository
from conversation_rag.embedding import SentenceTransformerProvider
from conversation_rag.vector_store import FAISSVectorStore
from conversation_rag.services import (
    ChunkingService,
    IngestionService,
    RetrievalService
)

# Initialize
config = Config()

repository = ConversationRepository(config.sqlite_db_path)

embedding_provider = SentenceTransformerProvider(
    model_name=config.embedding_model
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

# Ingestion
ingestion_service = IngestionService(
    repository=repository,
    embedding_provider=embedding_provider,
    vector_store=vector_store,
    chunking_service=chunking_service,
)

chunks = ingestion_service.ingest_conversation(messages)
vector_store.save()

# Retrieval
retrieval_service = RetrievalService(
    repository=repository,
    embedding_provider=embedding_provider,
    vector_store=vector_store,
)

results = retrieval_service.search("your query")
```

## Extension Points

### Custom Embedding Provider

```python
class CustomEmbeddingProvider:
    def embed(self, texts: List[str]) -> np.ndarray:
        # Your implementation
        pass
    
    def embed_query(self, query: str) -> np.ndarray:
        # Your implementation
        pass
    
    @property
    def embedding_dim(self) -> int:
        return 1536  # Your dimension
```

### Custom Vector Store

```python
class CustomVectorStore:
    def add(self, vectors: np.ndarray, ids: List[str]) -> None:
        # Your implementation
        pass
    
    def search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        # Your implementation
        pass
    
    # ... implement other methods
```

### Custom Message Source

```python
class CustomMessageSource:
    def read_messages(self) -> List[Message]:
        # Your implementation
        return messages
```

## Dependencies

Minimal dependencies:

```
sentence-transformers>=2.3.0
faiss-cpu>=1.7.4
numpy>=1.24.0
tiktoken>=0.5.2
python-dotenv>=1.0.0
```

Optional:
- `faiss-gpu` for GPU support
- `psycopg2` for PostgreSQL

## Testing

The module is designed for easy testing:

```python
# Mock embedding provider
class MockEmbeddingProvider:
    def embed(self, texts):
        return np.random.rand(len(texts), 384)
    
    def embed_query(self, query):
        return np.random.rand(384)
    
    @property
    def embedding_dim(self):
        return 384

# Use in tests
embedding_provider = MockEmbeddingProvider()
# ... rest of setup
```

## Performance Characteristics

- **Ingestion**: ~100-200 messages/minute (CPU, BGE-M3)
- **Retrieval**: <1 second for typical queries
- **Memory**: ~2-3GB with BGE-M3 model loaded
- **Disk**: ~10MB per 1000 messages

## Thread Safety

- `ConversationRepository`: SQLite connection per operation (thread-safe)
- `FAISSVectorStore`: Thread-safe for reads, use locks for writes
- `SentenceTransformerProvider`: Thread-safe after model load

## Migration to PostgreSQL

The schema is PostgreSQL-compatible. To migrate:

1. Replace `TEXT` with `JSONB` for metadata columns
2. Use `psycopg2` instead of `sqlite3`
3. Update connection handling in `repository.py`

Schema modifications needed:

```sql
-- Change from:
metadata TEXT

-- To:
metadata JSONB
```

## License

Part of the Cursor Conversation RAG project.
