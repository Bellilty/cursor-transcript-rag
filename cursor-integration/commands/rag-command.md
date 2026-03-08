# /rag Command

## Usage
```
/rag <your question about past conversations>
```

## Description
The `/rag` command triggers semantic search over all past Cursor conversation history and returns relevant context before answering your question.

## Examples

**Search for architectural decisions:**
```
/rag what did we decide about the API architecture?
```

**Find discussions about specific topics:**
```
/rag show me discussions about Hebrew text handling
```

**Recall implementation details:**
```
/rag how did we solve the embedding performance issue?
```

**Review past decisions:**
```
/rag what vector store did we choose and why?
```

## How It Works

1. Your query is embedded using the same multilingual model as your conversation history
2. FAISS performs similarity search across all stored conversation chunks
3. The top-k most relevant snippets are retrieved with timestamps and context
4. The agent uses this retrieved context to answer your question accurately

## Note

The `/rag` command searches across ALL your Cursor conversations that have been ingested into the system. To ingest new conversations, run:

```bash
python scripts/ingest_transcripts.py
```
