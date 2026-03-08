# Quick Start Guide

Get the Conversation RAG system running in 5 minutes.

## Prerequisites

- Python 3.11+
- ~2GB free disk space (for BGE-M3 model)
- ~3GB RAM for running

## Step 1: Install Dependencies (1 minute)

```bash
pip install sentence-transformers faiss-cpu numpy tiktoken python-dotenv mcp
```

Or with uv (faster):

```bash
uv pip install sentence-transformers faiss-cpu numpy tiktoken python-dotenv mcp
```

## Step 2: Run Setup (2 minutes)

```bash
python scripts/setup.py
```

This will:
- Create the database
- Download BGE-M3 model (~2GB)
- Verify installation

## Step 3: Ingest Conversations (1 minute)

```bash
python scripts/ingest_transcripts.py
```

This scans `~/.cursor/projects/*/agent-transcripts/*.jsonl` and indexes everything.

## Step 4: Configure MCP Server (1 minute)

### macOS/Linux

Edit: `~/Library/Application Support/Cursor/User/globalStorage/mcp.json`

Add:

```json
{
  "mcpServers": {
    "conversation-rag": {
      "command": "python3",
      "args": ["/absolute/path/to/cursor-rag-dev/mcp-server/server.py"],
      "env": {
        "DATA_DIR": "/absolute/path/to/cursor-rag-dev/data"
      }
    }
  }
}
```

**Replace `/absolute/path/to/cursor-rag-dev` with your actual path!**

To get your absolute path:

```bash
cd /path/to/cursor-rag-dev
pwd
```

### Windows

Edit: `%APPDATA%\Cursor\User\globalStorage\mcp.json`

Use `python` instead of `python3` and Windows-style paths.

## Step 5: Install Cursor Rule (30 seconds)

```bash
# For this project only
mkdir -p .cursor/rules
cp cursor-integration/rules/conversation-rag.mdrule .cursor/rules/

# OR for all projects (global)
cp cursor-integration/rules/conversation-rag.mdrule ~/.cursor/rules/
```

## Step 6: Restart Cursor

Restart Cursor to load the MCP server and rule.

## Step 7: Test It

Open a new Cursor Agent chat and use a direct MCP-first prompt:

```
Call search_conversation_history now.
Answer only from that tool.
Query: how did we create this Cursor transcript RAG?
```

If Cursor shows a tool call for `search_conversation_history`, the MCP integration is working.

Alternative test queries:
```
Use search_conversation_history first.
Do not answer from repository files or README unless the tool returns nothing.
Query: what embedding model did we choose?
```

**Note:** The MCP tool name is `search_conversation_history`, not a `/rag` slash command. Results quality depends on query phrasing.

## Troubleshooting

### Model Download Fails

If BGE-M3 download is too slow/fails, use a lighter model:

```bash
export EMBEDDING_MODEL=paraphrase-multilingual-mpnet-base-v2
python scripts/setup.py
```

### No Transcripts Found

Check if transcripts exist:

```bash
ls ~/.cursor/projects/*/agent-transcripts/
```

If the path is different, create a `.env` file:

```bash
echo "CURSOR_TRANSCRIPTS_DIR=/your/actual/path" > .env
```

### MCP Server Not Starting

Check logs:

```bash
python mcp-server/server.py 2>&1 | tee debug.log
```

Common issues:
- Wrong path in mcp.json (use absolute paths)
- Missing dependencies (run pip install again)
- Python version <3.11 (upgrade Python)

### Empty Search Results

1. Check if indexes exist:

```bash
ls -lh data/faiss/
```

Or check specific index files:

```bash
ls -lh data/faiss/primary_index.faiss
ls -lh data/faiss/secondary_index.faiss
```

2. If missing, re-run ingestion:

```bash
python scripts/ingest_transcripts.py
```

3. Check database:

```bash
sqlite3 data/conversations.db "SELECT COUNT(*) FROM messages;"
```

## Verify Installation

Run the test scripts:

```bash
# Test embeddings (should show high similarity for similar text)
python scripts/test_embeddings.py

# Test retrieval (should return relevant results)
python scripts/test_retrieval.py
```

## Next Steps

### Test with Example Data

```bash
# Use the sample transcript
python scripts/ingest_transcripts.py

# The example contains multilingual content
python scripts/test_retrieval.py
```

### Customize Configuration

Create `.env` file:

```bash
cp .env.example .env
```

Edit to customize:
- Embedding model
- Chunk size
- Similarity threshold
- etc.

### Regular Maintenance

To keep your index up to date:

```bash
# Re-run ingestion periodically
python scripts/ingest_transcripts.py

# Or rebuild entire index
python scripts/rebuild_index.py
```

## Common Use Cases

### Search Previous Decisions

```
Use search_conversation_history with query: "what database did we choose and why?"
```

### Find Implementation Details

```
Use search_conversation_history with query: "how did we implement the chunking service?"
```

### Multilingual Search

```
Use search_conversation_history with query: "comment avons-nous implémenté l'embedding?"
```

```
Use search_conversation_history with query: "איך יישמנו את מערכת החיפוש?"
```

### Cross-Project Search

If you've ingested transcripts from multiple projects, you can search across all of them!

## Performance Tips

### Faster Ingestion

```bash
# Use smaller chunks for faster processing
export CHUNK_SIZE=256
export CHUNK_OVERLAP=64
python scripts/ingest_transcripts.py
```

### Lower Memory Usage

```bash
# Use lighter model
export EMBEDDING_MODEL=all-MiniLM-L6-v2
python scripts/rebuild_index.py
```

### GPU Acceleration

```bash
# Install GPU version
pip install faiss-gpu

# Models will automatically use GPU if available
```

## Need Help?

1. Check [README.md](README.md) for detailed documentation
2. Check [PORTING.md](PORTING.md) for reuse in other projects
3. Review [IMPLEMENTATION.md](IMPLEMENTATION.md) for architecture details
4. Check examples in `examples/` directory

## Summary

You now have:
- ✅ Local conversation history RAG system
- ✅ Multilingual search (Hebrew/French/English)
- ✅ MCP tool available in Cursor
- ✅ Raw conversation-history search working from Agent chat
- ⚠️ Retrieval quality still depends on query phrasing

Total setup time: **~5 minutes** (plus model download)

Enjoy semantic search over your conversation history! 🚀
