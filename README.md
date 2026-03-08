# Cursor Transcript RAG

Local-first RAG prototype for indexing Cursor agent transcripts and retrieving past discussion context with multilingual embeddings.

## What it does

This project ingests Cursor transcript `.jsonl` files, stores metadata in SQLite, builds semantic indexes with FAISS, and lets you search past conversations using local embeddings.

Current focus:
- local-first
- free by default
- multilingual retrieval (English / French / Hebrew)
- portable architecture
- MCP-ready foundation

## Current status

This is an **experimental MVP**, not a finished production memory system.

What works:
- transcript discovery from Cursor folders
- transcript parsing
- local embedding pipeline with `BAAI/bge-m3`
- FAISS indexing
- SQLite metadata storage
- raw retrieval test scripts
- MCP tool integration in Cursor for conversation-history search
- reusable project structure

What does **not** work reliably yet:
- retrieval precision is still inconsistent on raw transcript chunks
- results can still surface planning / summary text instead of best implementation details
- structured memory extraction is still experimental and not merged into main
- query quality depends heavily on phrasing

## Architecture

Main pieces:
- `packages/conversation_rag/` → reusable core
- `adapters/cursor_transcripts/` → Cursor transcript parsing
- `scripts/` → setup / ingest / test scripts
- `mcp-server/` → working MCP server for raw conversation-history retrieval
- `cursor-integration/` → Cursor rule / command docs

## Default stack

- Python
- sentence-transformers
- `BAAI/bge-m3`
- FAISS
- SQLite

## Quickstart

### 1. Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install
```bash
python -m pip install -e .
```

### 3. Optional: Hugging Face local cache
```bash
export HF_HOME="$PWD/data/hf_home"
export HUGGINGFACE_HUB_CACHE="$PWD/data/hf_home/hub"
export TRANSFORMERS_CACHE="$PWD/data/hf_home/transformers"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"
```

### 4. Setup
```bash
python scripts/setup.py
```

### 5. Ingest Cursor transcripts
```bash
python scripts/ingest_transcripts.py
```

### 6. Test retrieval
```bash
python scripts/test_retrieval.py
```

## Transcript source

By default the project looks under:

```
~/.cursor/projects
```

Expected transcript structure includes nested `agent-transcripts/.../*.jsonl` files.

## Why this exists

Cursor conversations often contain useful architecture decisions, debugging context, implementation notes, and design iterations. This project is an attempt to make that history searchable locally instead of relying only on chat memory.

## Known limitations

- Raw transcript chunks include noise
- Retrieval can surface summaries or planning text instead of the best implementation detail
- Query quality depends a lot on phrasing
- Results should be verified manually
- This is not a replacement for a true structured memory system yet

## Roadmap

Short term:
- improve retrieval quality
- reduce noisy chunks
- better ranking and filtering
- stabilize structured memory extraction

Long term:
- memory-first retrieval
- cleaner MCP integration
- stronger project portability
- higher precision for implementation / schema / decision recall

## Publishing note

This repository is shared as an open prototype / engineering experiment.

It is useful today for:
- personal transcript search
- local RAG experimentation
- architecture inspiration
- building a more robust memory layer on top

It is not yet positioned as a production-ready memory engine.

## License

MIT
