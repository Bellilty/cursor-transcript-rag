#!/usr/bin/env python3
"""
MCP Server for Conversation History RAG

Exposes the search_conversation_history tool to Cursor Agent.
"""

import asyncio
import sys
import os
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from conversation_rag.config import Config
from conversation_rag.storage import ConversationRepository
from conversation_rag.embedding.sentence_transformer import SentenceTransformerProvider
from conversation_rag.vector_store import FAISSVectorStore
from conversation_rag.services import RetrievalService

# Import from local tools module
sys.path.insert(0, str(Path(__file__).parent))
from tools import create_search_tool_definition, handle_search_tool_call


app = Server("conversation-rag")

config = None
retrieval_service = None


def initialize_services():
    """Initialize RAG services."""
    global config, retrieval_service
    
    config = Config()
    
    print(f"Initializing RAG system...", file=sys.stderr)
    print(f"Data directory: {config.data_dir}", file=sys.stderr)
    print(f"Embedding model: {config.embedding_model}", file=sys.stderr)
    
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
    
    if config.faiss_index_path.exists():
        print(f"Loading existing index...", file=sys.stderr)
        try:
            vector_store.load()
            print(f"Loaded index with {vector_store.size} vectors", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Could not load index: {e}", file=sys.stderr)
    else:
        print("No existing index found", file=sys.stderr)
    
    retrieval_service = RetrievalService(
        repository=repository,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        min_similarity_threshold=config.min_similarity_threshold,
    )
    
    print("RAG system initialized successfully", file=sys.stderr)


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    tool_def = create_search_tool_definition()
    
    return [
        Tool(
            name=tool_def["name"],
            description=tool_def["description"],
            inputSchema=tool_def["inputSchema"]
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    if name != "search_conversation_history":
        raise ValueError(f"Unknown tool: {name}")
    
    query = arguments.get("query", "")
    top_k = arguments.get("top_k", 5)
    conversation_id = arguments.get("conversation_id")
    
    result = handle_search_tool_call(
        query=query,
        top_k=top_k,
        conversation_id=conversation_id,
        retrieval_service=retrieval_service
    )
    
    return [TextContent(type="text", text=result)]


async def main():
    """Main entry point."""
    initialize_services()
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
