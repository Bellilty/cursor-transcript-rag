"""
MCP tools for conversation history RAG.
"""

from typing import Any, Dict


def create_search_tool_definition() -> Dict[str, Any]:
    """Create the tool definition for search_conversation_history."""
    return {
        "name": "search_conversation_history",
        "description": "Search past conversation history using semantic similarity. "
                      "Returns relevant snippets from previous discussions with timestamps and context.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query - can be a question, topic, or keywords to search for in past conversations"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5)",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20
                },
                "conversation_id": {
                    "type": "string",
                    "description": "Optional: Filter results to a specific conversation ID"
                }
            },
            "required": ["query"]
        }
    }


def handle_search_tool_call(
    query: str,
    top_k: int = 5,
    conversation_id: str | None = None,
    retrieval_service = None,
) -> str:
    """
    Handle a search_conversation_history tool call.
    
    Args:
        query: Search query
        top_k: Number of results to return
        conversation_id: Optional conversation filter
        retrieval_service: RetrievalService instance
        
    Returns:
        Formatted search results
    """
    if not retrieval_service:
        return "Error: Retrieval service not initialized"
    
    try:
        results = retrieval_service.search(
            query=query,
            top_k=top_k,
            conversation_id=conversation_id
        )
        
        return retrieval_service.format_for_agent(results)
    except Exception as e:
        return f"Error searching conversation history: {str(e)}"
