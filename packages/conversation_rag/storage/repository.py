"""
SQLite-based repository for conversations, messages, and chunks.

Schema is PostgreSQL-compatible for easy migration.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from ..types import Message, Chunk


class ConversationRepository:
    """Repository for managing conversation data in SQLite."""
    
    def __init__(self, db_path: Path):
        """
        Initialize repository.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database with schema."""
        schema_path = Path(__file__).parent / "schema.sql"
        with open(schema_path) as f:
            schema_sql = f.read()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(schema_sql)
    
    def save_conversation(self, conversation_id: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Save or update a conversation.
        
        Args:
            conversation_id: Unique conversation ID
            metadata: Optional metadata dictionary
        """
        metadata_json = json.dumps(metadata) if metadata else None
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO conversations (id, created_at, metadata)
                VALUES (?, ?, ?)
                """,
                (conversation_id, datetime.now().isoformat(), metadata_json),
            )
    
    def save_message(self, message: Message):
        """
        Save a message.
        
        Args:
            message: Message object to save
        """
        metadata_json = json.dumps(message.metadata) if message.metadata else None
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO messages 
                (id, conversation_id, role, content, timestamp, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.id,
                    message.conversation_id,
                    message.role,
                    message.content,
                    message.timestamp.isoformat(),
                    metadata_json,
                    datetime.now().isoformat(),
                ),
            )
    
    def save_chunk(self, chunk: Chunk):
        """
        Save a chunk.
        
        Args:
            chunk: Chunk object to save
        """
        metadata_json = json.dumps(chunk.metadata) if chunk.metadata else None
        vector_index = chunk.metadata.get("vector_index") if chunk.metadata else None
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO chunks 
                (id, message_id, conversation_id, content, chunk_index, vector_index, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.id,
                    chunk.message_id,
                    chunk.conversation_id,
                    chunk.content,
                    chunk.chunk_index,
                    vector_index,
                    metadata_json,
                    datetime.now().isoformat(),
                ),
            )
    
    def get_message(self, message_id: str) -> Optional[Message]:
        """
        Get a message by ID.
        
        Args:
            message_id: Message ID
            
        Returns:
            Message object or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM messages WHERE id = ?",
                (message_id,),
            )
            row = cursor.fetchone()
        
        if not row:
            return None
        
        return Message(
            id=row["id"],
            conversation_id=row["conversation_id"],
            role=row["role"],
            content=row["content"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )
    
    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """
        Get a chunk by ID.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            Chunk object or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM chunks WHERE id = ?",
                (chunk_id,),
            )
            row = cursor.fetchone()
        
        if not row:
            return None
        
        return Chunk(
            id=row["id"],
            message_id=row["message_id"],
            conversation_id=row["conversation_id"],
            content=row["content"],
            chunk_index=row["chunk_index"],
            embedding=None,  # Embeddings stored separately in FAISS
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )
    
    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Chunk]:
        """
        Get multiple chunks by IDs.
        
        Args:
            chunk_ids: List of chunk IDs
            
        Returns:
            List of Chunk objects
        """
        if not chunk_ids:
            return []
        
        placeholders = ",".join("?" * len(chunk_ids))
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                f"SELECT * FROM chunks WHERE id IN ({placeholders})",
                chunk_ids,
            )
            rows = cursor.fetchall()
        
        chunks = []
        for row in rows:
            chunks.append(
                Chunk(
                    id=row["id"],
                    message_id=row["message_id"],
                    conversation_id=row["conversation_id"],
                    content=row["content"],
                    chunk_index=row["chunk_index"],
                    embedding=None,
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                )
            )
        
        return chunks
    
    def get_all_chunks(self) -> List[Chunk]:
        """
        Get all chunks in the database.
        
        Returns:
            List of all Chunk objects
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM chunks ORDER BY created_at")
            rows = cursor.fetchall()
        
        chunks = []
        for row in rows:
            chunks.append(
                Chunk(
                    id=row["id"],
                    message_id=row["message_id"],
                    conversation_id=row["conversation_id"],
                    content=row["content"],
                    chunk_index=row["chunk_index"],
                    embedding=None,
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                )
            )
        
        return chunks
    
    def count_messages(self) -> int:
        """Count total messages in database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM messages")
            return cursor.fetchone()[0]
    
    def count_chunks(self) -> int:
        """Count total chunks in database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM chunks")
            return cursor.fetchone()[0]
