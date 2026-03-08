-- PostgreSQL-compatible schema for conversation history
-- SQLite implementation uses TEXT for JSONB columns

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    created_at TIMESTAMP NOT NULL,
    metadata TEXT  -- JSON in SQLite, JSONB in PostgreSQL
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    metadata TEXT,  -- JSON in SQLite, JSONB in PostgreSQL
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    vector_index INTEGER,  -- Index position in FAISS
    metadata TEXT,  -- JSON in SQLite, JSONB in PostgreSQL
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (message_id) REFERENCES messages(id),
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);

CREATE INDEX IF NOT EXISTS idx_chunks_conversation ON chunks(conversation_id);
CREATE INDEX IF NOT EXISTS idx_chunks_message ON chunks(message_id);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_chunks_vector_index ON chunks(vector_index);
