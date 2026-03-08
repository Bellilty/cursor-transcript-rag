"""
Text chunking service with multilingual support.

Splits text into overlapping chunks preserving context for Hebrew, French, and English.
"""

import re
from typing import List
import tiktoken


class ChunkingService:
    """Service for chunking text with overlap."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128):
        """
        Initialize chunking service.
        
        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Number of overlapping tokens between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        if self.tokenizer:
            return self._chunk_with_tokenizer(text)
        else:
            return self._chunk_with_sentences(text)
    
    def _chunk_with_tokenizer(self, text: str) -> List[str]:
        """Chunk text using token-based splitting."""
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            start += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def _chunk_with_sentences(self, text: str) -> List[str]:
        """Chunk text using sentence-based splitting (fallback)."""
        sentences = self._split_sentences(text)
        
        if not sentences:
            return [text]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                
                overlap_words = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    s_len = len(s.split())
                    if overlap_length + s_len <= self.chunk_overlap:
                        overlap_words.insert(0, s)
                        overlap_length += s_len
                    else:
                        break
                
                current_chunk = overlap_words
                current_length = overlap_length
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Handles multiple languages and preserves code blocks.
        """
        code_block_pattern = r'```[\s\S]*?```'
        code_blocks = re.findall(code_block_pattern, text)
        
        if code_blocks:
            text_parts = re.split(code_block_pattern, text)
            
            sentences = []
            for i, part in enumerate(text_parts):
                sentences.extend(self._split_text_sentences(part))
                if i < len(code_blocks):
                    sentences.append(code_blocks[i])
            
            return [s for s in sentences if s.strip()]
        else:
            return self._split_text_sentences(text)
    
    def _split_text_sentences(self, text: str) -> List[str]:
        """Split plain text into sentences."""
        sentence_endings = r'[.!?؟]\s+'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]
