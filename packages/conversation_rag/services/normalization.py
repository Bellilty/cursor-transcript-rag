"""
Text normalization utilities for multilingual content.

Preserves important characters for Hebrew, French, and English.
"""

import re


def normalize_text(text: str, preserve_case: bool = False) -> str:
    """
    Normalize text for embedding and retrieval.
    
    Args:
        text: Text to normalize
        preserve_case: If True, preserve original case
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    # Optionally convert to lowercase (preserve for case-sensitive languages)
    if not preserve_case:
        text = text.lower()
    
    return text
