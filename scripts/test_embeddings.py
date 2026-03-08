#!/usr/bin/env python3
"""
Test multilingual embedding quality.

Verifies that the embedding model works well for Hebrew, French, and English.
"""

import numpy as np

from conversation_rag.config import Config
from conversation_rag.embedding.sentence_transformer import SentenceTransformerProvider


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def main():
    """Test embedding quality."""
    print("=" * 60)
    print("Multilingual Embedding Quality Test")
    print("=" * 60)
    
    config = Config()
    
    print(f"\nLoading model: {config.embedding_model}")
    embedding_provider = SentenceTransformerProvider(
        model_name=config.embedding_model,
        device="cpu"
    )
    
    print(f"Embedding dimension: {embedding_provider.embedding_dim}\n")
    
    test_cases = [
        {
            "name": "English - Technical",
            "query": "How do I implement a vector database?",
            "similar": "What is the best way to build a vector store?",
            "different": "I like pizza and ice cream."
        },
        {
            "name": "French - Technical",
            "query": "Comment implémenter une base de données vectorielle?",
            "similar": "Quelle est la meilleure façon de construire un stockage vectoriel?",
            "different": "J'aime la pizza et la glace."
        },
        {
            "name": "Hebrew - Technical",
            "query": "איך ליישם מסד נתונים וקטורי?",
            "similar": "מה הדרך הטובה ביותר לבנות אחסון וקטורים?",
            "different": "אני אוהב פיצה וגלידה."
        },
        {
            "name": "Cross-lingual (English→French)",
            "query": "vector database implementation",
            "similar": "implémentation de base de données vectorielle",
            "different": "recette de cuisine française"
        },
    ]
    
    print("Testing semantic similarity:\n")
    
    for test in test_cases:
        print(f"Test: {test['name']}")
        print(f"  Query: {test['query']}")
        
        query_emb = embedding_provider.embed_query(test['query'])
        similar_emb = embedding_provider.embed_query(test['similar'])
        different_emb = embedding_provider.embed_query(test['different'])
        
        similar_score = cosine_similarity(query_emb, similar_emb)
        different_score = cosine_similarity(query_emb, different_emb)
        
        print(f"  Similar text similarity: {similar_score:.3f}")
        print(f"  Different text similarity: {different_score:.3f}")
        
        if similar_score > different_score:
            print("  ✓ PASS - Similar text scored higher\n")
        else:
            print("  ✗ FAIL - Different text scored higher\n")
    
    print("=" * 60)
    print("Test complete!")
    print("=" * 60)
    print("\nExpected behavior:")
    print("- Similar texts should have high similarity (>0.7)")
    print("- Different texts should have low similarity (<0.5)")
    print("- Model should work across Hebrew, French, and English")
    print()


if __name__ == "__main__":
    main()
