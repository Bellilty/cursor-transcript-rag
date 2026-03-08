"""
Reranking service for improving retrieval quality.

Reranks FAISS candidates using quality signals and lexical overlap.
"""

import re
from typing import List, Tuple
from ..types import RetrievalResult


class RerankerService:
    """Reranks retrieval results using quality and relevance signals."""
    
    def __init__(self, quality_weight: float = 0.35, lexical_weight: float = 0.25):
        """
        Initialize reranker.
        
        Args:
            quality_weight: Weight for quality score (0-1)
            lexical_weight: Weight for lexical overlap (0-1)
        """
        self.quality_weight = quality_weight
        self.lexical_weight = lexical_weight
        self.semantic_weight = 1.0 - quality_weight - lexical_weight
    
    def rerank(
        self, 
        results: List[RetrievalResult],
        query: str,
        top_k: int,
    ) -> List[RetrievalResult]:
        """
        Rerank results using multiple signals.
        
        Args:
            results: Initial retrieval results from FAISS
            query: Original query
            top_k: Number of top results to return
            
        Returns:
            Reranked results with updated scores and metadata
        """
        if not results:
            return []
        
        # Detect query intent and language
        query_info = self._analyze_query(query)
        query_info["raw_query"] = query  # Store for setup query detection
        
        # Extract query terms
        query_terms = self._extract_terms(query.lower())
        
        # Score each result
        scored_results = []
        for result in results:
            meta = result.chunk.metadata
            
            # Get quality score from chunk metadata
            quality_score = meta.get("quality_score", 0.5)
            
            # Calculate lexical overlap (length-normalized)
            lexical_score = self._calculate_lexical_overlap(
                result.chunk.content.lower(),
                query_terms,
                len(result.chunk.content)
            )
            
            # Semantic score is the FAISS similarity
            semantic_score = result.similarity_score
            
            # Apply query-specific adjustments
            adjusted_quality = self._adjust_quality_for_query(
                quality_score, meta, query_info
            )
            
            # Adjust lexical weight for non-English queries
            lex_weight = self.lexical_weight
            if query_info["is_non_english"] and lexical_score < 0.2:
                # Lexical is unreliable, rely more on semantic
                lex_weight = 0.1
                adjusted_sem_weight = self.semantic_weight + 0.15
            else:
                adjusted_sem_weight = self.semantic_weight
            
            # Combined score
            final_score = (
                adjusted_sem_weight * semantic_score +
                self.quality_weight * adjusted_quality +
                lex_weight * lexical_score
            )
            
            # Store debug info
            result.chunk.metadata["rerank_semantic"] = semantic_score
            result.chunk.metadata["rerank_quality"] = adjusted_quality
            result.chunk.metadata["rerank_lexical"] = lexical_score
            result.chunk.metadata["rerank_final"] = final_score
            result.chunk.metadata["query_type"] = query_info["type"]
            
            scored_results.append((final_score, result))
        
        # Sort by final score (descending)
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        # Deduplicate near-duplicates
        deduped = self._deduplicate(scored_results, similarity_threshold=0.85)
        
        # Take top_k
        top_results = deduped[:top_k]
        
        # Update ranks
        for rank, (score, result) in enumerate(top_results, 1):
            result.rank = rank
            result.similarity_score = score  # Update with reranked score
        
        return [result for _, result in top_results]
    
    def _analyze_query(self, query: str) -> dict:
        """Analyze query for intent and language."""
        query_lower = query.lower()
        
        # Detect query type (multilingual)
        implementation_terms_en = [
            'implement', 'build', 'create', 'develop', 'code',
            'how did we', 'what did we', 'how was', 'what was'
        ]
        implementation_terms_fr = [
            'implémenté', 'implémenter', 'comment avons-nous implémenté',
            'construit', 'créé', 'développé'
        ]
        implementation_terms_he = [
            'יישמנו', 'מימשנו', 'בנינו', 'איך יישמנו'
        ]
        
        decision_terms_en = [
            'why', 'decided', 'chose', 'selected', 'picked',
            'reason', 'rationale', 'decision'
        ]
        decision_terms_fr = [
            'pourquoi', 'décidé', 'choisi', 'sélectionné',
            'décision', 'raison'
        ]
        decision_terms_he = [
            'החלטנו', 'החלטה', 'למה', 'בחרנו'
        ]
        
        discussion_terms_en = [
            'discussion', 'talked', 'mentioned', 'said', 'conversation', 'show me'
        ]
        discussion_terms_fr = [
            'discussion', 'parlé', 'mentionné', 'conversation', 'montre-moi'
        ]
        discussion_terms_he = [
            'דיון', 'שיחה', 'דיברנו', 'הזכרנו'
        ]
        
        # Check all language variants
        is_implementation_query = (
            any(term in query_lower for term in implementation_terms_en) or
            any(term in query_lower for term in implementation_terms_fr) or
            any(term in query for term in implementation_terms_he)
        )
        
        is_decision_query = (
            any(term in query_lower for term in decision_terms_en) or
            any(term in query_lower for term in decision_terms_fr) or
            any(term in query for term in decision_terms_he)
        )
        
        is_discussion_query = (
            any(term in query_lower for term in discussion_terms_en) or
            any(term in query_lower for term in discussion_terms_fr) or
            any(term in query for term in discussion_terms_he)
        )
        
        # Detect non-English (non-ASCII)
        has_non_ascii = any(ord(c) > 127 for c in query)
        
        # Determine primary query type
        query_type = "general"
        if is_implementation_query:
            query_type = "implementation"
        elif is_decision_query:
            query_type = "decision"
        elif is_discussion_query:
            query_type = "discussion"
        
        return {
            "type": query_type,
            "is_implementation_query": is_implementation_query,
            "is_decision_query": is_decision_query,
            "is_discussion_query": is_discussion_query,
            "is_non_english": has_non_ascii,
        }
    
    def _adjust_quality_for_query(
        self, 
        base_quality: float,
        metadata: dict,
        query_info: dict
    ) -> float:
        """
        Adjust quality score based on query type and chunk metadata.
        
        Apply STRONG penalties for docs/setup/meta content.
        """
        adjusted = base_quality
        
        # Extract all flags
        is_setup = metadata.get("is_setup_like", False)
        is_install = metadata.get("is_install_like", False)
        is_docs_summary = metadata.get("is_docs_summary_like", False)
        is_project_struct = metadata.get("is_project_structure_like", False)
        is_meta_report = metadata.get("is_meta_report_like", False)
        is_limitations = metadata.get("is_limitations_like", False)
        
        is_req_blob = metadata.get("is_requirement_blob", False)
        is_doc = metadata.get("is_doc_like", False)
        
        is_impl = metadata.get("is_implementation_like", False)
        is_decision = metadata.get("is_decision_like", False)
        
        # VERY STRONG penalties for docs/setup/meta (unless query is about setup/docs)
        is_setup_query = any(term in query_info.get("raw_query", "").lower() 
                            for term in ['setup', 'install', 'configure', 'how to run'])
        
        if not is_setup_query:
            if is_setup:
                adjusted -= 0.70
            if is_install:
                adjusted -= 0.70
            if is_docs_summary:
                adjusted -= 0.70
            if is_project_struct:
                adjusted -= 0.70
            if is_meta_report:
                adjusted -= 0.65
            if is_limitations:
                adjusted -= 0.60
        
        # For implementation/decision queries
        if query_info["is_implementation_query"] or query_info["is_decision_query"]:
            # EXTRA penalties for requirement blobs and docs
            if is_req_blob:
                adjusted -= 0.60
            if is_doc:
                adjusted -= 0.40
            
            # Strong boosts for real implementation/decision
            if is_impl:
                adjusted += 0.40
            if is_decision:
                adjusted += 0.35
        
        # For discussion queries, lighter penalties
        elif query_info["is_discussion_query"]:
            if is_req_blob:
                adjusted -= 0.35
            if is_setup or is_docs_summary:
                adjusted -= 0.30
        
        return max(0.0, min(1.0, adjusted))
    
    def _extract_terms(self, text: str) -> List[str]:
        """Extract meaningful terms from text."""
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text)
        terms = text.split()
        
        # Filter stopwords (simple list)
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its',
            'our', 'their',
        }
        
        return [term for term in terms if term and term not in stopwords and len(term) > 2]
    
    def _calculate_lexical_overlap(
        self, 
        content: str, 
        query_terms: List[str],
        content_length: int
    ) -> float:
        """
        Calculate lexical overlap with length normalization.
        
        Penalizes giant blobs that match everything.
        """
        if not query_terms:
            return 0.0
        
        content_terms = self._extract_terms(content)
        if not content_terms:
            return 0.0
        
        # Count matching terms
        content_term_set = set(content_terms)
        matches = sum(1 for term in query_terms if term in content_term_set)
        
        if matches == 0:
            return 0.0
        
        # Base overlap ratio
        overlap_ratio = matches / len(query_terms)
        
        # Length normalization: penalize very long chunks
        # Focused matches are better than broad keyword soup
        if content_length > 1500:
            length_penalty = 0.5
        elif content_length > 1000:
            length_penalty = 0.7
        elif content_length > 500:
            length_penalty = 0.85
        else:
            length_penalty = 1.0
        
        overlap_ratio *= length_penalty
        
        # Boost if rare/long query terms appear
        if matches > 0:
            avg_match_length = sum(
                len(term) for term in query_terms if term in content_term_set
            ) / matches
            if avg_match_length > 8:  # Long/rare terms
                overlap_ratio *= 1.2
        
        # Concentration bonus: matching many terms in small space is better
        if matches >= len(query_terms) * 0.8 and content_length < 500:
            overlap_ratio *= 1.3
        
        return min(1.0, overlap_ratio)
    
    def _deduplicate(
        self,
        scored_results: List[Tuple[float, RetrievalResult]],
        similarity_threshold: float = 0.85,
    ) -> List[Tuple[float, RetrievalResult]]:
        """
        Remove near-duplicate results.
        
        Args:
            scored_results: List of (score, result) tuples
            similarity_threshold: Jaccard similarity threshold for duplicates
            
        Returns:
            Deduplicated list
        """
        if len(scored_results) <= 1:
            return scored_results
        
        deduped = []
        seen_contents = []
        
        for score, result in scored_results:
            content = result.chunk.content.lower()
            
            # Check if too similar to any already-seen content
            is_duplicate = False
            for seen in seen_contents:
                if self._jaccard_similarity(content, seen) > similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduped.append((score, result))
                seen_contents.append(content)
        
        return deduped
    
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts."""
        terms1 = set(self._extract_terms(text1))
        terms2 = set(self._extract_terms(text2))
        
        if not terms1 or not terms2:
            return 0.0
        
        intersection = len(terms1 & terms2)
        union = len(terms1 | terms2)
        
        return intersection / union if union > 0 else 0.0
