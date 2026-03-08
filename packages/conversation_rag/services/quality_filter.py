"""
Quality filtering for conversation messages and chunks.

Filters out low-value content before indexing.
"""

import re
from typing import Dict
from ..types import Message


class QualityFilter:
    """Filter for identifying and scoring message/chunk quality."""
    
    def __init__(self, min_length: int = 20):
        """
        Initialize quality filter.
        
        Args:
            min_length: Minimum content length for valid messages
        """
        self.min_length = min_length
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        # Progress chatter
        self.progress_patterns = [
            re.compile(r"^(now let me|let me now|excellent!?|perfect!?|great!?)", re.I),
            re.compile(r"^(i'll now|i will now|next,? i'll)", re.I),
        ]
        
        # Setup/docs terms
        self.setup_terms = [
            'install', 'installation', 'setup', 'configure', 'configuration',
            'getting started', 'quick start', 'quickstart', 'next steps',
            'how to run', 'how to use', 'usage', 'prerequisites'
        ]
        
        self.docs_terms = [
            'readme', 'documentation', 'summary', 'table of contents',
            'file structure', 'project structure', 'folder structure',
            'architecture overview', 'limitations', 'porting guide',
            'troubleshooting', 'checklist', 'status report'
        ]
        
        # Technical terms that indicate real implementation
        self.impl_evidence_terms = [
            'created file', 'modified file', 'implemented in', 'added to',
            'updated file', 'wrote', 'built', 'defined in', 'located in'
        ]
        
        # Decision language
        self.decision_terms = [
            'we chose', 'we decided', 'we selected', 'we picked',
            'we opted for', 'because', 'instead of', 'rather than',
            'the reason', 'tradeoff', 'trade-off'
        ]
    
    def should_skip_message(self, message: Message) -> bool:
        """Determine if message should be skipped entirely."""
        content = message.content.strip()
        
        if len(content) < self.min_length:
            return True
        
        return False
    
    def should_skip_chunk(self, chunk_text: str) -> bool:
        """Determine if chunk should be skipped."""
        content = chunk_text.strip()
        
        if len(content) < self.min_length:
            return True
        
        # Skip extremely meta/docs chunks
        if self._is_pure_meta_docs(chunk_text):
            return True
        
        return False
    
    def score_chunk_quality(self, chunk_text: str, role: str) -> Dict:
        """
        Score chunk quality and compute diagnostic features.
        
        Returns dict with quality_score and all metadata flags.
        """
        # Compute diagnostic features
        diagnostics = self._compute_diagnostics(chunk_text)
        
        # Detect content types
        is_setup_like = self._is_setup_like(chunk_text, diagnostics)
        is_docs_summary_like = self._is_docs_summary_like(chunk_text, diagnostics)
        is_install_like = self._is_install_like(chunk_text)
        is_project_structure_like = self._is_project_structure_like(chunk_text, diagnostics)
        is_meta_report_like = self._is_meta_report_like(chunk_text)
        is_limitations_like = self._is_limitations_like(chunk_text)
        
        is_requirement_blob = self._is_requirement_blob(chunk_text, diagnostics)
        is_doc_like = self._is_doc_like(chunk_text, diagnostics)
        is_checklist_like = self._is_checklist_like(chunk_text, diagnostics)
        
        # STRICTER implementation/decision detection
        is_implementation_like = self._is_implementation_like(chunk_text, role, diagnostics)
        is_decision_like = self._is_decision_like(chunk_text, diagnostics)
        
        is_explanation_like = self._is_explanation_like(chunk_text, role)
        is_markdown_heavy = self._is_markdown_heavy(chunk_text, diagnostics)
        is_code_heavy = diagnostics["code_block_count"] >= 2
        
        # Calculate quality score
        score = 0.5
        
        # STRONG penalties for docs/setup/meta
        if is_setup_like:
            score -= 0.50
        if is_install_like:
            score -= 0.50
        if is_docs_summary_like:
            score -= 0.50
        if is_project_structure_like:
            score -= 0.50
        if is_meta_report_like:
            score -= 0.50
        if is_limitations_like:
            score -= 0.45
        
        # Penalties for requirement blobs and checklists
        if is_requirement_blob:
            score -= 0.40
        if is_doc_like:
            score -= 0.30
        if is_checklist_like:
            score -= 0.25
        if is_markdown_heavy:
            score -= 0.20
        
        # Boosts for real implementation/decision
        if is_implementation_like:
            score += 0.35
        if is_decision_like:
            score += 0.30
        if is_explanation_like:
            score += 0.20
        if is_code_heavy:
            score += 0.15
        
        # Length penalties (stronger)
        length = diagnostics["content_length"]
        if length > 1800:
            score -= 0.25
        elif length > 1200:
            score -= 0.15
        elif length < 50:
            score -= 0.20
        
        # Focused medium chunks get boost
        if 200 < length < 800 and diagnostics["file_path_reference_count"] > 0:
            score += 0.15
        
        # Progress chatter penalty
        if any(p.match(chunk_text[:100]) for p in self.progress_patterns):
            score -= 0.30
        
        # Role-based (but only if not docs/setup)
        if not (is_setup_like or is_docs_summary_like or is_install_like):
            if role == "user" and not is_requirement_blob:
                score += 0.10
            elif role == "assistant" and is_implementation_like:
                score += 0.15
        
        final_score = max(0.0, min(1.0, score))
        
        return {
            "quality_score": final_score,
            "is_setup_like": is_setup_like,
            "is_install_like": is_install_like,
            "is_docs_summary_like": is_docs_summary_like,
            "is_project_structure_like": is_project_structure_like,
            "is_meta_report_like": is_meta_report_like,
            "is_limitations_like": is_limitations_like,
            "is_requirement_blob": is_requirement_blob,
            "is_doc_like": is_doc_like,
            "is_checklist_like": is_checklist_like,
            "is_implementation_like": is_implementation_like,
            "is_decision_like": is_decision_like,
            "is_explanation_like": is_explanation_like,
            "is_markdown_heavy": is_markdown_heavy,
            "is_code_heavy": is_code_heavy,
            **diagnostics,
        }
    
    def _compute_diagnostics(self, text: str) -> Dict:
        """Compute diagnostic features for a chunk."""
        return {
            "content_length": len(text),
            "heading_count": len(re.findall(r'^#+\s+', text, re.MULTILINE)),
            "bullet_count": len(re.findall(r'^\s*[-*]\s+', text, re.MULTILINE)),
            "code_block_count": text.count('```') // 2,
            "file_path_reference_count": len(re.findall(r'\b[\w_]+\.[\w]+\b', text)),
            "technical_term_count": sum(
                1 for term in ['schema', 'function', 'class', 'module', 'import', 'def', 'const']
                if term in text.lower()
            ),
            "setup_term_count": sum(1 for term in self.setup_terms if term in text.lower()),
            "doc_term_count": sum(1 for term in self.docs_terms if term in text.lower()),
        }
    
    def _is_pure_meta_docs(self, text: str) -> bool:
        """Detect purely meta/docs chunks to skip entirely."""
        text_lower = text.lower()
        
        # File tree dumps
        if text.count('├─') > 5 or text.count('└─') > 5:
            return True
        
        # Table of files
        if '|' in text and text.count('|') > 10 and 'file' in text_lower:
            return True
        
        return False
    
    def _is_setup_like(self, text: str, diagnostics: Dict) -> bool:
        """Detect setup/installation/configuration instructions."""
        text_lower = text.lower()
        
        indicators = 0
        
        # Setup terms in first 300 chars
        start_text = text_lower[:300]
        if diagnostics["setup_term_count"] >= 2:
            indicators += 2
        
        # Contains command examples for setup
        if any(phrase in text_lower for phrase in [
            'npm install', 'pip install', 'python scripts/setup',
            'mkdir -p', 'export ', 'source .venv'
        ]):
            indicators += 2
        
        # Numbered setup steps
        if len(re.findall(r'^\s*\d+[\.)]\s+(install|setup|configure|run)', text, re.I | re.M)) >= 2:
            indicators += 2
        
        # "Next steps" sections
        if 'next steps' in text_lower or 'getting started' in text_lower:
            indicators += 1
        
        return indicators >= 3
    
    def _is_install_like(self, text: str) -> bool:
        """Detect installation instructions."""
        text_lower = text.lower()
        
        install_phrases = [
            'install rule', 'install the', 'installation',
            'configure mcp', 'setup mcp', 'start mcp server'
        ]
        
        return sum(1 for phrase in install_phrases if phrase in text_lower) >= 2
    
    def _is_docs_summary_like(self, text: str, diagnostics: Dict) -> bool:
        """Detect documentation summary/overview chunks."""
        text_lower = text.lower()
        
        indicators = 0
        
        # Docs terms
        if diagnostics["doc_term_count"] >= 2:
            indicators += 2
        
        # Starts with docs headers
        start = text_lower[:200]
        if any(phrase in start for phrase in [
            '# readme', '## documentation', 'documentation summary',
            '## overview', 'table of contents', '## files'
        ]):
            indicators += 2
        
        # Has markdown table with file/purpose columns
        if '|' in text and any(word in text_lower for word in ['file', 'purpose', 'documentation']):
            if text.count('|') > 8:
                indicators += 2
        
        # References to other docs
        if text.count('[') > 5 and text.count(']') > 5:
            if any(ext in text_lower for ext in ['.md', 'readme', 'guide']):
                indicators += 1
        
        return indicators >= 3
    
    def _is_project_structure_like(self, text: str, diagnostics: Dict) -> bool:
        """Detect project structure/file tree dumps."""
        # File tree characters
        if text.count('├─') > 3 or text.count('└─') > 3 or text.count('│') > 5:
            return True
        
        # Many lines starting with path-like structure
        path_lines = len(re.findall(r'^\s*[\w-]+/', text, re.MULTILINE))
        if path_lines > 5:
            return True
        
        # "Project Structure" heading
        if 'project structure' in text.lower() or 'folder structure' in text.lower():
            return True
        
        return False
    
    def _is_meta_report_like(self, text: str) -> bool:
        """Detect status reports, summaries, completion reports."""
        text_lower = text.lower()
        
        meta_phrases = [
            'status report', 'completion report', '✓ complete',
            'success criteria', 'what changed', 'files created',
            'files modified', 'implementation complete', 'ready to test'
        ]
        
        return sum(1 for phrase in meta_phrases if phrase in text_lower) >= 2
    
    def _is_limitations_like(self, text: str) -> bool:
        """Detect limitations/caveats/known issues sections."""
        text_lower = text.lower()
        
        limitation_phrases = [
            'limitations', 'known issues', 'caveats', 'not supported',
            'does not', 'cannot', 'future work', 'todo', 'remaining issues'
        ]
        
        # Starts with limitations header
        if text_lower[:100].count('limitation') > 0:
            return True
        
        return sum(1 for phrase in limitation_phrases if phrase in text_lower) >= 3
    
    def _is_requirement_blob(self, text: str, diagnostics: Dict) -> bool:
        """Detect requirement/specification blobs."""
        if diagnostics["content_length"] < 1200:
            return False
        
        indicators = 0
        text_lower = text.lower()
        
        # Many bullets
        if diagnostics["bullet_count"] > 15:
            indicators += 2
        
        # Requirement language
        req_terms = ['must', 'required', 'requirement', 'deliverable', 'constraint']
        if sum(1 for term in req_terms if term in text_lower) >= 4:
            indicators += 2
        
        # Numbered sections
        if len(re.findall(r'^\s*\d+[\.)]\s+', text, re.MULTILINE)) > 8:
            indicators += 1
        
        # Starts with requirement phrases
        start = text_lower[:400]
        if any(phrase in start for phrase in [
            'i want you to', 'your mission', 'core objective',
            'product requirements', 'implementation order'
        ]):
            indicators += 2
        
        return indicators >= 3
    
    def _is_doc_like(self, text: str, diagnostics: Dict) -> bool:
        """Detect README/guide-style documentation."""
        if diagnostics["doc_term_count"] >= 3:
            return True
        
        if diagnostics["code_block_count"] > 4 and diagnostics["heading_count"] > 4:
            return True
        
        return False
    
    def _is_checklist_like(self, text: str, diagnostics: Dict) -> bool:
        """Detect checklists and phase lists."""
        checkbox_count = text.count('- [ ]') + text.count('- [x]') + text.count('✓')
        
        if checkbox_count > 8:
            return True
        
        step_count = len(re.findall(r'\b(step|phase|stage)\s+\d+', text.lower()))
        if step_count > 6:
            return True
        
        return False
    
    def _is_implementation_like(self, text: str, role: str, diagnostics: Dict) -> bool:
        """
        VERY STRICT detection of implementation content.
        
        Requires CONCRETE technical evidence, not just completion claims.
        """
        if role != "assistant":
            return False
        
        text_lower = text.lower()
        indicators = 0
        
        # MUST have concrete file/module references (not just mentioned)
        file_refs = diagnostics["file_path_reference_count"]
        if file_refs >= 2:
            # Check if actually referring to implementation, not just listing
            has_concrete_file_context = any(phrase in text_lower for phrase in [
                'in ', 'to ', 'from ', 'modified ', 'created ', 'updated '
            ])
            if has_concrete_file_context:
                indicators += 2
        
        # MUST have code blocks with actual implementation code
        if diagnostics["code_block_count"] >= 1:
            # But NOT if it's just setup/install commands
            if diagnostics["setup_term_count"] == 0:
                # Check if code block is substantial (not just a one-liner example)
                if text.count('\n') > 10:  # Has substantial code
                    indicators += 2
        
        # Has specific class/function/module names (not generic)
        has_specific_names = False
        for pattern in [r'class \w+', r'def \w+', r'function \w+', r'const \w+', r'module \w+']:
            if re.search(pattern, text_lower):
                has_specific_names = True
                break
        if has_specific_names:
            indicators += 1
        
        # Has SQL/schema/API specifics
        if any(term in text_lower for term in ['schema', 'table', 'column', 'query', 'endpoint', 'route']):
            # Check it's not just mentioned, but explained
            if diagnostics["technical_term_count"] >= 3:
                indicators += 1
        
        # NOT progress chatter
        if not any(p.match(text[:100]) for p in self.progress_patterns):
            indicators += 1
        else:
            # Progress chatter disqualifies
            return False
        
        # NOT setup/docs
        if diagnostics["setup_term_count"] == 0 and diagnostics["doc_term_count"] < 2:
            indicators += 1
        
        # NOT summary/report language
        summary_phrases = [
            "i've updated", "i updated", "i've completed", "i completed",
            "perfect!", "excellent!", "here's the", "here is the",
            "i've implemented", "i implemented", "done", "✓", "✅"
        ]
        has_summary_language = any(phrase in text_lower[:200] for phrase in summary_phrases)
        if has_summary_language:
            # Summary language strongly suggests this is a report, not implementation detail
            return False
        
        # REQUIRE very high threshold
        return indicators >= 5
    
    def _is_decision_like(self, text: str, diagnostics: Dict) -> bool:
        """
        VERY STRICT detection of design decision content.
        
        Requires clear choice + rationale + technical context.
        """
        text_lower = text.lower()
        
        # MUST have decision language
        decision_count = sum(1 for term in self.decision_terms if term in text_lower)
        if decision_count < 2:
            return False
        
        # MUST have reasoning (because/instead of/rather than/tradeoff)
        has_reasoning = any(word in text_lower for word in ['because', 'instead', 'rather', 'tradeoff', 'trade-off'])
        if not has_reasoning:
            return False
        
        # MUST have technical context (not just generic)
        if diagnostics["technical_term_count"] < 2:
            return False
        
        # NOT a requirement blob or setup doc
        if diagnostics["setup_term_count"] >= 2 or diagnostics["doc_term_count"] >= 2:
            return False
        
        # NOT summary/report language
        summary_phrases = [
            "i've decided", "we've decided", "decision summary", "here's what"
        ]
        if any(phrase in text_lower[:100] for phrase in summary_phrases):
            # Report about a decision, not the decision itself
            return False
        
        return True
    
    def _is_explanation_like(self, text: str, role: str) -> bool:
        """Detect explanation/reasoning content."""
        if role != "assistant":
            return False
        
        text_lower = text.lower()
        
        explanation_phrases = [
            'works by', 'how it works', 'the way', 'mechanism',
            'process is', 'pipeline', 'flow is'
        ]
        
        has_explanation = any(phrase in text_lower for phrase in explanation_phrases)
        
        paragraphs = [p for p in text.split('\n\n') if len(p.strip()) > 100]
        
        return has_explanation and len(paragraphs) >= 1
    
    def _is_markdown_heavy(self, text: str, diagnostics: Dict) -> bool:
        """Detect chunks that are mostly markdown formatting."""
        lines = text.split('\n')
        if not lines:
            return False
        
        # Many headings + bullets relative to content
        if diagnostics["heading_count"] + diagnostics["bullet_count"] > len(lines) * 0.4:
            return True
        
        return False
