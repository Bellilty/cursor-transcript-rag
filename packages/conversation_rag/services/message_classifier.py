"""
Message-level source type classification.

Classifies messages before chunking to filter out low-value content.
"""

import re
from typing import Dict
from ..types import Message


class MessageClassifier:
    """Classify messages by source type for filtering."""
    
    SOURCE_TYPES = [
        "implementation_explanation",
        "technical_decision",
        "code_or_schema",
        "user_requirement_prompt",
        "setup_install_doc",
        "generated_project_doc",
        "progress_chatter",
        "status_report",
        "limitations_meta",
        "generic_other",
    ]
    
    def __init__(self):
        """Initialize classifier."""
        pass
    
    def classify_message(self, message: Message) -> str:
        """
        Classify message into source type.
        
        Returns one of SOURCE_TYPES.
        """
        content = message.content
        content_lower = content.lower()
        role = message.role
        
        # Compute features
        length = len(content)
        heading_count = len(re.findall(r'^#+\s+', content, re.MULTILINE))
        bullet_count = len(re.findall(r'^\s*[-*]\s+', content, re.MULTILINE))
        code_block_count = content.count('```') // 2
        
        # Progress chatter (very short assistant messages)
        if role == "assistant" and length < 150:
            if any(phrase in content_lower[:100] for phrase in [
                'now let me', 'excellent', 'perfect', 'great',
                "i'll now", "let me continue"
            ]):
                return "progress_chatter"
        
        # User requirement prompts (long, many bullets, requirement language)
        if role == "user" and length > 1500:
            if bullet_count > 15 and heading_count > 5:
                req_terms = ['must', 'required', 'requirement', 'deliverable', 'constraint']
                if sum(1 for term in req_terms if term in content_lower) >= 4:
                    return "user_requirement_prompt"
        
        # Setup/install docs
        setup_indicators = 0
        if any(phrase in content_lower for phrase in [
            'install rule', 'configure mcp', 'setup mcp', 'npm install',
            'pip install', 'python scripts/setup', 'getting started',
            'quick start', 'installation', 'how to run'
        ]):
            setup_indicators += 2
        
        if heading_count > 3 and any(h in content_lower for h in [
            '## installation', '## setup', '## configuration', '## usage'
        ]):
            setup_indicators += 2
        
        if setup_indicators >= 3:
            return "setup_install_doc"
        
        # Generated project docs (README, PORTING, summaries)
        if heading_count > 5 and length > 1000:
            doc_markers = [
                'readme', 'documentation', 'file structure', 'project structure',
                'architecture overview', 'porting guide', 'folder structure'
            ]
            if sum(1 for marker in doc_markers if marker in content_lower) >= 2:
                return "generated_project_doc"
            
            # File trees
            if content.count('├─') > 3 or content.count('└─') > 3:
                return "generated_project_doc"
        
        # Status reports / completion summaries
        status_indicators = 0
        if '✓' in content or '✅' in content:
            status_indicators += 1
        
        if any(phrase in content_lower for phrase in [
            'status report', 'completion report', 'implementation complete',
            'ready to test', 'files created', 'files modified', 'what changed'
        ]):
            status_indicators += 2
        
        if heading_count > 4 and bullet_count > 10:
            if 'complete' in content_lower or 'summary' in content_lower:
                status_indicators += 1
        
        if status_indicators >= 3:
            return "status_report"
        
        # Limitations / caveats
        if 'limitation' in content_lower[:200] or 'known issue' in content_lower[:200]:
            return "limitations_meta"
        
        if sum(1 for phrase in ['limitation', 'caveat', 'not supported', 'cannot', 'does not']
               if phrase in content_lower) >= 3:
            return "limitations_meta"
        
        # Implementation explanation (VERY STRICT: requires concrete technical detail)
        if role == "assistant":
            impl_indicators = 0
            
            # MUST have concrete file/module/class references
            has_file_refs = len(re.findall(r'\b[\w_]+\.[\w]+\b', content)) >= 3
            has_class_func = len(re.findall(r'\b(class|function|def|const)\s+\w+', content_lower)) >= 1
            
            if has_file_refs or has_class_func:
                impl_indicators += 2
            
            # MUST have substantial code or schema
            if code_block_count >= 1 and content.count('\n') > 15:
                impl_indicators += 2
            
            # Has technical specifics (not just mentioned)
            tech_detail_count = sum(1 for term in [
                'schema', 'table', 'column', 'query', 'endpoint', 'route', 
                'function', 'class', 'module', 'import', 'api'
            ] if term in content_lower)
            
            if tech_detail_count >= 3:
                impl_indicators += 1
            
            # NOT progress chatter or summary
            is_summary = any(phrase in content_lower[:200] for phrase in [
                "i've updated", "i updated", "i've completed", "i completed",
                "perfect!", "excellent!", "here's the", "here is the",
                "i've implemented", "i implemented", "done", "✓", "✅"
            ])
            
            if is_summary:
                # This is a report/summary, NOT implementation detail
                return "generic_other"
            
            # NOT too short (summaries are often short)
            if length > 300:
                impl_indicators += 1
            
            # NOT progress chatter
            if not any(phrase in content_lower[:100] for phrase in [
                'now let me', 'excellent', 'perfect', 'great', "i'll now"
            ]):
                impl_indicators += 1
            
            if impl_indicators >= 5:
                return "implementation_explanation"
        
        # Technical decision (contains decision + rationale + technical context)
        decision_terms = ['we chose', 'we decided', 'we selected', 'we picked', 'we opted for']
        reasoning_terms = ['because', 'instead of', 'rather than', 'the reason', 'tradeoff']
        
        has_decision = sum(1 for term in decision_terms if term in content_lower) >= 1
        has_reasoning = sum(1 for term in reasoning_terms if term in content_lower) >= 1
        has_tech_context = sum(1 for term in ['schema', 'function', 'class', 'api', 'architecture', 'design', 'implementation']
                               if term in content_lower) >= 2
        
        if has_decision and has_reasoning and has_tech_context and heading_count < 5:
            # NOT a summary/report
            is_report = any(phrase in content_lower[:100] for phrase in [
                "decision summary", "here's what", "we've decided", "decisions were"
            ])
            if not is_report:
                return "technical_decision"
        
        # Code or schema (lots of code blocks, SQL, technical content)
        if code_block_count >= 2:
            if any(lang in content_lower for lang in ['sql', 'python', 'typescript', 'javascript', 'schema']):
                return "code_or_schema"
        
        # Default
        return "generic_other"
    
    def should_index_by_default(self, source_type: str) -> bool:
        """
        Determine if source type should be indexed by default.
        
        Only high-signal types are indexed by default.
        """
        high_signal_types = {
            "implementation_explanation",
            "technical_decision",
            "code_or_schema",
            "generic_other",  # Keep unknown stuff by default for now
        }
        
        return source_type in high_signal_types
    
    def get_index_namespace(self, source_type: str) -> str:
        """
        Determine which index namespace this source type belongs to.
        
        Returns:
            "primary" for high-confidence implementation content
            "secondary" for everything else
        """
        primary_types = {
            "implementation_explanation",
            "technical_decision",
            "code_or_schema",
        }
        
        if source_type in primary_types:
            return "primary"
        else:
            return "secondary"
