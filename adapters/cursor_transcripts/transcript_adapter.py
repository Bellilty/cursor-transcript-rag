"""
Adapter for reading Cursor agent transcript files.

Maps Cursor's .jsonl transcript format to the core Message interface.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from conversation_rag.types import Message


class CursorTranscriptAdapter:
    """Adapter for reading Cursor agent transcripts."""
    
    def __init__(self, transcript_path: Path):
        """
        Initialize adapter.
        
        Args:
            transcript_path: Path to .jsonl transcript file
        """
        self.transcript_path = transcript_path
        
        # Extract conversation ID from path
        # Handles: agent-transcripts/<uuid>/<uuid>.jsonl
        if transcript_path.parent.name != "agent-transcripts":
            self.conversation_id = transcript_path.parent.name
        else:
            self.conversation_id = transcript_path.stem
        
        # Extract workspace name
        workspace_path = transcript_path.parent.parent.parent
        self.workspace_name = workspace_path.name if workspace_path.exists() else "unknown"
        
        # Get file modification time as fallback timestamp
        self.file_mtime = datetime.fromtimestamp(transcript_path.stat().st_mtime)
    
    def read_messages(self) -> List[Message]:
        """
        Read messages from the transcript.
        
        Returns:
            List of Message objects
        """
        if not self.transcript_path.exists():
            return []
        
        messages = []
        stats = {
            "total_lines": 0,
            "roles_found": {},
            "extracted": 0,
            "skipped_empty": 0,
            "skipped_error": 0,
        }
        
        try:
            with open(self.transcript_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    stats["total_lines"] += 1
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        entry = json.loads(line)
                        message = self._parse_entry(entry, line_num)
                        
                        if message:
                            messages.append(message)
                            stats["extracted"] += 1
                            role = message.role
                            stats["roles_found"][role] = stats["roles_found"].get(role, 0) + 1
                        else:
                            stats["skipped_empty"] += 1
                            
                    except json.JSONDecodeError as e:
                        stats["skipped_error"] += 1
                        continue
                    except Exception as e:
                        stats["skipped_error"] += 1
                        continue
        
        except Exception as e:
            print(f"  Error reading transcript {self.transcript_path}: {e}")
            return []
        
        # Report stats
        if stats["total_lines"] > 0:
            print(f"  File: {self.transcript_path.name}")
            print(f"    Lines: {stats['total_lines']}, "
                  f"Extracted: {stats['extracted']}, "
                  f"Skipped: {stats['skipped_empty'] + stats['skipped_error']}")
            if stats["roles_found"]:
                roles_str = ", ".join(f"{k}={v}" for k, v in stats["roles_found"].items())
                print(f"    Roles: {roles_str}")
        
        return messages
    
    def _parse_entry(self, entry: Dict[str, Any], line_num: int) -> Optional[Message]:
        """Parse a single transcript entry into a Message."""
        
        # Extract role from top level
        role = entry.get("role")
        if not role:
            return None
        
        # Map role names
        role_map = {
            "user": "user",
            "assistant": "assistant",
            "system": "system",
        }
        mapped_role = role_map.get(role, role)
        
        # Extract content from message.content array
        message_obj = entry.get("message", {})
        content_parts = message_obj.get("content", [])
        
        if not isinstance(content_parts, list):
            return None
        
        # Extract and join text parts
        raw_texts = []
        for part in content_parts:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text", "")
                if text:
                    raw_texts.append(text)
        
        if not raw_texts:
            return None
        
        raw_content = "\n".join(raw_texts)
        
        # Clean content for indexing
        cleaned_content = self._clean_content(raw_content)
        
        # Skip if content is empty after cleaning
        if not cleaned_content or not cleaned_content.strip():
            return None
        
        # Extract timestamp (fallback to file mtime + line offset)
        timestamp = self._extract_timestamp(entry, line_num)
        
        # Generate message ID
        message_id = f"{self.conversation_id}:{line_num}"
        
        # Build metadata
        metadata = {
            "workspace": self.workspace_name,
            "transcript_file": str(self.transcript_path),
            "line_number": line_num,
            "raw_role": role,
            "raw_content": raw_content[:500],  # Store first 500 chars of raw content
            "content_cleaned": True,
        }
        
        return Message(
            id=message_id,
            conversation_id=self.conversation_id,
            role=mapped_role,
            content=cleaned_content,
            timestamp=timestamp,
            metadata=metadata,
        )
    
    def _clean_content(self, text: str) -> str:
        """
        Clean Cursor wrapper noise from content.
        
        Removes:
        - <user_query>...</user_query> wrappers (keep inner text)
        - <attached_files>...</attached_files> (remove entirely)
        - <code_selection>...</code_selection> (remove entirely)
        - Extra whitespace
        """
        if not text:
            return ""
        
        # Remove <attached_files> blocks entirely
        text = re.sub(r'<attached_files>.*?</attached_files>', '', text, flags=re.DOTALL)
        
        # Remove <code_selection> blocks entirely
        text = re.sub(r'<code_selection[^>]*>.*?</code_selection>', '', text, flags=re.DOTALL)
        
        # Remove <terminal_selection> blocks entirely
        text = re.sub(r'<terminal_selection[^>]*>.*?</terminal_selection>', '', text, flags=re.DOTALL)
        
        # Extract content from <user_query> tags
        text = re.sub(r'<user_query>(.*?)</user_query>', r'\1', text, flags=re.DOTALL)
        
        # Remove other common wrapper tags but keep content
        text = re.sub(r'<open_and_recently_viewed_files>.*?</open_and_recently_viewed_files>', '', text, flags=re.DOTALL)
        text = re.sub(r'<agent_skills>.*?</agent_skills>', '', text, flags=re.DOTALL)
        text = re.sub(r'<system_reminder>.*?</system_reminder>', '', text, flags=re.DOTALL)
        
        # Normalize whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple newlines to double
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single
        text = text.strip()
        
        return text
    
    def _extract_timestamp(self, entry: Dict[str, Any], line_num: int) -> datetime:
        """
        Extract timestamp from entry or use fallback.
        
        Args:
            entry: JSON entry
            line_num: Line number in file
            
        Returns:
            datetime object
        """
        # Try to find timestamp in various locations
        timestamp_fields = [
            "timestamp",
            "created_at",
            "time",
            ["message", "timestamp"],
            ["message", "created_at"],
        ]
        
        for field in timestamp_fields:
            if isinstance(field, list):
                # Nested field
                value = entry
                for key in field:
                    if isinstance(value, dict):
                        value = value.get(key)
                    else:
                        value = None
                        break
            else:
                value = entry.get(field)
            
            if value:
                try:
                    # Try parsing ISO format
                    return datetime.fromisoformat(value.replace("Z", "+00:00"))
                except Exception:
                    pass
        
        # Fallback: file mtime + line offset (seconds)
        from datetime import timedelta
        return self.file_mtime + timedelta(seconds=line_num)

