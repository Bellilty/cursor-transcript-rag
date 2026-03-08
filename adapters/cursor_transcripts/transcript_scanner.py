"""
Scanner for discovering and reading Cursor transcript files.

Scans ~/.cursor/projects/.../agent-transcripts/ directories recursively.
"""

from pathlib import Path
from typing import List, Iterator
from .transcript_adapter import CursorTranscriptAdapter


class CursorTranscriptScanner:
    """Scanner for Cursor agent transcripts."""
    
    def __init__(self, cursor_projects_dir: Path, include_subagents: bool = False):
        """
        Initialize scanner.
        
        Args:
            cursor_projects_dir: Base directory for Cursor projects
                (typically ~/.cursor/projects)
            include_subagents: If True, include subagent transcripts
                (default False to avoid duplicates)
        """
        self.cursor_projects_dir = cursor_projects_dir
        self.include_subagents = include_subagents
    
    def find_transcript_files(self) -> List[Path]:
        """
        Find all transcript files recursively.
        
        Searches for:
        - agent-transcripts/*.jsonl
        - agent-transcripts/<uuid>/<uuid>.jsonl
        
        Excludes by default:
        - agent-transcripts/<uuid>/subagents/*.jsonl
        
        Returns:
            List of paths to .jsonl transcript files
        """
        if not self.cursor_projects_dir.exists():
            print(f"Warning: Cursor projects directory not found: {self.cursor_projects_dir}")
            return []
        
        transcript_files = []
        
        # Recursively find all .jsonl files under agent-transcripts directories
        for workspace_dir in self.cursor_projects_dir.iterdir():
            if not workspace_dir.is_dir():
                continue
            
            # Look for agent-transcripts directory
            transcript_dir = workspace_dir / "agent-transcripts"
            if not transcript_dir.exists():
                continue
            
            # Recursively find all .jsonl files
            for jsonl_file in transcript_dir.rglob("*.jsonl"):
                # Skip subagent files unless explicitly included
                if not self.include_subagents and "subagents" in jsonl_file.parts:
                    continue
                
                transcript_files.append(jsonl_file)
        
        return sorted(transcript_files)
    
    def iter_adapters(self) -> Iterator[CursorTranscriptAdapter]:
        """
        Iterate over transcript adapters.
        
        Yields:
            CursorTranscriptAdapter instances
        """
        for transcript_file in self.find_transcript_files():
            yield CursorTranscriptAdapter(transcript_file)
    
    def get_workspace_transcripts(self, workspace_name: str) -> List[Path]:
        """
        Get transcripts for a specific workspace.
        
        Args:
            workspace_name: Name of the workspace directory
            
        Returns:
            List of transcript paths for that workspace
        """
        workspace_dir = None
        
        for wdir in self.cursor_projects_dir.iterdir():
            if wdir.is_dir() and wdir.name == workspace_name:
                workspace_dir = wdir
                break
        
        if not workspace_dir:
            return []
        
        transcript_dir = workspace_dir / "agent-transcripts"
        if not transcript_dir.exists():
            return []
        
        # Recursively find all .jsonl files
        transcript_files = []
        for jsonl_file in transcript_dir.rglob("*.jsonl"):
            # Skip subagent files unless explicitly included
            if not self.include_subagents and "subagents" in jsonl_file.parts:
                continue
            
            transcript_files.append(jsonl_file)
        
        return sorted(transcript_files)
